"""
routing.py  —  Running Route Recommendation System  —  Team 11
==============================================================
Four FastAPI endpoints:
    GET /route/basic          — single path toward a target node
    GET /route/out-and-back   — go out, come back the same way
    GET /route/loop           — return via a different path
    GET /route/random         — exploratory wandering route

Install:
    pip install fastapi uvicorn osmnx networkx scipy pydantic

Run:
    uvicorn routing:app --reload
Then open http://127.0.0.1:8000/docs
"""
#interpreter base (3.11.7)
#testing data: lon: -75.1652, lat: 39.9526, dist: 5000

import math
import random

import networkx as nx
import osmnx as ox
from fastapi import FastAPI, HTTPException

# ---------------------------------------------------------------------------
# 1.  LOAD THE GRAPH
# ---------------------------------------------------------------------------

#local saved graph file for philly walking paths
GRAPH_PATH = "philadelphia_walk.graphml"

try: # "try loading the saved graph"
    G = ox.load_graphml(GRAPH_PATH)
    print(f"[startup] Graph loaded — {len(G.nodes):,} nodes, {len(G.edges):,} edges")
except FileNotFoundError: # "if the local saved graph does not exist, download philly walking data from OSM"
    print("[startup] Graph file not found — downloading from OSM (~30 s) …")
    G = ox.graph_from_place("Philadelphia, Pennsylvania, USA", network_type="walk")
    #save the graph locally, so future runs can reuse it
    ox.save_graphml(G, GRAPH_PATH)
    print(f"[startup] Graph saved to {GRAPH_PATH}")

# ---------------------------------------------------------------------------
# 2.  COST FUNCTION  (all values >= 0 so Dijkstra works correctly)
# ---------------------------------------------------------------------------

# if users don't input what they want, these are the defaults weights 
# the higher the number/weight, the higher the penalty 
# so these are the penalty weights for highway routes: 
HIGHWAY_PENALTY: dict[str, float] = {
    "motorway":      10.0,
    "trunk":          8.0,
    "primary":        5.0,
    "secondary":      3.0,
    "tertiary":       2.0,
    "residential":    0.5,
    "living_street":  0.2,
    "footway":        0.0,
    "path":           0.0,
    "track":          0.1,
    "cycleway":       0.2,
    "steps":          3.0,
    "unclassified":   1.0,
}

# these are the default weights for trail routes:
TRAIL_MULTIPLIER: dict[str, float] = {
    "footway":        0.4,
    "path":           0.4,
    "track":          0.5,
    "cycleway":       0.6,
    "residential":    0.9,
    "living_street":  0.85,
}

# these are the default weigthts for different surfaces
SURFACE_PENALTY: dict[str, float] = {
    "asphalt":  0.0,
    "paved":    0.0,
    "concrete": 0.05,
    "gravel":   0.2,
    "dirt":     0.15,
    "grass":    0.3,
    "unpaved":  0.2,
    "sand":     0.6,
}

# basically just reads one value from map data
def _get(edge_data: dict, key: str, default):
    """Safely get an edge attribute; handles osmnx list values."""
    val = edge_data.get(key, default) # reads a value from one edge of the map graph
    if isinstance(val, list): # OSM data sometimes stores values as lists, so use the first item
        val = val[0] if val else default
    return val if val is not None else default # if the value is missing, fall back to the default

# just calculates the cost of one edge
def edge_cost(u: int, v: int, edge_data: dict, prefs: dict) -> float:
    """Return a strictly positive cost for one directed edge."""
    # these are the user controlled weights that are inputted from the API page
    # higher values mean the user cares more about that factor
    w_road  = float(prefs.get("w_road",       1.0))
    w_trail = float(prefs.get("w_trail",      1.0))
    w_elev  = float(prefs.get("w_elevation",  1.0))
    w_pop   = float(prefs.get("w_popularity", 0.5))

    # base cost starts with edge length in meters
    length_m = max(float(_get(edge_data, "length", 1.0)), 0.1)

    hw = str(_get(edge_data, "highway", "unclassified")) # get current road type
    road_pen = HIGHWAY_PENALTY.get(hw, 1.5) # start with a penalty based on road type
    try: # if speed data exists, faster roads get more penalty
        speed = float(str(_get(edge_data, "maxspeed", "30")).split()[0])
        if speed >= 60:
            road_pen += 3.0
        elif speed >= 45:
            road_pen += 1.5
    except (ValueError, AttributeError): # if speed info is missing or messy, just skip it
        pass 

    # creates a surface adjustment factor so the route can favor nicer running paths when the user increases w_trail
    # basically just cost adjustment for surfaces
    trail_base = TRAIL_MULTIPLIER.get(hw, 1.0)
    surface = str(_get(edge_data, "surface", ""))
    trail_base += SURFACE_PENALTY.get(surface, 0.0)
    trail_factor = 1.0 + w_trail * (trail_base - 1.0)
    trail_factor = max(trail_factor, 0.1)

    # grade is the steepness of the segment
    grade = abs(float(_get(edge_data, "grade", 0.0)))
    # steeper edges get a bigger penalty
    # if slope is under 3%, it's basically flat
    # if the slope is between 3% and 6%, there's a small hill penalty
    # if the slope is 6%, there's a much bigger hill penalty
    if grade < 0.03:
        elev_pen = 0.0
    elif grade < 0.06:
        elev_pen = grade * 5.0
    else:
        elev_pen = grade * 15.0

    # this is the popularity cost, which we talked about removing, so i guess ignore this
    pop = min(float(_get(edge_data, "popularity", 0.0)), 10.0)
    pop_reduction = w_pop * pop * 0.02

    # combining all penalities into one number
    additive = max(w_road * road_pen + w_elev * elev_pen - pop_reduction, 0.0)

    # final edge cost used by the routing algorithm 
    # lower cost means the edge is more likely to be chosen
    return length_m * trail_factor * (1.0 + additive / 50.0)


def build_cost_graph(G: nx.MultiDiGraph, prefs: dict) -> nx.MultiDiGraph:
    """Copy of G with a 'cost' attribute on every edge."""
    H = G.copy() # make a copy so we don't overwrite the original graph
    # add a cost value to every edge on the map based on user preferences 
    for u, v, key, data in H.edges(keys=True, data=True):
        H[u][v][key]["cost"] = edge_cost(u, v, data, prefs)
    return H


# ---------------------------------------------------------------------------
# 3.  GRAPH UTILITIES
# ---------------------------------------------------------------------------

def snap_to_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    return ox.distance.nearest_nodes(G, lon, lat) # convert the user's starting coordinate to the nearest graph node


def node_latlon(G: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    # get the latitude and longitude for one graph node
    d = G.nodes[node]
    return d["y"], d["x"]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # approximate straight line distance between two coordinates in meters
    #* mathematically what this does is:
    # finds the angular seperation between two map points
    # uses sphere geometry to turn that into an angle
    # and then multiply by Earth's readius to get meters *#
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def path_length_m(G: nx.MultiDiGraph, path: list[int]) -> float:
    # add up the length of each edge in a route
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += min(d.get("length", 1.0) for d in G[u][v].values())
    return total


def path_elevation_gain_m(G: nx.MultiDiGraph, path: list[int]) -> float:
    # add up only uphill elevation gain along the route
    gain = 0.0
    for u, v in zip(path[:-1], path[1:]):
        eu = G.nodes[u].get("elevation", 0.0)
        ev = G.nodes[v].get("elevation", 0.0)
        if ev > eu:
            gain += ev - eu
    return gain

# convert graph nodes into [lat, lon] points for the JSON output
def path_to_coords(G: nx.MultiDiGraph, path: list[int]) -> list[list[float]]:
    return [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in path]


def dijkstra(H: nx.MultiDiGraph, src: int, dst: int) -> list[int] | None:
    # finds the lowest cost path from source to destination and cost is based on user preferences, not just distance
    try:
        return nx.shortest_path(H, src, dst, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # return none if no valid route can be found
        return None


def candidates_near_distance( # narrows down routes by finding nodes that are near the desired distance of the run
    G: nx.MultiDiGraph, # G represents the graph
    origin: int, # starting point
    target_m: float, # distance
    epsilon_m: float = 400, # allowed error range because its hard to find nodes that is the exact distance away
    max_n: int = 25, # if there are too many nodes that fit these requirements, just return the top 25 nodes
) -> list[int]:
    # find nodes whose straight line distance is close to a target value
    # this is used to find turnaround points or loop waypoints
    olat, olon = node_latlon(G, origin) 
    found = [] #stores matching candidate nodes
    for node in G.nodes: # loop through every node in the graph
        nlat, nlon = node_latlon(G, node)
        d = haversine_m(olat, olon, nlat, nlon) # straight line distance froom the origin node to the current node
        if abs(d - target_m) <= epsilon_m: # checks if this node is close enough to the desired distance
            found.append((node, abs(d - target_m)))
    found.sort(key=lambda x: x[1])
    return [n for n, _ in found[:max_n]]


# ---------------------------------------------------------------------------
# 4.  ROUTING STRATEGIES
# ---------------------------------------------------------------------------

# basically what this function does is it takes a start point and tries to build a one way route heading towards a point
def generate_basic_route(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    distance_m: float,
    prefs: dict,
) -> dict:
    """Single path from start toward a node roughly distance to the east."""
    H = build_cost_graph(G, prefs) # build graph where every edge has a custom running cost
    s = snap_to_node(G, start_lat, start_lon) # get a snapshot of the runners input coordinate to the nearest graph node

    lon_offset = (distance_m) / (111_320 * math.cos(math.radians(start_lat)))
    target = snap_to_node(G, start_lat, start_lon + lon_offset)

    # finds the best path from start to target using dijkstra's algo
    path = dijkstra(H, s, target)
    if not path:
        raise ValueError("No path found. Try a different start location or shorter distance.")

    # return route info as JSON friendly data
    return {
        "route_type":         "basic",
        "geometry":           path_to_coords(G, path),
        "distance_m":         round(path_length_m(G, path), 1),
        "elevation_gain_m":   round(path_elevation_gain_m(G, path), 1),
    }

# this is for the out and back route, so it finds the best halfway turnaround point, and then just reverses the path
def generate_out_and_back(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    distance_m: float,
    prefs: dict,
) -> dict:
    """Out-and-back: route to a turnaround node, then reverse."""
    H = build_cost_graph(G, prefs)
    s = snap_to_node(G, start_lat, start_lon)

    candidates = candidates_near_distance(G, s, distance_m / 2, epsilon_m=500, max_n=20)
    if not candidates:
        raise ValueError("No turnaround candidates found. Try a longer distance.")

    best_path, best_err = None, float("inf")
    for cand in candidates:
        p = dijkstra(H, s, cand)
        if not p:
            continue
        err = abs(path_length_m(G, p) * 2 - distance_m)
        if err < best_err:
            best_err, best_path = err, p

    if best_path is None:
        raise ValueError("Could not build an out-and-back route.")

    turnaround_coord = path_to_coords(G, best_path)[-1]
    full = best_path + list(reversed(best_path[:-1]))

    return {
        "route_type":         "out-and-back",
        "turnaround":         turnaround_coord,
        "geometry":           path_to_coords(G, full),
        "distance_m":         round(path_length_m(G, full), 1),
        "elevation_gain_m":   round(path_elevation_gain_m(G, full), 1),
    }

#* this is for the loop route, so it starts at the runners location, goes out to a waypoint, and 
# then comes back to the start using a different path if possible 
def generate_loop(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    distance_m: float,
    prefs: dict,
) -> dict:
    """Loop: start → waypoint → start via a DIFFERENT path."""
    H = build_cost_graph(G, prefs)
    s = snap_to_node(G, start_lat, start_lon)

    candidates = candidates_near_distance(G, s, distance_m / 2, epsilon_m=600, max_n=25)
    if not candidates:
        raise ValueError("No loop waypoint candidates found. Try a longer distance.")

    best_route, best_score = None, float("inf")
    for wp in candidates:
        path_out = dijkstra(H, s, wp) # calculates the first half of the loop, like going out to the turn around point
        if not path_out:
            continue

        H_ret = H.copy() #this line creates a copy of the first half of the loop
        out_edges = set(zip(path_out[:-1], path_out[1:])) # this keeps track of of the streets already used on the first half of the loop
        for u, v, key in H_ret.edges(keys=True): # loops through every edge in the return graph 
            if (u, v) in out_edges or (v, u) in out_edges: # checks whether that edge was already used in path_out
                H_ret[u][v][key]["cost"] *= 6 # if yes, it multiples that edge's cost by 6

        path_back = dijkstra(H_ret, wp, s) # computes the return route using dijkstas algo
        if not path_back:
            continue

        # after finding the return path, it checks how successful it was:
        full = path_out + path_back[1:]
        total_len = path_length_m(G, full)
        back_edges = set(zip(path_back[:-1], path_back[1:])) # edges used on the return trip
        overlap = len(out_edges & back_edges) / max(len(out_edges), 1) # fraction of the outgoing route that got reused (you want a lower overlap score)
        # after penalizing for the above things, then we calculate the score 
        score = abs(total_len - distance_m) / distance_m + overlap * 2 

        # so if the loop is better than the best one we've seen so far (because this is all in the for loop), 
        # save it as the new best loop
        if score < best_score:
            best_score, best_route = score, full

    if best_route is None:
        raise ValueError("Could not build a loop route.")
    #return the best loop we found in the for loop if its valid
    return {
        "route_type":         "loop",
        "geometry":           path_to_coords(G, best_route),
        "distance_m":         round(path_length_m(G, best_route), 1),
        "elevation_gain_m":   round(path_elevation_gain_m(G, best_route), 1),
    }

# this creates a random route by choosing nearby random waypoint locations, connecting them with the best paths, 
# and then trying to return back to the start while staying close to the requested total distance
def generate_random_explore(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    distance_m: float,
    prefs: dict,
    randomness_level: float = 0.5,
) -> dict:
    """Exploratory route through randomly sampled waypoints."""
    H = build_cost_graph(G, prefs)
    s = snap_to_node(G, start_lat, start_lon)
    s_lat, s_lon = node_latlon(G, s)

    # define how far away waypoint candidates are allowed to be, also higher randomness level means larger search radius
    radius_m = distance_m / 3 * (1 + randomness_level)
    tolerance = 0.15 # this allows the generated route to be within 15& of the requested distance\
    # (i think we should actually lower this value so that its more accurate to what the user wants)

    # take all graph nodes and shuffle them so wapoint selection starts random
    all_nodes = list(G.nodes)
    random.shuffle(all_nodes)

    # keep only nodes that fall within the allowed radius of the start point, these become candidate random waypoints 
    # the number of waypoint candidates increases as randomness level increases 
    waypoints = [
        n for n in all_nodes
        if haversine_m(s_lat, s_lon, *node_latlon(G, n)) <= radius_m
    ][:max(10, int(25 * randomness_level))]
    random.shuffle(waypoints) # shuffles again

    H_exp = H.copy()
    full_path = [s] # final route
    cumulative = 0.0 
    used_edges: set[tuple[int, int]] = set() # this keeps track of the edges so we don't repeat 
    current = s

    for wp in waypoints: # visits random waypoints one by one
        if wp == current: # skip if the waypoint is the same as the current node
            continue
        seg = dijkstra(H_exp, current, wp) # find the most direct path from the current node to this waypoint
        if not seg: # if this path doesn't exist, skip this waypoint
            continue
        seg_len = path_length_m(G, seg)
        if cumulative + seg_len > distance_m * (1 + tolerance): # makes sure the random route being generated doesn't go past the distance the user wants to run
            break

        for u, v in zip(seg[:-1], seg[1:]): # go through each edge in the segment
            key = (min(u, v), max(u, v)) 
            if key in used_edges: # basically if we have already used this edge, increase its cost so the algorithm wont repeat it again
                for k in H_exp[u][v]:
                    H_exp[u][v][k]["cost"] *= 4
            used_edges.add(key) 

        full_path += seg[1:] # add the segment to the final route, skipping the first node so we dont duplicate the connection point
        cumulative += seg_len
        current = wp

        # if we are close enough to the target distance the user wants to run, then stop, if not keep looping
        if abs(cumulative - distance_m) / distance_m <= tolerance:
            break

    # after the loop, if we are not close to back where the runner started, then figure out the path from the current location 
    # to go back to the starting point
    if current != s:
        home = dijkstra(H_exp, current, s)
        if home:
            full_path += home[1:]

    return { # return the route
        "route_type":         "random",
        "geometry":           path_to_coords(G, full_path),
        "distance_m":         round(path_length_m(G, full_path), 1),
        "elevation_gain_m":   round(path_elevation_gain_m(G, full_path), 1),
    }


# ---------------------------------------------------------------------------
# 5.  FASTAPI APP
# ---------------------------------------------------------------------------

# This entire chunk of code basically makes the API page, defines the API endpoints, and runs the server: 
app = FastAPI(
    title="Running Route Recommendation API",
    version="0.3.0",
    description="Team 11 — DS440 Capstone",
)

SHARED_DESC = """
**distance** is in metres (e.g. 5000 = 5 km).

**Philadelphia test coords:** start_lat=39.9526, start_lon=**-75.1652** (negative!)

Preference weights (all optional, default 1.0):
- **w_road** — higher avoids busy/fast roads more strongly
- **w_trail** — higher prefers trails & footpaths more strongly
- **w_elevation** — higher avoids steep hills more strongly
- **w_popularity** — higher prefers Strava-popular segments (coming soon)
"""

# Endpoint: generates a basic one way route
@app.get("/route/basic", summary="Single-path route", description=SHARED_DESC)
def basic_route(
    start_lat: float,
    start_lon: float,
    distance: float,
    w_road: float = 1.0,
    w_trail: float = 1.0,
    w_elevation: float = 1.0,
    w_popularity: float = 0.5,
):
    prefs = dict(w_road=w_road, w_trail=w_trail, w_elevation=w_elevation, w_popularity=w_popularity) # puts the reference values into one dictionary
    try:
        return generate_basic_route(G, start_lat, start_lon, distance, prefs) # returns the route asked for
    except ValueError as e:
        raise HTTPException(400, str(e)) # an error if it doesn't work just so the whole thing doesnt crash

# Endpoint: generates an out and back route
@app.get("/route/out-and-back", summary="Out-and-back route", description=SHARED_DESC)
def out_and_back(
    start_lat: float,
    start_lon: float,
    distance: float,
    w_road: float = 1.0,
    w_trail: float = 1.0,
    w_elevation: float = 1.0,
    w_popularity: float = 0.5,
):
    prefs = dict(w_road=w_road, w_trail=w_trail, w_elevation=w_elevation, w_popularity=w_popularity)
    try:
        return generate_out_and_back(G, start_lat, start_lon, distance, prefs)
    except ValueError as e:
        raise HTTPException(400, str(e))

# Endpoint: generates a loop route
@app.get("/route/loop", summary="Loop route", description=SHARED_DESC)
def loop_route(
    start_lat: float,
    start_lon: float,
    distance: float,
    w_road: float = 1.0,
    w_trail: float = 1.0,
    w_elevation: float = 1.0,
    w_popularity: float = 0.5,
):
    prefs = dict(w_road=w_road, w_trail=w_trail, w_elevation=w_elevation, w_popularity=w_popularity)
    try:
        return generate_loop(G, start_lat, start_lon, distance, prefs)
    except ValueError as e:
        raise HTTPException(400, str(e))

# Endpoint: generates a random route
@app.get("/route/random", summary="Random exploratory route", description=SHARED_DESC)
def random_route(
    start_lat: float,
    start_lon: float,
    distance: float,
    randomness_level: float = 0.5,
    w_road: float = 1.0,
    w_trail: float = 1.0,
    w_elevation: float = 1.0,
    w_popularity: float = 0.5,
    prefer_trails: bool = False,
):
    prefs = dict(
        w_road=w_road, w_trail=w_trail, w_elevation=w_elevation,
        w_popularity=w_popularity, prefer_trails=prefer_trails,
    )
    try:
        return generate_random_explore(G, start_lat, start_lon, distance, prefs, randomness_level)
    except ValueError as e:
        raise HTTPException(400, str(e))

# starts the FastAPI server locally:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("routing:app", host="0.0.0.0", port=8000, reload=True)