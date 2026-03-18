"""
routing.py  —  Running Route Recommendation System  —  Team 11
==============================================================
Four FastAPI endpoints:
    GET /route/basic          — single path toward a target node
    GET /route/out-and-back   — go out, come back the same way
    GET /route/loop           — return via a different path
    GET /route/random         — exploratory wandering route

Scoring functions (road_penalty, trail_bonus, elevation_cost, safety_score)
taken from route_recommender_draft.py.

Install:
    pip install fastapi uvicorn osmnx networkx scipy pydantic pandas numpy

Run:
    uvicorn routing:app --reload
Then open http://127.0.0.1:8000/docs
"""
# interpreter: base (3.11.7)
# test coords: lon=-75.1652, lat=39.9526, dist=5000

import json
import math
import random

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

# ---------------------------------------------------------------------------
# 1.  LOAD THE GRAPH
# ---------------------------------------------------------------------------

# local saved graph file for philly walking paths
GRAPH_PATH = "philadelphia_walk.graphml"

try:  # try loading the saved graph
    G = ox.load_graphml(GRAPH_PATH)
    print(f"[startup] Graph loaded — {len(G.nodes):,} nodes, {len(G.edges):,} edges")
except FileNotFoundError:  # if the local saved graph does not exist, download philly walking data from OSM
    print("[startup] Graph file not found — downloading from OSM (~30 s) …")
    G = ox.graph_from_place("Philadelphia, Pennsylvania, USA", network_type="walk")
    ox.save_graphml(G, GRAPH_PATH)  # save the graph locally so future runs can reuse it
    print(f"[startup] Graph saved to {GRAPH_PATH}")

# compute mean edge length at startup — used to normalize road_penalty and trail_bonus scores
mean_edge_length = sum(
    d.get("length", 0) for u, v, d in G.edges(data=True)
) / max(G.number_of_edges(), 1)
print(f"[startup] Mean edge length: {mean_edge_length:.2f} m")

# Check if grade data already exists
sample_edge = next(iter(G.edges(data=True)))
has_grades = "grade" in sample_edge[2]

if not has_grades:
    print("[startup] Grade data missing — fetching elevation and computing grades...")
    # --- Fetch elevation for every node ---
    OPEN_ELEVATION_URL = 'https://api.open-elevation.com/api/v1/lookup'
    node_coords = [{"latitude": d["y"], "longitude": d["x"], "osmid": n}
               for n, d in G.nodes(data=True)]
    batch_size = 5000
    fetch_success = True
    for i in range(0, len(node_coords), batch_size):
        batch = node_coords[i:i + batch_size]
        payload = json.dumps({"locations": [{"latitude": b["latitude"], "longitude": b["longitude"]} for b in batch]})
        try:
            response = requests.post(OPEN_ELEVATION_URL, data=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            for j, result in enumerate(response.json()["results"]):
                G.nodes[batch[j]["osmid"]]["elevation"] = result["elevation"]
            print(f"[startup] Fetched elevation for {i + len(batch)} nodes...")
        except Exception as e:
            print(f"[startup] Elevation fetch failed: {e}")
            fetch_success = False
            break

    if fetch_success:
        G = ox.elevation.add_edge_grades(G)
        ox.save_graphml(G, GRAPH_PATH)
        print("[startup] Graph re-saved with elevation and grade data.")
    else:
        print("[startup] WARNING: Elevation fetch incomplete — grade data may be partial.")
else:
    print("[startup] Grade data found — skipping elevation fetch.")


# ---------------------------------------------------------------------------
# 2.  SCORING FUNCTIONS  (from route_recommender_draft.py)
# ---------------------------------------------------------------------------

# road penalty:
def road_penalty(road_category, edge_length, normalization_length):
    """Penalizes edges based on how busy/fast the road type is."""
    high_traffic   = ['trunk', 'trunk_link', 'primary', 'primary_link']
    medium_traffic = ['secondary', 'secondary_link', 'tertiary', 'tertiary_link']
    low_traffic    = ['residential', 'service']
    no_traffic     = ['pedestrian', 'path', 'track', 'corridor', 'living_street', 'steps']
    base_score = 0
    if road_category in high_traffic:
        base_score = 1.0
    elif road_category in medium_traffic:
        base_score = 0.7
    elif road_category in low_traffic:
        base_score = 0.3
    elif road_category in no_traffic:
        base_score = 0.0
    return base_score * (edge_length / normalization_length)


# trail bonus:
def get_trail_alpha(trail_preference):
    """Converts a trail preference string into a numeric multiplier for trail_bonus."""
    if trail_preference == "none":
        return 0.0
    elif trail_preference == "normal":
        return 1.0
    elif trail_preference == "high":
        return 2.0
    return 1.0  # default to normal if unrecognized value


def trail_bonus(type_category, edge_length, normalization_length, alpha):
    """Returns a negative cost for trail edges (lower cost = more preferred)."""
    if type_category == 'trail':
        return -alpha * 1.0 * (edge_length / normalization_length)
    return 0.0


# elevation cost
def elevation_cost(grade, beta1, beta2, edge_length, threshold):
    """
    Two-tier elevation penalty:
      - Gentle slopes (grade <= threshold): small linear penalty (beta1)
      - Steep slopes  (grade >  threshold): larger quadratic penalty (beta2)
    """
    if grade <= 0:
        return 0
    elif grade <= threshold:
        return beta1 * grade * edge_length
    return beta2 * (grade ** 2) * edge_length


# maxspeed parser (used by safety_score):
def _parse_maxspeed(maxspeed_value):
    """Parses raw OSM maxspeed values into a numeric km/h float."""
    if isinstance(maxspeed_value, list):
        maxspeed_value = maxspeed_value[0] if maxspeed_value else None
    if isinstance(maxspeed_value, str):
        if "mph" in maxspeed_value:
            try:
                return float(maxspeed_value.replace(" mph", "").strip()) * 1.60934
            except ValueError:
                return np.nan
        else:
            try:
                return float(maxspeed_value)
            except ValueError:
                return np.nan
    return np.nan


# safety score: 
def safety_score(edge, alpha_speed=0.02, alpha_sidewalk=2.0, alpha_lanes=0.1):
    """
    Returns a non-negative safety cost (higher = less safe).
    Considers: road speed, presence of sidewalks, and number of lanes.
    """
    length       = edge.get("length", 0)
    highway      = edge.get("highway", "")
    maxspeed_raw = edge.get("maxspeed", None)
    sidewalk     = edge.get("sidewalk", "no")
    lanes        = edge.get("lanes", None)

    # ensure highway is always a list for consistent handling
    if isinstance(highway, str):
        highway = [highway]
    elif not isinstance(highway, list):
        highway = []

    pedestrian_highways = {"footway", "path", "pedestrian", "steps", "track", "living_street"}
    speed_val_kmh = _parse_maxspeed(maxspeed_raw)

    if pd.isna(speed_val_kmh):
        # if speed is unknown, assume low speed for pedestrian ways and high speed for vehicle roads
        speed_val = 5.0 if any(h in pedestrian_highways for h in highway) else 70.0
    else:
        speed_val = speed_val_kmh
    speed_val    = max(0.0, speed_val)
    speed_factor = alpha_speed * speed_val

    # penalize roads without sidewalks
    if isinstance(sidewalk, str):
        sidewalk = [sidewalk]
    elif not isinstance(sidewalk, list):
        sidewalk = []
    sidewalk_factor = 0.2 if any(s in {"both", "left", "right"} for s in sidewalk) else 1.0

    # penalize roads with more lanes
    lane_factor = 1.0
    if isinstance(lanes, list):
        lanes = lanes[0] if lanes else None
    if lanes is not None and not pd.isna(lanes):
        try:
            num_lanes = int(float(lanes))
            if num_lanes > 1:
                lane_factor = 1 + alpha_lanes * (num_lanes - 1)
        except ValueError:
            pass

    base_risk = speed_factor * (alpha_sidewalk * sidewalk_factor) * lane_factor
    return base_risk * (length / 100.0)  # scale by 100m units


# combined edge cost:
def edge_cost(edge_data: dict, prefs: dict) -> float:
    """
    Combines all scoring components into a single edge cost using the formula:

        cost = w_length * length
             + w_elev   * elevation_cost
             + w_safe   * safety_score
             + w_road   * road_penalty
             + w_trail  * trail_bonus      ← negative for trail edges

    Clamped to >= 0.01 so Dijkstra always receives positive weights.
    """
    # user-controllable weights (defaults from route_recommender_draft.py)
    w_length = float(prefs.get("w_length", 1.0))
    w_elev   = float(prefs.get("w_elev",   0.7))
    w_safe   = float(prefs.get("w_safe",   1.2))
    w_road   = float(prefs.get("w_road",   0.8))
    w_trail  = float(prefs.get("w_trail",  0.8))

    # trail preference controls the strength of the trail bonus
    alpha = get_trail_alpha(prefs.get("trail_preference", "normal"))

    # elevation tuning parameters
    beta1     = float(prefs.get("beta1",     0.5))
    beta2     = float(prefs.get("beta2",     2.0))
    threshold = float(prefs.get("threshold", 0.05))

    length = max(float(edge_data.get("length", 1.0)), 0.1)

    # get the highway type, handling OSM list values
    hw = edge_data.get("highway", "unclassified")
    if isinstance(hw, list):
        hw = hw[0] if hw else "unclassified"
    hw = str(hw)

    # classify as trail or road for trail_bonus
    trail_types = {'footway', 'path', 'cycleway', 'steps', 'pedestrian', 'track', 'bridleway'}
    type_cat = 'trail' if hw in trail_types else 'road'

    # compute each component
    grade = float(edge_data.get("grade", 0.0))
    r_pen   = road_penalty(hw, length, mean_edge_length)
    t_bonus = trail_bonus(type_cat, length, mean_edge_length, alpha)
    e_cost  = elevation_cost(grade, beta1, beta2, length, threshold)
    s_score = safety_score(edge_data)

    # combine and clamp — w_length * length (~100m) dominates, so cost is almost always positive
    cost = (
        w_length * length
        + w_elev * e_cost
        + w_safe * s_score
        + w_road * r_pen
        + w_trail * t_bonus
    )
    return max(cost, 0.01)


def build_cost_graph(G: nx.MultiDiGraph, prefs: dict) -> nx.MultiDiGraph:
    """Returns a copy of G with a 'cost' attribute on every edge."""
    H = G.copy()  # make a copy so we don't overwrite the original graph
    for u, v, key, data in H.edges(keys=True, data=True):
        H[u][v][key]["cost"] = edge_cost(data, prefs)
    return H


# ---------------------------------------------------------------------------
# 3.  GRAPH UTILITIES
# ---------------------------------------------------------------------------
def snap_to_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Converts a lat/lon coordinate to the nearest graph node."""
    return ox.distance.nearest_nodes(G, lon, lat)


def node_latlon(G: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    """Returns the (lat, lon) of a graph node."""
    d = G.nodes[node]
    return d["y"], d["x"]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in metres between two lat/lon points using sphere geometry."""
    R  = 6_371_000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def path_length_m(G: nx.MultiDiGraph, path: list[int]) -> float:
    """Adds up the length of each edge in a route."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += min(d.get("length", 1.0) for d in G[u][v].values())
    return total


def path_elevation_gain_m(G: nx.MultiDiGraph, path: list[int]) -> float:
    """Adds up only the uphill elevation gain along a route (requires 'elevation' node attribute)."""
    gain = 0.0
    for u, v in zip(path[:-1], path[1:]):
        eu = G.nodes[u].get("elevation", 0.0)
        ev = G.nodes[v].get("elevation", 0.0)
        if ev > eu:
            gain += ev - eu
    return gain


def path_to_coords(G: nx.MultiDiGraph, path: list[int]) -> list[list[float]]:
    """Converts graph node IDs into [lat, lon] pairs for JSON output."""
    return [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in path]


def dijkstra(H: nx.MultiDiGraph, src: int, dst: int) -> list[int] | None:
    """Finds the lowest-cost path from src to dst. Returns None if no path exists."""
    try:
        return nx.shortest_path(H, src, dst, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def candidates_near_distance(
    G: nx.MultiDiGraph,
    origin: int, # starting node
    target_m: float, # desired straight-line distance from origin
    epsilon_m: float = 400, # allowed error margin (hard to find nodes at an exact distance)
    max_n: int = 25, # cap candidates to avoid unnecessary computation
) -> list[int]:
    """Returns nodes whose straight-line distance from origin is close to target_m."""
    olat, olon = node_latlon(G, origin)
    found = []
    for node in G.nodes:
        nlat, nlon = node_latlon(G, node)
        d = haversine_m(olat, olon, nlat, nlon)
        if abs(d - target_m) <= epsilon_m:
            found.append((node, abs(d - target_m)))
    found.sort(key=lambda x: x[1])
    return [n for n, _ in found[:max_n]]


# ---------------------------------------------------------------------------
# 4.  ROUTING STRATEGIES
# ---------------------------------------------------------------------------
def generate_basic_route(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    distance_m: float,
    prefs: dict,
) -> dict:
    """Single one-way path heading east toward a node roughly distance_m away."""
    H = build_cost_graph(G, prefs)
    s = snap_to_node(G, start_lat, start_lon)

    # shift longitude east by ~distance_m to pick a target direction
    lon_offset = distance_m / (111_320 * math.cos(math.radians(start_lat)))
    target = snap_to_node(G, start_lat, start_lon + lon_offset)

    path = dijkstra(H, s, target)
    if not path:
        raise ValueError("No path found. Try a different start location or shorter distance.")

    return {
        "route_type":       "basic",
        "geometry":         path_to_coords(G, path),
        "distance_m":       round(path_length_m(G, path), 1),
        "elevation_gain_m": round(path_elevation_gain_m(G, path), 1),
    }


def generate_out_and_back(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    distance_m: float,
    prefs: dict,
) -> dict:
    """Finds the best turnaround point at ~distance/2 away, routes there, then reverses."""
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
        "route_type":       "out-and-back",
        "turnaround":       turnaround_coord,
        "geometry":         path_to_coords(G, full),
        "distance_m":       round(path_length_m(G, full), 1),
        "elevation_gain_m": round(path_elevation_gain_m(G, full), 1),
    }


def generate_loop(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    distance_m: float,
    prefs: dict,
) -> dict:
    """
    Starts at the runner's location, goes out to a waypoint, and comes back
    via a different path. Edges used on the outward leg are penalized 6x on
    the return trip to discourage retracing the same streets.
    """
    H = build_cost_graph(G, prefs)
    s = snap_to_node(G, start_lat, start_lon)

    candidates = candidates_near_distance(G, s, distance_m / 2, epsilon_m=600, max_n=25)
    if not candidates:
        raise ValueError("No loop waypoint candidates found. Try a longer distance.")

    best_route, best_score = None, float("inf")
    for wp in candidates:
        path_out = dijkstra(H, s, wp)  # first half of the loop
        if not path_out:
            continue

        # build return graph with heavy penalties on already-used edges
        H_ret     = H.copy()
        out_edges = set(zip(path_out[:-1], path_out[1:]))  # streets used on the way out
        for u, v, key in H_ret.edges(keys=True):
            if (u, v) in out_edges or (v, u) in out_edges:
                H_ret[u][v][key]["cost"] *= 6  # heavily discourage reuse

        path_back = dijkstra(H_ret, wp, s)  # return route via different streets
        if not path_back:
            continue

        full       = path_out + path_back[1:]
        total_len  = path_length_m(G, full)
        back_edges = set(zip(path_back[:-1], path_back[1:]))
        overlap    = len(out_edges & back_edges) / max(len(out_edges), 1)  # fraction of streets reused

        # score: penalize distance error and street overlap
        score = abs(total_len - distance_m) / distance_m + overlap * 2
        if score < best_score:
            best_score, best_route = score, full

    if best_route is None:
        raise ValueError("Could not build a loop route.")

    return {
        "route_type":       "loop",
        "geometry":         path_to_coords(G, best_route),
        "distance_m":       round(path_length_m(G, best_route), 1),
        "elevation_gain_m": round(path_elevation_gain_m(G, best_route), 1),
    }


def generate_random_explore(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    distance_m: float,
    prefs: dict,
    randomness_level: float = 0.5,
) -> dict:
    """
    Creates an exploratory route by choosing random nearby waypoints, connecting
    them with the best paths, and returning to the start near the target distance.
    Higher randomness_level = wider search radius and more waypoints.
    """
    H = build_cost_graph(G, prefs)
    s = snap_to_node(G, start_lat, start_lon)
    s_lat, s_lon = node_latlon(G, s)

    # higher randomness = larger search radius for waypoints
    radius_m  = distance_m / 3 * (1 + randomness_level)
    tolerance = 0.15  # accept routes within ±15% of the requested distance

    # shuffle all nodes and keep only those within the allowed radius
    all_nodes = list(G.nodes)
    random.shuffle(all_nodes)
    waypoints = [
        n for n in all_nodes
        if haversine_m(s_lat, s_lon, *node_latlon(G, n)) <= radius_m
    ][:max(10, int(25 * randomness_level))]
    random.shuffle(waypoints)

    H_exp      = H.copy()
    full_path  = [s]
    cumulative = 0.0
    used_edges: set[tuple[int, int]] = set()  # tracks visited edges to avoid repetition
    current    = s

    for wp in waypoints:
        if wp == current:
            continue
        seg = dijkstra(H_exp, current, wp)  # best path to the next waypoint
        if not seg:
            continue
        seg_len = path_length_m(G, seg)
        if cumulative + seg_len > distance_m * (1 + tolerance):
            break  # adding this segment would overshoot the target distance

        # penalize already-used edges so the route keeps exploring new streets
        for u, v in zip(seg[:-1], seg[1:]):
            key = (min(u, v), max(u, v))
            if key in used_edges:
                for k in H_exp[u][v]:
                    H_exp[u][v][k]["cost"] *= 4
            used_edges.add(key)

        full_path  += seg[1:]  # skip the first node to avoid duplicating the connection point
        cumulative += seg_len
        current     = wp

        if abs(cumulative - distance_m) / distance_m <= tolerance:
            break  # close enough to target distance, stop adding waypoints

    # route back to start if we haven't returned yet
    if current != s:
        home = dijkstra(H_exp, current, s)
        if home:
            full_path += home[1:]

    return {
        "route_type":       "random",
        "geometry":         path_to_coords(G, full_path),
        "distance_m":       round(path_length_m(G, full_path), 1),
        "elevation_gain_m": round(path_elevation_gain_m(G, full_path), 1),
    }


# ---------------------------------------------------------------------------
# 5.  FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Running Route Recommendation API",
    version="0.4.0",
    description="Team 11 — DS440 Capstone",
)

SHARED_DESC = """
**distance** is in metres (e.g. 5000 = 5 km).

**Philadelphia test coords:** start_lat=39.9526, start_lon=**-75.1652** (negative!)

**Preference weights** (all optional — defaults from route_recommender_draft.py):
- **w_length** (1.0) — scales raw edge length; keeps routes reasonably direct
- **w_elev** (0.7)   — avoids big climbs, but not at all costs
- **w_safe** (1.2)   — prioritizes safety (speed limits, sidewalks, lane count)
- **w_road** (0.8)   — avoids car-heavy roads
- **w_trail** (0.8)  — prefers trails and parks (trail edges get a negative bonus)
- **trail_preference** — "none" | "normal" | "high" (controls trail bonus strength)
"""


def _build_prefs(
    w_length: float, w_elev: float, w_safe: float,
    w_road: float, w_trail: float, trail_preference: str,
) -> dict:
    """Packages API query params into the prefs dict used by all routing functions."""
    return dict(
        w_length=w_length, w_elev=w_elev, w_safe=w_safe,
        w_road=w_road, w_trail=w_trail, trail_preference=trail_preference,
    )


@app.get("/route/basic", summary="Single-path route", description=SHARED_DESC)
def basic_route(
    start_lat: float, start_lon: float, distance: float,
    w_length: float = 1.0, w_elev: float = 0.7, w_safe: float = 1.2,
    w_road: float = 0.8, w_trail: float = 0.8, trail_preference: str = "normal",
):
    prefs = _build_prefs(w_length, w_elev, w_safe, w_road, w_trail, trail_preference)
    try:
        return generate_basic_route(G, start_lat, start_lon, distance, prefs)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/route/out-and-back", summary="Out-and-back route", description=SHARED_DESC)
def out_and_back(
    start_lat: float, start_lon: float, distance: float,
    w_length: float = 1.0, w_elev: float = 0.7, w_safe: float = 1.2,
    w_road: float = 0.8, w_trail: float = 0.8, trail_preference: str = "normal",
):
    prefs = _build_prefs(w_length, w_elev, w_safe, w_road, w_trail, trail_preference)
    try:
        return generate_out_and_back(G, start_lat, start_lon, distance, prefs)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/route/loop", summary="Loop route", description=SHARED_DESC)
def loop_route(
    start_lat: float, start_lon: float, distance: float,
    w_length: float = 1.0, w_elev: float = 0.7, w_safe: float = 1.2,
    w_road: float = 0.8, w_trail: float = 0.8, trail_preference: str = "normal",
):
    prefs = _build_prefs(w_length, w_elev, w_safe, w_road, w_trail, trail_preference)
    try:
        return generate_loop(G, start_lat, start_lon, distance, prefs)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/route/random", summary="Random exploratory route", description=SHARED_DESC)
def random_route(
    start_lat: float, start_lon: float, distance: float,
    randomness_level: float = 0.5,
    w_length: float = 1.0, w_elev: float = 0.7, w_safe: float = 1.2,
    w_road: float = 0.8, w_trail: float = 0.8, trail_preference: str = "normal",
):
    prefs = _build_prefs(w_length, w_elev, w_safe, w_road, w_trail, trail_preference)
    try:
        return generate_random_explore(G, start_lat, start_lon, distance, prefs, randomness_level)
    except ValueError as e:
        raise HTTPException(400, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("routing:app", host="0.0.0.0", port=8000, reload=True)