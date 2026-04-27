"""
ablation_test.py — Weight Sensitivity Ablation Test
=====================================================
Tests whether each preference weight (w_elev, w_safe, w_road, w_trail)
meaningfully affects its corresponding route metric by varying one weight
at a time while holding all others fixed at neutral (1.0).

Run from the same directory as routing.py:
    python ablation_test.py

Outputs (one file per weight):
    ablation_w_elev.json
    ablation_w_safe.json
    ablation_w_road.json
    ablation_w_trail.json
"""

import json
import statistics
from datetime import datetime

from routing import (
    G,
    mi_to_m,
    _build_prefs,
    generate_loop,
    generate_out_and_back,
    generate_random_explore,
    generate_point_to_point,
    M_PER_MILE,
)

# ---------------------------------------------------------------------------
# SHARED CONFIG  (same locations used for every weight's ablation)
# ---------------------------------------------------------------------------

WEIGHT_VALUES    = [0.0, 0.5, 1.0, 1.5, 2.0]   # tested for whichever weight is active
NEUTRAL          = 1.0                           # fixed value for all inactive weights
TEST_DISTANCE_MI = 3.0

FIXED_START_POINTS = [
    (39.9526, -75.1652),   # Center City
    (40.0379, -75.1330),   # Chestnut Hill (hilly)
    (39.9200, -75.1900),   # Southwest Philly (flat)
    
    (40.0150, -75.1780),   # Germantown
    (39.9700, -75.1300),   # Fishtown
    (39.9900, -75.2200),   # Roxborough (hilly)
    (39.9400, -75.1600),   # South Philly (flat)
    (40.0600, -75.1000),   # Far Northeast
    (39.9800, -75.1500),   # North Philly
    (39.9550, -75.2000),   # West Philly
    (39.9480, -75.1800),   # University City
    (40.0250, -75.1600),   # Oak Lane
    (39.9650, -75.2300),   # Overbrook (hilly)
    (40.0450, -75.1150),   # Somerton
    (39.9350, -75.1750),   # Grays Ferry
    (40.0050, -75.1050),   # Holmesburg
    (39.9750, -75.2500),   # Manayunk (hilly)
    (39.9600, -75.1450),   # Northern Liberties
    (40.0300, -75.2000),   # East Falls (hilly)
    (39.9450, -75.1350),   # Pennsport
    
]


P2P_PAIRS = [
    # Pairs where straight-line distance is well under 2 miles
    (39.9526, -75.1652,  39.9600, -75.1450,  "CenterCity->NorthernLiberties"),
    (39.9526, -75.1652,  39.9480, -75.1800,  "CenterCity->UniversityCity"),
    (39.9480, -75.1800,  39.9400, -75.1600,  "UniversityCity->SPhilly"),
    (39.9700, -75.1300,  39.9600, -75.1450,  "Fishtown->NorthernLiberties"),
    (39.9800, -75.1500,  39.9700, -75.1300,  "NPhilly->Fishtown"),
    (39.9550, -75.2000,  39.9480, -75.1800,  "WPhilly->UniversityCity"),
    (39.9350, -75.1750,  39.9450, -75.1350,  "GraysFerry->Pennsport"),
    (39.9450, -75.1350,  39.9526, -75.1652,  "Pennsport->CenterCity"),
    (40.0150, -75.1780,  39.9900, -75.2200,  "Germantown->Roxborough"),
    (40.0250, -75.1600,  40.0150, -75.1780,  "OakLane->Germantown"),
    (40.0300, -75.2000,  39.9900, -75.2200,  "EastFalls->Roxborough"),
    (39.9900, -75.2200,  39.9750, -75.2500,  "Roxborough->Manayunk"),
]


ROUTE_GENERATORS = {
    "loop":         generate_loop,
    "out-and-back": generate_out_and_back,
    "random":       generate_random_explore,
}

# ---------------------------------------------------------------------------
# ABLATION PLAN
# Each entry defines which weight varies and what metric to watch.
# ---------------------------------------------------------------------------

ABLATION_PLAN = [
    {
        "weight_name":   "w_elev",
        "target_metric": "elevation_ft",
        "output_file":   "ablation_w_elev.json",
        "description":   "Elevation weight — expect lower elevation gain as w_elev increases",
    },
    
    {
        "weight_name":   "w_safe",
        "target_metric": "safety_score",
        "output_file":   "ablation_w_safe.json",
        "description":   "Safety weight — expect higher safety score as w_safe increases",
    },
    {
        "weight_name":   "w_road",
        "target_metric": "road_avoidance_score",
        "output_file":   "ablation_w_road.json",
        "description":   "Road weight — expect higher road avoidance score as w_road increases",
    },
    {
        "weight_name":   "w_trail",
        "target_metric": "trail_pct",
        "output_file":   "ablation_w_trail.json",
        "description":   "Trail weight — expect higher trail % as w_trail increases",
    },
    
]

ABLATION_PLAN = ABLATION_PLAN[1:]
# ---------------------------------------------------------------------------
# HELPER — run all routes for one weight value and collect results
# ---------------------------------------------------------------------------

def run_one_weight(weight_name: str, weight_value: float) -> list:
    """
    Runs all start points and p2p pairs for a single weight value.
    All other weights are held at NEUTRAL (1.0).
    Returns a list of result dicts.
    """
    w_elev  = weight_value if weight_name == "w_elev"  else NEUTRAL
    w_safe  = weight_value if weight_name == "w_safe"  else NEUTRAL
    w_road  = weight_value if weight_name == "w_road"  else NEUTRAL
    w_trail = weight_value if weight_name == "w_trail" else NEUTRAL

    prefs = _build_prefs(
        w_elev=w_elev,
        w_safe=w_safe,
        w_road=w_road,
        w_trail=w_trail,
    )

    results = []

    # ── Single-start routes ─────────────────────────────────────────────────
    for start_lat, start_lon in FIXED_START_POINTS:
        for route_type, generator in ROUTE_GENERATORS.items():
            try:
                result = generator(
                    G,
                    start_lat, start_lon,
                    mi_to_m(TEST_DISTANCE_MI),
                    prefs,
                )
                routes = result.get("routes", [])
                if not routes:
                    continue

                route = routes[0]
                stats = route.get("stats", {})
                results.append({
                    "timestamp":            datetime.now().isoformat(),
                    "route_type":           route_type,
                    "start_lat":            start_lat,
                    "start_lon":            start_lon,
                    "end_lat":              None,
                    "end_lon":              None,
                    "pair_label":           None,
                    "w_elev":               w_elev,
                    "w_safe":               w_safe,
                    "w_road":               w_road,
                    "w_trail":              w_trail,
                    "requested_mi":         TEST_DISTANCE_MI,
                    "actual_mi":            route["distance_mi"],
                    "dist_error_pct":       round(
                        abs(route["distance_mi"] - TEST_DISTANCE_MI) / TEST_DISTANCE_MI * 100, 2
                    ),
                    "elevation_ft":         route["elevation_gain_ft"],
                    "trail_pct":            stats.get("trail_pct", 0),
                    "safety_score":         stats.get("safety_score", 0),
                    "road_avoidance_score": stats.get("road_avoidance_score", 0),
                })

            except Exception as e:
                print(f"    -> Failed ({route_type} @ {start_lat:.4f},{start_lon:.4f}): {e}")

    # ── Point-to-point ──────────────────────────────────────────────────────
    for start_lat, start_lon, end_lat, end_lon, label in P2P_PAIRS:
        try:
            result = generate_point_to_point(
                G,
                start_lat, start_lon,
                end_lat, end_lon,
                mi_to_m(TEST_DISTANCE_MI),
                prefs,
            )
            routes = result.get("routes", [])
            if not routes:
                continue

            route = routes[0]
            stats = route.get("stats", {})
            results.append({
                "timestamp":            datetime.now().isoformat(),
                "route_type":           "point-to-point",
                "start_lat":            start_lat,
                "start_lon":            start_lon,
                "end_lat":              end_lat,
                "end_lon":              end_lon,
                "pair_label":           label,
                "w_elev":               w_elev,
                "w_safe":               w_safe,
                "w_road":               w_road,
                "w_trail":              w_trail,
                "requested_mi":         TEST_DISTANCE_MI,
                "actual_mi":            route["distance_mi"],
                "dist_error_pct":       round(
                    abs(route["distance_mi"] - TEST_DISTANCE_MI) / TEST_DISTANCE_MI * 100, 2
                ),
                "elevation_ft":         route["elevation_gain_ft"],
                "trail_pct":            stats.get("trail_pct", 0),
                "safety_score":         stats.get("safety_score", 0),
                "road_avoidance_score": stats.get("road_avoidance_score", 0),
            })

        except Exception as e:
            print(f"    -> Failed (p2p {label}): {e}")

    return results


# ---------------------------------------------------------------------------
# PRINT SUMMARY for one completed ablation
# ---------------------------------------------------------------------------

def print_summary(weight_name: str, target_metric: str, results: list):
    print(f"\n  --- {weight_name} -> {target_metric} (all route types) ---")
    by_weight = {}
    for r in results:
        key = r[weight_name]
        by_weight.setdefault(key, []).append(r[target_metric])

    for w in sorted(by_weight.keys()):
        vals = by_weight[w]
        print(f"    {weight_name}={w:.1f}: n={len(vals):3d} | "
              f"mean={statistics.mean(vals):.3f} | "
              f"median={statistics.median(vals):.3f}")


# ---------------------------------------------------------------------------
# MAIN — iterate over all four ablation plans
# ---------------------------------------------------------------------------

routes_per_weight = (
    len(WEIGHT_VALUES) * len(FIXED_START_POINTS) * len(ROUTE_GENERATORS)
    + len(WEIGHT_VALUES) * len(P2P_PAIRS)
)
grand_total = routes_per_weight * len(ABLATION_PLAN)

print(f"Starting full ablation test")
print(f"  {len(ABLATION_PLAN)} weights x {routes_per_weight} routes each = {grand_total} total generations\n")

for plan in ABLATION_PLAN:
    weight_name   = plan["weight_name"]
    target_metric = plan["target_metric"]
    output_file   = plan["output_file"]
    description   = plan["description"]

    print(f"\n{'='*60}")
    print(f"  Ablating: {weight_name}")
    print(f"  {description}")
    print(f"  Output -> {output_file}")
    print(f"{'='*60}")

    all_results = []

    for i, weight_value in enumerate(WEIGHT_VALUES):
        routes_this_value = len(FIXED_START_POINTS) * len(ROUTE_GENERATORS) + len(P2P_PAIRS)
        print(f"\n  [{i+1}/{len(WEIGHT_VALUES)}] {weight_name}={weight_value:.1f} "
              f"({routes_this_value} routes) ...")

        batch = run_one_weight(weight_name, weight_value)
        all_results.extend(batch)
        print(f"    -> {len(batch)} routes logged")

    # Save this weight's results to its own file
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved {len(all_results)} results to {output_file}")

    # Print summary table for quick validation
    print_summary(weight_name, target_metric, all_results)

print(f"\n{'='*60}")
print("All ablation tests complete.")
print("Output files:")
for plan in ABLATION_PLAN:
    print(f"  {plan['output_file']}")
