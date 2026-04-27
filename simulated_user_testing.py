"""
random_user_testing.py — Simulated User Testing
================================================
Simulates realistic user behavior by randomly sampling weights
and distances, only keeping routes within 20% distance error.

Output: simulated_user_testing.json
"""

import json
import random
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
)

OUTPUT_PATH    = "C:/Users/Claire/OneDrive/Desktop/DS 440/simulated_user_testing.json"
MAX_DIST_ERROR = 20.0
TARGET_ROUTES  = 400   # total valid routes to collect
MAX_ATTEMPTS   = 2000  # safety cap to avoid infinite loop

# Realistic distance range a user might request (miles)
DISTANCE_RANGE = (1.0, 10.0)

# Realistic weight range — users pick anywhere from 0 to 2
WEIGHT_RANGE = (0.0, 2.0)

# Route type distribution — weighted to reflect realistic usage
ROUTE_TYPE_POOL = (
    ["loop"] * 3 +
    ["out-and-back"] * 3 +
    ["random"] * 3 +
    ["point-to-point"] * 1   # less common since it needs two points
)

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

# Nearby p2p pairs (straight-line well under 3 mi)
P2P_PAIRS = [
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

GENERATORS = {
    "loop":           generate_loop,
    "out-and-back":   generate_out_and_back,
    "random":         generate_random_explore,
}

def tier(w):
    if w <= 0.5:  return "low"
    if w <= 1.5:  return "medium"
    return "high"

def random_weights():
    return {
        "w_elev":  round(random.uniform(*WEIGHT_RANGE), 1),
        "w_safe":  round(random.uniform(*WEIGHT_RANGE), 1),
        "w_road":  round(random.uniform(*WEIGHT_RANGE), 1),
        "w_trail": round(random.uniform(*WEIGHT_RANGE), 1),
    }

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
results  = []
attempts = 0
skipped  = 0

print(f"Collecting {TARGET_ROUTES} valid simulated user routes...\n")

while len(results) < TARGET_ROUTES and attempts < MAX_ATTEMPTS:
    attempts += 1
    route_type  = random.choice(ROUTE_TYPE_POOL)
    distance_mi = round(random.uniform(*DISTANCE_RANGE), 1)
    weights     = random_weights()
    prefs       = _build_prefs(**weights)

    try:
        if route_type == "point-to-point":
            pair = random.choice(P2P_PAIRS)
            start_lat, start_lon, end_lat, end_lon, label = pair
            result = generate_point_to_point(
                G, start_lat, start_lon, end_lat, end_lon,
                mi_to_m(distance_mi), prefs,
            )
        else:
            start_lat, start_lon = random.choice(FIXED_START_POINTS)
            end_lat, end_lon, label = None, None, None
            result = GENERATORS[route_type](
                G, start_lat, start_lon,
                mi_to_m(distance_mi), prefs,
            )

        routes = result.get("routes", [])
        for route in routes:
            dist_error = abs(route["distance_mi"] - distance_mi) / distance_mi * 100
            if dist_error <= MAX_DIST_ERROR:
                stats = route.get("stats", {})
                results.append({
                    "timestamp":            datetime.now().isoformat(),
                    "route_type":           route_type,
                    "start_lat":            start_lat,
                    "start_lon":            start_lon,
                    "end_lat":              end_lat,
                    "end_lon":              end_lon,
                    "pair_label":           label,
                    "requested_mi":         distance_mi,
                    "actual_mi":            route["distance_mi"],
                    "dist_error_pct":       round(dist_error, 2),
                    "elevation_ft":         route["elevation_gain_ft"],
                    "trail_pct":            stats.get("trail_pct", 0),
                    "safety_score":         stats.get("safety_score", 0),
                    "road_avoidance_score": stats.get("road_avoidance_score", 0),
                    **weights,
                    "elev_pref":  tier(weights["w_elev"]),
                    "safe_pref":  tier(weights["w_safe"]),
                    "road_pref":  tier(weights["w_road"]),
                    "trail_pref": tier(weights["w_trail"]),
                })
                if len(results) % 25 == 0:
                    print(f"  Collected {len(results)}/{TARGET_ROUTES} "
                          f"(attempts={attempts}, skipped={skipped})")
            else:
                skipped += 1

    except Exception as e:
        skipped += 1

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone.")
print(f"  Valid routes collected: {len(results)}")
print(f"  Total attempts:         {attempts}")
print(f"  Skipped (failed/error): {skipped}")
print(f"  Pass rate:              {len(results)/max(attempts,1)*100:.1f}%")
print(f"  Saved to: {OUTPUT_PATH}")