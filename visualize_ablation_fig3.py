"""
visualize_ablation_fig3.py — Ablation Fig 3 (Weight Sensitivity Validation)
=============================================================================
Combines all 4 ablation JSON files, classifies each route into a
low/medium/high tier based on the weight that was varied, then
recreates Fig 3 using the controlled ablation data.

Run from the same directory as the ablation JSON files:
    python visualize_ablation_fig3.py

Output: fig3_ablation_pref_validation.png
"""

import json
import statistics
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS  — update if your files are elsewhere
# ---------------------------------------------------------------------------
BASE_DIR   = "C:/Users/Claire/OneDrive/Desktop/DS 440/"
INPUT_FILES = {
    "w_elev":  BASE_DIR + "ablation_w_elev.json",
    "w_safe":  BASE_DIR + "ablation_w_safe.json",
    "w_road":  BASE_DIR + "ablation_w_road.json",
    "w_trail": BASE_DIR + "ablation_w_trail.json",
}
OUTPUT_PATH = BASE_DIR + "fig3_ablation_pref_validation.png"

# ---------------------------------------------------------------------------
# COLOUR PALETTE  (matches your existing figures)
# ---------------------------------------------------------------------------
ACCENT  = "#7c6af7"
ACCENT2 = "#4fd1c5"
BG      = "#0d0f14"
PANEL   = "#13161e"
BORDER  = "#1f2433"
MUTED   = "#94a3b8"
WHITE   = "#ffffff"

# ---------------------------------------------------------------------------
# TIER CLASSIFICATION
# w_elev is inverted: low w_elev = high elevation preference (wants hills)
# All other weights: high value = high preference
# ---------------------------------------------------------------------------
def classify_tier(weight_name: str, weight_value: float) -> str:
    """Map a raw weight value to low / medium / high tier."""
    if weight_value <= 0.5:
        return "low"
    elif weight_value <= 1.5:
        return "medium"
    else:
        return "high"

# ---------------------------------------------------------------------------
# LOAD & COMBINE ALL 4 ABLATION FILES
# ---------------------------------------------------------------------------
all_routes = []
for weight_name, filepath in INPUT_FILES.items():
    try:
        with open(filepath) as f:
            routes = json.load(f)
        # Tag each route with which weight was being ablated
        for r in routes:
            r["ablated_weight"] = weight_name
            r["tier"] = classify_tier(weight_name, r[weight_name])
        all_routes.extend(routes)
        print(f"Loaded {len(routes):4d} routes from {Path(filepath).name}")
    except FileNotFoundError:
        print(f"WARNING: {filepath} not found — skipping")

print(f"\nTotal routes loaded: {len(all_routes)}\n")

# After the loading loop, add:
for r in all_routes:
    if r.get("actual_mi", 0) > 0:
        r["elevation_per_mile"] = r["elevation_ft"] / r["actual_mi"]
    else:
        r["elevation_per_mile"] = 0.0
# ---------------------------------------------------------------------------
# AGGREGATE  mean & median per (ablated_weight, tier, target_metric)
# ---------------------------------------------------------------------------
def aggregate(routes, weight_name, tier, metric):
    """Return (mean, median) for a given weight/tier/metric slice."""
    vals = [
        r[metric] for r in routes
        if r["ablated_weight"] == weight_name and r["tier"] == tier
    ]
    if not vals:
        return 0.0, 0.0
    return statistics.mean(vals), statistics.median(vals)

# ---------------------------------------------------------------------------
# FIGURE SETUP  (matches dark_fig() in your original script)
# ---------------------------------------------------------------------------
def dark_fig(nrows=1, ncols=1, figsize=(14, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG)
    fig.patch.set_facecolor(BG)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    for ax in np.array(axes).flatten():
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
    return fig, np.array(axes).flatten()

# ---------------------------------------------------------------------------
# FIG 3 — one subplot per weight, x-axis = tier, y-axis = target metric
# ---------------------------------------------------------------------------
checks = [
    ("Trail Pref → Trail %",       "w_trail", "trail_pct",            "Avg Trail Coverage (%)"),
    ("Elev Avoidance → Elev/Mile", "w_elev",  "elevation_per_mile",   "Avg Elevation Gain (ft/mi)"),  # ← updated
    ("Safety Pref → Safety Score", "w_safe",  "safety_score",         "Avg Safety Score (0–1)"),
    ("Road Pref → Road Avoidance", "w_road",  "road_avoidance_score", "Avg Road Avoidance (0–1)"),
]

tiers = ["low", "medium", "high"]
x     = np.arange(len(tiers))
w     = 0.35

fig3, axes3 = dark_fig(1, 4, figsize=(16, 5))
fig3.suptitle(
    "Does Preference Weight Actually Affect Route Composition? (Ablation)",
    color=WHITE, fontsize=13, fontweight="bold", y=1.02
)

for ax, (title, weight_name, metric, ylabel) in zip(axes3, checks):
    means   = []
    medians = []
    ns      = []

    for tier in tiers:
        mn, med = aggregate(all_routes, weight_name, tier, metric)
        means.append(mn)
        medians.append(med)
        ns.append(sum(
            1 for r in all_routes
            if r["ablated_weight"] == weight_name and r["tier"] == tier
        ))

    b1 = ax.bar(x - w/2, means,   w, label="Mean",   color=ACCENT,  edgecolor=BORDER, linewidth=0.8)
    b2 = ax.bar(x + w/2, medians, w, label="Median", color=ACCENT2, edgecolor=BORDER, linewidth=0.8)

    # Annotate n= above each pair of bars
    for i, (mn, med, n) in enumerate(zip(means, medians, ns)):
        ax.text(i, max(mn, med) + max(means + medians) * 0.03,
                f"n={n}", ha="center", va="bottom",
                color=MUTED, fontsize=7)

    ax.set_title(title, color=WHITE, fontsize=9, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, labelcolor=MUTED, facecolor=PANEL, edgecolor=BORDER,
          loc="lower right", bbox_to_anchor=(0.02, 0.98))
    ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

fig3.tight_layout()
fig3.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved fig3_ablation_pref_validation.png")

# ---------------------------------------------------------------------------
# CONSOLE SUMMARY — quick sanity check of the trend per weight
# ---------------------------------------------------------------------------
print("\n--- Ablation Summary ---")
for title, weight_name, metric, _ in checks:
    print(f"\n  {title}  ({weight_name} -> {metric})")
    for tier in tiers:
        mn, med = aggregate(all_routes, weight_name, tier, metric)
        n = sum(1 for r in all_routes
                if r["ablated_weight"] == weight_name and r["tier"] == tier)
        print(f"    {tier:6s}: mean={mn:.3f}  median={med:.3f}  n={n}")
