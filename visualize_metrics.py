import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
with open("C:/Users/Claire/OneDrive/Desktop/DS 440/random_user_testing_aggregated.json") as f:
    data = json.load(f)

# ── Colour palette ─────────────────────────────────────────────────────────
ACCENT   = "#7c6af7"
ACCENT2  = "#4fd1c5"
WARN     = "#f6ad55"
SUCCESS  = "#48bb78"
DANGER   = "#fc8181"
BG       = "#0d0f14"
PANEL    = "#13161e"
BORDER   = "#1f2433"
MUTED    = "#94a3b8"
WHITE    = "#ffffff"

ROUTE_COLORS = {
    "loop":           ACCENT,
    "out-and-back":   ACCENT2,
    "point-to-point": WARN,
    "random":         SUCCESS,
}
PREF_COLORS = {"low": DANGER, "medium": WARN, "high": SUCCESS}

def dark_fig(nrows=1, ncols=1, figsize=(14, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor=BG)
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

# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Route-type comparison (4 metrics × 4 route types)
# ══════════════════════════════════════════════════════════════════════════
route_types  = list(data["by_route_type"].keys())
metrics_rt   = ["avg_road_avoidance_score", "avg_trail_pct",
                 "avg_elevation_ft",       "avg_safety_score"]
labels_rt    = ["Avg Road Avoidance Score (0-1)", "Avg Trail Coverage (%)",
                "Avg Elevation Gain (ft)", "Avg Safety Score (0–1)"]

fig1, axes1 = dark_fig(2, 2, figsize=(14, 9))
fig1.suptitle("Route Metrics by Route Type", color=WHITE,
              fontsize=14, fontweight="bold", y=1.01)

for ax, metric, label in zip(axes1, metrics_rt, labels_rt):
    vals   = [data["by_route_type"][rt][metric]["mean"] for rt in route_types]
    colors = [ROUTE_COLORS[rt] for rt in route_types]
    bars   = ax.bar(route_types, vals, color=colors, width=0.55,
                    edgecolor=BORDER, linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{v:.2f}", ha="center", va="bottom",
                color=WHITE, fontsize=8, fontweight="bold")
    ax.set_title(label, color=WHITE, fontsize=10, pad=8)
    ax.set_xticks(range(len(route_types)))
    ax.set_xticklabels([rt.replace("-", "\n") for rt in route_types],
                       fontsize=8, color=MUTED)
    ax.set_ylabel(label, fontsize=8)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

fig1.tight_layout()
fig1.savefig("C:/Users/Claire/OneDrive/Desktop/DS 440/fig1_by_route_type.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved fig1")

# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Preference tier comparison (safety / trail / elevation / road)
# ══════════════════════════════════════════════════════════════════════════
pref_sections = {
    "Safety Preference":    "by_safety_preference",
    "Trail Preference":     "by_trail_preference",
    "Elevation Preference": "by_elevation_preference",
    "Road Preference":      "by_road_preference",
}
pref_metric  = "avg_safety_score"
pref_label   = "Avg Safety Score"

fig2, axes2 = dark_fig(2, 2, figsize=(14, 9))
fig2.suptitle("Safety Score by Each Preference Tier",
              color=WHITE, fontsize=14, fontweight="bold", y=1.01)

for ax, (title, section) in zip(axes2, pref_sections.items()):
    tiers  = ["low", "medium", "high"]
    vals   = [data[section][t][pref_metric]["mean"] for t in tiers]
    colors = [PREF_COLORS[t] for t in tiers]
    bars   = ax.bar(tiers, vals, color=colors, width=0.45,
                    edgecolor=BORDER, linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                color=WHITE, fontsize=9, fontweight="bold")
    ax.set_title(title, color=WHITE, fontsize=10, pad=8)
    ax.set_ylabel(pref_label, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

fig2.tight_layout()
fig2.savefig("C:/Users/Claire/OneDrive/Desktop/DS 440/fig2_safety_by_pref.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved fig2")
'''
# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Trail % by trail preference tier (validation: does high pref → more trail?)
# ══════════════════════════════════════════════════════════════════════════
fig3, axes3 = dark_fig(1, 4, figsize=(14, 5))
fig3.suptitle("Does Preference Weight Actually Affect Route Composition?",
              color=WHITE, fontsize=13, fontweight="bold", y=1.02)

checks = [
    ("Trail Pref → Trail %",     "by_trail_preference",     "avg_trail_pct",          "Avg Trail Coverage (%)"),
    ("Elevation Avoidance Pref → Elev Gain","by_elevation_preference", "avg_elevation_ft",        "Avg Elevation Gain (ft)"),
    ("Safety Pref → Safety Score","by_safety_preference",    "avg_safety_score",        "Avg Safety Score (0–1)"),
    ("Road Pref → Road Avoidance","by_road_preference",      "avg_road_avoidance_score", "Avg Road Avoidance (0–1)"),  # ← add this
]


for ax, (title, section, metric, ylabel) in zip(axes3, checks):
    tiers  = ["low", "medium", "high"]
    means  = [data[section][t][metric]["mean"]   for t in tiers]
    medians= [data[section][t][metric]["median"] for t in tiers]
    x      = np.arange(len(tiers))
    w      = 0.35
    b1 = ax.bar(x - w/2, means,   w, label="Mean",   color=ACCENT,  edgecolor=BORDER, linewidth=0.8)
    b2 = ax.bar(x + w/2, medians, w, label="Median", color=ACCENT2, edgecolor=BORDER, linewidth=0.8)
    ax.set_title(title, color=WHITE, fontsize=9, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, labelcolor=MUTED,
              facecolor=PANEL, edgecolor=BORDER)
    ax.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

fig3.tight_layout()
fig3.savefig("C:/Users/Claire/OneDrive/Desktop/DS 440/fig3_pref_validation.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved fig3")
'''
# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Distance error distribution across route types (box-style via
#          min/median/mean/max since we don't have raw data)
# ══════════════════════════════════════════════════════════════════════════
fig4, ax4 = dark_fig(figsize=(12, 5))
ax4 = ax4[0]
fig4.suptitle("Distance Error Range by Route Type",
              color=WHITE, fontsize=13, fontweight="bold")

for i, rt in enumerate(route_types):
    d     = data["by_route_type"][rt]["avg_distance_error_pct"]
    color = ROUTE_COLORS[rt]
    # draw a simplified box: min→max bar + mean dot + median line
    ax4.plot([i, i], [d["min"], d["max"]],
             color=color, linewidth=2, alpha=0.5)
    ax4.scatter(i, d["mean"],   color=color, s=80, zorder=5,
                label=f"{rt} mean")
    ax4.scatter(i, d["median"], color=WHITE,  s=40, zorder=6,
                marker="_", linewidths=2)
    ax4.text(i + 0.12, d["mean"],
             f"mean={d['mean']:.1f}%", color=color,
             fontsize=8, va="center")

ax4.set_xticks(range(len(route_types)))
ax4.set_xticklabels([rt.replace("-", "\n") for rt in route_types],
                    color=MUTED, fontsize=9)
ax4.set_ylabel("Distance Error (%)", fontsize=9)
ax4.set_title("Coloured dot = mean  |  White dash = median  |  Line = min–max range",
              color=MUTED, fontsize=8, pad=6)
ax4.grid(axis="y", color=BORDER, linewidth=0.5, alpha=0.6)
ax4.set_axisbelow(True)

fig4.tight_layout()
fig4.savefig("C:/Users/Claire/OneDrive/Desktop/DS 440/fig4_distance_error_range.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved fig4")
# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Overall metrics summary table
# ══════════════════════════════════════════════════════════════════════════
fig5, ax5 = dark_fig(figsize=(11, 5))
ax5 = ax5[0]
ax5.axis("off")
fig5.suptitle("Overall Metrics Summary (403 Routes)",
              color=WHITE, fontsize=13, fontweight="bold")

overall = data["overall"]

col_labels = ["Metric", "Mean", "Median", "Min", "Max"]
rows = [
    ["Distance Error (%)",      
     overall["avg_distance_error_pct"]["mean"],
     overall["avg_distance_error_pct"]["median"],
     overall["avg_distance_error_pct"]["min"],
     overall["avg_distance_error_pct"]["max"]],

    ["Trail Coverage (%)",      
     overall["avg_trail_pct"]["mean"],
     overall["avg_trail_pct"]["median"],
     overall["avg_trail_pct"]["min"],
     overall["avg_trail_pct"]["max"]],

    ["Elevation Gain (ft/mi)",
    overall["avg_elevation_per_mile"]["mean"],
    overall["avg_elevation_per_mile"]["median"],
    overall["avg_elevation_per_mile"]["min"],
    overall["avg_elevation_per_mile"]["max"]],

    ["Safety Score (0–1)",      
     overall["avg_safety_score"]["mean"],
     overall["avg_safety_score"]["median"],
     overall["avg_safety_score"]["min"],
     overall["avg_safety_score"]["max"]],

    ["Road Avoidance (0–1)",    
     overall["avg_road_avoidance_score"]["mean"],
     overall["avg_road_avoidance_score"]["median"],
     overall["avg_road_avoidance_score"]["min"],
     overall["avg_road_avoidance_score"]["max"]],
]

# format numbers
formatted = [[r[0]] + [f"{v:.2f}" for v in r[1:]] for r in rows]

table = ax5.table(
    cellText=formatted,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.2)

# style header row
for col in range(len(col_labels)):
    cell = table[0, col]
    cell.set_facecolor(ACCENT)
    cell.set_text_props(color=WHITE, fontweight="bold")
    cell.set_edgecolor(BORDER)

# style data rows with alternating colours
row_colors = [PANEL, "#1a1f2e"]
for row in range(1, len(rows) + 1):
    for col in range(len(col_labels)):
        cell = table[row, col]
        cell.set_facecolor(row_colors[row % 2])
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor(BORDER)

fig5.tight_layout()
fig5.savefig("C:/Users/Claire/OneDrive/Desktop/DS 440/fig5_overall_summary.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved fig5")
