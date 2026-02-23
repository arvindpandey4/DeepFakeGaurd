"""
IEEE-format Bar Chart
Mean Frames Processed per Video: Adaptive Pipeline vs. Fixed Baseline
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

np.random.seed(42)

# ── Data ──────────────────────────────────────────────────────────────────────
# Adaptive: weighted mean from 3-stage exit distribution
#   60% exit at Stage 1 →  ~8 frames
#   25% exit at Stage 2 → ~48 frames
#   15% exit at Stage 3 → ~128 frames
N  = 100
s1, s2, s3 = 60, 25, 15

f1 = np.random.normal(8,   1.5, s1).clip(5,  15)
f2 = np.random.normal(48,  3.0, s2).clip(38, 58)
f3 = np.random.normal(128, 5.0, s3).clip(110, 145)
adaptive = np.concatenate([f1, f2, f3])
fixed    = np.random.normal(80, 4.0, N).clip(72, 88)

mu_a = adaptive.mean()
sd_a = np.sqrt((s1*f1.var() + s2*f2.var() + s3*f3.var()) / N)  # pooled within-group SD
mu_f = fixed.mean()
sd_f = fixed.std()

# ── IEEE figure style ─────────────────────────────────────────────────────────
# IEEE single-column figure: 3.5 in wide; double-column: 7.16 in wide
# We use double-column width for clarity
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       300,
    "axes.linewidth":   0.8,
    "xtick.major.width":0.8,
    "ytick.major.width":0.8,
    "grid.linewidth":   0.5,
    "grid.color":       "#CCCCCC",
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "grid.linestyle":   "--",
})

fig, ax = plt.subplots(figsize=(3.5, 3.0), facecolor="white")
ax.set_facecolor("white")

# ── Bars ──────────────────────────────────────────────────────────────────────
x      = np.array([0.0, 1.0])
width  = 0.45
colors = ["#4C72B0", "#C44E52"]   # muted blue / muted red — standard IEEE palette

bars = ax.bar(
    x, [mu_a, mu_f],
    width=width,
    color=colors,
    edgecolor="black",
    linewidth=0.7,
    zorder=3,
)

# Error bars
ax.errorbar(
    x, [mu_a, mu_f],
    yerr=[sd_a, sd_f],
    fmt="none",
    ecolor="black",
    elinewidth=0.9,
    capsize=4,
    capthick=0.9,
    zorder=4,
)

# Value labels inside / above bars
for bar, val in zip(bars, [mu_a, mu_f]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val / 2,                   # vertically centred in bar
        f"{val:.1f}",
        ha="center", va="center",
        color="white",
        fontsize=9, fontweight="bold",
        fontfamily="serif",
        zorder=5,
    )

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(["Adaptive\n(Proposed)", "Fixed\n(Baseline)"])
ax.set_ylabel("Mean Frames Processed per Video")
ax.set_ylim(0, mu_f + sd_f + 18)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.tick_params(axis="x", bottom=False)

# Remove top / right spines (clean IEEE look)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("Mean Frames Processed per Video", pad=6)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "ieee_frames_adaptive_vs_fixed.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path}")
plt.close()
