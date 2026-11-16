# -*- coding: utf-8 -*-
"""Figure 1, fourth revision: shared panel baselines, marker-based nodes,
colour convention (orange = low coverage, blue = sufficient, gray = missing)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 7.6,
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

LOW = "#e9b89c"
SUF = "#c8d5e6"
MISS = "#9a9a9a"
DARK = "#232323"

fig = plt.figure(figsize=(4.803, 2.10))
gs = fig.add_gridspec(1, 3, left=0.015, right=0.995, top=0.995, bottom=0.135,
                      wspace=0.14)
AX = [fig.add_subplot(gs[0, k]) for k in range(3)]
for ax in AX:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ------------------------------------------------------------------ panel (a)
ax = AX[0]

def strip(ax, y, keep, label):
    n = 12
    w = 1.0 / n
    ax.plot([0, 1], [y, y], color=DARK, linewidth=0.7)
    for k in range(n):
        if k in keep:
            ax.plot([(k + 0.5) * w] * 2, [y, y + 0.16], color=DARK,
                    linewidth=1.2, solid_capstyle="butt")
    ax.text(0.0, y + 0.21, label, fontsize=7.6, va="bottom", ha="left")

strip(ax, 0.66, set(range(11)), "high $c^{\\mathrm{obs}}$")
strip(ax, 0.34, {0, 5, 9}, "low $c^{\\mathrm{obs}}$")
ax.annotate("", xy=(1.0, 0.12), xytext=(0.0, 0.12),
            arrowprops=dict(arrowstyle="-|>", linewidth=0.6, color=DARK,
                            mutation_scale=5))
ax.text(0.5, 0.03, "time", fontsize=7.4, ha="center", va="bottom")

# ------------------------------------------------------------------ panel (b)
ax = AX[1]
px = [0.20, 0.50, 0.80]
py = [0.34, 0.60, 0.86]
pos = {(i, j): (px[i], py[j]) for i in range(3) for j in range(3)}

# faint road network
for i in range(3):
    for j in range(3):
        for di, dj in ((1, 0), (0, 1)):
            a, b = (i, j), (i + di, j + dj)
            if a in pos and b in pos and b != (2, 2):
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        color="#d8d8d8", linewidth=0.6, zorder=0)

strong = [((1, 1), (1, 2)), ((1, 1), (2, 1)), ((1, 1), (0, 1))]
weak = [((1, 1), (0, 2)), ((1, 1), (1, 0))]
for a, b in strong:
    ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=DARK,
            linewidth=1.5, zorder=1)
for a, b in weak:
    ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=DARK,
            linewidth=0.8, linestyle=(0, (2, 1.4)), zorder=1)

ax.plot([pos[(1, 1)][0], pos[(2, 2)][0] - 0.05],
        [pos[(1, 1)][1], pos[(2, 2)][1] - 0.05], color=MISS, linewidth=0.7,
        linestyle=(0, (1.4, 1.4)), zorder=1)

for k, (x, y) in pos.items():
    if k == (2, 2):
        ax.scatter([x], [y], s=120, facecolor="white", edgecolor=MISS,
                   linewidth=0.8, linestyle=(0, (1.5, 1.5)), zorder=2)
    elif k == (1, 1):
        ax.scatter([x], [y], s=190, facecolor=LOW, edgecolor=DARK,
                   linewidth=0.8, zorder=3)
        ax.text(x, y, "$i$", fontsize=7.6, ha="center", va="center", zorder=4)
    else:
        ax.scatter([x], [y], s=120, facecolor="white", edgecolor=DARK,
                   linewidth=0.8, zorder=3)

ly = 0.12
ax.plot([0.02, 0.12], [ly, ly], color=DARK, linewidth=1.5)
ax.text(0.15, ly, "strong", fontsize=7.0, va="center", ha="left")
ax.plot([0.43, 0.53], [ly, ly], color=DARK, linewidth=0.8,
        linestyle=(0, (2, 1.4)))
ax.text(0.56, ly, "weak", fontsize=7.0, va="center", ha="left")
ax.scatter([0.80], [ly], s=45, facecolor="white", edgecolor=MISS,
           linewidth=0.8, linestyle=(0, (1.5, 1.5)))
ax.text(0.86, ly, "missing", fontsize=7.0, va="center", ha="left")

# ------------------------------------------------------------------ panel (c)
ax = AX[2]
ax.set_xlim(-0.02, 1.02)
rng = np.random.default_rng(11)
suf = 0.38 + rng.normal(0, 0.045, 8)
low = 0.72 + rng.normal(0, 0.05, 8)
xs_s = 0.30 + rng.normal(0, 0.045, 8)
xs_l = 0.72 + rng.normal(0, 0.045, 8)

ax.plot([0.0, 0.0], [0.16, 0.92], color=DARK, linewidth=0.6)
for ty in (0.16, 0.54, 0.92):
    ax.plot([-0.03, 0.0], [ty, ty], color=DARK, linewidth=0.6)
ax.text(-0.09, 0.54, "MAE", fontsize=7.4, rotation=90, va="center", ha="center")

ax.scatter(xs_s, suf, s=16, facecolor=SUF, edgecolor=DARK, linewidth=0.5,
           zorder=3)
ax.scatter(xs_l, low, s=16, facecolor=LOW, edgecolor=DARK, linewidth=0.5,
           zorder=3)
ax.plot([0.17, 0.43], [suf.mean()] * 2, color=DARK, linewidth=1.6, zorder=4)
ax.plot([0.59, 0.85], [low.mean()] * 2, color=DARK, linewidth=1.6, zorder=4)

overall = 0.1 * low.mean() + 0.9 * suf.mean()
ax.plot([0.0, 0.97], [overall, overall], color=MISS, linewidth=0.7,
        linestyle=(0, (2.5, 1.6)), zorder=2)
ax.text(1.02, overall, "overall", fontsize=6.8, color=MISS, va="center",
        ha="left", clip_on=False)

ax.annotate("", xy=(0.51, low.mean()), xytext=(0.51, suf.mean()),
            arrowprops=dict(arrowstyle="<|-|>", linewidth=0.7, color=DARK,
                            mutation_scale=5))
ax.text(0.55, (suf.mean() + low.mean()) / 2, "Gap", fontsize=7.4,
        va="center", ha="left")

ax.text(0.30, 0.11, "sufficient", fontsize=7.0, ha="center", va="bottom")
ax.text(0.72, 0.11, "low", fontsize=7.0, ha="center", va="bottom")

# ------------------------------------------------------------------ titles
for k, txt in enumerate(["(a) Observation coverage",
                         "(b) Neighborhood support",
                         "(c) Group-wise risk"]):
    xc = (k + 0.5) / 3.0
    fig.text(xc, 0.012, txt, fontsize=7.6, ha="center", va="bottom")

fig.savefig("fig1_problem_v3.pdf")
fig.savefig("fig1_problem_v3.png", dpi=400)
print("written")


