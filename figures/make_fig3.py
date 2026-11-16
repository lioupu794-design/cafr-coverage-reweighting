# -*- coding: utf-8 -*-
"""Redraw Figure 3 (sparsity sweep) from the exact values reported in draft 10."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "mathtext.fontset": "stix",
    "savefig.pad_inches": 0.01,
})

xs = [0.2, 0.5, 0.8, 0.9]
low = [16.3848, 18.2829, 20.8560, 24.3017]
gap = [-0.1019, 1.5563, 4.2306, 7.6501]

fig = plt.figure(figsize=(4.05, 1.62))
ax1 = fig.add_axes([0.085, 0.235, 0.385, 0.66])
ax2 = fig.add_axes([0.585, 0.235, 0.385, 0.66])

for ax, ys, ylab, panel in (
    (ax1, low, "Low MAE", "(a) Low-coverage MAE"),
    (ax2, gap, "Gap", "(b) Group-risk gap"),
):
    ax.plot(xs, ys, color="black", lw=1.0, marker="o", ms=3.4,
            mfc="black", mec="black", clip_on=False, zorder=3)
    ax.grid(True, axis="y", color="#dcdcdc", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:.1f}" for v in xs])
    ax.set_xlabel("Historical observation sparsity", fontsize=7.6, labelpad=2)
    ax.set_ylabel(ylab, fontsize=7.6, labelpad=2)
    ax.set_xlim(0.13, 0.97)
    ax.tick_params(length=2.4, pad=2)
    ax.set_title(panel, fontsize=7.6, pad=4)

# value labels
for x, y in zip(xs, low):
    ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                 xytext=(-13, 4) if x in (0.8, 0.9) else (0, 6),
                 ha="center", fontsize=6.8)
for x, y in zip(xs, gap):
    off = (0, 5) if y <= 0 else ((-14, 5) if x == 0.8 else (0, 6))
    ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=off,
                 ha="center", fontsize=6.8)

ax1.set_ylim(15.4, 25.6)
ax2.set_ylim(-0.9, 8.4)

fig.savefig(r"C:\Users\李欧浦\Documents\deepseek-harness\default-workspace\fig3_sparsity_v2.pdf")
fig.savefig(r"C:\Users\李欧浦\Documents\deepseek-harness\default-workspace\fig3_sparsity_v2.png", dpi=400)
print("written")


