# -*- coding: utf-8 -*-
"""Figure 2, medium version: five-step pipeline with the inserted block highlighted."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 7.2,
    "mathtext.fontset": "stix",
    "savefig.pad_inches": 0.0,
})

ACC = "#e9b89c"
GRY = "#8f8f8f"
ACCENT = "#b03a2e"
DARK = "#232323"

fig = plt.figure(figsize=(4.30, 1.24))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x0, y0, x1, y1, fc="white", ec=DARK, lw=0.7, ls="solid", z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="square,pad=0", facecolor=fc,
                                edgecolor=ec, linewidth=lw, linestyle=ls,
                                zorder=z))


def arrow(x0, y0, x1, y1, ls="solid", color=DARK, lw=0.7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", linewidth=lw, color=color,
                                linestyle=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=4.5))


W = 0.158
xs = [0.010, 0.206, 0.402, 0.598, 0.794]
YT, YB = 0.98, 0.46
cx = [x + W / 2 for x in xs]

labels = [
    ("inputs", r"$\mathbf{X},\ \mathbf{B}$", "training period"),
    ("coverage", r"$c_i^{\mathrm{obs}},\ c_i^{\mathrm{spa}}$", "pre-training"),
    ("group", r"$S_\rho$", r"bottom $\rho N$ nodes"),
    ("backbone", r"$f_\theta \rightarrow \ell_i$", "unchanged"),
    ("CAFR", r"$\dfrac{\sum m\,w\,\ell}{\sum m\,w}$", "training"),
]

for k, (x, (t1, t2, t3)) in enumerate(zip(xs, labels)):
    fc = ACC if k == 4 else "white"
    lw = 1.1 if k == 4 else 0.7
    box(x, YB, x + W, YT, fc=fc, lw=lw)
    ax.text(x + W / 2, 0.905, t1, fontsize=7.6, ha="center",
            color=DARK)
    ax.text(x + W / 2, 0.720, t2, fontsize=8.8, ha="center", va="center")
    ax.text(x + W / 2, 0.545, t3, fontsize=6.8, ha="center", va="center",
            style="italic", color=ACCENT)

for k in range(4):
    arrow(xs[k] + W, 0.700, xs[k + 1], 0.700)

# lambda branch under the inserted block
box(0.762, 0.05, 0.984, 0.28, ec=ACCENT, ls=(0, (2.4, 1.2)))
ax.text(0.873, 0.165, r"$\lambda$ set on validation", fontsize=6.8, ha="center",
        color=ACCENT)

arrow(0.873, 0.280, 0.873, 0.460, ls=(0, (2.4, 1.2)), color=ACCENT)

ax.text(0.180, 0.310,
        r"$w_i=\lambda$ for $i\in S_\rho$, $w_i=1$ otherwise",
        fontsize=6.8, ha="left", va="center", color="#3a3a3a")

fig.savefig("fig2_framework_v3.pdf")
fig.savefig("fig2_framework_v3.png", dpi=400)
print("written")








