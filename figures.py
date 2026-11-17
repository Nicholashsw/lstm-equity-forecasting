"""Hero chart for the repo README. Reads committed results only."""
# --- shared chart style (identical in every repo's figures.py) ---------------------------
# Palette: first three categorical slots of the validated reference palette (pass all-pairs
# CVD checks in light mode). Ink and grid are text tokens, never series colours.
import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "regular", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.labelcolor": INK2, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
})
FIGSIZE = (7.2, 4.0)   # one hero size for all six projects


def finish(fig, ax, title, subtitle, path):
    """Title (what the chart says) and subtitle (what is plotted, from which file), then save."""
    ax.set_title(title, color=INK, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
# -------------------------------------------------------------------------------------------


import pandas as pd


def hero(path="figures/hero.png"):
    r = pd.read_csv("results/xs_regime_results.csv")
    r = r[r.model == "lightgbm"].set_index("regime")
    regs = [("normal", "Calm\n732 days"), ("highvol", "Stressed\n183 days")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1, 1.25]})
    # left: mean daily rank IC with 95% interval = mean +/- 1.96 * mean / t
    for i, (k, lab) in enumerate(regs):
        m, t = r.loc[k, "ic_mean"], r.loc[k, "ic_t"]
        h = 1.96 * abs(m / t)
        a1.errorbar(i, m, yerr=h, color=BLUE, marker="o", ms=6, capsize=4, lw=1.6)
        a1.text(i + 0.12, m, f"{m:.4f}\nt = {t:.2f}", color=INK, fontsize=8.5, va="center")
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_xlim(-0.5, 1.9); a1.set_ylim(-0.02, 0.075)
    a1.set_xticks([0, 1], [l for _, l in regs])
    a1.set_ylabel("Mean daily rank IC (95% interval)")
        # right: gross vs net Sharpe of the decile long-short
    w = 0.36
    for j, (col, c, name) in enumerate([("ls_sharpe_gross", BLUE, "before costs"),
                                        ("ls_sharpe_net", ORANGE, "after costs")]):
        for i, (k, _) in enumerate(regs):
            v = r.loc[k, col]
            x = i + (j - 0.5) * w
            a2.bar(x, v, w * 0.92, color=c)
            a2.text(x, v + (0.15 if v >= 0 else -0.15), f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", color=INK, fontsize=8.5)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xticks([0, 1], [l for _, l in regs]); a2.set_xlim(-0.6, 1.6)
    a2.set_ylim(-6, 3.4)
    a2.set_ylabel("Long-short Sharpe (decile)")
    a2.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["before costs", "after costs"], loc="upper left", fontsize=8.5)
    finish(fig, a1, "The LightGBM signal shows up on stressed days and is gone after costs",
           "LightGBM, calm vs stressed test days (xs_regime_results.csv)", path)


if __name__ == "__main__":
    hero()
