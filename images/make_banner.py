"""
    @file:              make_banner.py
    @Author:            Maxence Larose

    @Creation Date:     09/2026
    @Last modification: 09/2026

    @Description:       Draws the repository banner. Kept in the repository so the image can be regenerated
                        rather than hand-edited.

                        The left panel is a dijet event: a hard scattering at the vertex throwing two showers
                        that recoil against each other, so the two jets are drawn back to back, as a 2 -> 2
                        process requires. One is coloured as a quark jet and one as a gluon jet, using the
                        same two colours the article's figures use, and the gluon jet is drawn wider and
                        busier because it radiates more.

                        Usage: python images/make_banner.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Polygon

# Text is embedded as outlines, so the banner renders identically without the fonts installed.
matplotlib.rcParams["svg.fonttype"] = "path"

INK = "#12141E"
PAPER = "#F4F2EC"
MUTED = "#8B90A3"
QUARK = "#D87363"
GLUON = "#87C07E"
BEAM = "#4A5068"

WIDTH, HEIGHT = 8.45, 2.0
OUT = Path(__file__).resolve().parent

# The jet axis, and its recoil partner exactly opposite it.
JET_ANGLE = 32.0
JET_LENGTH = 0.88


def face(*candidates, size, fallback):
    """
    Load a font file by name, falling back when it is not installed.

    Ubuntu Bold ships as its own file whose metadata reports a regular weight, so matplotlib will not
    select it by family and weight alone; it has to be addressed by filename.
    """
    for name in candidates:
        for path in Path("/usr/share/fonts").rglob(name):
            return FontProperties(fname=str(path), size=size)

    return FontProperties(fname=findfont(FontProperties(family=fallback)), size=size)


WORDMARK = face("Ubuntu-B.ttf", size=40, fallback="DejaVu Sans:bold")
TAGLINE = face("Ubuntu-R.ttf", size=12.5, fallback="DejaVu Sans")


def shower(ax, x0, y0, angle, opening, n, length, color, seed):
    """Draw one jet: a cone of constituents fanning out from the vertex along ``angle``."""
    rng = np.random.default_rng(seed)

    ax.add_patch(Polygon(
        [(x0, y0),
         (x0 + length * np.cos(angle - opening), y0 + length * np.sin(angle - opening)),
         (x0 + length * np.cos(angle + opening), y0 + length * np.sin(angle + opening))],
        closed=True, facecolor=color, alpha=0.10, edgecolor="none", zorder=1,
    ))

    for _ in range(n):
        # Constituents cluster towards the jet axis, so bias the spread inwards.
        theta = angle + opening * rng.normal(0.0, 0.42)
        r = length * rng.uniform(0.45, 1.0)
        weight = 1.0 - abs(theta - angle) / opening
        ax.plot(
            [x0, x0 + r * np.cos(theta)], [y0, y0 + r * np.sin(theta)],
            color=color, linewidth=0.5 + 2.0 * weight ** 2,
            alpha=0.30 + 0.6 * weight ** 2, solid_capstyle="round", zorder=2,
        )


fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=200)
fig.patch.set_facecolor(INK)
ax.set_facecolor(INK)
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.axis("off")

# --- the event ---------------------------------------------------------------------------------
vx, vy = 1.90, HEIGHT / 2

ax.plot([0.42, 3.38], [vy, vy], color=BEAM, linewidth=1.4, alpha=0.55,
        solid_capstyle="round", zorder=0)

shower(ax, vx, vy, angle=np.deg2rad(JET_ANGLE), opening=np.deg2rad(19), n=26,
       length=JET_LENGTH, color=QUARK, seed=7)
shower(ax, vx, vy, angle=np.deg2rad(JET_ANGLE + 180.0), opening=np.deg2rad(27), n=42,
       length=JET_LENGTH, color=GLUON, seed=11)

ax.plot([vx], [vy], marker="o", markersize=5.5, color=PAPER, zorder=4)
ax.plot([vx], [vy], marker="o", markersize=12.0, color=PAPER, alpha=0.14, zorder=3)

# --- the rule ----------------------------------------------------------------------------------
ax.plot([3.72, 3.72], [0.38, HEIGHT - 0.38], color=BEAM, linewidth=1.0, alpha=0.7, zorder=1)

# --- the wordmark ------------------------------------------------------------------------------
ax.text(4.18, vy + 0.19, "JETGO", fontproperties=WORDMARK, color=PAPER, ha="left", va="center")

ax.text(4.24, vy - 0.45, "Jet Event Toolkit for Generator-level Observables",
        fontproperties=TAGLINE, color=MUTED, ha="left", va="center")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

for suffix in ("svg", "png"):
    path = OUT / f"banner.{suffix}"
    fig.savefig(path, facecolor=INK, dpi=200)
    print("wrote", path)
