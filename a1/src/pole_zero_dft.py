"""
Script associated with Q5.

Determines the roots of a certain polynomials and produces a pole-zero plot.
Evaluates the magnitude of the polynomial around the unit circle using the DFT.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path

from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from numpy.polynomial.polynomial import Polynomial

from config import A1_ROOT, PLT_CONFIG, SAVEFIG_CONFIG

### VISUALISATION ##############################################################

def plot_poles_or_zeros(F: Polynomial, type: str, ax: Axes) -> Axes:
    """
    Plots the roots of the polynomial in the complex plane on the given axes.
    """
    roots = F.roots()

    marker = {"poles": "X", "zeros": "o"}[type]
    sns.scatterplot(x=np.real(roots), y=np.imag(roots), ax=ax, marker=marker)

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")

    return ax

def axes_ratio_scale(ax: Axes, ratio: float) -> Axes:
    """
    Sets axes aspect as equal and autoscales the axes. If the axes limits ratio
    does not match the given aspect ratio (i.e. the ratio height / width), the
    x- or y-axis is lengthened to the desired ratio. Returns the modified axes.
    """
    ax.set_aspect("equal")
    ax.autoscale()

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xrng, yrng = xlim[1] - xlim[0], ylim[1] - ylim[0]
    curr_ratio = yrng / xrng

    if curr_ratio > ratio: # i.e. the current ratio is too tall and narrow
        add_xlim = (yrng / ratio - xrng) * 0.5
        new_xlim = (xlim[0] - add_xlim, xlim[1] + add_xlim)
        ax.set_xlim(new_xlim)

    if curr_ratio < ratio: # i.e. the current ratio is too short and wide
        add_ylim = (xrng * ratio - yrng) * 0.5
        new_ylim = (ylim[0] - add_ylim, ylim[1] + add_ylim)
        ax.set_ylim(new_ylim)

    return ax

def draw_unit_circle(ax: Axes) -> Axes:
    """
    Draws dotted axes and unit circle on the given axes, similar in style to
    MATLAB's zplane function.
    """
    style_config = {"ls": "dotted", "lw": 0.9, "color": "cadetblue", "zorder": 0}

    u_circ = Circle(xy=(0, 0), radius=1, fill=False, **style_config)
    ax.add_patch(u_circ)

    x_axis = Line2D(xdata=ax.get_xlim(), ydata=(0, 0), **style_config)
    y_axis = Line2D(xdata=(0, 0), ydata=ax.get_ylim(), **style_config)
    ax.add_line(x_axis)
    ax.add_line(y_axis)
    ax.set_aspect("equal")

    return ax

### ENTRYPOINT #################################################################

def main():

    F = Polynomial([1, 5, 3, 4, 4, 2, 1])
    print(f"{F.roots() = }")

    # Override default style to hide grid
    sns.set_style("dark")

    # Re-set the plot text customisation, which gets overriden by set_style
    plt.rcParams.update(PLT_CONFIG)

    fig, ax = plt.subplots()

    ax = plot_poles_or_zeros(F, "zeros", ax)
    ax = axes_ratio_scale(ax, ratio=9/16)
    ax = draw_unit_circle(ax)
    ax.set_title("Zero Plot for $F(z)$")

    fname = Path(A1_ROOT, "output", "q5_polezero.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)


if __name__ == "__main__":
    main()
