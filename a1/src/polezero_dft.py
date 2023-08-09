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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from numpy.polynomial.polynomial import Polynomial, polyval
from scipy.fft import fft

from config import A1_ROOT, PLT_CONFIG, SAVEFIG_CONFIG

### DSP FUNCTIONS ##############################################################

def zdft(poly_coef: np.array, N: int) -> np.array:
    """
    Computes the 1D `n`-point discrete Fourier transform of some sequence from
    its Z transform, given by `poly`.
    """
    return np.array([polyval(np.exp(-1j*2*np.pi*k/N), poly_coef) for k in range(N)])

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

def axes_ratio_scale(ax: Axes, ratio: float, padto: str = None) -> Axes:
    """
    Sets axes aspect as equal and autoscales the axes. If the axes limits ratio
    does not match the given aspect ratio (i.e. the ratio height / width), the
    x- or y-axis is lengthened to the desired ratio. Returns the modified axes.
    """
    if padto and padto not in ("upper", "lower", "left", "right", "center"):
        raise ValueError("invalid 'padto' specified")
    padto = padto or "center"

    ax.set_aspect("equal")
    ax.autoscale()

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xrng, yrng = xlim[1] - xlim[0], ylim[1] - ylim[0]
    curr_ratio = yrng / xrng

    if curr_ratio > ratio: # i.e. the current ratio is too tall and narrow
        add_xlim = (yrng / ratio - xrng) * 0.5
        if padto == "right":
            new_xlim = (xlim[0], xlim[1] + 2 * add_xlim)
        elif padto == "left":
            new_xlim = (xlim[0] - 2 * add_xlim, xlim[1])
        else:
            new_xlim = (xlim[0] - add_xlim, xlim[1] + add_xlim)
        ax.set_xlim(new_xlim)

    if curr_ratio < ratio: # i.e. the current ratio is too short and wide
        add_ylim = (xrng * ratio - yrng) * 0.5
        if padto == "upper":
            new_ylim = (ylim[0], ylim[1] + 2 * add_ylim)
        elif padto == "lower":
            new_ylim = (ylim[0] - 2 * add_ylim, ylim[1])
        else:
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

### SUBPARTS ###################################################################

def run_part_a(F: Polynomial) -> None:
    """
    Plots the roots of the polynomial with given coefficients on the complex
    plane, with a unit circle underlay.
    """
    print(f"{F.roots() = }")

    # Override default style to hide grid
    sns.set_style("dark")

    # Re-set the plot text customisation, which gets overriden by set_style
    plt.rcParams.update(PLT_CONFIG)

    fig, ax = plt.subplots()

    ax = plot_poles_or_zeros(F, "zeros", ax)
    ax = axes_ratio_scale(ax, ratio=9/16, padto="center")
    ax = draw_unit_circle(ax)
    ax.set_title("Zero Plot for $F(z)$")

    fname = Path(A1_ROOT, "output", "q5a_polezero.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)

def run_part_b(poly: Polynomial) -> None:
    """
    Plots the magnitude of the polynomial with the given coefficients at 128
    uniformly spaced points around the unit circle using the DFT.
    """
    y_fft = np.abs(fft(poly.coef, n=128))
    y_dft = np.abs(zdft(poly.coef, N=128))

    fig, ax = plt.subplots()

    sns.lineplot(x=np.arange(128), y=y_fft, ax=ax, lw=3, label=r"$\texttt{scipy.fft}$")
    sns.lineplot(x=np.arange(128), y=y_dft, ax=ax, lw=1, label=r"Own DFT")

    ax = axes_ratio_scale(ax, ratio=1/4, padto="upper")

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("$|F(z)|$")

    ax.legend(loc="upper center")

    fname = Path(A1_ROOT, "output", "q5b_dftsample.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)

### ENTRYPOINT #################################################################

def main():

    poly = Polynomial([1, 5, 3, 4, 4, 2, 1])
    # run_part_a(poly)
    run_part_b(poly)


if __name__ == "__main__":
    main()
