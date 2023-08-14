"""
Script associated with Q9.

Plots a number of headings on a polar plot and calculates the average heading.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.axes import Axes

from config import A1_ROOT, SAVEFIG_CONFIG, SNS_PALETTE

### DSP FUNCTIONS ##############################################################

def average_heading(headings: list[float]) -> float:
    """
    Returns the average heading in degrees of the given list of headings.
    """
    return np.rad2deg(np.angle(np.average(np.exp(1j * np.deg2rad(headings)))))

### VISUALISATION ##############################################################

def draw_arrow(x: float, y: float, dx: float, dy: float, ax: Axes,
        arrowprops: dict = None):
    """
    Draws an arrow on the given axes from (x, y) to (x+dx, y+dy).
    """
    arrowprops = arrowprops or {
        "arrowstyle": "-|>", "color": sns.color_palette(SNS_PALETTE)[1]}

    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y), arrowprops=arrowprops)

def visualise_headings(headings: list[float]) -> None:
    """
    Visualises the given headings in degrees on a polar plot.
    """
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(projection='polar')

    for phi in np.deg2rad(headings):
        draw_arrow(0, 0, phi, 0.7, ax)

    avg = average_heading(headings)
    print("Average heading:", np.round(avg, decimals=3), "[deg]")

    draw_arrow(0, 0, np.deg2rad(avg), 0.9, ax, arrowprops={"arrowstyle": "-|>",
        "color": sns.color_palette(SNS_PALETTE)[3], "lw": 2})

    # Hide magnitude labels
    ax.set_yticklabels([])

    fname = Path(A1_ROOT, "output", "q9_headings.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)

### ENTRYPOINT #################################################################

def main():

    headings = [11, 15, 350, 330, 23, 347, 17, 356, 6, 358] # in degrees
    visualise_headings(headings)


if __name__ == "__main__":
    main()
