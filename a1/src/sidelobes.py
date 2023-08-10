"""
Script associated with Q1.

Plots Fourier transform of rectangular and triangular pulse functions, enabling
comparison of rates at which sidelobes fall off.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.fft import fft, fftfreq

from config import A1_ROOT, SAVEFIG_CONFIG

### DSP FUNCTIONS ##############################################################

def rectangular_pulse(_t: np.array, A: float = 1, T: float = 1) -> np.array:
    """
    Constructs a rectangular pulse of amplitude `A` and width `T`, centred about
    t=0 on `_t`. Returns array of y-values corresponding to `_t`.
    """
    return A * (np.abs(_t) <= T / 2)

def triangular_pulse(_t: np.array, A: float = 1, T: float = 1) -> np.array:
    """
    Constructs a triangular pulse of maximum amplitude `A` and width `T`,
    centred about t=0 on `_t`. Returns array of y-values corresponding to `_t`.
    """
    return A * (1 - 2 * np.abs(_t) / T) * (np.abs(_t) <= T / 2)

### VISUALISATION ##############################################################

def visualise_rectangular() -> None:
    """
    Produces a series of plots visualising the rectangular pulse and its first
    derivative.
    """
    t = np.linspace(-1, 1, 1001)

    fig, axs = plt.subplots(1, 2, figsize=(6, 2))

    # Rectangular pulse
    sns.lineplot(x=t, y=(np.abs(t)<=0.5), ax=axs[0])

    # 1st derivative
    sns.lineplot(x=t, y=(-np.sign(t)*(np.abs(t)==0.5)), ax=axs[1])

    axs[0].set_title("Rectangular pulse")
    axs[1].set_title("1st derivative")
    for i in range(2):
        axs[i].set_xticks(np.linspace(-1, 1, 5))
        axs[i].set_xticklabels(["", "$-T/2$", "", "$T/2$", ""])
        axs[i].set_yticks(np.linspace(-1, 1, 5))
        axs[i].set_yticklabels([])
    axs[0].set_yticklabels(["$-A$", "", 0, "", "$A$"])

    fname = Path(A1_ROOT, "output", "q1_rectangular.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)

def visualise_triangular() -> None:
    """
    Produces a series of plots visualising the triangular pulse and its first
    and second derivatives.
    """
    t = np.linspace(-1, 1, 1001)

    fig, axs = plt.subplots(1, 3, figsize=(9, 2))

    # Triangular pulse
    sns.lineplot(x=t, y=((1-2*np.abs(t))*(np.abs(t)<=0.5)), ax=axs[0])

    # 1st derivative
    sns.lineplot(x=t, y=(-np.sign(t)*(np.abs(t)<=0.5)), ax=axs[1])

    # 2nd derivative
    ddy = np.zeros(t.shape); ddy[np.abs(t)==0.5] = 1; ddy[t==0] = -2
    sns.lineplot(x=t, y=ddy, ax=axs[2])

    axs[0].set_title("Triangular pulse")
    axs[1].set_title("1st derivative")
    axs[2].set_title("2nd derivative")
    for i in range(3):
        axs[i].set_xticks(np.linspace(-1, 1, 5))
        axs[i].set_xticklabels(["", "$-T/2$", "", "$T/2$", ""])
        axs[i].set_yticks(np.linspace(-2, 2, 9))
        axs[i].set_yticklabels([])
    axs[0].set_yticklabels(["$-2A$", "", "$-A$", "", 0, "", "$A$", "", "$2A$"])

    fname = Path(A1_ROOT, "output", "q1_triangular.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)

### ENTRYPOINT #################################################################

def main():

    t_min = -50    # Configuration of input time sequence
    t_max =  50
    N = 1001

    t = np.linspace(t_min, t_max, N)
    f = fftfreq(N, (t_max - t_min) / (N - 1))

    # (Optionally) visualise the rectangular/triangular pulses and derivatives
    visualise_rectangular()
    visualise_triangular()

    H_rect = np.abs(fft(rectangular_pulse(t)))
    H_tria = np.abs(fft(triangular_pulse(t)))

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.lineplot(x=f, y=H_rect, ax=ax, label="Rectangular")
    sns.lineplot(x=f, y=H_tria, ax=ax, label="Triangular")

    ax.set_title("Sidelobe fall off comparison")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")

    fname = Path(A1_ROOT, "output", "q1_sidelobes.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)


if __name__ == "__main__":
    main()
