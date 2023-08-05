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

from config import A1_ROOT

### DSP FUNCTIONS ##############################################################

def rectangular_pulse(_t: np.array, A: float = 1, T: float = 1) -> np.array:
    """
    Constructs a rectangular pulse of amplitude `A` and width `T`, centred about
    t=0 on `_t`. Returns array of y-values corresponding to `_t`.
    """
    return A * (np.abs(_t) <= T)

def triangular_pulse(_t: np.array, A: float = 1, T: float = 1) -> np.array:
    """
    Constructs a triangular pulse of maximum amplitude `A` and width `T`,
    centred about t=0 on `_t`. Returns array of y-values corresponding to `_t`.
    """
    return A * (1 - np.abs(_t) / T) * (np.abs(_t) <= T)

### ENTRYPOINT #################################################################

def main():

    N = 800     # Number of sample points
    T = 1e-1    # Sample spacing

    _t = np.linspace(-N*T/2, N*T/2, N, endpoint=False)
    _f = fftfreq(N, T)[:N//2]
    _f = np.concatenate((-1 * np.flip(_f), _f))

    _y1 = rectangular_pulse(_t)
    _H1 = 2 / N * np.abs(fft(_y1)[0:N//2])
    _H1 = np.concatenate((np.flip(_H1), _H1))

    _y2 = triangular_pulse(_t)
    _H2 = 2 / N * np.abs(fft(_y2)[0:N//2])
    _H2 = np.concatenate((np.flip(_H2), _H2))

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.lineplot(x=_f, y=_H1, ax=ax, label="Rectangular")
    sns.lineplot(x=_f, y=_H2, ax=ax, label="Triangular")

    ax.set_title("Sidelobe fall off comparison")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude")

    fname = Path(A1_ROOT, "output", "q1_sidelobes.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
