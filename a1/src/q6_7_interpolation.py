"""
Script associated with Q6 and 7.

Performs two methods of upsamping a sine wave: sinc interpolation and
zero-padding in the Fourier domain.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.fft import fft, fftfreq, ifft

from config import A1_ROOT, SAVEFIG_CONFIG

### DSP FUNCTIONS ##############################################################

def sinc_interpolate(x: np.array, n: int, viz: bool = False) -> np.array:
    """
    Upsamples the given signal by the specified factor using sinc interpolation.
    """
    # Increases the sampling rate of x by inserting n-1 zeros between samples
    x_upsamp = np.concatenate([[p]+[0]*(n-1) for p in x])

    # (Optionally) Visualise the intermediate upsampled result
    if viz:
        fig, axs = time_fourier_plot(np.linspace(0, 1, len(x_upsamp)), x_upsamp)
        axs[1].set_xlim(-5, 105)
        axs[1].add_patch(Rectangle(
            (0, 0), 10, 10, color=sns.color_palette()[1], fill=False, ls="--"))

        fname = Path(A1_ROOT, "output", "q6_intermediate.png")
        fig.savefig(fname, **SAVEFIG_CONFIG)

    # Convolve with sinc in time domain by applying rect window in freq. domain
    H_upsamp = fft(x_upsamp)
    H_upsamp[10:-10] = 0
    x_upsamp = ifft(H_upsamp)

    return x_upsamp

def fourier_interpolate(x: np.array, n: int) -> np.array:
    """
    Upsamples the given signal by the specified factor by applying the Fourier
    transform, zero-padding, then inverse Fourier transforming.
    """
    H = fft(x); N = len(H)

    # Pad N*(n-1) zeros between positive and negative frequencies
    H_upsamp = np.concatenate([H[:N//2], np.zeros(len(x)*(n-1)), H[N//2:]])
    x_upsamp = ifft(H_upsamp)

    return x_upsamp

### VISUALISATION ##############################################################

def time_fourier_plot(t: np.array, x: np.array) -> tuple[Figure, list[Axes]]:
    """
    Plot the given signal and its discrete Fourier transform.
    """
    f = fftfreq(n=len(t), d=(t[1]-t[0]))[:len(t)//2]
    H = np.abs(fft(x))[:len(t)//2]

    fig, axs = plt.subplots(2, figsize=(8, 4.5))
    fig.tight_layout()

    sns.lineplot(x=t, y=x, ax=axs[0])
    sns.lineplot(x=f, y=H, ax=axs[1])

    axs[0].set_xlabel("Time [s]")
    axs[1].set_xlabel("Frequency [Hz]")

    return fig, axs

### ENTRYPOINT #################################################################

def run_question_6(x_samp: np.array):
    """
    Performs sinc interpolation to upsample the given signal to 80 Hz and plots
    the results in the time and Fourier domains.
    """
    # Upsample from 20 Hz to 80 Hz
    x_upsamp = sinc_interpolate(x_samp, 4, viz=True)
    t_upsamp = np.linspace(0, 1, 80)

    fig, axs = time_fourier_plot(t_upsamp, x_upsamp)
    axs[1].set_xlim(-5, 105)

    fname = Path(A1_ROOT, "output", "q6_upsampled.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)

def run_question_7(x_samp: np.array):
    """
    Performs zero-padding in the Fourier domain to upsample the given signal to
    80 Hz and plots the results in the the time and Fourier domains.
    """
    # Upsample from 20 Hz to 80 Hz
    x_upsamp = fourier_interpolate(x_samp, 4)
    t_upsamp = np.linspace(0, 1, 80)

    # Plot a before/after frequency domain comparison
    fig, axs = plt.subplots(2, figsize=(8, 4.5))
    fig.tight_layout()

    f_samp = fftfreq(n=20, d=1/20)[:len(x_samp)//2]
    H_samp = np.abs(fft(x_samp))[:len(x_samp)//2]

    N_upsamp = len(x_upsamp)
    f_upsamp = fftfreq(n=80, d=1/80)[:N_upsamp//2]
    H_upsamp = np.abs(fft(x_upsamp))[:N_upsamp//2]

    sns.lineplot(x=f_samp,   y=H_samp,   ax=axs[0], label="20Hz sampled")
    sns.lineplot(x=f_upsamp, y=H_upsamp, ax=axs[1], label="80Hz sinc interpolated")

    axs[0].set_xlim([-5, 105])
    axs[1].set_xlim([-5, 105])

    axs[1].set_xlabel("Frequency [Hz]")

    fname = Path(A1_ROOT, "output", "q7_freqcompare.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)

    # Plot a comparison this method and sinc interpolation
    x_upsinc = sinc_interpolate(x_samp, 4)
    H_upsinc = np.abs(fft(x_upsinc))[:N_upsamp//2]

    fig, axs = plt.subplots(2, figsize=(8, 4.5))
    fig.tight_layout()

    sns.lineplot(x=t_upsamp, y=x_upsamp, ax=axs[0], ls="-",  lw=2,
        label="Fourier interpolation")
    sns.lineplot(x=t_upsamp, y=x_upsinc, ax=axs[0], ls="--", lw=1,
        label="Sinc interpolation")

    sns.lineplot(x=f_upsamp, y=H_upsamp, ax=axs[1], ls="-",  lw=2,
        label="Fourier interpolation")
    sns.lineplot(x=f_upsamp, y=H_upsinc, ax=axs[1], ls="--", lw=1,
        label="Sinc interpolation")

    axs[0].set_xlabel("Time [s]")
    axs[1].set_xlabel("Frequency [Hz]")

    axs[0].legend(loc="center right")
    axs[1].legend(loc="center right")

    axs[1].set_xlim(-5, 105)

    fname = Path(A1_ROOT, "output", "q7_upsampled.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)

def main():

    # "Continuous time" 7 Hz sine wave (actually sampled at 1 kHz)
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 7 * t)

    # Plot in the time and Fourier domains
    fig, axs = time_fourier_plot(t, x)
    axs[1].set_xlim(-5, 105)

    fname = Path(A1_ROOT, "output", "q6_sine7hz.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)

    # Sine wave sampled at 20 Hz
    t_samp = t[::1000//20]
    x_samp = x[::1000//20]

    # Plot in the time and Fourier domains
    fig, axs = time_fourier_plot(t_samp, x_samp)
    axs[1].set_xlim(-5, 105)

    fname = Path(A1_ROOT, "output", "q6_sampled.png")
    fig.savefig(fname, **SAVEFIG_CONFIG)

    ### QUESTION 6 ###
    run_question_6(x_samp)

    ### QUESTION 7 ###
    run_question_7(x_samp)


if __name__ == "__main__":
    main()
