"""
Script associated with Q2 and 3.

Multiplies polynomials/numbers in vector representation using convolution and
Fourier transform methods.
"""

import numpy as np
import scipy.signal as signal

from scipy.fft import fft, ifft

### ENTRYPOINT #################################################################

def main():

    ### QUESTION 2 ###

    vx = np.array([1, 2, 6, 11, 15, 12])
    vy = np.array([1, -3, -3, 7, -7, 3])
    res = signal.convolve(vx, vy, mode="full", method="direct")
    print("Q2 by direct convolution:\n", res)

    pad = len(vx) + len(vy) - 1
    hx = fft(np.pad(vx, (0, pad - len(vx))))
    hy = fft(np.pad(vy, (0, pad - len(vy))))
    res = ifft(hx * hy)
    print("Q2 by FFT, multiply and IFFT:\n", res.real)


if __name__ == "__main__":
    main()
