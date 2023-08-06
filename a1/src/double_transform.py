"""
Script associated with Q4.

Implementation of the double transform algorithm to Fourier transform two real
N-point sequences using one commplex N-point transform.
"""

import numpy as np

from scipy.fft import fft

### DSP FUNCTIONS ##############################################################

def ev(H: np.array) -> np.array:
    """
    Returns the even component of the given sequence `H`.
    """
    H_minus = np.concatenate([H[:1], H[-1:0:-1]])
    return 0.5 * (H + H_minus)

def od(H: np.array) -> np.array:
    """
    Returns the odd component of the given sequence `H`.
    """
    H_minus = np.concatenate([H[:1], H[-1:0:-1]])
    return 0.5 * (H - H_minus)

### ENTRYPOINT #################################################################

def main():

    x = np.array([1, 2, 4, 4, 5, 3, 7, 8])
    y = np.array([1, 5, 3, 1, 3, 5, 3, 7])  # PART A: comment out for the other
    # y = np.array([1, 5, 3, 1, 3, 5, 3, 0])  # PART B: comment out for the other
    print(f"{fft(x) = }")
    print(f"{fft(y) = }")

    Z = fft(np.array([a+b*1j for a, b in zip(x, y)]))
    print(f"{Z = }")

    print(f"{ev(np.real(Z)) = }")
    print(f"{od(np.imag(Z)) = }")
    print(f"{od(np.real(Z)) = }")
    print(f"{ev(np.imag(Z)) = }")

    X = (ev(np.real(Z)) + 1j * od(np.imag(Z)))
    Y = (ev(np.imag(Z)) - 1j * od(np.real(Z)))
    print(f"{X = }")
    print(f"{Y = }")


if __name__ == "__main__":
    main()
