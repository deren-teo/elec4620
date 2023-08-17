"""
Script associated with Q2 and 3.

Multiplies polynomials/numbers in vector representation using convolution and
Fourier transform methods.
"""

import numpy as np

from scipy.signal import convolve
from scipy.fft import fft, ifft

### UTILITY FUNCTIONS ##########################################################

def multiply_carry(z: np.array) -> np.array:
    """
    Perform the "carry" steps of the multiplication process. Starting from the
    right end of `z`, each digit is taken modulo 10 and the remainder is added
    to the value immediately to the left. Returns an array of single digits,
    possibly except the first value (though the answer will still be correct).
    """
    r = 0; ret = []
    for n in z[::-1]:
        n += r
        ret.append(n % 10)
        r = n // 10
    ret.append(r)
    return np.array(ret[::-1])

### DSP FUNCTIONS ##############################################################

def multiply_conv(x: np.array, y: np.array, carry: bool = False) -> np.array:
    """
    Convolve arrays `x` and `y` using direct convolution. If `carry` is True,
    the output array is modified to be equivalent to the vectorised integer
    product of vectorised integers `x` and `y`. Otherwise, the convolution
    result is returned "as-is".
    """
    z = convolve(x, y, mode="full", method="direct")
    if carry:
        z = multiply_carry(z)
    return z

def multiply_fft(x: np.array, y: np.array, carry: bool = False) -> np.array:
    """
    Convolve arrays `x` and `y` by performing the FFT on `x` and `y`,
    multiplying the results, then performing the inverse FFT. If `carry` is
    True, the output array is modified to be equivalent to the vectorised
    integer product of vectorised integers `x` and `y`. Otherwise, the
    convolution result is returned "as-is".
    """
    xpad = np.pad(x, (0, len(y) - 1))
    ypad = np.pad(y, (0, len(x) - 1))
    z = ifft(fft(xpad) * fft(ypad)).real
    if carry:
        z = multiply_carry(z)
    return z

### ENTRYPOINT #################################################################

def main():

    ### QUESTION 2 ###

    vx = np.array([1, 2, 6, 11, 15, 12])
    vy = np.array([1, -3, -3, 7, -7, 3])

    print("Q2 by convolution:\n", multiply_conv(vx, vy))
    print("Q2 by FFT and IFFT:\n", multiply_fft(vx, vy))

    ### QUESTION 3 ###

    vx = np.array([8, 7, 5, 5, 7, 9, 0])
    vy = np.array([1, 3, 6, 7, 2, 6, 7])

    print("Q3 by multiplication:", 8755790 * 1367267)
    print("Q3 by convolution:", multiply_conv(vx, vy, carry=True))
    print("Q3 by FFT and IFFT:", multiply_fft(vx, vy, carry=True))


if __name__ == "__main__":
    main()
