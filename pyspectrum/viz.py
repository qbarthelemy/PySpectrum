"""Functions for visualization."""

import numpy as np
from scipy.fft import fft2, fftshift
from scipy.signal import welch


def remove_ticks(ax):
    """Remove x and y ticks from axes."""
    for i in range(len(ax)):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    return ax


def plot_welch(ax, x, fs, nperseg, *, label=None, semilogy=True):
    """Plot spectrum obtained by Welch.

    Parameters
    ----------
    ax : matplotlib axis
        Axis of figure.
    x : ndarray, shape (n_times,)
        Signal.
    fs : float
        Sampling frequency.

    Returns
    -------
    ax : matplotlib axis
        Axis of figure.
    """
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, scaling="spectrum")

    ax.plot(f, Pxx, label=label)
    if semilogy:
        ax.semilogy()
    ax.set_xlabel("Frequency in Hz")
    ax.set_ylabel("Power spectrum of the ECG")
    ax.set_xlim(f[[0, -1]])
    if label:
        ax.legend(loc='upper right')
    return ax


def plot_spectrum2(ax, img):
    """Plot spectrum of 2D image.

    Parameters
    ----------
    ax : matplotlib axis
        Axis of figure.
    img : ndarray, shape (n_height, n_width)
        2D image.

    Returns
    -------
    ax : matplotlib axis
        Axis of figure.
    """
    dft = fft2(img)
    dft_shift = fftshift(dft)
    mag = 20 * np.log(np.abs(dft_shift))
    ax.imshow(mag, cmap='gray')
    return ax
