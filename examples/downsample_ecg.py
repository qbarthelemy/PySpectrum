"""
===============================================================================
Time series downsampling
===============================================================================

Spectral analyses after downsampling of an electrocardiogram (ECG) time series.
"""
# Author: Quentin Barthélemy

from fractions import Fraction
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import (
    resample, firwin, upfirdn, resample_poly, dlti, cheby1
)
from matplotlib import pyplot as plt

from pyspectrum.src import upiirdn
from pyspectrum.viz import plot_welch


###############################################################################
# Raw ECG
# -------
#
# Signal is a 5 minute long ECG sampled at 360 Hz [1]_,
# with a low powerline contamination at 60 Hz and 120 Hz.

ecg, fi = electrocardiogram(), 360
n_times = len(ecg)

# We add a high powerline contamination at 50 Hz and 100 Hz, to highlight the
# effect of aliasing.
fp = 50
ecg += 0.5 * np.cos(2 * np.pi * (fp / fi) * np.arange(n_times))
ecg += 0.1 * np.cos(2 * np.pi * (2 * fp / fi) * np.arange(n_times))

fig, ax = plt.subplots()
fig.suptitle('Raw ECG', fontsize=12)
ax = plot_welch(ax, ecg, fi, nperseg=2*fi)
plt.show()

# You can observe on the spectrum the powerline peak at 60 Hz.


###############################################################################
# Resampled ECG using Fourier method in spectral domain
# -----------------------------------------------------
#
# ECG is resampled from 360 Hz to 180 Hz, using Fourier method in spectral
# domain [2]_.

fo = 180       # output sampling frequency
src = fo / fi  # sample rate conversion

ecg_re = resample(ecg, int(n_times * src))
fig, ax = plt.subplots()
fig.suptitle('ECG resampled in spectral domain', fontsize=12)
ax = plot_welch(ax, ecg_re, fo, nperseg=2*fo)
plt.show()


###############################################################################
# Resampled ECG in time domain
# ----------------------------
#
# ECG is resampled from 360 Hz to 180 Hz, in time domain.
# Note that `scipy.signal.decimate` is valid only for a non-fractional
# resampling [3]_.

frac = Fraction(fo, fi)  # useful for fractional src
up, down = frac.numerator, frac.denominator

# FIR filters
fir = firwin(int(20 / src) + 1, src, window='hamming')
# Upsampling, low-pass FIR filtering, and downsampling
ecg_fir = upfirdn(fir, ecg, up=up, down=down)
# Upsampling, zero-phase low-pass FIR filtering, and downsampling
ecg_firzp = resample_poly(ecg, up, down, window=fir)

# IIR filters
iir = dlti(*cheby1(8, 0.05, 0.8 * src, btype='lowpass'))
# Upsampling, low pass IIR filtering, and downsampling
ecg_iir = upiirdn(ecg, iir, up=up, down=down, zero_phase=False)
# Upsampling, zero-phase low pass IIR filtering, and downsampling
ecg_iirzp = upiirdn(ecg, iir, up=up, down=down, zero_phase=True)

fig, ax = plt.subplots()
fig.suptitle('ECG resampled in time domain\nwith anti-aliasing filter',
             fontsize=12)
ax = plot_welch(ax, ecg_fir, fo, nperseg=2*fo, label='FIR')
ax = plot_welch(ax, ecg_firzp, fo, nperseg=2*fo, label='FIR zero-phase')
ax = plot_welch(ax, ecg_iir, fo, nperseg=2*fo, label='IIR')
ax = plot_welch(ax, ecg_iirzp, fo, nperseg=2*fo, label='IIR zero-phase')
plt.legend(loc='lower left')
plt.show()


###############################################################################
# Resampled ECG in time domain, but without anti-aliasing filter
# --------------------------------------------------------------
#
# ECG is resampled from 360 Hz to 180 Hz, signal domain but without
# anti-aliasing filter.

if up == 1:  # only for non-fractional src

    ecg_down = ecg[::down]  # equivalent to pyspectrum.src.downsample in 1D

    fig, ax = plt.subplots()
    fig.suptitle('ECG resampled in time domain\nwithout anti-aliasing filter',
                 fontsize=12)
    ax = plot_welch(ax, ecg_down, fo, nperseg=2*fo)
    plt.show()

# You can observe a huge artifact peak at 80 Hz.


###############################################################################
# References
# ----------
# .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.electrocardiogram.html
#
# .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html
#
# .. [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html
