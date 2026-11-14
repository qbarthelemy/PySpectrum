"""
===============================================================================
Automatic alpha peak detection in EEG
===============================================================================

Automatic detection of alpha peak in the spectrum of an electroencephalogram
(EEG), compared to peak finders provided by SciPy [1]_ and MNE-Python [2]_.
"""
# Author: Quentin Barthélemy

import os

import numpy as np
from mne.datasets import sample
from mne.io import read_raw_fif
from mne.preprocessing import peak_finder
from scipy.signal import find_peaks, welch
from matplotlib import pyplot as plt

from pyspectrum.spectral import detect_peak


###############################################################################
# Load EEG data
# -------------

raw_fname = os.path.join(
    sample.data_path(), 'MEG', 'sample', 'sample_audvis_raw.fif'
)
raw = read_raw_fif(raw_fname, preload=True, verbose=False)
raw.pick("eeg")
fs = int(raw.info['sfreq'])  # 600 Hz


###############################################################################
# Compute spectrum and find peak
# ------------------------------

n_times = 8 * fs  # short period to have a noisy spectrum
eeg = 5e5 * raw.get_data()[40, :n_times]

freqs, spectrum = welch(eeg, fs=fs, nperseg=fs, noverlap=0.1*fs)

band_low, band_high = 8, 12
foi = ((freqs >= band_low) & (freqs < band_high))
idx_off = np.flatnonzero(foi)[0]  # index of the first frequency of the band

# SciPy and MNE return indices relative to the band: shift them to be
# indices of the whole spectrum
idx_sp, _ = find_peaks(spectrum[foi])
idx_sp = idx_sp + idx_off
idx_mne, _ = peak_finder(spectrum[foi], verbose=False)
idx_mne = np.asarray(idx_mne, dtype=int) + idx_off

# PySpectrum returns the frequency of the peak, NaN if no peak is found
peak_ps = detect_peak(spectrum, freqs=freqs, band_target=(band_low, band_high))
peak_ps = peak_ps[~np.isnan(peak_ps)]
idx_ps = np.searchsorted(freqs, peak_ps)


###############################################################################
# Display found peaks
# -------------------

fig, ax = plt.subplots()
fig.suptitle('Comparison of peak finders', fontsize=12)
ax.plot(freqs, spectrum)
ax.scatter(freqs[idx_sp], spectrum[idx_sp], c='C1', s=50, marker='v',
           label='SciPy')
ax.scatter(freqs[idx_mne], spectrum[idx_mne], c='C2', s=50, marker='o',
           label='MNE')
ax.scatter(freqs[idx_ps], spectrum[idx_ps], c='C3', s=60, marker='x',
           label='PySpectrum')
ax.set_xlim(0, 30)
ax.set_xlabel("Frequency in Hz")
ax.set_ylabel("Power spectrum of the EEG")
ax.legend(loc='upper right')
plt.show()


###############################################################################
# References
# ----------
# .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
# .. [2] https://mne.tools/stable/generated/mne.preprocessing.peak_finder.html
