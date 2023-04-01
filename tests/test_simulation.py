"""Tests for module simulation.

To execute tests:
>>> pytest -k test_simulation
"""

import pytest
from pyspectrum.simulation import (
    spectrum_noise_pink, spectrum_eeg, signal_noise_pink
)


n_channels = 3


@pytest.mark.parametrize("n_freqs", [10, 11, 12, 32, 33])
@pytest.mark.parametrize("fs", [8, 9, 10])
def test_spectrum_noise_pink(n_freqs, fs):
    spect, freqs = spectrum_noise_pink(n_channels, n_freqs, fs)
    assert spect.shape == (n_channels, n_freqs)
    assert freqs.shape == (n_freqs,)


@pytest.mark.parametrize("n_freqs", [10, 11, 12, 32, 33])
@pytest.mark.parametrize("fs", [8, 9, 10])
def test_spectrum_eeg(n_freqs, fs):
    spect, freqs = spectrum_eeg(n_channels, n_freqs, fs)
    assert spect.shape == (n_channels, n_freqs)
    assert freqs.shape == (n_freqs,)


@pytest.mark.parametrize("n_samples", [10, 12, 32])
@pytest.mark.parametrize("fs", [8, 9, 10])
def test_signal_noise_pink(n_samples, fs):
    sig = signal_noise_pink(n_channels, n_samples, fs)
    assert sig.shape == (n_channels, n_samples)
