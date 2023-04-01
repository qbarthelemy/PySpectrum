"""Functions to simulate signals or spectra."""

import numpy as np


def spectrum_noise_pink(n_channels, n_freqs, fs):
    """Generate complex spectrum of pink noise.

    Generate a complex spectrum of pink noise, with a spectrum in 1/f.

    Parameters
    ----------
    n_channels : int
        Number of channels to generate.

    n_freqs : int
        Number of frequencies to generate.

    fs : int
        Sampling frequency of the spectrum, in Hz.

    Returns
    -------
    spectrum : ndarray, shape (n_channels, n_freqs)
        Spectrum of pink noise.

    freqs : ndarray, shape (n_freqs,)
        The frequencies associated to spectrum.
    """

    freqs = np.fft.rfftfreq(2 * (n_freqs - 1), 1/fs)
    assert len(freqs) == n_freqs
    spectrum = np.zeros((n_channels, n_freqs), dtype=np.complex128)
    spec = np.empty_like(freqs, dtype=np.complex128)

    for c in range(n_channels):
        # assume amplitude of spectrum is inversely proportional to frequency
        spec[1:] = 1 / (freqs[1:])
        spec[0] = spec[1]
        # add random noise
        spec += np.random.uniform(low=0, high=0.1, size=spec.shape)
        # amplify
        spec *= 5e3
        # define a random phase for each frequency
        phase = np.random.uniform(low=0, high=2*np.pi, size=spec.shape)
        # convert power to a random complex
        spec = spec * np.exp(1j*phase)
        # zero and Nyquist bins should not have an imaginary part
        spec[0] = spec[0].real
        if n_freqs % 2 == 0:  # even case
            spec[-1] = spec[-1].real
        spectrum[c] = spec

    return spectrum, freqs


def spectrum_eeg(n_channels, n_freqs, fs):
    """Generate complex spectrum of EEG.

    Generate a complex EEG spectrum, ie pink noise in 1/f,
    with additional alpha and beta peaks, and a 50Hz powerline noise.

    Parameters
    ----------
    n_channels : int
        Number of channels to generate.

    n_freqs : int
        Number of frequencies to generate.

    fs : int
        Sampling frequency of the spectrum, in Hz.

    Returns
    -------
    spectrum : ndarray, shape (n_channels, n_freqs)
        Spectrum of EEG.

    freqs : ndarray, shape (n_freqs,)
        The frequencies associated to spectrum.
    """

    freqs = np.fft.rfftfreq(2 * (n_freqs - 1), 1/fs)
    assert len(freqs) == n_freqs
    df = np.diff(freqs)[0]
    spectrum = np.zeros((n_channels, n_freqs), dtype=np.complex128)
    spec = np.empty_like(freqs, dtype=np.complex128)

    for c in range(n_channels):
        # assume amplitude of spectrum is inversely proportional to frequency
        spec[1:] = 1 / (freqs[1:])
        spec[0] = spec[1]
        # add alpha as a Gaussian around 10Hz
        spec += 0.25 * np.exp(-0.5*(freqs-10)**2)
        # add beta as a Gaussian around 20Hz
        spec += 0.15 * np.exp(-0.1*(freqs-20)**2)
        # add 50Hz noise
        spec[np.isclose(freqs, 50, atol=df)] += 2
        # add random noise
        spec += np.random.uniform(low=0, high=0.1, size=spec.shape)
        # amplify
        spec *= 5e3
        # define a random phase for each frequency
        phase = np.random.uniform(low=0, high=2*np.pi, size=spec.shape)
        # convert power to a random complex
        spec = spec * np.exp(1j*phase)
        # zero and Nyquist bins should not have an imaginary part
        spec[0] = spec[0].real
        if n_freqs % 2 == 0: # even case
            spec[-1] = spec[-1].real
        spectrum[c] = spec

    return spectrum, freqs


def signal_noise_pink(n_channels, n_samples, fs):
    """Generate signal of pink noise.

    Generate a signal of pink noise, with a spectrum in 1/f.

    Parameters
    ----------
    n_channels : int
        Number of channels to generate.

    n_samples : int
        Number of samples to generate, even.

    fs : int
        Sampling frequency of the signal, in Hz.

    Returns
    -------
    sig : ndarray, shape (n_channels, n_samples)
        Signal of pink noise.
    """
    if n_samples % 2 == 1:
        raise ValueError('Number of samples must be even.')

    spectrum, _ = spectrum_noise_pink(n_channels, n_samples // 2 + 1, fs)
    sig = np.fft.irfft(spectrum, axis=1).real  # drop imaginary part

    return sig
