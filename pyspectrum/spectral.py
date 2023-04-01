"""Functions for spectral processing."""

import numpy as np
import statsmodels.api as sm


def _check_freqs(freqs, n_freqs):
    """This function checks freqs."""
    if freqs is None:
        freqs = np.arange(n_freqs)
    else:
        freqs = np.asarray(freqs)
        if freqs.shape != (n_freqs,):
            raise ValueError(
                'freqs does not have the same dimension of the frequency axis.'
                ' Should be %d but got %d.' % (n_freqs, freqs.shape[0]))
        if np.any(freqs < 0):
            raise ValueError('freqs must be strictly positive')

    return freqs


def rescale(x, freqs, *, rescaling='complete', foi=None):
    r"""Rescale spectra using a robust log-log linear regression.

    1. Robust linear regression [1]_ of log spectra `x` as a function of log
    frequency `freqs`:

    On log-log spectra:

    .. math:: \log(x) = a * \log(freqs) + b

    On raw spectra:

    .. math:: x = freqs^a * e^b

    2. Rescaling of spectra:

    - `complete`, using the y-intercept and slope:

    .. math:: x = (x / e^b)^{-1/a}

    - `y-intercept`, using the y-intercept:

    .. math:: x = x / e^b

    - `slope`, using the slope:

    .. math:: x = x^{-1/a}

    Parameters
    ----------
    x : ndarray, shape (n_channels, n_freqs)
        Matrix array containing spectra, with the first dimension representing
        the channels, and the second dimension representing the frequencies.

    freqs : None | ndarray, shape (n_freqs,), default None
        Vector array of size `n_freqs` and containing (positive) frequencies
        abscissa of the spectrum.
        If `freqs` is None, it is assumed that frequencies are
        `[0, 1, ... n_freqs-1]`.

    rescaling : {'complete', 'y-intercept', 'slope'}, default 'complete'
        Type of spectral rescaling, see above.

    foi : None | array_like, default None
        Frequencies of interest (advanced indexing by integers or booleans) for
        the regression, to exclude disturbing frequencies, like DC offset,
        alpha peak, or high frequencies.
        If `foi` is None, the rescaling considers all strictly positive
        frequencies.

    Returns
    -------
    x_r : ndarray, shape (n_channels, n_freqs)
        Rescaled spectra.

    logcoeffs : ndarray, shape (n_channels, 2)
        Coefficients `a` and `b` of the log-log linear regression, with the
        slopes `a` in the first column and y-intercepts `b` in the second one.

    References
    ----------
    .. [1] http://www.statsmodels.org/dev/rlm.html

    Examples
    -------

    Let `eeg` be a matrix of 10 seconds of random pink noise on 8 channels
    and 512 Hz sampling frequency:

    >>> import numpy as np
    >>> from pyspectrum.simulation import signal_noise_pink
    >>> from pyspectrum.spectral import rescale
    >>> n_channels, fs, n_samples = 8, 512, 512*10
    >>> frequency_resolution = fs / n_samples  # 0.1 Hz
    >>> eeg = signal_noise_pink(n_channels, n_samples, fs)

    Let `X` be the power spectra of `eeg`, and `freqs` the associated
    frequencies. Note that since this takes 5120 samples, there are 2560+1
    frequencies:

    >>> X = np.abs(np.fft.rfft(eeg, axis=1))
    >>> freqs = np.fft.rfftfreq(eeg.shape[1], 1/fs)

    To rescale using a boolean array that represents the [1--95) Hz band
    without the [8--12) Hz band (note the ``~`` operation):

    >>> foi = (freqs >= 1) & (freqs < 95) & ~((freqs >= 8) & (freqs < 12))
    >>> Xr, coeffs = rescale(X, freqs, foi=foi)

    To rescale using an integer array, we must take into account the frequency
    resolution:

    >>> foi = np.r_[int(np.floor(1/frequency_resolution)):int(np.floor(8/frequency_resolution)), \
    int(np.floor(12/frequency_resolution)):int(np.floor(95/frequency_resolution))]
    >>> Xr, coeffs = rescale(X, freqs, foi=foi)
    """

    x = np.atleast_2d(np.asarray(x))
    n_channels, n_freqs = x.shape
    x_r = np.zeros_like(x)
    freqs = _check_freqs(freqs, n_freqs)

    if foi is not None:
        foi = np.asarray(foi)
    else:
        foi = (freqs > 0)
    if np.any(x[..., foi] <= 0):
        raise ValueError(
            'x must be strictly positive on frequencies of interest'
        )

    # keep frequency of interest and log-transform
    logfreqs = np.log(freqs[foi])[:, np.newaxis]
    logcoeffs = np.empty((n_channels, 2), dtype=float)
    logspectrum = np.log(x[..., foi])

    for c in range(n_channels):
        # fit robust linear regression on the log-log spectrum
        rlm_model = sm.RLM(
            logspectrum[c],
            sm.tools.add_constant(logfreqs),
            M=sm.robust.norms.HuberT(),
        )
        rlm_results = rlm_model.fit(conv='coefs', tol=1e-3)
        intercept, slope = rlm_results.params

        # rescale
        if rescaling == 'complete':
            x_r[c] = np.power(x[c] / np.exp(intercept), -1 / slope)
        elif rescaling == 'y-intercept':
            x_r[c] = x[c] / np.exp(intercept)
        elif rescaling == 'slope':
            x_r[c] = np.power(x[c], -1 / slope)
        else:
            raise ValueError('Invalid rescaling')

        logcoeffs[c] = [slope, intercept]

    return x_r, logcoeffs


def detect_peak(x, freqs=None, *, band=(1, 95), band_target=(8, 12),
                huber_t=1.345):
    """Detect peak on spectra following a power law.

    This function detects peaks in a target frequency band of spectra,
    removing the 1/f baseline by linear regression.

    Parameters
    ----------
    x : ndarray, shape (n_channels, n_freqs)
        Array containing spectra, with the first dimension representing the
        channels, and the second dimension representing the frequencies.

    freqs : None | ndarray, shape (n_freqs,), default None
        Vector of size `n_freqs` and containing (positive) frequencies
        abscissa of the spectrum.
        If `freqs` is None, it is assumed that frequencies are
        `[0, 1, ... n_freqs-1]`.

    band : tuple, default (1, 95)
        Low and high frequencies (in Hz) of the spectral band where the linear
        regression is applied.

    band_target : tuple, default (8, 12)
        Low and high frequencies (in Hz) of the spectral band where the peak
        is searched.

    huber_t : float, default 1.345
        The tuning constant for Huber’s t function [1]_, to estimate bounds to
        detect if there is a peak.

    Returns
    -------
    peaks : ndarray, shape (n_channels,)
        Frequencies associated to the spectral peak in the target band, one
        value by channel. If no peak has been found, NaN.

    References
    ----------
    .. [1] http://www.statsmodels.org/dev/generated/statsmodels.robust.norms.HuberT.html
    """

    x = np.atleast_2d(np.asarray(x))
    n_channels, n_freqs = x.shape
    freqs = _check_freqs(freqs, n_freqs)

    if band[1] > freqs[-1]:
        raise ValueError('band[1] is superior to Nyquist frequency')

    # fit and apply robust linear regression on log-log spectra
    foi_reg = (freqs >= band[0]) & (freqs < band[1]) & \
        ~((freqs >= band_target[0]) & (freqs < band_target[1]))
    x, logcoeffs = rescale(x, freqs, rescaling='complete', foi=foi_reg)

    # remove the 1/f baseline from the rescaled log spectrum
    with np.errstate(divide='ignore', invalid='ignore'):
        logx_resids = np.log(x) + np.log(freqs)  # + because -(-1)

    # compute HuberT bounds
    resids_mad = np.median(
        np.abs(
            logx_resids[:, foi_reg]
            - np.median(logx_resids[:, foi_reg], axis=1)[:, np.newaxis]
        ),
        axis=1,
    )

    # find the max in the target band
    foi_max = ((freqs >= band_target[0]) & (freqs < band_target[1]))
    logx_resids = logx_resids[:, foi_max]
    peak_inds = np.argmax(logx_resids, axis=1)
    peaks = freqs[foi_max][peak_inds]

    # detect peak, comparing the residuals at peaks to HuberT bounds
    logx_scaled_resids = logx_resids / resids_mad[:, np.newaxis]
    for c in range(logx_scaled_resids.shape[0]):
        if np.abs(logx_scaled_resids[c, peak_inds[c]]) < huber_t:
            peaks[c] = np.nan

    return peaks


def generate_sig_noise_pink(n_channels, n_samples, fs):
    """Generate signal of pink noise.

    Generate a signal of pink noise, with a spectrum in 1/f.

    Parameters
    ----------
    n_channels : int
        Number of channels to generate.

    n_samples : int
        Number of samples to generate.

    fs : int
        Sampling frequency of the signal to generate, in Hz.

    Returns
    -------
    pink_noise : ndarray, shape (n_channels, n_samples)
        Signal of pink noise.
    """

    noise_pink = np.zeros((n_channels, n_samples))
    spectrum, _ = _pink(n_channels, n_samples, fs)

    for i_channel in range(n_channels):
        # transform from frequencies to time, dropping complex part that
        # may not be zero due to numerical errors
        noise_pink[i_channel] = np.fft.irfft(spectrum[i_channel]).real

    return noise_pink


def _pink(n_channels, n_samples, fs):
    """Generate complex spectrum of pink noise."""
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    spectrum = np.zeros((n_channels, len(freqs)), dtype=np.complex128)
    spec = np.empty_like(freqs, dtype=np.complex128)

    for i_channel in range(n_channels):
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
        if n_samples % 2 == 0:  # even case
            spec[-1] = spec[-1].real
        spectrum[i_channel] = spec

    return spectrum, freqs
