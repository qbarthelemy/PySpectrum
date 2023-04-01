"""
===============================================================================
Spectral rescaling
===============================================================================

Compare different spectral rescalings on simulated spectra.
"""
# Author: Quentin Barthélemy

import numpy as np
from matplotlib import pyplot as plt

from pyspectrum.simulation import spectrum_eeg
from pyspectrum.spectral import rescale


###############################################################################
# Simulate spectra
# ----------------

np.random.seed(1234)

n_channels, fs = 3, 128

spectra, freqs = spectrum_eeg(n_channels, fs, fs)
coeffs_mult = np.random.uniform(1, 10, n_channels)
coeffs_pow = np.random.uniform(0.5, 1.5, n_channels)
spectra = np.multiply(
    np.power(np.absolute(spectra), coeffs_pow[:, np.newaxis]),
    coeffs_mult[:, np.newaxis]
)


###############################################################################
# Plot spectral rescaling using a robust log-log linear regression
# ----------------------------------------------------------------

# define frequencies of interest
foi = (freqs >= 1) & (freqs <= 100) & ~((freqs > 8) & (freqs < 12))

spectra_complete, lr_logcoeffs = rescale(
    spectra,
    freqs,
    rescaling='complete',
    foi=foi
)


# exclusion of 0Hz, to avoid to compute log(0)
spectra_, freqs_ = spectra[:, 1:], freqs[1:]
logfreqs = np.log(freqs_)

y_pred = np.outer(lr_logcoeffs[:, 0], logfreqs) + lr_logcoeffs[:, 1, np.newaxis]
res = np.log(spectra_) - y_pred
mad_scaled = np.median(np.abs(res - np.median(res, axis=1)[:, np.newaxis]), axis=1)
res_scaled = res / mad_scaled[:, np.newaxis]
huber_t = 1.345

fig, ax = plt.subplots(n_channels, 3, sharex='col', figsize=(16, 6))
fig.suptitle('Spectral rescaling using a robust log-log linear regression',
             fontsize=14)
for c in range(n_channels):
    ax[c, 0].plot(freqs_, spectra_[c])
    ax[c, 0].plot(freqs_, np.exp(y_pred[c]))
    ax[c, 0].set_ylabel("Channel " + str(c + 1))
    ax[c, 1].loglog(freqs_, spectra_[c])
    ax[c, 1].loglog(freqs_, np.exp(y_pred[c]), 'o')
    ax[c, 2].fill_between(freqs_, -3*huber_t, 3*huber_t, color='r', alpha=0.25)
    ax[c, 2].fill_between(freqs_, -huber_t, huber_t, color='g', alpha=0.25)
    ax[c, 2].semilogx(freqs_, res_scaled[c], 'o')
ax[0, 0].set_title('Plot raw')
ax[0, 1].set_title('Plot log-log')
ax[0, 2].set_title('Residuals log-log')
plt.show(block=False)


###############################################################################


fig, ax = plt.subplots(n_channels, 2, figsize=(10, 6))
fig.suptitle('Raw versus rescaled spectra using a robust log-log linear regression',
             fontsize=14)
for c in range(n_channels):
    ax[c, 0].plot(freqs, spectra[c])
    ax[c, 0].set_ylabel("Channel " + str(c + 1))
    ax[c, 1].plot(freqs, spectra_complete[c])
ax[0, 0].set_title('Original spectrum')
ax[0, 1].set_title('Rescaled spectrum')
plt.show(block=False)


###############################################################################
# Compare different spectral rescalings
# -------------------------------------

# Rescaling "Global Frobenius norm":
# it divides spectra by its Frobenius norm.
# This rescaling suffers from inter-channel variabilities (in power).
spectra_fro = spectra / np.linalg.norm(spectra[:, foi], ord='fro')

# Rescaling "Channel-wise 2-norm":
# for each channel, it divides spectrum by its 2-norm.
spectra_channel2 = spectra / np.linalg.norm(spectra[:, foi], axis=1, ord=2, keepdims=True)

# Rescaling "Channel-wise 1/f y-intercept":
# rescaling using only the y-intercept of the robust log-log linear regression.
spectra_yintercept, _ = rescale(spectra, freqs, rescaling='y-intercept', foi=foi)

# Rescaling "Channel-wise 1/f slope":
# rescaling using only the slope of the robust log-log linear regression.
spectra_slope, _ = rescale(spectra, freqs, rescaling='slope', foi=foi)

# Rescaling "Frequency-wise 2-norm":
# for each frequency, it divides spectrum by its 2-norm.
# This normalization removes the volume conduction in sources spectra in BSS.
spectra_freq2 = spectra / np.linalg.norm(spectra, axis=0, ord=2, keepdims=True)


fig, ax = plt.subplots(2, 6, figsize=(20, 10))
fig.suptitle('Compare different spectral rescalings', fontsize=14)
for c in range(n_channels):
    ax[0, 0].plot(freqs, spectra_fro[c])
    ax[1, 0].plot(logfreqs, np.log10(spectra_fro[c, 1:]))
    ax[0, 1].plot(freqs, spectra_channel2[c])
    ax[1, 1].plot(logfreqs, np.log10(spectra_channel2[c, 1:]))
    ax[0, 2].plot(freqs, spectra_yintercept[c])
    ax[1, 2].plot(logfreqs, np.log10(spectra_yintercept[c, 1:]))
    ax[0, 3].plot(freqs, spectra_slope[c])
    ax[1, 3].plot(logfreqs, np.log10(spectra_slope[c, 1:]))
    ax[0, 4].plot(freqs, spectra_complete[c])
    ax[1, 4].plot(logfreqs, np.log10(spectra_complete[c, 1:]))
    ax[0, 5].plot(freqs, spectra_freq2[c])
    ax[1, 5].plot(logfreqs, np.log10(spectra_freq2[c, 1:]))
ax[0, 0].set_title('Global Frobenius norm')
ax[0, 1].set_title('Channel 2 norm')
ax[0, 2].set_title('Channel 1/f y-intercept')
ax[0, 3].set_title('Channel 1/f slope')
ax[0, 4].set_title('Channel 1/f')
ax[0, 5].set_title('Frequency 2 norm')
ax[0, 0].set_ylabel('Plot')
ax[1, 0].set_ylabel('Plot log-log')
plt.show()
