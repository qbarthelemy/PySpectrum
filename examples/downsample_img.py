"""
===============================================================================
2D image downsampling
===============================================================================

Spatial and spectral analyses of artifacts after downsampling of a 2D image.
"""
# Author: Quentin Barthélemy

import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image

from pyspectrum.src import downsample
from pyspectrum.viz import plot_spectrum2, remove_ticks


###############################################################################
# Original image
# --------------

img = scipy.misc.ascent()


###############################################################################
# Downsample image with anti-aliasing filter
# ------------------------------------------

factor = 4

height, width = img.shape
img_r = np.array(
    Image.fromarray(img).resize(
        (height // factor, width // factor),
        resample=Image.HAMMING
    )
)


###############################################################################
# Downsample image without anti-aliasing filter
# ---------------------------------------------

img_d = downsample(img, factor=factor, axis=None)


###############################################################################
# Plot images
# -----------

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
fig.suptitle('Downsampled images', fontsize=14)
plt.gray()
ax[0].imshow(img)
ax[1].imshow(img_r)
ax[2].imshow(img_d)
ax[0].set_xlabel('Original')
ax[1].set_xlabel('With anti-aliasing filter')
ax[2].set_xlabel('Without anti-aliasing filter')
ax = remove_ticks(ax)
plt.show()


###############################################################################
# Plot zooms on barrier
# ---------------------

s1, s2, s3 = 15, 50, 40
zoom = (slice(s1 * factor, s2 * factor, 1), slice(0, s3 * factor, 1))
zoom_d = (slice(s1, s2, 1), slice(0, s3, 1))

patch = img[zoom]
patch_r = img_r[zoom_d]
patch_d = img_d[zoom_d]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
fig.suptitle('Zoom in downsampled images', fontsize=14)
ax[0].imshow(patch)
ax[1].imshow(patch_r)
ax[2].imshow(patch_d)
ax[0].set_xlabel('Original')
ax[1].set_xlabel('With anti-aliasing filter')
ax[2].set_xlabel('Without anti-aliasing filter')
ax = remove_ticks(ax)
plt.show()


###############################################################################
# Plot spectra
# ------------

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
fig.suptitle('Spectra of downsampled images', fontsize=14)
plot_spectrum2(ax[0], img)
plot_spectrum2(ax[1], img_r)
plot_spectrum2(ax[2], img_d)
right, bottom = ax[0].get_xlim()[1], ax[0].get_ylim()[0]
ax[0].set_xlim(3 * right // 8, 5 * right // 8)
ax[0].set_ylim(5 * bottom // 8, 3 * bottom // 8)
ax[0].set_xlabel('Original')
ax[1].set_xlabel('With anti-aliasing filter')
ax[2].set_xlabel('Without anti-aliasing filter')
ax = remove_ticks(ax)
plt.show()


###############################################################################
# References
# ----------
# .. [1] https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
