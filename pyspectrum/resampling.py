"""Functions for sample-rate conversion of multidimensional arrays."""

from fractions import Fraction
import numbers

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple
from scipy.signal import dlti, lfilter, filtfilt


def _check_args(factor, offset):
    """This function checks arguments."""
    if factor != int(factor):
        raise ValueError('Parameter factor must be an integer')
    if factor <= 0:
        raise ValueError('Parameter factor must be strictly positive')
    if offset != int(offset):
        raise ValueError('Parameter offset must be an integer')


def _check_axis(x, axis):
    """This function checks axis and return axis in tuple format.

    Negative axes are converted to positive ones, and repeated or
    out-of-bounds axes raise a ValueError.
    """
    if axis is None:
        return tuple(range(x.ndim))
    if isinstance(axis, numbers.Integral):
        axis = (axis,)
    if not (isinstance(axis, tuple)
            and all(isinstance(a, numbers.Integral) for a in axis)):
        raise ValueError('Parameter axis must be None, int or tuple of int')
    return normalize_axis_tuple(axis, x.ndim, argname='axis')


def get_src(fi, fo):
    """Compute quantities useful for sample rate conversion.

    Parameters
    ----------
    fi : int
        Input sampling frequency.

    fo : int
        Output sampling frequency.

    Returns
    -------
    src : float
        Sample rate conversion.

    up : int
        Upsampling factor.

    down : int
        Downsampling factor.
    """
    src = fo / fi
    frac = Fraction(fo, fi)  # useful for fractional src
    up, down = frac.numerator, frac.denominator
    return src, up, down


def downsample(x, factor, *, offset=0, axis=None):
    """Downsample multidimensional array.

    This function reduces sampling rate of a multidimensional array x,
    by integer factor with included offset.
    There is no anti-aliasing filter in this function.

    Parameters
    ----------
    x : array_like
        Array to downsample.

    factor : int
        Downsampling factor, strictly positive.

    offset : int, default=0
        Offset for sampling, circular.

    axis : None | int | tuple of int, default=None
        Axis or axes along which to downsample.
        If None, input array is downsampled along all its dimensions.

    Returns
    -------
    y : ndarray
        Downsampled array.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Downsampling_(signal_processing)
    """
    _check_args(factor, offset)
    x = np.asarray(x)
    if factor == 1:
        return x

    axis = _check_axis(x, axis)
    idx = [slice(None)] * x.ndim
    for a in axis:
        idx[a] = np.s_[::factor]

    x = np.roll(x, -offset, axis=axis)
    y = x[tuple(idx)]

    return y


def upsample(x, factor, *, offset=0, axis=None):
    """Upsample multidimensional array.

    This function increases sampling rate of a multidimensional array x,
    by integer factor with included offset, adding zeros.
    There is no interpolation filter in this function.

    Parameters
    ----------
    x : array_like
        Array to upsample.

    factor : int
        Upsampling factor, strictly positive.

    offset : int, default=0
        Offset for sampling, circular.

    axis : None | int | tuple of int, default=None
        Axis or axes along which to upsample.
        If None, input array is upsampled along all its dimensions.

    Returns
    -------
    y : ndarray
        Upsampled array.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Upsampling
    """
    _check_args(factor, offset)
    x = np.asarray(x)
    if factor == 1:
        return x

    axis = _check_axis(x, axis)
    dims = [1] * x.ndim
    for a in axis:
        dims[a] = factor
    u = np.zeros(dims)
    u[tuple([0] * x.ndim)] = 1

    y_ = np.kron(x, u)
    y = np.roll(y_, offset, axis=axis)

    return y


def upiirdn(x, iir, *, up=1, down=1, zero_phase=True, axis=-1):
    """Upsample, IIR filter, and downsample.

    This function performs an IIR-based resampling: upsampling,
    IIR anti-aliasing filtering, and downsampling.
    It is the ``upfirdn`` [1]_ counterpart for IIR filtering.

    Parameters
    ----------
    x : array_like
        Array to resample.

    iir : dlti
        One-dimensional discrete-time IIR (infinite-impulse response) filter
        [2]_, in any representation: transfer function, zeros-poles-gain or
        state-space.

    up : int, default=1
        Upsampling factor.

    down : int, default=1
        Downsampling factor.

    zero_phase : bool, default True
        Prevent phase shift by filtering with ``filtfilt`` instead of
        ``lfilter``.

    axis : int, default=-1
        Axis along which to resample: upsample, filter (applied to each
        subarray along this axis), and downsample.

    Returns
    -------
    y : ndarray
        Resampled array.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlti.html
    """
    if axis != int(axis):
        raise ValueError('Parameter axis must be an integer')
    if not isinstance(iir, dlti):
        raise TypeError('Parameter iir must be a discrete-time dlti object')
    tf = iir.to_tf()  # convert zeros-poles-gain or state-space representation

    x = upsample(x, up, axis=axis) * up  # to keep the signal amplitude

    if zero_phase:
        x = filtfilt(tf.num, tf.den, x, axis=axis)
    else:
        x = lfilter(tf.num, tf.den, x, axis=axis)

    x = downsample(x, down, axis=axis)

    return x
