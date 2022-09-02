"""Functions for sample-rate conversion of multidimensional arrays."""

import numbers

import numpy as np
from scipy.signal import lfilter, filtfilt


def _check_args(factor, offset):
    """This function checks arguments."""
    if factor != int(factor):
        raise ValueError('Parameter factor must be an integer')
    if factor <= 0:
        raise ValueError('Parameter factor must be strictly positive')
    if offset != int(offset):
        raise ValueError('Parameter offset must be an integer')


def _check_axis(x, axis):
    """This function checks axis and return axis in tuple format."""
    if axis is None:
        axis = tuple(range(x.ndim))
    else:
        if isinstance(axis, numbers.Integral):
            axis = (axis,)
        if isinstance(axis, tuple):
            if max(axis) >= x.ndim:
                raise ValueError(
                    'Parameter axis higher than the number of dimensions of x')
        else:
            raise ValueError(
                'Parameter axis must be None, int or tuple of int')
    return axis


def downsample(x, factor, *, offset=0, axis=None):
    """Downsample multidimensional array.

    This function reduces sampling rate of a multidimensional array x,
    by integer factor with included offset.
    There is no anti-aliasing filter in this function.

    Parameters
    ----------
    x : array_like
        Array to downsample.

    factor : integer
        Downsampling factor, strictly positive.

    offset : integer, default 0
        Offset for sampling.

    axis : None | int | tuple of int, default None
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

    factor : integer
        Upsampling factor, strictly positive.

    offset : integer, default 0
        Offset for sampling.

    axis : None | int | tuple of int, default None
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
    It is the upfirdn [1]_ counterpart for IIR filtering.

    Parameters
    ----------
    x : array_like
        Array to resample.

    iir : dlti object
        One-dimensional IIR (infinite-impulse response) filter [2]_.

    up : int, default 1
        Upsampling factor.

    down : int, default 1
        Downsampling factor.

    zero_phase : bool, default True
        Prevent phase shift by filtering with filtfilt instead of lfilter.

    axis : int, default -1
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

    x = upsample(x, up, axis=axis) * up  # to keep the signal power

    if zero_phase:
        x = filtfilt(iir.num, iir.den, x, axis=axis)
    else:
        x = lfilter(iir.num, iir.den, x, axis=axis)

    x = downsample(x, down, axis=axis)

    return x
