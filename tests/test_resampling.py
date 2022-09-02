"""Tests for module resampling.

To execute tests:
>>> pytest -k test_resampling
"""

import pytest
import numpy as np
from scipy.signal import dlti, cheby1
from pyspectrum.resampling import downsample, upsample, upiirdn

np.random.seed(17)


@pytest.mark.parametrize("fun", [downsample, upsample])
@pytest.mark.parametrize("axis", [1.1, "blabla", 2])
def test_resample_axis_errors(fun, axis):
    """Check resample errors for axis."""
    x = np.random.randn(5, 5)
    with pytest.raises((ValueError, IndexError)):
        fun(x, 2, axis=axis)


@pytest.mark.parametrize("fun", [downsample, upsample])
@pytest.mark.parametrize("factor", [0, 2.5, -3])
def test_resample_factor_errors(fun, factor):
    """Check resample errors for factor."""
    with pytest.raises(ValueError):
        fun(None, factor)


@pytest.mark.parametrize("fun", [downsample, upsample])
@pytest.mark.parametrize("ndim", range(1, 3))
@pytest.mark.parametrize("offset", range(0, 3))
def test_resample_factor_1(fun, ndim, offset):
    """Check resample with factor=1 is identity."""
    dims = np.random.randint(10, high=20, size=ndim)
    x = np.random.randint(-10, high=10, size=dims)
    y = fun(x, factor=1, offset=offset)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal(x.shape, y.shape), 'Input and output must have the same dimensions.'
    np.testing.assert_equal(x, y), 'Input and output must be equal.'


@pytest.mark.parametrize("fun", [downsample, upsample])
@pytest.mark.parametrize("offset", [2.5, -1.1])
def test_resample_offset_errors(fun, offset):
    """Checks resample errors for offset."""
    with pytest.raises(ValueError):
        fun(None, 2, offset=offset)


@pytest.mark.parametrize("ndim", range(1, 4))
@pytest.mark.parametrize("factor", range(1, 5))
def test_downsample_axisnone(ndim, factor):
    """Test function downsample for axis=None."""
    dims = np.random.randint(10, high=50, size=ndim)
    x = np.random.randn(*dims * factor)
    y = downsample(x, factor)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal(y.shape, dims), 'Output dimensions must be integer divisions of input dimensions.'


@pytest.mark.parametrize("ndim", range(1, 4))
@pytest.mark.parametrize("factor", range(1, 5))
def test_downsample_axis(ndim, factor):
    """Test function downsample for a specific axis."""
    axis = np.random.randint(0, high=ndim, size=1)[0]
    dims = np.random.randint(10, high=50, size=ndim)
    x = np.random.randn(*dims * factor)
    y = downsample(x, factor, axis=axis)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal(y.shape[axis], dims[axis]), 'Output dimension to downsample must be an integer division of input dimension.'

    x_shape = [s for i, s in enumerate(x.shape) if i != axis]
    y_shape = [s for i, s in enumerate(y.shape) if i != axis]
    np.testing.assert_equal(x_shape, y_shape), 'Output dimensions not downsampled must be equal to input dimensions.'


@pytest.mark.parametrize("ndim", range(2, 5))
@pytest.mark.parametrize("factor", range(1, 4))
def test_downsample_axes(ndim, factor):
    """Test function downsample for several axes."""
    n_axes = np.random.randint(1, high=ndim, size=1)[0]
    axis = tuple(np.random.randint(0, high=ndim, size=n_axes))
    dims = np.random.randint(10, high=50, size=ndim)
    x = np.random.randn(*dims * factor)
    y = downsample(x, factor, axis=axis)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal([y.shape[a] for a in axis], [dims[a] for a in axis]), 'Output dimension to downsample must be an integer division of input dimension.'

    x_shape = [s for i, s in enumerate(x.shape) if i not in axis]
    y_shape = [s for i, s in enumerate(y.shape) if i not in axis]
    np.testing.assert_equal(x_shape, y_shape), 'Output dimensions not downsampled must be equal to input dimensions.'


def test_downsample_offset():
    """Check downsample offset on 2x2 matrices."""
    x = np.array([[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 4, 0], [0, 0, 0, 0]])
    y = downsample(x, 2)
    z = np.array([[1, 2], [3, 4]])
    np.testing.assert_equal(y, z)

    x = np.array([[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [0, 3, 0, 4]])
    y = downsample(x, 2, offset=1)
    np.testing.assert_equal(y, z)

    x = np.array([[0, 0, 0, 0], [0, 4, 0, 3], [0, 0, 0, 0], [0, 2, 0, 1]])
    y = downsample(x, 2, offset=-1)
    np.testing.assert_equal(y, z)


@pytest.mark.parametrize("ndim", range(1, 4))
@pytest.mark.parametrize("factor", range(1, 5))
def test_upsample_axisnone(ndim, factor):
    """Test function upsample for axis=None."""
    dims = np.random.randint(10, high=50, size=ndim)
    x = np.random.randn(*dims)
    shape = tuple([d * factor for d in x.shape])
    y = upsample(x, factor)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal(y.shape, shape), 'Output dimensions must be multiplicative factors of input dimensions.'


@pytest.mark.parametrize("ndim", range(1, 4))
@pytest.mark.parametrize("factor", range(1, 5))
def test_upsample_axis(ndim, factor):
    """Test function upsample for a specific axis."""
    axis = np.random.randint(0, high=ndim, size=1)[0]
    dims = np.random.randint(10, high=50, size=ndim)
    x = np.random.randn(*dims)
    y = upsample(x, factor, axis=axis)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal(y.shape[axis], dims[axis] * factor), 'Output dimension to upsample must be an integer factor of input dimension.'

    x_shape = [s for i, s in enumerate(x.shape) if i != axis]
    y_shape = [s for i, s in enumerate(y.shape) if i != axis]
    np.testing.assert_equal(x_shape, y_shape), 'Output dimensions not upsampled must be equal to input dimensions.'


@pytest.mark.parametrize("ndim", range(2, 5))
@pytest.mark.parametrize("factor", range(1, 4))
def test_upsample_axes(ndim, factor):
    """Test function upsample for several axes."""
    n_axes = np.random.randint(1, high=ndim, size=1)[0]
    axis = tuple(np.random.randint(0, high=ndim, size=n_axes))
    dims = np.random.randint(10, high=50, size=ndim)
    x = np.random.randn(*dims)
    y = upsample(x, factor, axis=axis)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal([y.shape[a] for a in axis], [dims[a] * factor for a in axis]), 'Output dimension to upsample must be an integer factor of input dimension.'

    x_shape = [s for i, s in enumerate(x.shape) if i not in axis]
    y_shape = [s for i, s in enumerate(y.shape) if i not in axis]
    np.testing.assert_equal(x_shape, y_shape), 'Output dimensions not upsampled must be equal to input dimensions.'


def test_upsample_offset():
    """Check upsample offset on 2x2 matrices."""
    x = np.array([[1, 2], [3, 4]])

    y = upsample(x, 2)
    z = np.array([[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 4, 0], [0, 0, 0, 0]])
    np.testing.assert_equal(y, z)

    y = upsample(x, 2, offset=1)
    z = np.array([[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [0, 3, 0, 4]])
    np.testing.assert_equal(y, z)

    y = upsample(x, 2, offset=-1)
    z = np.array([[0, 0, 0, 0], [0, 4, 0, 3], [0, 0, 0, 0], [0, 2, 0, 1]])
    np.testing.assert_equal(y, z)


@pytest.mark.parametrize("ndim", range(1, 4))
@pytest.mark.parametrize("factor", range(1, 5))
@pytest.mark.parametrize("offset", range(0, 3))
def test_downsample_upsample(ndim, factor, offset):
    """Check that downsample(upsample()) is identity."""
    dims = np.random.randint(3, high=9, size=ndim)
    x = np.random.randint(-10, high=10, size=dims)
    y_ = upsample(x, factor, offset=offset)
    y = downsample(y_, factor, offset=offset)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal(x.shape, y.shape), 'Input and output must have the same dimensions.'
    np.testing.assert_equal(x, y), 'Input and output must be equal.'


@pytest.mark.parametrize("ndim", range(1, 4))
@pytest.mark.parametrize("factor", range(1, 5))
@pytest.mark.parametrize("offset", range(0, 3))
def test_upsample_downsample(ndim, factor, offset):
    """Check that upsample(downsample()) have same dimensions."""
    dims = np.random.randint(3, high=9, size=ndim) * factor
    x = np.random.randint(-10, high=10, size=dims)
    y_ = downsample(x, factor, offset=offset)
    y = upsample(y_, factor, offset=offset)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal(x.shape, y.shape), 'Input and output must have the same dimensions.'


@pytest.mark.parametrize("axis", [2.3, None, "blabla", (1, 2)])
def test_upiirdn_axis_errors(axis):
    """Check upiirdn errors for axis."""
    with pytest.raises((ValueError, TypeError)):
        upiirdn(None, None, axis=axis)


@pytest.mark.parametrize("zero_phase", [True, False])
def test_upiirdn(zero_phase):
    """Check function upiirdn."""
    x = np.random.randn(100)
    up, down = 9, 15
    iir = dlti(*cheby1(8, 0.05, 0.8 * up / down, btype='lowpass'))
    upiirdn(x, iir, up=up, down=down, zero_phase=zero_phase)
