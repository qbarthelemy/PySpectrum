"""Tests for module resampling.

To execute tests:
>>> pytest -k test_resampling
"""

import pytest
import numpy as np
from scipy.signal import dlti, lti, cheby1

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
    assert x.shape == y.shape, 'Input and output must have the same dimensions.'
    np.testing.assert_equal(
        x,
        y,
        'Input and output must be equal.'
    )


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
    np.testing.assert_equal(
        y.shape,
        dims,
        'Output dimensions must be integer divisions of input dimensions.'
    )


@pytest.mark.parametrize("ndim", range(1, 4))
@pytest.mark.parametrize("factor", range(1, 5))
def test_downsample_axis(ndim, factor):
    """Test function downsample for a specific axis."""
    axis = np.random.randint(0, high=ndim, size=1)[0]
    dims = np.random.randint(10, high=50, size=ndim)
    x = np.random.randn(*dims * factor)
    y = downsample(x, factor, axis=axis)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    assert y.shape[axis] == dims[axis], 'Output dimension to downsample must be an integer division of input dimension.'

    x_shape = [s for i, s in enumerate(x.shape) if i != axis]
    y_shape = [s for i, s in enumerate(y.shape) if i != axis]
    assert x_shape == y_shape, 'Output dimensions not downsampled must be equal to input dimensions.'


@pytest.mark.parametrize("ndim", range(2, 5))
@pytest.mark.parametrize("factor", range(1, 4))
def test_downsample_axes(ndim, factor):
    """Test function downsample for several axes."""
    n_axes = np.random.randint(1, high=ndim, size=1)[0]
    axis = tuple(np.random.choice(ndim, size=n_axes, replace=False))
    dims = np.random.randint(10, high=50, size=ndim)
    x = np.random.randn(*dims * factor)
    y = downsample(x, factor, axis=axis)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal(
        [y.shape[a] for a in axis],
        [dims[a] for a in axis],
        'Output dimension to downsample must be an integer division of input dimension.'
    )

    x_shape = [s for i, s in enumerate(x.shape) if i not in axis]
    y_shape = [s for i, s in enumerate(y.shape) if i not in axis]
    assert x_shape == y_shape, 'Output dimensions not downsampled must be equal to input dimensions.'


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
    assert y.shape == shape, 'Output dimensions must be multiplicative factors of input dimensions.'


@pytest.mark.parametrize("ndim", range(1, 4))
@pytest.mark.parametrize("factor", range(1, 5))
def test_upsample_axis(ndim, factor):
    """Test function upsample for a specific axis."""
    axis = np.random.randint(0, high=ndim, size=1)[0]
    dims = np.random.randint(10, high=50, size=ndim)
    x = np.random.randn(*dims)
    y = upsample(x, factor, axis=axis)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    assert y.shape[axis] == dims[axis] * factor, 'Output dimension to upsample must be an integer factor of input dimension.'

    x_shape = [s for i, s in enumerate(x.shape) if i != axis]
    y_shape = [s for i, s in enumerate(y.shape) if i != axis]
    assert x_shape == y_shape, 'Output dimensions not upsampled must be equal to input dimensions.'


@pytest.mark.parametrize("ndim", range(2, 5))
@pytest.mark.parametrize("factor", range(1, 4))
def test_upsample_axes(ndim, factor):
    """Test function upsample for several axes."""
    n_axes = np.random.randint(1, high=ndim, size=1)[0]
    axis = tuple(np.random.choice(ndim, size=n_axes, replace=False))
    dims = np.random.randint(10, high=50, size=ndim)
    x = np.random.randn(*dims)
    y = upsample(x, factor, axis=axis)

    assert x.ndim == y.ndim, 'Input and output must have the same number of dimensions.'
    np.testing.assert_equal(
        [y.shape[a] for a in axis],
        [dims[a] * factor for a in axis],
        'Output dimension to upsample must be an integer factor of input dimension.'
    )

    x_shape = [s for i, s in enumerate(x.shape) if i not in axis]
    y_shape = [s for i, s in enumerate(y.shape) if i not in axis]
    assert x_shape == y_shape, 'Output dimensions not upsampled must be equal to input dimensions.'


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
    assert x.shape == y.shape, 'Input and output must have the same dimensions.'
    np.testing.assert_equal(x, y, 'Input and output must be equal.')


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
    assert x.shape == y.shape, 'Input and output must have the same dimensions.'


@pytest.mark.parametrize("fun", [downsample, upsample])
@pytest.mark.parametrize("axis", [(0, 0), (1, -1), -3, (0, 2), (0, 1.0)])
def test_resample_axis_invalid(fun, axis):
    """Check resample raises ValueError for repeated or out-of-bounds axes."""
    x = np.random.randn(4, 4)
    with pytest.raises(ValueError):
        fun(x, 2, offset=1, axis=axis)


@pytest.mark.parametrize("fun", [downsample, upsample])
@pytest.mark.parametrize("offset", [0, 1, -1])
def test_resample_negative_axis(fun, offset):
    """Check negative axis is equivalent to positive axis."""
    x = np.random.randn(4, 6, 8)
    np.testing.assert_equal(
        fun(x, 2, offset=offset, axis=-1),
        fun(x, 2, offset=offset, axis=2),
    )
    np.testing.assert_equal(
        fun(x, 2, offset=offset, axis=(-1, 0)),
        fun(x, 2, offset=offset, axis=(2, 0)),
    )


@pytest.mark.parametrize("axis", [2.3, None, "blabla", (1, 2)])
def test_upiirdn_axis_errors(axis):
    """Check upiirdn errors for axis."""
    with pytest.raises((ValueError, TypeError)):
        upiirdn(None, None, axis=axis)


UP_DOWN = [(1, 2), (1, 4), (2, 3), (3, 2), (9, 15)]


def _lowpass(up, down):
    """Anti-aliasing / anti-imaging IIR filter, designed at the upsampled rate."""
    return dlti(*cheby1(8, 0.05, 0.8 / max(up, down), btype='lowpass'))


def _amplitude(y, freq, fs):
    """Amplitude of a sinusoid at freq, by projection, ignoring 10% edges."""
    m = len(y) // 10
    t = np.arange(len(y))[m:-m] / fs
    return 2 * np.abs(np.mean(y[m:-m] * np.exp(-2j * np.pi * freq * t)))


@pytest.mark.parametrize("up, down", UP_DOWN)
@pytest.mark.parametrize("n_times", [100, 101, 997])
@pytest.mark.parametrize("zero_phase", [True, False])
def test_upiirdn_length(up, down, n_times, zero_phase):
    """Check output length is ceil(n_times * up / down)."""
    x = np.random.randn(n_times)
    y = upiirdn(x, _lowpass(up, down), up=up, down=down, zero_phase=zero_phase)
    assert y.shape == (int(np.ceil(n_times * up / down)),)


@pytest.mark.parametrize("zero_phase", [True, False])
def test_upiirdn_identity(zero_phase):
    """Check upiirdn with up=down=1 and identity filter is identity."""
    x = np.random.randn(100)
    iir = dlti([1., 0.], [1., 0.])  # identity filter, of length 2 for filtfilt
    y = upiirdn(x, iir, zero_phase=zero_phase)
    np.testing.assert_allclose(y, x, err_msg='Output must be equal to input.')


@pytest.mark.parametrize("up, down", UP_DOWN)
@pytest.mark.parametrize("zero_phase", [True, False])
def test_upiirdn_passband(up, down, zero_phase):
    """Check a sinusoid in the passband keeps its amplitude."""
    fs_in, n_times = 1000, 4000
    fs_out = fs_in * up / down
    freq = 0.1 * min(fs_in, fs_out)
    x = np.cos(2 * np.pi * freq * np.arange(n_times) / fs_in)
    y = upiirdn(x, _lowpass(up, down), up=up, down=down, zero_phase=zero_phase)
    np.testing.assert_allclose(
        _amplitude(y, freq, fs_out), 1, atol=0.02,
        err_msg='Amplitude in passband must be preserved.',
    )


@pytest.mark.parametrize("up, down", [(1, 2), (1, 4), (2, 3), (9, 15)])
@pytest.mark.parametrize("zero_phase", [True, False])
def test_upiirdn_antialiasing(up, down, zero_phase):
    """Check a sinusoid above output Nyquist is removed, not aliased."""
    fs_in, n_times = 1000, 4000
    freq = 0.45 * fs_in  # above output Nyquist, below input Nyquist
    assert freq > fs_in * up / down / 2
    x = np.cos(2 * np.pi * freq * np.arange(n_times) / fs_in)
    y = upiirdn(x, _lowpass(up, down), up=up, down=down, zero_phase=zero_phase)
    m = len(y) // 10
    rms = np.sqrt(2 * np.mean(y[m:-m] ** 2))  # 1 for a non attenuated sinusoid
    assert rms < 1e-2, 'Frequencies above output Nyquist must be attenuated.'


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("zero_phase", [True, False])
def test_upiirdn_axis(axis, zero_phase):
    """Check resampling a 2D array along an axis equals resampling each 1D signal."""
    up, down = 2, 3
    x = np.random.randn(60, 50)
    iir = _lowpass(up, down)
    y = upiirdn(x, iir, up=up, down=down, zero_phase=zero_phase, axis=axis)
    x_ = np.moveaxis(x, axis, -1)
    y_ref = np.stack([
        upiirdn(xi, iir, up=up, down=down, zero_phase=zero_phase) for xi in x_
    ])
    np.testing.assert_allclose(np.moveaxis(y, axis, -1), y_ref, atol=1e-10)


@pytest.mark.parametrize("zero_phase", [True, False])
def test_upiirdn_dlti_representations(zero_phase):
    """Check upiirdn gives same output for tf, zpk and ss representations."""
    up, down = 2, 3
    x = np.random.randn(500)
    iir_tf = _lowpass(up, down)
    y_tf = upiirdn(x, iir_tf, up=up, down=down, zero_phase=zero_phase)
    for iir in [iir_tf.to_zpk(), iir_tf.to_ss()]:
        y = upiirdn(x, iir, up=up, down=down, zero_phase=zero_phase)
        np.testing.assert_allclose(
            y, y_tf, rtol=1e-6, atol=1e-8,
            err_msg=f'Output differs for {type(iir).__name__}.',
        )


@pytest.mark.parametrize("iir", [None, lti([1.], [1., 1.]), ([1.], [1.])])
def test_upiirdn_iir_errors(iir):
    """Check upiirdn raises TypeError if iir is not a discrete-time dlti."""
    with pytest.raises(TypeError):
        upiirdn(np.random.randn(10), iir)
