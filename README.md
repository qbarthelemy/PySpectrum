# PySpectrum

[![Code PythonVersion](https://img.shields.io/badge/python-3.7+-blue)](https://img.shields.io/badge/python-3.7+-blue)
[![License](https://img.shields.io/badge/licence-BSD--3--Clause-green)](https://img.shields.io/badge/license-BSD--3--Clause-green)

PySpectrum is a Python package for spectral analyses,
looking at sampling rate conversion and its potential aliasing artifacts.

PySpectrum is distributed under the open source 3-clause BSD license.

## Description

### Sample-rate conversion (SRC)

In signal processing,
[sample-rate conversion](https://en.wikipedia.org/wiki/Sample-rate_conversion)
(SRC) is the process of changing the sampling rate of a discrete signal: 
[downsampling](https://en.wikipedia.org/wiki/Downsampling_(signal_processing))
or [upsampling](https://en.wikipedia.org/wiki/Upsampling).
SRC can generate
[aliasing artifacts](https://en.wikipedia.org/wiki/Aliasing) when
[Nyquist–Shannon sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem)
is not respected.

This package implements several functions to complete the
[Signal processing module of SciPy](https://docs.scipy.org/doc/scipy/reference/signal.html#filtering):
- `downsample` and `upsample` for multidimensional arrays,
- `upiirdn` the [scipy.signal.upfirdn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html)
counterpart for IIR filtering, allowing **fractional** downsampling in sample domain contrary to
[scipy.signal.decimate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html).

#### 2D image

Image downsampled without anti-aliasing filter shows
[aliasing artifacts](https://en.wikipedia.org/wiki/Aliasing).
This kind of downsampling is present in max-pooling, strided-convolution
and more generally in strided-layers.
These architectural components are widely used in convolutional neural networks
like [ResNets](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf),
[DenseNets](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf),
[MobileNets](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf),
degrading performances in terms of
[shift-invariance and classification accuracy](http://proceedings.mlr.press/v97/zhang19a/zhang19a.pdf).

![](/doc/fig_downsample_img.png)

See `examples\downsample_img.py` for the complete analysis.

#### Time series

Time series of this example is an electrocardiogram (ECG) signal downsampled from 360 Hz to 180 Hz.
Signal downsampled without anti-aliasing filter shows aliasing artifacts,
ie. a huge artifact peak at 80 Hz. This kind of downsampling can be present in quickly coded drivers.

![](/doc/fig_downsample_ecg.png)

See `examples\downsample_ecg.py` for the complete analysis.

#### Real-time SRC

Offline resampling can use non-causal processing like Fourier transform,
or bilateral / forward-backward filters to provide zero-phase distortion.

Online resampling is limited to causal processing, ie forward filters:
FIR filters generating a delay but no phase distortion, or
IIR filters generating no delay but a phase distortion.

Real-time resampling requires the smallest delay, restricting usage to IIR
filters.

C++ libraries for real-time SRC:
[DSPFilters](https://github.com/vinniefalco/DSPFilters)
and [r8brain](https://github.com/avaneev/r8brain-free-src).


## Installation

#### From sources

To install PySpectrum as a standard module:
```shell 
pip install path/to/PySpectrum
```

To install PySpectrum in editable / development mode, in the folder:
```shell
pip install poetry
poetry install
```

## Testing

Use `pytest`.

