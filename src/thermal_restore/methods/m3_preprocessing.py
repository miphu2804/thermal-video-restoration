"""Point, spatial, frequency and wavelet preprocessing candidates.

These operators are deliberately registered as opt-in baselines. They can
change the input distribution seen by the frozen detector, so a method must be
validated before it is used for inference.

The stretching operators (``fshs``, ``log``) remap each frame to a full output
range. With the default 0-255 range their output is far outside the fixed
statistics a raw-14-bit detector was trained on; pass ``out_low``/``out_high``
near the detector's training range to keep the stretch in distribution.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy import ndimage

from thermal_restore.methods.registry import register


def _frames(frames: np.ndarray) -> np.ndarray:
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("Expected finite frames with shape (T, H, W)")
    return array


def _odd_size(size: int) -> int:
    if size < 1 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd integer, got {size}")
    return size


def _clip(frames: np.ndarray) -> np.ndarray:
    return np.clip(frames, 0.0, 255.0).astype(np.float32)


def _output_range(out_low: float, out_high: float) -> tuple[float, float]:
    if not (np.isfinite(out_low) and np.isfinite(out_high)):
        raise ValueError("out_low and out_high must be finite")
    if not 0.0 <= out_low < out_high <= 255.0:
        raise ValueError("output range must satisfy 0 <= out_low < out_high <= 255")
    return float(out_low), float(out_high)


@register("linear")
def linear(frames: np.ndarray, gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Apply ``gain * f + offset`` and clip to the project scale."""
    if not np.isfinite(gain) or not np.isfinite(offset):
        raise ValueError("gain and offset must be finite")
    return _clip(float(gain) * _frames(frames) + float(offset))


@register("fshs")
def full_scale_stretch(
    frames: np.ndarray,
    low_percentile: float | None = None,
    high_percentile: float | None = None,
    out_low: float = 0.0,
    out_high: float = 255.0,
) -> np.ndarray:
    """Stretch each frame to ``[out_low, out_high]`` using min/max or percentiles.

    The default 0-255 range is a display-style stretch. For a detector trained
    on linear raw-14-bit frames, choose an output range around its training
    mean/std so the stretched frame stays in distribution.
    """
    out_low, out_high = _output_range(out_low, out_high)
    if (low_percentile is None) != (high_percentile is None):
        raise ValueError("low_percentile and high_percentile must be paired")
    if low_percentile is not None and (
        not 0 <= low_percentile < high_percentile <= 100
    ):
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    array = _frames(frames)
    if low_percentile is None:
        low = array.min(axis=(-2, -1), keepdims=True)
        high = array.max(axis=(-2, -1), keepdims=True)
    else:
        low, high = np.percentile(
            array, [low_percentile, high_percentile], axis=(-2, -1), keepdims=True
        )
    scale = (out_high - out_low) / np.maximum(high - low, 1e-6)
    return np.clip((array - low) * scale + out_low, out_low, out_high).astype(
        np.float32
    )


@register("log")
def logarithmic(
    frames: np.ndarray, out_low: float = 0.0, out_high: float = 255.0
) -> np.ndarray:
    """Apply ``log(1 + f)`` then stretch each frame to ``[out_low, out_high]``."""
    array = np.maximum(_frames(frames), 0.0)
    return full_scale_stretch(np.log1p(array), out_low=out_low, out_high=out_high)


@register("mean")
def arithmetic_mean(frames: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a spatial arithmetic-mean filter independently per frame."""
    size = _odd_size(size)
    return ndimage.uniform_filter(
        _frames(frames), size=(1, size, size), mode="reflect"
    ).astype(np.float32)


@register("geometric")
def geometric_mean(
    frames: np.ndarray, size: int = 3, epsilon: float = 1e-3
) -> np.ndarray:
    """Apply a spatial geometric-mean filter to non-negative intensities."""
    size = _odd_size(size)
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    array = np.maximum(_frames(frames), 0.0)
    result = (
        np.exp(
            ndimage.uniform_filter(
                np.log(array + epsilon), size=(1, size, size), mode="reflect"
            )
        )
        - epsilon
    )
    return _clip(result)


@register("harmonic")
def harmonic_mean(
    frames: np.ndarray, size: int = 3, epsilon: float = 1e-3
) -> np.ndarray:
    """Apply a spatial harmonic-mean filter to non-negative intensities."""
    size = _odd_size(size)
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    array = np.maximum(_frames(frames), 0.0)
    reciprocal = ndimage.uniform_filter(
        1.0 / (array + epsilon), size=(1, size, size), mode="reflect"
    )
    return _clip(1.0 / np.maximum(reciprocal, 1e-12) - epsilon)


@register("contraharmonic")
def contraharmonic_mean(
    frames: np.ndarray, size: int = 3, q: float = 1.5, epsilon: float = 1e-3
) -> np.ndarray:
    """Apply a contraharmonic mean filter with exponent ``q``."""
    size = _odd_size(size)
    if not np.isfinite(q) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("q must be finite and epsilon must be positive")
    array = np.maximum(_frames(frames), 0.0) + epsilon
    numerator = ndimage.uniform_filter(
        array ** (q + 1.0), size=(1, size, size), mode="reflect"
    )
    denominator = ndimage.uniform_filter(array**q, size=(1, size, size), mode="reflect")
    return _clip(numerator / np.maximum(denominator, 1e-12) - epsilon)


@register("fourier_lowpass")
def fourier_lowpass(frames: np.ndarray, cutoff: float = 0.15) -> np.ndarray:
    """Apply a Gaussian low-pass filter in the 2-D Fourier domain."""
    if not 0 < cutoff <= 0.5 or not np.isfinite(cutoff):
        raise ValueError("cutoff must be finite and in (0, 0.5]")
    array = _frames(frames)
    fy = np.fft.fftfreq(array.shape[-2])[:, None]
    fx = np.fft.fftfreq(array.shape[-1])[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    transfer = np.exp(-0.5 * (radius / cutoff) ** 2)
    spectrum = np.fft.fft2(array, axes=(-2, -1))
    restored = np.fft.ifft2(spectrum * transfer, axes=(-2, -1)).real
    return _clip(restored)


@register("fourier_notch")
def fourier_notch(
    frames: np.ndarray,
    bandwidth: float = 0.005,
    preserve: float = 0.01,
    axis: str = "x",
) -> np.ndarray:
    """Suppress stripes along the zero-frequency line of the stripe axis.

    Column (vertical) stripes vary only along x, so their energy lies on the
    ``fy = 0`` line of the 2-D spectrum. ``axis="x"`` attenuates that line;
    ``axis="y"`` handles row stripes on ``fx = 0``. Frequencies along the line
    below ``preserve`` cycles/pixel are kept so the DC level and slow scene
    gradients survive.

    Args:
        frames: Sequence of shape ``(T, H, W)`` on the 0-255 scale.
        bandwidth: Gaussian half-width of the notch across the line, in
            cycles/pixel.
        preserve: Gaussian half-width of the protected low-frequency region
            along the line, in cycles/pixel.
        axis: Axis along which the stripe offsets vary: ``"x"`` for column
            stripes, ``"y"`` for row stripes.
    """
    if not 0 < bandwidth <= 0.5 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be finite and in (0, 0.5]")
    if not 0 < preserve <= 0.5 or not np.isfinite(preserve):
        raise ValueError("preserve must be finite and in (0, 0.5]")
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    array = _frames(frames)
    fy = np.fft.fftfreq(array.shape[-2])[:, None]
    fx = np.fft.fftfreq(array.shape[-1])[None, :]
    across, along = (fy, fx) if axis == "x" else (fx, fy)
    on_line = np.exp(-0.5 * (across / bandwidth) ** 2)
    protected = np.exp(-0.5 * (along / preserve) ** 2)
    transfer = 1.0 - on_line * (1.0 - protected)
    spectrum = np.fft.fft2(array, axes=(-2, -1))
    restored = np.fft.ifft2(spectrum * transfer, axes=(-2, -1)).real
    return _clip(restored)


@register("wavelet_soft")
def wavelet_soft(
    frames: np.ndarray,
    wavelet: str = "db2",
    level: int = 1,
    threshold_scale: float = 1.0,
) -> np.ndarray:
    """Denoise detail coefficients with universal soft thresholding.

    The noise level is the robust MAD estimate on the finest diagonal band
    (HH1), which contains little scene structure, so edges do not inflate it.
    """
    if level < 1 or not np.isfinite(threshold_scale) or threshold_scale <= 0:
        raise ValueError("level must be positive and threshold_scale must be positive")
    array = _frames(frames)
    restored = []
    for frame in array:
        coefficients = pywt.wavedec2(
            frame, wavelet=wavelet, level=level, mode="symmetric"
        )
        finest_diagonal = coefficients[-1][2]
        sigma = np.median(np.abs(finest_diagonal)) / 0.6745
        threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(frame.size))
        if threshold <= 0:
            # Flat or heavily quantized frames (e.g. 8-bit compressed video)
            # can give a zero HH1 median; pywt would then divide 0 by 0.
            restored.append(frame)
            continue
        filtered = [coefficients[0]]
        filtered.extend(
            tuple(pywt.threshold(band, threshold, mode="soft") for band in bands)
            for bands in coefficients[1:]
        )
        restored.append(
            pywt.waverec2(filtered, wavelet=wavelet, mode="symmetric")[
                : frame.shape[0], : frame.shape[1]
            ]
        )
    return _clip(np.stack(restored))
