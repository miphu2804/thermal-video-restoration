"""Degradation simulation module for thermal video sequences (proposal 3.3).

Provides atomic degradation operators, preset severity levels ('light', 'medium', 'heavy'),
and an end-to-end pipeline function `degrade()`.
"""

from thermal_restore.degradation.ops import (
    compress_contrast,
    fixed_stripes,
    gaussian_blur,
    gaussian_noise,
    temporal_stripes,
)
from thermal_restore.degradation.pipeline import degrade
from thermal_restore.degradation.presets import DEFAULT_STEPS, LEVELS, get_preset

__all__ = [
    "DEFAULT_STEPS",
    "LEVELS",
    "compress_contrast",
    "degrade",
    "fixed_stripes",
    "gaussian_blur",
    "gaussian_noise",
    "get_preset",
    "temporal_stripes",
]
