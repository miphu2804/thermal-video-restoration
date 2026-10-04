"""Pure A/B/C helpers; source I/O and detector execution stay outside here."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np

from thermal_restore.degradation import degrade, fixed_stripes, get_preset
from thermal_restore.methods import get_method
from thermal_restore.metrics.image import mse, ssim


def _rng(seed: int, key: str) -> np.random.Generator:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 32, 4)]
    return np.random.default_rng(np.random.SeedSequence([seed, *words]))


def degrade_frame(
    frame: np.ndarray,
    *,
    sequence_key: str,
    frame_key: str,
    level: str = "medium",
    seed: int = 0,
    steps: Sequence[str] | None = None,
) -> np.ndarray:
    """Stream degradation independent of batching, selection and method order.

    Fixed offsets are keyed by sequence, while temporal/pixel noise is keyed
    by frame. Reusing seed across severity levels scales the same draws.
    Never regenerate degradation independently for competing restorations.
    """
    array = np.asarray(frame, dtype=np.float32)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("Expected a finite non-empty (H, W) frame")
    params = get_preset(level)
    offsets = fixed_stripes(
        np.zeros((1, 1, array.shape[1]), dtype=np.float32),
        sigma=params["fixed_stripes"],
        rng=_rng(seed, f"fixed:{sequence_key}"),
    )
    return degrade(
        array[None],
        params,
        _rng(seed, f"frame:{sequence_key}:{frame_key}"),
        steps=steps,
        fixed_offsets=offsets,
    )[0]


def build_restoration(stages: list[dict]) -> Callable[[np.ndarray], np.ndarray]:
    """Build a registry-based single-frame M0-M2 chain with explicit params."""
    if not isinstance(stages, list) or not stages:
        raise ValueError("A restoration pipeline requires a non-empty stage list")
    methods = []
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) - {"name", "params"}:
            raise ValueError("Each stage must contain name and optional params")
        if not isinstance(stage.get("name"), str) or not isinstance(
            stage.get("params", {}), dict
        ):
            raise TypeError("Invalid method name or params")
        methods.append(get_method(stage["name"], **stage.get("params", {})))

    def restore(frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("Expected a finite (H, W) restoration input")
        out = array[None].copy()
        for method in methods:
            out = np.asarray(method(out), dtype=np.float32)
            if out.shape != (1, *array.shape) or not np.isfinite(out).all():
                raise ValueError(
                    "Restoration must preserve frame shape and finite values"
                )
        return np.clip(out[0], 0, 255).astype(np.float32)

    return restore


class QualityTotals:
    """Own pixel-weighted MSE and frame-averaged SSIM accumulators."""

    def __init__(self):
        self.squared_error = 0.0
        self.pixels = 0
        self.ssim_total = 0.0
        self.frames = 0

    def update(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        if reference.ndim != 2 or candidate.ndim != 2:
            raise ValueError("QualityTotals.update expects individual (H, W) frames")
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise ValueError("Image metrics require finite inputs")
        error = mse(reference, candidate)
        structural = ssim(reference, candidate)
        self.squared_error += error * reference.size
        self.pixels += reference.size
        self.ssim_total += structural
        self.frames += 1

    def result(self) -> dict:
        if not self.frames:
            raise ValueError("No image quality samples accumulated")
        error = self.squared_error / self.pixels
        return {
            "mse": error,
            "psnr_db": float(10 * np.log10(255**2 / error)) if error else None,
            "psnr_infinite": error == 0,
            "ssim": self.ssim_total / self.frames,
            "frames": self.frames,
            "pixels": self.pixels,
            "aggregation": "pixel-weighted global MSE/PSNR; frame-mean SSIM; data_range=255",
        }
