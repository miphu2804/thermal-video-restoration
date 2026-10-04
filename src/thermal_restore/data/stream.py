"""Lazy thermal sources for interactive detector inference."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from thermal_restore.data.loader import (
    frame_dtype,
    frame_index,
    list_sequences,
    load_frame,
)


@dataclass(frozen=True)
class StreamFrame:
    """One source frame, with its original scale and source identity."""

    index: int
    pixels: np.ndarray
    dtype: np.dtype
    label: str


def select_tiff_paths(root: Path, sequence: str | None = None) -> list[Path]:
    """Select one sequence, avoiding concatenation of unrelated FLIR videos."""
    root = Path(root)
    if root.is_file():
        if sequence or root.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError("A TIFF file cannot be combined with --sequence")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    groups = list_sequences(root)
    if sequence is not None:
        if sequence not in groups:
            raise ValueError(
                f"Unknown sequence {sequence!r}; available: {list(groups)}"
            )
        return groups[sequence]
    if len(groups) > 1:
        raise ValueError(f"Select --sequence from: {list(groups)}")
    if groups:
        return next(iter(groups.values()))
    paths = [
        path for path in root.rglob("*") if path.suffix.lower() in {".tif", ".tiff"}
    ]
    if not paths:
        raise ValueError(f"No TIFF frames found under {root}")

    def natural_key(path):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path.relative_to(root)))
        ]

    return sorted(paths, key=natural_key)


def iter_tiff_frames(
    paths: list[Path], *, allow_uint8: bool = False
) -> Iterator[StreamFrame]:
    """Decode one TIFF at a time through the project's canonical loader."""
    for ordinal, path in enumerate(paths):
        try:
            index = frame_index(path)
        except ValueError:
            index = ordinal
        yield StreamFrame(
            index,
            load_frame(path, allow_uint8=allow_uint8),
            frame_dtype(path),
            path.name,
        )


def capture_to_thermal(raw: np.ndarray, *, allow_uint8: bool) -> np.ndarray:
    """Convert explicitly allowed grayscale/AGC video to the project scale.

    Color video is converted to grayscale, not recovered radiometric data.
    VideoCapture uint16 output is rejected: its sensor depth/calibration is
    unknown. Use the TIFF source for supported 14-bit raw frames.
    """
    if not allow_uint8:
        raise ValueError("Video/camera inputs require --allow-uint8 for AGC frames")
    if raw.dtype != np.uint8 or raw.size == 0:
        raise ValueError("Video/camera input must contain non-empty uint8 frames")
    if raw.ndim == 2:
        gray = raw
    elif raw.ndim == 3 and raw.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if raw.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(raw, code)
    else:
        raise ValueError(f"Unsupported video frame shape: {raw.shape}")
    return gray.astype(np.float32)


def iter_capture_frames(
    source: Path | int, *, allow_uint8: bool
) -> Iterator[StreamFrame]:
    """Read a local video/camera lazily and release it on close or error."""
    if not allow_uint8:
        raise ValueError("Video/camera inputs require --allow-uint8")
    capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video/camera: {source}")
        index = 0
        while True:
            ok, raw = capture.read()
            if not ok:
                break
            pixels = capture_to_thermal(raw, allow_uint8=allow_uint8)
            yield StreamFrame(index, pixels, np.dtype(np.uint8), f"frame {index}")
            index += 1
    finally:
        capture.release()
