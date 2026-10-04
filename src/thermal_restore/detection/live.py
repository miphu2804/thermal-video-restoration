"""Display-only rendering helpers for live thermal detector demos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from thermal_restore.detection import Detection


def render_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    category_names: Mapping[int, str],
    *,
    status: str = "",
    stretch: bool = True,
) -> np.ndarray:
    """Return a BGR uint8 display image; never change the model input.

    Percentile stretching is exclusively for raw-TIFF visualization. AGC
    video can instead retain its source grayscale intensities.
    """
    array = np.asarray(frame)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("Display expects a finite, non-empty (H, W) frame")
    if stretch:
        low, high = np.percentile(array, [1, 99])
        gray = np.clip((array - low) / max(high - low, 1e-6) * 255, 0, 255)
    else:
        gray = np.clip(array, 0, 255)
    image = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    height, width = array.shape
    for item in detections:
        x, y, w, h = item.bbox
        if w <= 0 or h <= 0 or x >= width or y >= height or x + w <= 0 or y + h <= 0:
            continue
        x1, y1 = int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))
        x2, y2 = int(np.clip(x + w, 0, width - 1)), int(np.clip(y + h, 0, height - 1))
        color = (
            50 + item.category_id * 41 % 180,
            220,
            50 + item.category_id * 83 % 180,
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        label = f"{category_names.get(item.category_id, str(item.category_id))} {item.score:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            image, label, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )
    if status:
        cv2.rectangle(image, (0, 0), (width - 1, 27), (0, 0, 0), -1)
        cv2.putText(
            image, status, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
    return image


def ppm_bytes(bgr: np.ndarray) -> bytes:
    """Encode an RGB PPM image for Tk without a second imaging dependency."""
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("PPM encoding expects a uint8 BGR image")
    height, width = bgr.shape[:2]
    return f"P6\n{width} {height}\n255\n".encode("ascii") + bgr[:, :, ::-1].tobytes()
