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
    display_limits: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return a BGR uint8 display image; never change the model input.

    Percentile stretching is exclusively for raw-TIFF visualization. AGC
    video can instead retain its source grayscale intensities.
    """
    array = np.asarray(frame)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("Display expects a finite, non-empty (H, W) frame")
    if stretch or display_limits is not None:
        low, high = (
            np.percentile(array, [1, 99]) if display_limits is None else display_limits
        )
        if not np.isfinite([low, high]).all() or high < low:
            raise ValueError("Invalid display limits")
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


def _render_panels(
    frames, predictions, names, *, labels, panel_width, raw14, own_range
):
    """Render labelled panels side by side with source-shared display limits.

    Panels flagged in ``own_range`` hold frames remapped to a full intensity
    range (e.g. CLAHE, full-scale stretch); the source limits would saturate
    them, so they get their own percentile stretch instead.
    """
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("Comparison frames must have identical shapes")
    own_range = tuple(own_range or (False,) * len(frames))
    if len(own_range) != len(frames):
        raise ValueError("own_range must have one flag per frame")
    shared = tuple(np.percentile(frames[0], [1, 99])) if raw14 else (0.0, 255.0)
    panels = []
    for frame, candidates, label, independent in zip(
        frames, predictions, labels, own_range
    ):
        limits = tuple(np.percentile(frame, [1, 99])) if independent else shared
        image = render_detections(frame, candidates, names, display_limits=limits)
        height = max(1, round(image.shape[0] * panel_width / image.shape[1]))
        image = cv2.resize(image, (panel_width, height))
        header = np.zeros((34, panel_width, 3), dtype=np.uint8)
        cv2.putText(
            header, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )
        panels.append(np.vstack([header, image]))
    return np.hstack(panels)


def render_comparison(
    frames, predictions, names, *, labels, panel_width=480, raw14=True, own_range=None
):
    """Render aligned A/B/C panels; B and C share A's display limits by default.

    Pass ``own_range`` to give range-remapped panels their own stretch.
    """
    if len(frames) != 3 or len(predictions) != 3 or len(labels) != 3 or panel_width < 1:
        raise ValueError("Expected three frames, predictions and labels")
    return _render_panels(
        frames,
        predictions,
        names,
        labels=labels,
        panel_width=panel_width,
        raw14=raw14,
        own_range=own_range,
    )


def render_preprocessing_comparison(
    frames, predictions, names, *, labels, panel_width=480, raw14=True, own_range=None
):
    """Render source and preprocessed detector results side by side.

    Both panels use the source frame's display range so visual contrast changes
    are not hidden by independent stretching, unless ``own_range`` flags a
    panel whose frame was remapped to a full intensity range. The limits affect
    display only; detector tensors are prepared before this function is called.
    """
    if len(frames) != 2 or len(predictions) != 2 or len(labels) != 2 or panel_width < 1:
        raise ValueError("Expected two frames, predictions and labels")
    return _render_panels(
        frames,
        predictions,
        names,
        labels=labels,
        panel_width=panel_width,
        raw14=raw14,
        own_range=own_range,
    )


def fit_to_window(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale a display image to fit ``width x height``, keeping aspect ratio."""
    if width < 1 or height < 1:
        return bgr
    scale = min(width / bgr.shape[1], height / bgr.shape[0])
    size = (max(1, round(bgr.shape[1] * scale)), max(1, round(bgr.shape[0] * scale)))
    if size == (bgr.shape[1], bgr.shape[0]):
        return bgr
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(bgr, size, interpolation=interpolation)


def ppm_bytes(bgr: np.ndarray) -> bytes:
    """Encode an RGB PPM image for Tk without a second imaging dependency."""
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("PPM encoding expects a uint8 BGR image")
    height, width = bgr.shape[:2]
    return f"P6\n{width} {height}\n255\n".encode("ascii") + bgr[:, :, ::-1].tobytes()
