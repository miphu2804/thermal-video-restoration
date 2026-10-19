"""Backend-neutral object-detection data structures."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One predicted object using COCO ``xywh`` box coordinates."""

    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float

    def __post_init__(self) -> None:
        box = tuple(float(value) for value in self.bbox)
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("bbox must contain four finite values")
        if box[2] < 0 or box[3] < 0:
            raise ValueError("bbox width and height must be non-negative")
        if not math.isfinite(float(self.score)) or not 0 <= self.score <= 1:
            raise ValueError("score must be finite and in [0, 1]")
        object.__setattr__(self, "bbox", box)

    def to_coco(self) -> dict[str, int | float | list[float]]:
        """Serialize this prediction as one COCO result record."""
        return {
            "image_id": int(self.image_id),
            "category_id": int(self.category_id),
            "bbox": list(self.bbox),
            "score": float(self.score),
        }
