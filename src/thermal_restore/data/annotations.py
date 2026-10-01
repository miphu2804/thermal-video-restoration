"""COCO annotation loading for FLIR ADAS thermal frames."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CocoImage:
    """A COCO image entry resolved to a local thermal file."""

    image_id: int
    file_name: str
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class CocoAnnotation:
    """A COCO bounding-box annotation in ``xywh`` format."""

    annotation_id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    area: float
    iscrowd: bool


@dataclass(frozen=True)
class CocoDataset:
    """A validated COCO split and its local image paths."""

    images: tuple[CocoImage, ...]
    annotations: tuple[CocoAnnotation, ...]
    categories: dict[int, str]

    @property
    def image_ids(self) -> tuple[int, ...]:
        """Return image IDs in annotation-file order."""
        return tuple(image.image_id for image in self.images)

    @property
    def used_category_ids(self) -> tuple[int, ...]:
        """Return category IDs used by non-crowd annotations."""
        return tuple(
            sorted(
                {
                    annotation.category_id
                    for annotation in self.annotations
                    if not annotation.iscrowd
                }
            )
        )

    def annotations_for_image(self, image_id: int) -> tuple[CocoAnnotation, ...]:
        """Return annotations belonging to one image."""
        return tuple(
            annotation
            for annotation in self.annotations
            if annotation.image_id == image_id
        )


def _box(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be a four-value COCO bbox")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box):
        raise ValueError(f"{label} must contain finite values")
    if box[2] < 0 or box[3] < 0:
        raise ValueError(f"{label} width and height must be non-negative")
    return box


def _resolve_image(file_name: str, image_root: Path) -> Path:
    """Resolve a COCO file name to a downloaded TIFF."""
    source = Path(file_name)
    stem = source.stem
    names = [f"{stem}.tiff", f"{stem}.tif"]
    if source.suffix.lower() in {".tif", ".tiff"}:
        names.append(source.name)

    candidates = [
        image_root / "analyticsData" / name for name in dict.fromkeys(names)
    ] + [image_root / name for name in dict.fromkeys(names)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = [
        match
        for name in dict.fromkeys(names)
        for match in image_root.rglob(name)
        if match.is_file()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve COCO image {file_name!r} below {image_root}"
        )
    raise ValueError(f"COCO image {file_name!r} resolves to multiple files: {matches}")


def load_coco(annotation_path: Path, image_root: Path) -> CocoDataset:
    """Load and validate a COCO file whose images are local thermal TIFFs.

    Args:
        annotation_path: Path to a split-level ``coco.json`` file.
        image_root: Directory containing the split and its ``analyticsData``.

    Returns:
        A validated dataset with every image path resolved locally.

    Raises:
        ValueError: If IDs, boxes, categories, or references are invalid.
        FileNotFoundError: If an image entry has no local TIFF match.
    """
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("COCO document must be a JSON object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise TypeError(f"COCO document must contain a list named {key!r}")

    categories: dict[int, str] = {}
    for item in payload["categories"]:
        category_id = int(item["id"])
        if category_id in categories:
            raise ValueError(f"Duplicate COCO category ID: {category_id}")
        categories[category_id] = str(item["name"])

    images: list[CocoImage] = []
    image_ids: set[int] = set()
    for item in payload["images"]:
        image_id = int(item["id"])
        if image_id in image_ids:
            raise ValueError(f"Duplicate COCO image ID: {image_id}")
        image_ids.add(image_id)
        file_name = str(item["file_name"])
        images.append(
            CocoImage(
                image_id=image_id,
                file_name=file_name,
                path=_resolve_image(file_name, Path(image_root)),
                width=int(item["width"]),
                height=int(item["height"]),
            )
        )

    annotations: list[CocoAnnotation] = []
    annotation_ids: set[int] = set()
    for item in payload["annotations"]:
        annotation_id = int(item["id"])
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate COCO annotation ID: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = int(item["image_id"])
        category_id = int(item["category_id"])
        if image_id not in image_ids:
            raise ValueError(
                f"Annotation {annotation_id} references unknown image {image_id}"
            )
        if category_id not in categories:
            raise ValueError(
                f"Annotation {annotation_id} references unknown category {category_id}"
            )
        bbox = _box(item["bbox"], f"annotation {annotation_id} bbox")
        annotations.append(
            CocoAnnotation(
                annotation_id=annotation_id,
                image_id=image_id,
                category_id=category_id,
                bbox=bbox,
                area=float(item.get("area", bbox[2] * bbox[3])),
                iscrowd=bool(item.get("iscrowd", False)),
            )
        )

    return CocoDataset(tuple(images), tuple(annotations), categories)
