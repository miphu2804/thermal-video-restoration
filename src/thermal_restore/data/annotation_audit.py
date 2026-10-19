"""Structural annotation checks without modifying images or labels."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from thermal_restore.data.annotations import CocoDataset


def audit_annotations(dataset: CocoDataset) -> dict:
    """Report box validity and class support; this cannot verify label semantics."""
    images = {image.image_id: image for image in dataset.images}
    grouped = defaultdict(list)
    issues = []
    duplicates = set()
    seen = set()
    for annotation in dataset.annotations:
        if annotation.iscrowd:
            continue
        image = images[annotation.image_id]
        x, y, width, height = annotation.bbox
        reasons = []
        if width <= 0 or height <= 0:
            reasons.append("zero_area")
        if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
            reasons.append("outside_image")
        if not np.isfinite(annotation.area) or annotation.area < 0:
            reasons.append("invalid_area")
        key = (annotation.image_id, annotation.category_id, annotation.bbox)
        if key in seen:
            duplicates.add(annotation.annotation_id)
            reasons.append("duplicate_box_same_class")
        seen.add(key)
        if reasons:
            issues.append(
                {
                    "annotation_id": annotation.annotation_id,
                    "image_id": annotation.image_id,
                    "category_id": annotation.category_id,
                    "bbox": list(annotation.bbox),
                    "reasons": reasons,
                }
            )
        grouped[annotation.category_id].append(annotation)
    classes = {}
    for category_id, class_annotations in sorted(grouped.items()):
        sizes = np.array([item.bbox[2:] for item in class_annotations])
        classes[str(category_id)] = {
            "name": dataset.categories[category_id],
            "instances": len(class_annotations),
            "images": len({item.image_id for item in class_annotations}),
            "median_width": float(np.median(sizes[:, 0])),
            "median_height": float(np.median(sizes[:, 1])),
            "small_box_fraction": float(np.mean(sizes[:, 0] * sizes[:, 1] < 1024)),
        }
    return {
        "images": len(images),
        "noncrowd_annotations": sum(map(len, grouped.values())),
        "classes": classes,
        "issues": issues,
        "issue_count": len(issues),
        "duplicate_box_count": len(duplicates),
    }
