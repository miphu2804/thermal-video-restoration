"""Pure helpers for grouped detector metrics and reproducible inspection."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from thermal_restore.data.annotations import CocoAnnotation, CocoDataset
from thermal_restore.detection.types import Detection
from thermal_restore.metrics import bbox_iou, mean_average_precision


def assess_annotation(
    annotation: CocoAnnotation,
    predictions: Sequence[Detection],
    *,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict:
    """Describe candidate coverage for one GT box, not one-to-one TP or AP.

    Predictions must include the low-confidence candidates if low-confidence
    coverage is to be measured. Multiple GT boxes can share a candidate.
    """
    if not 0 <= score_threshold <= 1 or not 0 < iou_threshold <= 1:
        raise ValueError("Invalid confidence or IoU threshold")
    candidates = [
        (bbox_iou(annotation.bbox, item.bbox), item)
        for item in predictions
        if item.image_id == annotation.image_id
    ]
    same = [
        (iou, item)
        for iou, item in candidates
        if item.category_id == annotation.category_id
    ]
    overlaps = [(iou, item) for iou, item in same if iou >= iou_threshold]
    visible = [(iou, item) for iou, item in candidates if item.score >= score_threshold]
    best = max(visible, key=lambda pair: pair[0], default=None)
    if any(item.score >= score_threshold for _, item in overlaps):
        status = "covered_same_class"
    elif overlaps:
        status = "low_confidence_same_class"
    elif best is not None and best[0] >= iou_threshold:
        status = "overlap_wrong_class"
    elif any(
        iou >= 0.1 and item.category_id == annotation.category_id
        for iou, item in visible
    ):
        status = "poor_localization_same_class"
    else:
        status = "no_matching_candidate"
    return {
        "annotation_id": annotation.annotation_id,
        "image_id": annotation.image_id,
        "category_id": annotation.category_id,
        "bbox": list(annotation.bbox),
        "status": status,
        "same_class_best_iou": max((iou for iou, _ in same), default=0.0),
        "same_class_overlap_max_score": max(
            (item.score for _, item in overlaps), default=None
        ),
        "best_visible_candidate": (
            {**best[1].to_coco(), "iou": best[0]} if best is not None else None
        ),
    }


def subset_dataset(dataset: CocoDataset, image_ids: Sequence[int]) -> CocoDataset:
    """Select images and their annotations without remapping any IDs."""
    selected = set(image_ids)
    unknown = selected - set(dataset.image_ids)
    if unknown:
        raise ValueError(f"Unknown image IDs: {sorted(unknown)}")
    return CocoDataset(
        images=tuple(image for image in dataset.images if image.image_id in selected),
        annotations=tuple(
            annotation
            for annotation in dataset.annotations
            if annotation.image_id in selected
        ),
        categories=dict(dataset.categories),
    )


def group_metrics(
    dataset: CocoDataset,
    predictions: Sequence[Detection],
    image_ids: Sequence[int],
) -> dict:
    """Recompute AP within a group; never average per-image or per-group AP."""
    subset = subset_dataset(dataset, image_ids)
    selected = set(subset.image_ids)
    candidates = [item for item in predictions if item.image_id in selected]
    metrics = mean_average_precision(subset, candidates)
    counts = Counter(
        item.category_id for item in subset.annotations if not item.iscrowd
    )
    small_counts = Counter(
        item.category_id
        for item in subset.annotations
        if not item.iscrowd and item.bbox[2] * item.bbox[3] < 32**2
    )
    return {
        "images": len(subset.images),
        "image_ids": list(subset.image_ids),
        "predictions": len(candidates),
        "map50": metrics.map50,
        "map50_95": metrics.map50_95,
        "evaluated_category_ids": list(subset.used_category_ids),
        "ap50_by_category": {
            str(category_id): {
                "name": subset.categories[category_id],
                "value": metrics.ap50_by_category[category_id],
                "instances": counts[category_id],
                "small_box_fraction": small_counts[category_id] / counts[category_id],
            }
            for category_id in subset.used_category_ids
        },
    }


def select_preview_images(
    dataset: CocoDataset,
    sequence_by_image: Mapping[int, str],
    count: int,
    *,
    seed: int,
    focus_categories: Sequence[int] = (2, 4),
) -> list[int]:
    """Round-robin sequences, alternating focus-class and general samples."""
    if count < 0:
        raise ValueError("Preview count cannot be negative")
    focus = {
        annotation.image_id
        for annotation in dataset.annotations
        if not annotation.iscrowd and annotation.category_id in focus_categories
    }
    grouped: dict[str, list[int]] = defaultdict(list)
    for image in dataset.images:
        grouped[sequence_by_image[image.image_id]].append(image.image_id)
    rng = np.random.default_rng(seed)
    queues = {}
    for name, image_ids in sorted(grouped.items()):
        preferred = [image_id for image_id in image_ids if image_id in focus]
        general = [image_id for image_id in image_ids if image_id not in focus]
        rng.shuffle(preferred)
        rng.shuffle(general)
        queue = []
        while preferred or general:
            if preferred:
                queue.append(preferred.pop())
            if general:
                queue.append(general.pop())
        queues[name] = queue
    selected = []
    target = min(count, len(dataset.images))
    while len(selected) < target:
        for queue in queues.values():
            if queue:
                selected.append(queue.pop(0))
                if len(selected) == target:
                    break
    return selected


def match_preview_detections(
    dataset: CocoDataset,
    image_id: int,
    predictions: Sequence[Detection],
    *,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict:
    """Classify visible boxes with score-ordered one-to-one IoU matching.

    This is a display diagnostic, not the AP calculation. Same-class matches
    take precedence over overlapping wrong-class boxes. A class-confused box
    is a false positive and its ground truth remains a false negative.
    """
    if not 0 <= score_threshold <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("Thresholds must be in [0, 1]")
    annotations = [
        item for item in dataset.annotations_for_image(image_id) if not item.iscrowd
    ]
    visible = sorted(
        (
            item
            for item in predictions
            if item.image_id == image_id and item.score >= score_threshold
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    matched = set()
    boxes = []
    for prediction in visible:
        eligible = [
            (bbox_iou(prediction.bbox, annotation.bbox), annotation)
            for annotation in annotations
            if annotation.annotation_id not in matched
            and annotation.category_id == prediction.category_id
        ]
        best = max(eligible, key=lambda item: item[0], default=None)
        if best is not None and best[0] >= iou_threshold:
            matched.add(best[1].annotation_id)
            status = "true_positive"
        elif any(
            annotation.category_id != prediction.category_id
            and bbox_iou(prediction.bbox, annotation.bbox) >= iou_threshold
            for annotation in annotations
        ):
            status = "wrong_class"
        else:
            status = "false_positive"
        boxes.append({"prediction": prediction.to_coco(), "status": status})
    true_positives = len(matched)
    return {
        "image_id": image_id,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "true_positives": true_positives,
        "false_positives": len(visible) - true_positives,
        "false_negatives": len(annotations) - true_positives,
        "wrong_class_predictions": sum(
            item["status"] == "wrong_class" for item in boxes
        ),
        "missed_annotation_ids": [
            item.annotation_id
            for item in annotations
            if item.annotation_id not in matched
        ],
        "ground_truth": [
            {
                "annotation_id": item.annotation_id,
                "category_id": item.category_id,
                "bbox": list(item.bbox),
                "matched": item.annotation_id in matched,
            }
            for item in annotations
        ],
        "boxes": boxes,
    }
