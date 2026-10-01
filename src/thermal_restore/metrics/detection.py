"""Detection metrics for COCO-style boxes without a detector dependency."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from thermal_restore.data.annotations import CocoAnnotation, CocoDataset
from thermal_restore.detection.types import Detection

COCO_IOU_THRESHOLDS = tuple(0.5 + 0.05 * index for index in range(10))


@dataclass(frozen=True)
class MapMetrics:
    """Mean average precision at the two project reporting points."""

    map50: float
    map50_95: float
    ap50_by_category: dict[int, float]


def bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Compute IoU for two ``xywh`` boxes."""
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def detections_to_coco(predictions: Sequence[Detection]) -> list[dict]:
    """Serialize predictions while preserving image and category IDs."""
    return [prediction.to_coco() for prediction in predictions]


def _average_precision(
    true_positives: np.ndarray, false_positives: np.ndarray, n_ground_truth: int
) -> float:
    if n_ground_truth == 0:
        return float("nan")
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recall = cumulative_tp / n_ground_truth
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    sampled = []
    for level in np.linspace(0.0, 1.0, 101):
        eligible = precision[recall >= level]
        sampled.append(float(np.max(eligible)) if eligible.size else 0.0)
    return float(np.mean(sampled))


def _category_ap(
    annotations: Sequence[CocoAnnotation],
    predictions: Sequence[Detection],
    category_id: int,
    iou_threshold: float,
) -> float:
    ground_truths: dict[int, list[CocoAnnotation]] = defaultdict(list)
    for annotation in annotations:
        if annotation.category_id == category_id and not annotation.iscrowd:
            ground_truths[annotation.image_id].append(annotation)
    n_ground_truth = sum(len(items) for items in ground_truths.values())
    if n_ground_truth == 0:
        return float("nan")

    candidates = sorted(
        (
            prediction
            for prediction in predictions
            if prediction.category_id == category_id
        ),
        key=lambda prediction: prediction.score,
        reverse=True,
    )
    matched = {
        image_id: np.zeros(len(items), dtype=bool)
        for image_id, items in ground_truths.items()
    }
    true_positives = np.zeros(len(candidates), dtype=np.float64)
    false_positives = np.zeros(len(candidates), dtype=np.float64)

    for index, prediction in enumerate(candidates):
        items = ground_truths.get(prediction.image_id, [])
        if not items:
            false_positives[index] = 1
            continue
        ious = np.array(
            [bbox_iou(prediction.bbox, annotation.bbox) for annotation in items]
        )
        order = np.argsort(ious)[::-1]
        match = next(
            (
                item_index
                for item_index in order
                if not matched[prediction.image_id][item_index]
            ),
            None,
        )
        if match is not None and ious[match] >= iou_threshold:
            matched[prediction.image_id][match] = True
            true_positives[index] = 1
        else:
            false_positives[index] = 1

    return _average_precision(true_positives, false_positives, n_ground_truth)


def mean_average_precision(
    dataset: CocoDataset,
    predictions: Sequence[Detection],
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
) -> MapMetrics:
    """Compute mAP@0.5 and mAP@[0.5:0.95] for one COCO split.

    Crowd annotations are excluded from the current calculation. FLIR's
    downloaded annotations use ``iscrowd=False`` for the evaluated boxes.
    Categories with no non-crowd ground truth are excluded from the mean.
    """
    thresholds = tuple(float(threshold) for threshold in iou_thresholds)
    if not thresholds or 0.5 not in thresholds:
        raise ValueError("iou_thresholds must contain 0.5")
    if any(threshold < 0 or threshold > 1 for threshold in thresholds):
        raise ValueError("iou_thresholds must be in [0, 1]")

    categories = dataset.used_category_ids
    ap_by_threshold = []
    ap50_by_category = {}
    for threshold in thresholds:
        values = []
        for category_id in categories:
            value = _category_ap(
                dataset.annotations, predictions, category_id, float(threshold)
            )
            if np.isfinite(value):
                values.append(value)
                if threshold == 0.5:
                    ap50_by_category[category_id] = value
        ap_by_threshold.append(float(np.mean(values)) if values else 0.0)

    if not ap_by_threshold:
        return MapMetrics(0.0, 0.0, ap50_by_category)
    map50 = ap_by_threshold[thresholds.index(0.5)]
    return MapMetrics(map50, float(np.mean(ap_by_threshold)), ap50_by_category)
