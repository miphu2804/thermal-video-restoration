from pathlib import Path

import numpy as np
import pytest

from thermal_restore.data.annotations import CocoAnnotation, CocoDataset, CocoImage
from thermal_restore.detection import Detection
from thermal_restore.detection.diagnostics import (
    group_metrics,
    match_preview_detections,
    select_preview_images,
    subset_dataset,
)
from thermal_restore.detection.torchvision_backend import normalization_for_dtype


def _dataset():
    return CocoDataset(
        images=tuple(
            CocoImage(index, f"{index}.tiff", Path(f"{index}.tiff"), 64, 64)
            for index in (10, 20, 30, 40)
        ),
        annotations=(
            CocoAnnotation(100, 10, 2, (0, 0, 10, 10), 100, False),
            CocoAnnotation(200, 20, 1, (0, 0, 40, 40), 1600, False),
            CocoAnnotation(300, 30, 2, (0, 0, 10, 10), 100, False),
            CocoAnnotation(400, 40, 1, (0, 0, 40, 40), 1600, False),
        ),
        categories={1: "person", 2: "bike"},
    )


def test_subset_preserves_source_ids_and_excludes_other_annotations():
    subset = subset_dataset(_dataset(), [30, 10])
    assert subset.image_ids == (10, 30)
    assert [item.annotation_id for item in subset.annotations] == [100, 300]
    with pytest.raises(ValueError, match="Unknown image IDs"):
        subset_dataset(_dataset(), [999])


def test_group_ap_excludes_predictions_and_classes_outside_the_group():
    predictions = [
        Detection(10, 2, (0, 0, 10, 10), 0.9),
        Detection(20, 2, (0, 0, 10, 10), 0.99),
    ]
    result = group_metrics(_dataset(), predictions, [10])
    assert result["map50"] == 1.0
    assert result["predictions"] == 1
    assert result["evaluated_category_ids"] == [2]
    assert result["ap50_by_category"]["2"]["instances"] == 1
    assert result["ap50_by_category"]["2"]["small_box_fraction"] == 1.0
    assert group_metrics(_dataset(), [], [20])["map50"] == 0.0


def test_previews_are_seeded_unique_and_cover_sequences():
    sequence_by_image = {10: "a", 20: "a", 30: "b", 40: "b"}
    selected = select_preview_images(_dataset(), sequence_by_image, 2, seed=3)
    assert selected == select_preview_images(_dataset(), sequence_by_image, 2, seed=3)
    assert set(selected) == {10, 30}
    assert len(select_preview_images(_dataset(), sequence_by_image, 100, seed=3)) == 4
    assert select_preview_images(_dataset(), sequence_by_image, 0, seed=3) == []


def test_duplicate_predictions_do_not_match_one_annotation_twice():
    result = match_preview_detections(
        _dataset(),
        10,
        [
            Detection(10, 2, (0, 0, 10, 10), 0.9),
            Detection(10, 2, (0, 0, 10, 10), 0.8),
            Detection(10, 2, (30, 30, 10, 10), 0.1),
            Detection(20, 1, (0, 0, 40, 40), 0.9),
        ],
    )
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0
    assert len(result["boxes"]) == 2


def test_class_confusion_stays_false_positive_and_false_negative():
    result = match_preview_detections(
        _dataset(), 10, [Detection(10, 1, (0, 0, 10, 10), 0.9)]
    )
    assert result["wrong_class_predictions"] == 1
    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["missed_annotation_ids"] == [100]


def test_reported_normalization_matches_source_dtype_policy():
    assert normalization_for_dtype(
        np.dtype("uint16"), mean=113, std=5, allow_uint8=True
    ) == (113, 5)
    mean, std = normalization_for_dtype(
        np.dtype("uint8"), mean=113, std=5, allow_uint8=True
    )
    assert mean == pytest.approx(114.495)
    assert std == pytest.approx(57.63)
    assert normalization_for_dtype(
        np.dtype("uint8"), mean=None, std=None, allow_uint8=True
    ) == (None, None)
