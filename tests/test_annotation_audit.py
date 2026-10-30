from pathlib import Path

from thermal_restore.data.annotation_audit import audit_annotations
from thermal_restore.data.annotations import CocoAnnotation, CocoDataset, CocoImage
from thermal_restore.detection import Detection
from thermal_restore.detection.diagnostics import assess_annotation


def test_audit_reports_issues_without_modifying_source_boxes():
    annotations = (
        CocoAnnotation(1, 10, 2, (25, 0, 10, 10), 100, False),
        CocoAnnotation(2, 10, 2, (25, 0, 10, 10), 100, False),
        CocoAnnotation(3, 10, 2, (0, 0, 0, 10), 0, False),
    )
    dataset = CocoDataset(
        (CocoImage(10, "a.tiff", Path("a.tiff"), 32, 32),), annotations, {2: "bike"}
    )
    result = audit_annotations(dataset)
    assert result["issue_count"] == 3
    assert result["duplicate_box_count"] == 1
    assert result["issues"][0]["reasons"] == ["outside_image"]
    assert "zero_area" in result["issues"][2]["reasons"]
    assert result["classes"]["2"]["instances"] == 3
    assert result["classes"]["2"]["images"] == 1
    assert dataset.annotations == annotations


def test_coverage_distinguishes_low_confidence_and_class_confusion():
    annotation = CocoAnnotation(1, 10, 2, (0, 0, 10, 10), 100, False)
    wrong = Detection(10, 1, (0, 0, 10, 10), 0.9)
    low = Detection(10, 2, (0, 0, 10, 10), 0.1)
    assert assess_annotation(annotation, [wrong])["status"] == "overlap_wrong_class"
    result = assess_annotation(annotation, [wrong, low])
    assert result["status"] == "low_confidence_same_class"
    assert result["same_class_overlap_max_score"] == 0.1
    assert result["best_visible_candidate"]["category_id"] == 1
    assert (
        assess_annotation(annotation, [low], score_threshold=0.05)["status"]
        == "covered_same_class"
    )


def test_coverage_ignores_other_images_and_detects_poor_localization():
    annotation = CocoAnnotation(1, 10, 2, (0, 0, 10, 10), 100, False)
    other = Detection(11, 2, (0, 0, 10, 10), 0.9)
    assert assess_annotation(annotation, [other])["status"] == "no_matching_candidate"
    partial = Detection(10, 2, (0, 0, 4, 10), 0.9)
    assert (
        assess_annotation(annotation, [partial])["status"]
        == "poor_localization_same_class"
    )
