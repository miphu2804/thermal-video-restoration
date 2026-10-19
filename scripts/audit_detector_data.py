"""Audit source geometry and bike/motor crops without changing data or weights.

Optional checkpoint inference covers only seeded train/val preview frames.
Full test candidate analysis reuses an existing prediction dump. Candidate
coverage is descriptive and not a replacement for mAP or independent objects.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import tifffile
from tqdm.auto import tqdm

from thermal_restore.data.annotation_audit import audit_annotations
from thermal_restore.data.annotations import load_coco
from thermal_restore.data.loader import load_frame
from thermal_restore.detection import Detection
from thermal_restore.detection.diagnostics import assess_annotation
from thermal_restore.metrics import bbox_iou


def _sequence(path):
    match = re.match(r"^(video-[^-]+)-frame-", path.stem)
    return match.group(1) if match else "unidentified_sequence"


def _select(dataset, category_id, count, rng):
    images = {item.image_id: item for item in dataset.images}
    queues = defaultdict(list)
    for item in dataset.annotations:
        image = images[item.image_id]
        x, y, width, height = item.bbox
        if (
            item.category_id == category_id
            and not item.iscrowd
            and min(width, height) > 0
            and x < image.width
            and y < image.height
            and x + width > 0
            and y + height > 0
        ):
            queues[_sequence(images[item.image_id].path)].append(item)
    for queue in queues.values():
        rng.shuffle(queue)
    sequence_names = sorted(queues)
    rng.shuffle(sequence_names)
    result = []
    used_images = set()
    while len(result) < count:
        progressed = False
        for name in sequence_names:
            queue = queues[name]
            while queue and queue[-1].image_id in used_images:
                queue.pop()
            if queue:
                item = queue.pop()
                result.append(item)
                used_images.add(item.image_id)
                progressed = True
                if len(result) == count:
                    break
        if not progressed:
            break
    return result


def _crop_sheet(
    dataset, annotations, predictions_by_image, path, allow_uint8, threshold
):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from thermal_restore.data.visualization import stretch_contrast

    if not annotations:
        return []
    images = {item.image_id: item for item in dataset.images}
    rows = (len(annotations) + 1) // 2
    figure, axes = plt.subplots(rows, 4, figsize=(16, rows * 3.7), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    records = []
    for index, annotation in enumerate(annotations):
        image = images[annotation.image_id]
        frame = load_frame(image.path, allow_uint8=allow_uint8)
        display = stretch_contrast(frame)
        x, y, width, height = annotation.bbox
        cx, cy = x + width / 2, y + height / 2
        half_width, half_height = max(48, width * 1.5), max(48, height * 1.5)
        left, top = max(0, int(cx - half_width)), max(0, int(cy - half_height))
        right, bottom = min(frame.shape[1], int(np.ceil(cx + half_width))), min(
            frame.shape[0], int(np.ceil(cy + half_height))
        )
        candidates = predictions_by_image.get(annotation.image_id, [])
        assessment = assess_annotation(
            annotation, candidates, score_threshold=threshold
        )
        assessment.update(
            {
                "file_name": image.file_name,
                "sequence": _sequence(image.path),
                "frame_min": float(frame.min()),
                "frame_max": float(frame.max()),
                "frame_mean": float(frame.mean()),
                "frame_std": float(frame.std()),
            }
        )
        records.append(assessment)
        gt_axis, pred_axis = axes[index // 2, (index % 2) * 2 : (index % 2) * 2 + 2]
        for axis in (gt_axis, pred_axis):
            axis.imshow(
                display[top:bottom, left:right],
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=(left, right, bottom, top),
            )
            axis.set_xlim(left, right)
            axis.set_ylim(bottom, top)
        gt_axis.add_patch(
            Rectangle(
                (x, y), width, height, fill=False, edgecolor="cyan", linewidth=1.5
            )
        )
        gt_axis.set_title(
            f"GT {dataset.categories[annotation.category_id]} #{annotation.annotation_id}\n"
            f"image={annotation.image_id} box={width:.0f}x{height:.0f}px",
            fontsize=9,
        )
        visible = sorted(
            (
                item
                for item in candidates
                if item.score >= threshold
                and item.bbox[0] < right
                and item.bbox[1] < bottom
                and item.bbox[0] + item.bbox[2] > left
                and item.bbox[1] + item.bbox[3] > top
            ),
            key=lambda item: item.score,
            reverse=True,
        )[:8]
        for item in visible:
            overlap = bbox_iou(annotation.bbox, item.bbox) >= 0.5
            color = (
                "lime"
                if overlap and item.category_id == annotation.category_id
                else "orange" if overlap else "red"
            )
            pred_axis.add_patch(
                Rectangle(
                    item.bbox[:2],
                    item.bbox[2],
                    item.bbox[3],
                    fill=False,
                    edgecolor=color,
                    linewidth=1,
                )
            )
            pred_axis.text(
                item.bbox[0],
                item.bbox[1],
                f"{dataset.categories[item.category_id]} {item.score:.2f}",
                color=color,
                fontsize=6,
                clip_on=True,
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
            )
        pred_axis.set_title(
            f"Predictions >= {threshold:.2f} (top 8 in crop)\n{assessment['status']}",
            fontsize=9,
        )
    figure.suptitle(
        "Seeded annotation crops | display zoom only; no model input resizing\n"
        "Cyan: GT | Green: same class IoU >= 0.5 | Orange: different class overlap | Red: other candidates",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/FLIR_ADAS_v2"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--test-predictions", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-score-threshold", type=float, default=0.3)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/detector/data_audit")
    )
    args = parser.parse_args()
    if (
        args.samples_per_class < 1
        or args.seed < 0
        or not 0 <= args.preview_score_threshold <= 1
    ):
        parser.error(
            "samples-per-class must be positive; seed non-negative; threshold in [0, 1]"
        )
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    datasets = {
        split: load_coco(args.data_root / folder / "coco.json", args.data_root / folder)
        for split, folder in (
            ("train", "images_thermal_train"),
            ("val", "images_thermal_val"),
            ("test", "video_thermal_test"),
        )
    }
    report = {
        "seed": args.seed,
        "focus_category_ids": [2, 4],
        "preview_score_threshold": args.preview_score_threshold,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "splits": {},
        "sequence_overlap": {},
        "notes": [
            "Geometry validation does not establish semantic correctness of labels.",
            "Frame-instance counts do not represent unique tracked objects.",
            "Candidate coverage is per GT and not one-to-one matching, precision, recall or AP.",
            "Train/val predictions cover only seeded preview frames; test predictions reuse a full dump.",
            "No labels, pixels, split assignments or checkpoint weights are changed.",
            "Do not select training hyperparameters using the test audit.",
        ],
    }
    sequences = {}
    selections = {}
    for split, dataset in datasets.items():
        audit = audit_annotations(dataset)
        dtype_counts = Counter()
        shape_issues = []
        for image in tqdm(
            dataset.images, desc=f"audit {split} TIFF geometry", unit="image"
        ):
            with tifffile.TiffFile(image.path) as file:
                page = file.pages[0]
                dtype_counts[str(page.dtype)] += 1
                if tuple(page.shape) != (image.height, image.width):
                    shape_issues.append(
                        {
                            "image_id": image.image_id,
                            "coco_shape": [image.height, image.width],
                            "tiff_shape": list(page.shape),
                        }
                    )
        audit["dtype_counts"] = dict(dtype_counts)
        audit["shape_issues"] = shape_issues
        sequences[split] = {_sequence(image.path) for image in dataset.images}
        for category_id, item in audit["classes"].items():
            focus_ids = {
                a.image_id
                for a in dataset.annotations
                if a.category_id == int(category_id) and not a.iscrowd
            }
            item["sequences"] = len(
                {
                    _sequence(image.path)
                    for image in dataset.images
                    if image.image_id in focus_ids
                }
            )
        report["splits"][split] = audit
        selections[split] = {
            category: _select(dataset, category, args.samples_per_class, rng)
            for category in (2, 4)
        }
        print(
            f"{split}: boxes={audit['noncrowd_annotations']} issues={audit['issue_count']} shape_issues={len(shape_issues)}"
        )
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        report["sequence_overlap"][f"{first}_{second}"] = sorted(
            sequences[first] & sequences[second]
        )

    stored = None
    if args.test_predictions:
        stored = json.loads(args.test_predictions.read_text(encoding="utf-8"))
        evaluation = stored["evaluation"]
        if (
            evaluation.get("split") != "test"
            or evaluation.get("evaluation_scope") != "full_split"
        ):
            parser.error("test-predictions must come from a full test diagnostic run")
        if (
            args.checkpoint
            and Path(evaluation["checkpoint"]).resolve() != args.checkpoint.resolve()
        ):
            parser.error("test-predictions checkpoint path does not match --checkpoint")
        report["test_prediction_evaluation"] = evaluation
    detector = None
    if args.checkpoint:
        import torch

        from thermal_restore.detection.device import require_device
        from thermal_restore.detection.torchvision_backend import TorchvisionDetector

        device = require_device(torch, args.device)
        print(f"audit device={device.selected} cuda={device.as_dict()}")
        detector = TorchvisionDetector.from_checkpoint(args.checkpoint, device.selected)
    for split, dataset in datasets.items():
        predictions_by_image = defaultdict(list)
        if split == "test" and stored is not None:
            for item in stored["predictions"]:
                prediction = Detection(
                    int(item["image_id"]),
                    int(item["category_id"]),
                    tuple(item["bbox"]),
                    float(item["score"]),
                )
                predictions_by_image[prediction.image_id].append(prediction)
            unknown = set(predictions_by_image) - set(dataset.image_ids)
            if unknown:
                parser.error("test prediction dump contains unknown image IDs")
            coverage = {}
            for category in (2, 4):
                assessments = [
                    assess_annotation(
                        item,
                        predictions_by_image[item.image_id],
                        score_threshold=args.preview_score_threshold,
                    )
                    for item in dataset.annotations
                    if item.category_id == category and not item.iscrowd
                ]
                coverage[str(category)] = {
                    "instances": len(assessments),
                    "candidate_status_counts": dict(
                        Counter(item["status"] for item in assessments)
                    ),
                }
            report["test_focus_candidate_coverage"] = coverage
        elif split != "test" and detector is not None:
            from thermal_restore.detection.torchvision_backend import (
                TorchvisionCocoDataset,
            )

            torch_dataset = TorchvisionCocoDataset(
                dataset,
                detector.category_ids,
                cache_dir=args.cache_root / split if args.cache_root else None,
                input_mean=detector.input_mean,
                input_std=detector.input_std,
            )
            focus_ids = {
                item.image_id for items in selections[split].values() for item in items
            }
            indices = [
                i
                for i, image in enumerate(dataset.images)
                if image.image_id in focus_ids
            ]
            for start in tqdm(
                range(0, len(indices), 2),
                desc=f"infer {split} preview frames",
                unit="batch",
            ):
                batch = [torch_dataset[index] for index in indices[start : start + 2]]
                predictions = detector.predict(
                    [item[0] for item in batch],
                    [int(item[1]["image_id"].item()) for item in batch],
                )
                for prediction in predictions:
                    predictions_by_image[prediction.image_id].append(prediction)
        if not predictions_by_image:
            report["splits"][split][
                "preview_prediction_scope"
            ] = "no_predictions_available"
        else:
            report["splits"][split]["preview_prediction_scope"] = (
                "full_test_dump" if split == "test" else "seeded_focus_frames_only"
            )
        report["splits"][split]["crop_records"] = {}
        for category, selected_annotations in selections[split].items():
            records = _crop_sheet(
                dataset,
                selected_annotations,
                predictions_by_image,
                root / f"{split}_{dataset.categories[category]}_crops.png",
                split == "test",
                args.preview_score_threshold,
            )
            report["splits"][split]["crop_records"][str(category)] = records
    (root / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Detector data audit",
        "",
        "Seeded crop sheets are display-only. Structural checks do not certify label semantics.",
        "",
        "Small boxes have bounding-box area below 32 x 32 pixels at source resolution. Counts are frame instances, not unique objects.",
        "",
        "| Split | Class | Instances | Images | Sequences | Median box (px) | Small box fraction |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for split, audit in report["splits"].items():
        for category in ("2", "4"):
            item = audit["classes"].get(category)
            if item:
                lines.append(
                    f"| {split} | {item['name']} | {item['instances']} | {item['images']} | {item['sequences']} | {item['median_width']:.1f} x {item['median_height']:.1f} | {item['small_box_fraction']:.1%} |"
                )
    lines.extend(["", "## Geometry and sequence integrity", ""])
    for split, audit in report["splits"].items():
        lines.append(
            f"- {split}: {audit['issue_count']} structural box issues ({audit['duplicate_box_count']} exact duplicate boxes); {len(audit['shape_issues'])} TIFF/COCO shape mismatches."
        )
    lines.extend(["", f"Sequence overlap: `{json.dumps(report['sequence_overlap'])}`."])
    coverage = report.get("test_focus_candidate_coverage")
    if coverage:
        lines.extend(
            [
                "",
                "## Existing test prediction candidate coverage",
                "",
                f"Display confidence threshold: {args.preview_score_threshold}; overlap threshold: IoU 0.5. Predictions retain the original inference confidence floor.",
                "",
                "| Class | GT instances | Candidate status | Count |",
                "|---|---:|---|---:|",
            ]
        )
        for category, item in coverage.items():
            for status, count in sorted(item["candidate_status_counts"].items()):
                lines.append(
                    f"| {datasets['test'].categories[int(category)]} | {item['instances']} | {status} | {count} |"
                )
    lines.extend(["", "## Seeded visual samples", ""])
    for split, audit in report["splits"].items():
        lines.extend(
            [
                f"### {split}",
                "",
                f"Prediction scope: `{audit['preview_prediction_scope']}`.",
                "",
            ]
        )
        for category in (2, 4):
            filename = f"{split}_{datasets[split].categories[category]}_crops.png"
            if (root / filename).exists():
                lines.append(
                    f"![{split} {datasets[split].categories[category]} annotation crops]({filename})"
                )
                lines.append("")
    lines.extend(
        [
            "",
            "See `audit.json` for box issue IDs, per-crop candidate coverage, source pixel summaries and sequence overlap.",
            "Candidate coverage is not AP or one-to-one matching. Test diagnostics must not guide hyperparameter selection.",
            "",
        ]
    )
    (root / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"audit={root.resolve()}")
    print(json.dumps(report.get("test_focus_candidate_coverage", {}), indent=2))


if __name__ == "__main__":
    main()
