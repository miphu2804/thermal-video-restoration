"""Export grouped metrics, prediction records, and display-only comparison PNGs.

Called by evaluate_detector.py after one inference pass. No model parameters or
normalization settings are changed. Test groups are descriptive diagnostics;
they must not be used to select hyperparameters on the test split.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from tqdm.auto import tqdm

from thermal_restore.data.annotations import CocoDataset
from thermal_restore.data.loader import frame_dtype, load_frame
from thermal_restore.detection.diagnostics import (
    group_metrics,
    match_preview_detections,
    select_preview_images,
)
from thermal_restore.detection.torchvision_backend import normalization_for_dtype
from thermal_restore.detection.types import Detection


def _sequence_name(path: Path) -> str:
    match = re.match(r"^(video-[^-]+)-frame-", path.stem)
    return match.group(1) if match else "unidentified_sequence"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _render_preview(
    dataset: CocoDataset,
    image_id: int,
    frame: np.ndarray,
    diagnostic: dict,
    metadata: dict,
    destination: Path,
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from thermal_restore.data.visualization import stretch_contrast

    # Both panels use the same display image; it never reaches the detector.
    display = stretch_contrast(frame)
    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis in axes:
        axis.imshow(display, cmap="gray", vmin=0, vmax=1)
        axis.set_xlim(0, frame.shape[1])
        axis.set_ylim(frame.shape[0], 0)
        axis.axis("off")
    placed_labels = {axis: [] for axis in axes}

    def draw(axis, box, label, color):
        x, y, width, height = box
        axis.add_patch(
            Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=1.2)
        )
        # Offset crowded labels while keeping a line to their source box.
        label_width = min(frame.shape[1], max(12, 4 * len(label)))
        label_x = min(max(0, x), frame.shape[1] - label_width)
        label_y = max(10, y - 2)
        for offset in range(12):
            candidate_y = max(10, label_y - offset * 10)
            occupied = (label_x, candidate_y - 10, label_x + label_width, candidate_y)
            if not any(
                occupied[0] < old[2]
                and occupied[2] > old[0]
                and occupied[1] < old[3]
                and occupied[3] > old[1]
                for old in placed_labels[axis]
            ):
                label_y = candidate_y
                break
        placed_labels[axis].append(
            (label_x, label_y - 10, label_x + label_width, label_y)
        )
        axis.annotate(
            label,
            xy=(x, y),
            xytext=(label_x, label_y),
            color=color,
            fontsize=6,
            va="bottom",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5},
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 1, "edgecolor": "none"},
        )

    missed = set(diagnostic["missed_annotation_ids"])
    for annotation in dataset.annotations_for_image(image_id):
        if annotation.iscrowd:
            continue
        draw(
            axes[0],
            annotation.bbox,
            dataset.categories[annotation.category_id],
            "cyan" if annotation.annotation_id in missed else "lime",
        )
    colors = {"true_positive": "lime", "wrong_class": "orange", "false_positive": "red"}
    for item in diagnostic["boxes"]:
        prediction = item["prediction"]
        draw(
            axes[1],
            prediction["bbox"],
            f"{dataset.categories[prediction['category_id']]} {prediction['score']:.2f}",
            colors[item["status"]],
        )
    axes[0].set_title("Ground truth (cyan: missed at the display threshold)")
    axes[1].set_title(
        f"Predictions >= {diagnostic['score_threshold']:.2f} | "
        f"TP={diagnostic['true_positives']} FP={diagnostic['false_positives']} "
        f"FN={diagnostic['false_negatives']}"
    )
    figure.suptitle(
        f"image_id={image_id} | {metadata['sequence']} | {metadata['dtype']}\n"
        "Prediction colors: green=correct, orange=class confusion, red=unmatched | "
        "Display stretch only; matching IoU=0.5",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def export_diagnostics(
    dataset: CocoDataset,
    predictions: list[Detection],
    evaluation_result: dict,
    destination: Path,
    *,
    preview_count: int,
    preview_score_threshold: float,
    seed: int,
    show_progress: bool = True,
) -> dict:
    """Write audit artifacts using the unchanged evaluation predictions."""
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {}
    dtype_groups = defaultdict(list)
    sequence_groups = defaultdict(list)
    for image in tqdm(
        dataset.images,
        desc="inspect TIFF headers",
        unit="image",
        disable=not show_progress,
    ):
        dtype = frame_dtype(image.path)
        sequence = _sequence_name(image.path)
        mean, std = normalization_for_dtype(
            dtype,
            mean=evaluation_result["input_mean"],
            std=evaluation_result["input_std"],
            allow_uint8=evaluation_result["allow_uint8"],
        )
        metadata[image.image_id] = {
            "image_id": image.image_id,
            "file_name": image.file_name,
            "dtype": str(dtype),
            "sequence": sequence,
            "effective_normalization": {
                "mode": (
                    "fixed_standardization" if mean is not None else "divide_by_255"
                ),
                "mean": mean,
                "std": std,
            },
        }
        dtype_groups[str(dtype)].append(image.image_id)
        sequence_groups[sequence].append(image.image_id)

    report = {
        "evaluation": evaluation_result,
        "notes": [
            "All-image metrics remain the primary result; group metrics are diagnostic.",
            "Each group recomputes AP over its own images and annotated classes.",
            "Different class sets, scenes and box sizes confound dtype comparisons.",
            "uint16 describes storage; it does not independently verify sensor bit depth.",
            "Preview thresholds and display stretch do not affect mAP or model inputs.",
            "TP/FP/FN previews use greedy same-class IoU=0.5 matching, not AP.",
            "Do not use test diagnostics for hyperparameter selection.",
        ],
        "by_dtype": {},
        "by_sequence": {},
    }
    tasks = [
        (kind, name, ids)
        for kind, groups in (
            ("by_dtype", dtype_groups),
            ("by_sequence", sequence_groups),
        )
        for name, ids in sorted(groups.items())
    ]
    for kind, name, image_ids in tqdm(
        tasks, desc="compute grouped mAP", unit="group", disable=not show_progress
    ):
        metrics = group_metrics(dataset, predictions, image_ids)
        metrics["dtype_counts"] = {
            dtype: sum(metadata[image_id]["dtype"] == dtype for image_id in image_ids)
            for dtype in sorted({metadata[image_id]["dtype"] for image_id in image_ids})
        }
        metrics["effective_normalization_by_dtype"] = {
            metadata[image_id]["dtype"]: metadata[image_id]["effective_normalization"]
            for image_id in image_ids
        }
        report[kind][name] = metrics
        tqdm.write(
            f"{kind}={name} images={metrics['images']} "
            f"map50={metrics['map50']:.4f} map50_95={metrics['map50_95']:.4f}"
        )
    _write_json(destination / "group_metrics.json", report)
    _write_json(destination / "image_metadata.json", list(metadata.values()))
    # A separate dump makes future offline audits possible without GPU inference.
    with (destination / "predictions.json").open("w", encoding="utf-8") as output:
        json.dump(
            {
                "evaluation": evaluation_result,
                "predictions": [prediction.to_coco() for prediction in predictions],
            },
            output,
        )

    selected = select_preview_images(
        dataset,
        {image_id: item["sequence"] for image_id, item in metadata.items()},
        preview_count,
        seed=seed,
    )
    predictions_by_image = defaultdict(list)
    for prediction in predictions:
        predictions_by_image[prediction.image_id].append(prediction)
    images_by_id = {image.image_id: image for image in dataset.images}
    previews = []
    preview_directory = destination / "previews"
    preview_directory.mkdir(exist_ok=True)
    for image_id in tqdm(
        selected, desc="render comparison PNGs", unit="image", disable=not show_progress
    ):
        diagnostic = match_preview_detections(
            dataset,
            image_id,
            predictions_by_image[image_id],
            score_threshold=preview_score_threshold,
        )
        filename = f"{metadata[image_id]['sequence']}_{image_id}.png"
        _render_preview(
            dataset,
            image_id,
            load_frame(
                images_by_id[image_id].path,
                allow_uint8=evaluation_result["allow_uint8"],
            ),
            diagnostic,
            metadata[image_id],
            preview_directory / filename,
        )
        previews.append(
            {**metadata[image_id], **diagnostic, "png": f"previews/{filename}"}
        )
    _write_json(
        destination / "preview_manifest.json",
        {"seed": seed, "focus_category_ids": [2, 4], "previews": previews},
    )
    (destination / "README.md").write_text(
        "# Detector diagnostics\n\n"
        "See `group_metrics.json` for dtype and sequence AP, class support and actual "
        "normalization. `image_metadata.json` preserves source image IDs. "
        "`predictions.json` contains the unchanged low-threshold predictions.\n\n"
        "Comparison PNGs in `previews/` use the same contrast-stretched display image "
        "on both panels. This stretch never reaches the detector. Ground truth is "
        "green or cyan (missed at the preview threshold). Predictions are green "
        "(correct), orange (wrong class), or red (unmatched). The preview manifest "
        "records scores, box coordinates, missed annotation IDs and thresholds.\n\n"
        "Previews are seeded and distributed across sequences, alternating frames "
        "containing bike/motor annotations with general frames where available. "
        "They are not representative estimates of whole-dataset error rates.\n\n"
        "Group AP is recomputed, not averaged from frames. Class sets and scenes "
        "differ between groups; a dtype gap alone does not prove a preprocessing "
        "bug. Test diagnostics must not guide hyperparameter selection.\n",
        encoding="utf-8",
    )
    return report
