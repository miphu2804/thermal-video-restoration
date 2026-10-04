"""Compare A/B/C on the same labelled FLIR images with one frozen checkpoint.

Defaults to validation and raw uint16 TIFF references. Full experiments are
user-run; --max-images is an explicitly labelled seeded smoke sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from thermal_restore.data import FrameCache, load_coco
from thermal_restore.data.experiment_config import (
    load_restoration_config,
    source_sequence,
)
from thermal_restore.data.loader import frame_dtype, load_frame
from thermal_restore.degradation import DEFAULT_STEPS, get_preset
from thermal_restore.detection.device import require_device
from thermal_restore.detection.diagnostics import subset_dataset
from thermal_restore.detection.torchvision_backend import (
    TorchvisionDetector,
    frame_to_tensor,
    normalization_for_dtype,
)
from thermal_restore.experiments import QualityTotals, build_restoration, degrade_frame
from thermal_restore.metrics import mean_average_precision


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/FLIR_ADAS_v2"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--restoration-config", type=Path, default=Path("configs/abc_baselines.json")
    )
    parser.add_argument(
        "--pipelines", nargs="+", help="Configured pipelines; default: all."
    )
    parser.add_argument(
        "--levels", nargs="+", choices=("light", "medium", "heavy"), default=["medium"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--degradation-steps",
        nargs="+",
        choices=DEFAULT_STEPS,
        default=list(DEFAULT_STEPS),
    )
    parser.add_argument(
        "--source-policy",
        choices=("raw14", "mixed"),
        default="raw14",
        help="raw14 excludes mirror uint8 TIFFs; mixed is a separately labelled AGC experiment.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.001,
        help="Same low-confidence floor for A/B/C; not the demo threshold.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-images", type=int, help="Bounded diagnostic sample, not a full result."
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/detector/abc_val")
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or not 0 <= args.score_threshold <= 1
        or args.sample_seed < 0
        or any(seed < 0 for seed in args.seeds)
        or args.progress_interval < 0
        or (args.max_images is not None and args.max_images < 1)
    ):
        parser.error(
            "Invalid batch size, confidence, seed, progress interval or image limit"
        )
    for values in (
        args.seeds,
        args.levels,
        args.degradation_steps,
        args.pipelines or [],
    ):
        if len(values) != len(set(values)):
            parser.error("Duplicate seeds, levels, pipelines or steps are not allowed")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error(
            "output-dir must be absent or empty; existing results are never overwritten"
        )
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_condition():
    return {
        "predictions": [],
        "quality": QualityTotals(),
        "restoration_seconds": 0.0,
        "tensor_seconds": 0.0,
        "inference_seconds": 0.0,
        "quality_seconds": 0.0,
    }


def _predict_condition(
    state, frames, references, image_ids, dtypes, detector, args, torch
):
    started = time.perf_counter()
    tensors = []
    for frame, dtype in zip(frames, dtypes):
        mean, std = normalization_for_dtype(
            dtype,
            mean=detector.input_mean,
            std=detector.input_std,
            allow_uint8=args.source_policy == "mixed",
        )
        tensors.append(frame_to_tensor(frame, mean=mean, std=std))
    state["tensor_seconds"] += time.perf_counter() - started
    if detector.device.type == "cuda":
        torch.cuda.synchronize(detector.device)
    started = time.perf_counter()
    state["predictions"].extend(
        detector.predict(tensors, image_ids, args.score_threshold)
    )
    if detector.device.type == "cuda":
        torch.cuda.synchronize(detector.device)
    state["inference_seconds"] += time.perf_counter() - started
    started = time.perf_counter()
    for reference, frame in zip(references, frames):
        state["quality"].update(reference, frame)
    state["quality_seconds"] += time.perf_counter() - started


def _finish(
    state,
    dataset,
    args,
    condition,
    level,
    seed,
    stages,
    decode_seconds,
    degradation_seconds,
):
    print(f"Computing mAP: {condition} level={level} seed={seed}...", flush=True)
    started = time.perf_counter()
    metrics = mean_average_precision(dataset, state["predictions"])
    map_seconds = time.perf_counter() - started
    processing = (
        degradation_seconds
        + state["restoration_seconds"]
        + state["tensor_seconds"]
        + state["inference_seconds"]
    )
    row = {
        "condition": condition,
        "level": level,
        "seed": seed,
        "stages": stages,
        "images": len(dataset.images),
        "predictions": len(state["predictions"]),
        "map50": metrics.map50,
        "map50_95": metrics.map50_95,
        "ap50_by_category": {
            str(key): {"name": dataset.categories[key], "value": value}
            for key, value in metrics.ap50_by_category.items()
        },
        "image_quality": state["quality"].result(),
        "timing": {
            "decode_seconds": decode_seconds,
            "degradation_seconds": degradation_seconds,
            **{
                key: state[key]
                for key in (
                    "restoration_seconds",
                    "tensor_seconds",
                    "inference_seconds",
                    "quality_seconds",
                )
            },
            "map_seconds": map_seconds,
            "processing_seconds_per_image": processing / len(dataset.images),
            "inference_seconds_per_image": state["inference_seconds"]
            / len(dataset.images),
        },
    }
    if args.save_predictions:
        label = condition.replace(":", "_")
        path = (
            args.output_dir
            / f"predictions_{label}_{level or 'clean'}_{seed if seed is not None else 'none'}.jsonl"
        )
        with path.open("x", encoding="utf-8") as file:
            for prediction in state["predictions"]:
                file.write(json.dumps(prediction.to_coco(), allow_nan=False) + "\n")
        row["predictions_file"] = path.name
    print(
        f"{condition} level={level} seed={seed} map50={metrics.map50:.4f} map50_95={metrics.map50_95:.4f} mse={row['image_quality']['mse']:.4f} ssim={row['image_quality']['ssim']:.4f}"
    )
    return row


def main():
    args = parse_args()
    config = load_restoration_config(args.restoration_config)
    pipelines = args.pipelines or list(config["pipelines"])
    unknown = set(pipelines) - set(config["pipelines"])
    if unknown:
        raise SystemExit(f"Unknown pipelines: {sorted(unknown)}")
    restorers = {
        name: build_restoration(config["pipelines"][name]) for name in pipelines
    }
    folder = args.data_root / (
        "images_thermal_val" if args.split == "val" else "video_thermal_test"
    )
    original = load_coco(folder / "coco.json", folder)
    dtypes = {
        image.image_id: frame_dtype(image.path)
        for image in tqdm(
            original.images,
            desc="inspect source policy",
            unit="image",
            disable=args.progress_interval == 0,
        )
    }
    unsupported = {
        str(dtype)
        for dtype in dtypes.values()
        if dtype not in (np.dtype(np.uint16), np.dtype(np.uint8))
    }
    if unsupported:
        raise SystemExit(f"Unsupported TIFF source dtypes: {unsupported}")
    selected_ids = [
        image.image_id
        for image in original.images
        if args.source_policy == "mixed"
        or dtypes[image.image_id] == np.dtype(np.uint16)
    ]
    excluded_ids = sorted(set(original.image_ids) - set(selected_ids))
    dataset = subset_dataset(original, selected_ids)
    eligible_count = len(dataset.images)
    if args.max_images is not None and args.max_images < eligible_count:
        indices = np.sort(
            np.random.default_rng(args.sample_seed).choice(
                eligible_count, args.max_images, replace=False
            )
        )
        dataset = subset_dataset(
            dataset, [dataset.images[int(index)].image_id for index in indices]
        )
    if not dataset.images:
        raise SystemExit("No eligible source frames")
    allow_uint8 = args.source_policy == "mixed"
    cache = (
        FrameCache(args.cache_root / args.split, original, allow_uint8=allow_uint8)
        if args.cache_root
        else None
    )
    cache_indices = {
        image.image_id: index for index, image in enumerate(original.images)
    }
    import torch

    device = require_device(torch, args.device)
    detector = TorchvisionDetector.from_checkpoint(args.checkpoint, device.selected)
    detector.model.eval()
    detector.model.requires_grad_(False)
    checkpoint_hash = _sha256(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scope = (
        "smoke_sample"
        if len(dataset.images) < eligible_count
        else "full_source_policy_subset" if excluded_ids else "full_split"
    )
    print(f"device={device.selected} cuda={device.as_dict()}")
    print(
        f"scope={scope} images={len(dataset.images)}/{len(original.images)} excluded_by_source_policy={len(excluded_ids)}"
    )
    print(
        "Frozen A/B/C: same original image IDs, same noise for all restorations, same source normalization. Test must not guide parameter selection."
    )
    started = time.perf_counter()
    rows = []
    clean_row = None
    for level in args.levels:
        for seed in args.seeds:
            states = {
                "B": _new_condition(),
                **{f"C:{name}": _new_condition() for name in pipelines},
            }
            if clean_row is None:
                states = {"A": _new_condition(), **states}
            decode_seconds = degradation_seconds = 0.0
            batches = range(0, len(dataset.images), args.batch_size)
            progress = tqdm(
                batches,
                total=(len(dataset.images) + args.batch_size - 1) // args.batch_size,
                desc=f"A/B/C {level} seed={seed}",
                unit="batch",
                dynamic_ncols=True,
                disable=args.progress_interval == 0,
            )
            for batch_number, start in enumerate(progress, 1):
                images = dataset.images[start : start + args.batch_size]
                ids = [image.image_id for image in images]
                source_dtypes = [dtypes[image.image_id] for image in images]
                load_started = time.perf_counter()
                references = [
                    (
                        cache.load(cache_indices[image.image_id])
                        if cache
                        else load_frame(image.path, allow_uint8=allow_uint8)
                    )
                    for image in images
                ]
                decode_seconds += time.perf_counter() - load_started
                for image, frame in zip(images, references):
                    if frame.shape != (image.height, image.width):
                        raise ValueError(f"TIFF/COCO dimension mismatch: {image.path}")
                degrade_started = time.perf_counter()
                degraded = [
                    degrade_frame(
                        frame,
                        sequence_key=source_sequence(image.path),
                        frame_key=image.path.stem,
                        level=level,
                        seed=seed,
                        steps=args.degradation_steps,
                    )
                    for image, frame in zip(images, references)
                ]
                degradation_seconds += time.perf_counter() - degrade_started
                if "A" in states:
                    _predict_condition(
                        states["A"],
                        references,
                        references,
                        ids,
                        source_dtypes,
                        detector,
                        args,
                        torch,
                    )
                _predict_condition(
                    states["B"],
                    degraded,
                    references,
                    ids,
                    source_dtypes,
                    detector,
                    args,
                    torch,
                )
                for name, restore in restorers.items():
                    state = states[f"C:{name}"]
                    restore_started = time.perf_counter()
                    restored = [restore(frame) for frame in degraded]
                    state["restoration_seconds"] += (
                        time.perf_counter() - restore_started
                    )
                    _predict_condition(
                        state,
                        restored,
                        references,
                        ids,
                        source_dtypes,
                        detector,
                        args,
                        torch,
                    )
                if (
                    args.progress_interval
                    and batch_number % args.progress_interval == 0
                ):
                    progress.set_postfix(
                        images=min(start + args.batch_size, len(dataset.images)),
                        predictions=sum(
                            len(state["predictions"]) for state in states.values()
                        ),
                    )
            if "A" in states:
                clean_row = _finish(
                    states["A"], dataset, args, "A", None, None, [], decode_seconds, 0.0
                )
                rows.append(clean_row)
            degraded_row = _finish(
                states["B"],
                dataset,
                args,
                "B",
                level,
                seed,
                [],
                decode_seconds,
                degradation_seconds,
            )
            rows.append(degraded_row)
            for name in pipelines:
                row = _finish(
                    states[f"C:{name}"],
                    dataset,
                    args,
                    f"C:{name}",
                    level,
                    seed,
                    config["pipelines"][name],
                    decode_seconds,
                    degradation_seconds,
                )
                row["delta_map50_vs_B"] = row["map50"] - degraded_row["map50"]
                row["delta_map50_95_vs_B"] = row["map50_95"] - degraded_row["map50_95"]
                rows.append(row)
            # Release all candidate lists before the next seed/level.
            del states
            state = None
    selected_dtypes = Counter(str(dtypes[image.image_id]) for image in dataset.images)
    support = Counter(
        annotation.category_id
        for annotation in dataset.annotations
        if not annotation.iscrowd
    )
    summary = {
        "version": 1,
        "split": args.split,
        "evaluation_scope": scope,
        "source_policy": args.source_policy,
        "full_split_images": len(original.images),
        "eligible_source_images": eligible_count,
        "images": len(dataset.images),
        "image_ids": list(dataset.image_ids),
        "annotation_ids": [item.annotation_id for item in dataset.annotations],
        "excluded_by_source_policy_image_ids": excluded_ids,
        "dtype_counts": dict(selected_dtypes),
        "class_frame_instance_support": dict(support),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "annotations_path": str(folder / "coco.json"),
        "annotations_sha256": (
            _sha256(folder / "coco.json") if (folder / "coco.json").is_file() else None
        ),
        "model_category_ids": list(detector.category_ids),
        "input_normalization": detector.input_normalization,
        "simulation_version": "sequence-fixed-frame-keyed-v1",
        "device": device.as_dict(),
        "batch_size": args.batch_size,
        "score_threshold": args.score_threshold,
        "normalization": {
            str(dtype): dict(
                zip(
                    ("mean", "std"),
                    normalization_for_dtype(
                        np.dtype(dtype),
                        mean=detector.input_mean,
                        std=detector.input_std,
                        allow_uint8=allow_uint8,
                    ),
                )
            )
            for dtype in selected_dtypes
        },
        "restoration_config": config,
        "levels": {level: get_preset(level) for level in args.levels},
        "degradation_steps": args.degradation_steps,
        "seeds": args.seeds,
        "sample_seed": args.sample_seed,
        "max_images": args.max_images,
        "total_wall_seconds": time.perf_counter() - started,
        "conditions": rows,
        "notes": [
            "A is evaluated once and reused across noise seeds and levels.",
            "Fixed column noise is keyed by sequence; other noise by source frame; independent of method/batch order.",
            "Identical source-specific checkpoint normalization for A/B/C; enhancement can reduce AP or PSNR.",
            "Raw14 filtering changes the evaluated images/classes; never compare this AP directly to the full mixed test AP.",
            "Timings include warm-up; model loading excluded; metrics timed separately. Shared decode/degrade costs appear in each condition as pipeline costs, not additive experiment wall time.",
            "Infinite PSNR is serialized as null with psnr_infinite=true.",
            "Validation selects parameters; test reports a preselected configuration only.",
        ],
    }
    aggregates = []
    for level in args.levels:
        for condition in ["B", *[f"C:{name}" for name in pipelines]]:
            matching = [
                row
                for row in rows
                if row["level"] == level and row["condition"] == condition
            ]
            aggregates.append(
                {
                    "level": level,
                    "condition": condition,
                    "runs": len(matching),
                    "map50_mean": float(np.mean([row["map50"] for row in matching])),
                    "map50_std": float(np.std([row["map50"] for row in matching])),
                    "map50_95_mean": float(
                        np.mean([row["map50_95"] for row in matching])
                    ),
                    "map50_95_std": float(
                        np.std([row["map50_95"] for row in matching])
                    ),
                }
            )
    summary["seed_aggregates"] = aggregates
    summary["seed_aggregate_scope"] = (
        "Population standard deviation over noise seeds, not a confidence interval; one seed gives zero std."
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics.csv").open(
        "x", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "condition",
            "level",
            "seed",
            "images",
            "map50",
            "map50_95",
            "delta_map50_vs_B",
            "mse",
            "psnr_db",
            "psnr_infinite",
            "ssim",
            "processing_seconds_per_image",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(
                        key, row["image_quality"].get(key, row["timing"].get(key))
                    )
                    for key in fields
                }
            )
    lines = [
        "# Frozen detector A/B/C comparison",
        "",
        f"Scope: **{scope}**, split: **{args.split}**, source policy: **{args.source_policy}**.",
        "",
        f"Images: {len(dataset.images)} / {len(original.images)}; excluded by source policy: {len(excluded_ids)}. See metrics.json for IDs, checkpoint SHA256, parameters and per-category AP.",
        "",
        "| Condition | Level | Seed | mAP@0.5 | mAP@[0.5:0.95] | Delta vs B | MSE | PSNR (dB) | SSIM | Processing ms/image |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        quality = row["image_quality"]
        psnr = "inf" if quality["psnr_infinite"] else f"{quality['psnr_db']:.2f}"
        delta = f"{row['delta_map50_vs_B']:+.4f}" if "delta_map50_vs_B" in row else "-"
        lines.append(
            f"| {row['condition']} | {row['level'] or '-'} | {row['seed'] if row['seed'] is not None else '-'} | {row['map50']:.4f} | {row['map50_95']:.4f} | {delta} | {quality['mse']:.3f} | {psnr} | {quality['ssim']:.4f} | {row['timing']['processing_seconds_per_image'] * 1000:.1f} |"
        )
    lines.extend(
        [
            "",
            "No automatic winner selection. Configure methods on validation, not test. m0 should reproduce B. Contrast enhancement may lower detection quality despite making images look clearer.",
            "",
            "PSNR derives from pixel-weighted global MSE; SSIM is the frame mean, using data_range=255. Raw14-subset AP is not the previously reported mixed-bit-depth full-test AP.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
