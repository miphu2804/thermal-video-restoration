"""Train and freeze the Issue #4 torchvision detector on clean FLIR data.

This script is intentionally a user-run entry point. It downloads pretrained
weights only when the user starts it and writes checkpoints under ``outputs/``.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from thermal_restore.data.annotations import CocoDataset, load_coco
from thermal_restore.data.frame_cache import FrameCache
from thermal_restore.data.loader import load_frame
from thermal_restore.detection.device import require_device
from thermal_restore.detection.torchvision_backend import (
    DEFAULT_ANCHOR_SIZES,
    DEFAULT_MAX_SIZE,
    DEFAULT_MIN_SIZE,
    DEFAULT_MODEL_VARIANT,
    MODEL_MOBILENET_320,
    MODEL_MOBILENET_HIGH_RES,
    MODEL_RESNET50_FPN_V2,
    NORMALIZATION_IMAGENET,
    NORMALIZATION_THERMAL,
    TorchvisionCocoDataset,
    TorchvisionDetector,
    collate_detection_batch,
)
from thermal_restore.metrics import mean_average_precision
from thermal_restore.progress import format_duration


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit(
            "Torch is not installed. Run `uv sync --extra detector` first."
        ) from error
    return torch, DataLoader


def _split(root: Path, name: str) -> CocoDataset:
    split_root = root / name
    return load_coco(split_root / "coco.json", split_root)


def _estimate_input_stats(
    dataset: CocoDataset,
    cache_dir: Path | None,
    *,
    frame_stride: int = 10,
    pixel_stride: int = 8,
) -> tuple[float, float]:
    """Estimate fixed detector statistics from a deterministic train sample."""
    cache = FrameCache(cache_dir, dataset) if cache_dir is not None else None
    if cache is not None:
        return cache.sampled_mean_std(
            frame_stride=frame_stride,
            pixel_stride=pixel_stride,
        )
    total = 0
    value_sum = 0.0
    square_sum = 0.0
    for index in range(0, len(dataset.images), frame_stride):
        frame = load_frame(dataset.images[index].path)
        sample = frame[::pixel_stride, ::pixel_stride].astype(np.float64, copy=False)
        total += sample.size
        value_sum += float(sample.sum())
        square_sum += float(np.square(sample).sum())
    mean = value_sum / total
    variance = max(0.0, square_sum / total - mean * mean)
    std = variance**0.5
    if std <= 0:
        raise SystemExit("Cannot standardize detector input with zero variance")
    return mean, std


def _evaluate(
    detector: TorchvisionDetector,
    dataset: CocoDataset,
    loader,
    score_threshold: float,
    progress_interval: int,
    label: str,
) -> tuple[float, float, float]:
    predictions = []
    total_batches = len(loader)
    progress = tqdm(
        total=total_batches,
        desc=label,
        unit="batch",
        dynamic_ncols=True,
        disable=progress_interval == 0,
    )
    for batch_index, (images, targets) in enumerate(loader, start=1):
        image_ids = [int(target["image_id"].item()) for target in targets]
        predictions.extend(detector.predict(images, image_ids, score_threshold))
        progress.update(1)
        if progress_interval > 0 and (
            batch_index == total_batches or batch_index % progress_interval == 0
        ):
            progress.set_postfix(predictions=len(predictions), refresh=True)
    progress.close()
    if progress_interval > 0:
        tqdm.write(f"{label} inference complete; computing mAP...")
    metrics = mean_average_precision(dataset, predictions)
    return metrics.map50, metrics.map50_95, len(predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/FLIR_ADAS_v2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/detector/fasterrcnn_mobilenet_best.pt"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional cache root created by scripts/cache_detector_frames.py.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "adamw"),
        default="sgd",
        help="Optimizer; SGD is the torchvision detection baseline.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=5e-4,
        help="Lower learning rate for the pretrained detector backbone.",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--scheduler",
        choices=("cosine", "step", "none"),
        default="cosine",
        help="Epoch learning-rate schedule; cosine avoids early abrupt decay.",
    )
    parser.add_argument("--lr-step-size", type=int, default=8)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--warmup-iterations", type=int, default=1000)
    parser.add_argument("--warmup-start-factor", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Diagnostic-only batch limit; exits before validation/checkpointing.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help=(
            "Show tqdm progress bars and refresh metrics every N batches; "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.001,
        help="Low confidence floor used to build the validation precision-recall curve.",
    )
    parser.add_argument(
        "--model-variant",
        choices=(
            MODEL_MOBILENET_320,
            MODEL_MOBILENET_HIGH_RES,
            MODEL_RESNET50_FPN_V2,
        ),
        default=DEFAULT_MODEL_VARIANT,
    )
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument(
        "--anchor-sizes",
        type=int,
        nargs=5,
        default=list(DEFAULT_ANCHOR_SIZES),
        metavar=("A1", "A2", "A3", "A4", "A5"),
        help="Five RPN anchor sizes; FLIR contains many boxes smaller than 32 px.",
    )
    parser.add_argument(
        "--input-normalization",
        choices=(NORMALIZATION_THERMAL, NORMALIZATION_IMAGENET),
        default=NORMALIZATION_THERMAL,
        help="Use fixed train-split thermal statistics or legacy ImageNet scaling.",
    )
    parser.add_argument("--input-mean", type=float, default=None)
    parser.add_argument("--input-std", type=float, default=None)
    parser.add_argument(
        "--trainable-backbone-layers",
        type=int,
        default=None,
        help="Backbone layers to fine-tune; defaults to all layers for the model.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-horizontal-flip",
        action="store_true",
        help="Disable the training-only horizontal flip augmentation.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download or use torchvision COCO pretrained weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or args.lr_step_size < 1
        or args.learning_rate <= 0
        or args.backbone_learning_rate <= 0
        or args.weight_decay < 0
        or not 0 < args.lr_gamma <= 1
        or not 0 <= args.momentum < 1
        or not 0 <= args.score_threshold <= 1
        or args.min_size < 1
        or args.max_size < args.min_size
        or any(size < 1 for size in args.anchor_sizes)
        or (
            args.trainable_backbone_layers is not None
            and not 0 <= args.trainable_backbone_layers <= 6
        )
        or (args.input_mean is None) != (args.input_std is None)
        or (args.input_std is not None and args.input_std <= 0)
        or args.warmup_iterations < 0
        or not 0 < args.warmup_start_factor <= 1
        or args.gradient_clip_norm <= 0
        or (args.max_train_batches is not None and args.max_train_batches < 1)
        or args.progress_interval < 0
    ):
        raise SystemExit(
            "epochs, batch-size, learning rates, and lr-step-size must be positive; "
            "workers and weight-decay cannot be negative; momentum must be in [0, 1); "
            "lr-gamma must be in (0, 1]; score-threshold must be in [0, 1]; "
            "model and anchor sizes must be positive, with max-size >= min-size; "
            "input mean/std must be supplied together with a positive std; "
            "trainable-backbone-layers must be in [0, 6]; warmup iterations cannot "
            "be negative; warmup factor must be in (0, 1]; gradient clipping and "
            "max-train-batches must be positive; progress-interval cannot be negative"
        )

    torch, DataLoader = _require_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        device_info = require_device(torch, args.device)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    device = device_info.selected
    print(f"device={device} cuda={device_info.as_dict()}")

    train = _split(args.data_root, "images_thermal_train")
    validation = _split(args.data_root, "images_thermal_val")
    category_ids = train.used_category_ids
    unknown = set(validation.used_category_ids) - set(category_ids)
    if unknown:
        raise SystemExit(
            f"Validation contains categories absent from training: {unknown}"
        )

    input_mean = args.input_mean
    input_std = args.input_std
    if args.input_normalization == NORMALIZATION_THERMAL:
        if input_mean is None:
            input_mean, input_std = _estimate_input_stats(
                train,
                args.cache_root / "train" if args.cache_root else None,
            )
        print(
            f"input_normalization={args.input_normalization} "
            f"mean={input_mean:.6f} std={input_std:.6f}"
        )
    else:
        input_mean = None
        input_std = None

    max_backbone_layers = 5 if args.model_variant == MODEL_RESNET50_FPN_V2 else 6
    trainable_backbone_layers = (
        max_backbone_layers
        if args.trainable_backbone_layers is None
        else args.trainable_backbone_layers
    )
    if trainable_backbone_layers > max_backbone_layers:
        raise SystemExit(
            f"{args.model_variant} supports at most {max_backbone_layers} trainable "
            "backbone layers"
        )

    train_loader = DataLoader(
        TorchvisionCocoDataset(
            train,
            category_ids,
            horizontal_flip=not args.no_horizontal_flip,
            cache_dir=args.cache_root / "train" if args.cache_root else None,
            input_mean=input_mean,
            input_std=input_std,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_detection_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        TorchvisionCocoDataset(
            validation,
            category_ids,
            cache_dir=args.cache_root / "val" if args.cache_root else None,
            input_mean=input_mean,
            input_std=input_std,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_detection_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    detector = TorchvisionDetector.create(
        category_ids,
        device,
        pretrained=not args.no_pretrained,
        model_variant=args.model_variant,
        min_size=args.min_size,
        max_size=args.max_size,
        anchor_sizes=args.anchor_sizes,
        input_normalization=args.input_normalization,
        input_mean=input_mean,
        input_std=input_std,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    backbone_parameters = [
        parameter
        for parameter in detector.model.backbone.parameters()
        if parameter.requires_grad
    ]
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
    detector_parameters = [
        parameter
        for parameter in detector.model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_parameter_ids
    ]
    parameter_groups = [
        {
            "params": detector_parameters,
            "lr": args.learning_rate,
            "name": "detector",
        }
    ]
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": args.backbone_learning_rate,
                "name": "backbone",
            }
        )

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            parameter_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=args.weight_decay,
        )
    if args.scheduler == "cosine":

        def cosine_factor(epoch: int) -> float:
            progress = min(epoch, args.epochs) / args.epochs
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=cosine_factor,
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
        )
    else:
        scheduler = None
    best_map50 = -1.0
    run_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        detector.model.train()
        losses = []
        component_losses: dict[str, list[float]] = {}
        scheduled_lrs = [group["lr"] for group in optimizer.param_groups]
        train_total_batches = len(train_loader)
        if args.max_train_batches is not None:
            train_total_batches = min(train_total_batches, args.max_train_batches)
        train_progress = tqdm(
            total=train_total_batches,
            desc=f"train epoch={epoch}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            disable=args.progress_interval == 0,
        )
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            if epoch == 1 and batch_index <= args.warmup_iterations:
                progress = batch_index / max(1, args.warmup_iterations)
                warmup_factor = args.warmup_start_factor + progress * (
                    1.0 - args.warmup_start_factor
                )
                for group, scheduled_lr in zip(optimizer.param_groups, scheduled_lrs):
                    group["lr"] = scheduled_lr * warmup_factor
            optimizer.zero_grad(set_to_none=True)
            loss_dict = detector.training_losses(images, targets)
            loss_values = {
                name: float(value.detach().cpu()) for name, value in loss_dict.items()
            }
            if not all(math.isfinite(value) for value in loss_values.values()):
                raise SystemExit(
                    f"Non-finite loss at epoch={epoch} batch={batch_index}: "
                    f"{loss_values}. No checkpoint was saved."
                )
            loss = sum(loss_dict.values())
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                detector.model.parameters(),
                max_norm=args.gradient_clip_norm,
                error_if_nonfinite=False,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise SystemExit(
                    f"Non-finite gradient at epoch={epoch} batch={batch_index}: "
                    f"norm={float(gradient_norm)}. No optimizer step or checkpoint "
                    "was written."
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in loss_values.items():
                component_losses.setdefault(name, []).append(value)
            train_progress.update(1)
            if args.progress_interval > 0 and (
                batch_index == train_total_batches
                or batch_index % args.progress_interval == 0
            ):
                current_loss = float(np.mean(losses))
                current_lrs = ",".join(
                    f"{group.get('name', index)}={float(group['lr']):.3g}"
                    for index, group in enumerate(optimizer.param_groups)
                )
                train_progress.set_postfix_str(
                    f"loss={current_loss:.4f} lrs={current_lrs}",
                    refresh=True,
                )
            if (
                args.max_train_batches is not None
                and batch_index >= args.max_train_batches
            ):
                break
        train_progress.close()

        if epoch == 1 and len(losses) < args.warmup_iterations:
            for group, scheduled_lr in zip(optimizer.param_groups, scheduled_lrs):
                group["lr"] = scheduled_lr

        mean_loss = float(np.mean(losses)) if losses else 0.0
        loss_summary = " ".join(
            f"{name}={float(np.mean(values)):.4f}"
            for name, values in sorted(component_losses.items())
        )
        if args.max_train_batches is not None:
            print(
                f"smoke_batches={len(losses)} loss={mean_loss:.4f} "
                f"{loss_summary} elapsed={format_duration(time.perf_counter() - epoch_started)} "
                f"device={device}"
            )
            return

        map50, map50_95, n_predictions = _evaluate(
            detector,
            validation,
            validation_loader,
            args.score_threshold,
            args.progress_interval,
            f"val epoch={epoch}",
        )
        current_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        epoch_elapsed = time.perf_counter() - epoch_started
        run_elapsed = time.perf_counter() - run_started
        run_eta = run_elapsed / epoch * (args.epochs - epoch)
        print(
            f"epoch={epoch} loss={mean_loss:.4f} "
            f"val_map50={map50:.4f} val_map50_95={map50_95:.4f} "
            f"predictions={n_predictions} lrs={current_lrs} {loss_summary} "
            f"epoch_time={format_duration(epoch_elapsed)} "
            f"run_time={format_duration(run_elapsed)} "
            f"run_eta={format_duration(run_eta)} device={device}"
        )
        if math.isfinite(map50) and map50 > best_map50:
            best_map50 = map50
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": detector.model.state_dict(),
                    "category_ids": list(category_ids),
                    "epoch": epoch,
                    "val_map50": map50,
                    "val_map50_95": map50_95,
                    "device": device,
                    "seed": args.seed,
                    "optimizer": args.optimizer,
                    "learning_rate": args.learning_rate,
                    "backbone_learning_rate": args.backbone_learning_rate,
                    "warmup_iterations": args.warmup_iterations,
                    "gradient_clip_norm": args.gradient_clip_norm,
                    "scheduler": args.scheduler,
                    "cache_root": str(args.cache_root) if args.cache_root else None,
                    "model_config": {
                        "model_variant": args.model_variant,
                        "min_size": args.min_size,
                        "max_size": args.max_size,
                        "anchor_sizes": list(args.anchor_sizes),
                        "input_normalization": args.input_normalization,
                        "input_mean": input_mean,
                        "input_std": input_std,
                        "trainable_backbone_layers": trainable_backbone_layers,
                    },
                },
                args.output,
            )
            print(f"saved={args.output}")
        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    main()
