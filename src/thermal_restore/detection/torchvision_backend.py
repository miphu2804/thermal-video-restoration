"""Optional torchvision detector backend for Issue #4.

Torch and torchvision are optional because the repository's data and metric
tests must remain runnable without a detector installation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from thermal_restore.data.annotations import CocoDataset
from thermal_restore.data.frame_cache import FrameCache
from thermal_restore.data.loader import frame_dtype, load_frame
from thermal_restore.detection.types import Detection

MODEL_MOBILENET_320 = "mobilenet_320"
MODEL_MOBILENET_HIGH_RES = "mobilenet_high_res"
MODEL_RESNET50_FPN_V2 = "resnet50_fpn_v2"
DEFAULT_MODEL_VARIANT = MODEL_MOBILENET_HIGH_RES
DEFAULT_MIN_SIZE = 512
DEFAULT_MAX_SIZE = 640
DEFAULT_ANCHOR_SIZES = (8, 16, 32, 64, 128)

LEGACY_MODEL_VARIANT = MODEL_MOBILENET_320
LEGACY_MIN_SIZE = 320
LEGACY_MAX_SIZE = 640
LEGACY_ANCHOR_SIZES = (32, 64, 128, 256, 512)

NORMALIZATION_IMAGENET = "imagenet"
NORMALIZATION_THERMAL = "thermal_standardize"
DEFAULT_NORMALIZATION = NORMALIZATION_THERMAL
# Grayscale equivalents of the ImageNet RGB channel statistics. These are
# used only for FLIR mirror frames that are already contrast-stretched uint8.
UINT8_DISPLAY_MEAN = 0.449 * 255.0
UINT8_DISPLAY_STD = 0.226 * 255.0


def normalization_for_dtype(
    dtype: np.dtype,
    *,
    mean: float | None,
    std: float | None,
    allow_uint8: bool = False,
) -> tuple[float | None, float | None]:
    """Resolve the fixed statistics actually applied to a source frame."""
    if mean is not None and allow_uint8 and np.dtype(dtype) == np.dtype(np.uint8):
        return UINT8_DISPLAY_MEAN, UINT8_DISPLAY_STD
    return mean, std


def _torchvision_imports() -> tuple[Any, ...]:
    """Import optional detector dependencies with an actionable error."""
    try:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
            fasterrcnn_mobilenet_v3_large_fpn,
            fasterrcnn_resnet50_fpn_v2,
        )
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as error:
        raise RuntimeError(
            "The torchvision detector requires the 'detector' extra. "
            "Run `uv sync --extra detector` before training or inference. "
            "The extra installs the CUDA 13.0 build on Windows/Linux."
        ) from error
    return (
        torch,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
        AnchorGenerator,
        FastRCNNPredictor,
    )


def normalize_detector_frame(
    frame: np.ndarray,
    *,
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """Convert a detector frame to either [0, 1] or fixed standardized scale."""
    array = np.asarray(frame, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"Expected one thermal frame with shape (H, W), got {array.shape}"
        )
    if mean is None and std is None:
        return array / np.float32(255.0)
    if mean is None or std is None or not np.isfinite(std) or std <= 0:
        raise ValueError("Detector normalization requires a positive mean/std pair")
    return (array - np.float32(mean)) / np.float32(std)


def frame_to_tensor(
    frame: np.ndarray,
    *,
    mean: float | None = None,
    std: float | None = None,
) -> Any:
    """Convert a 0–255 thermal frame to a three-channel model tensor.

    The pretrained torchvision backbone expects three channels. The normalized
    thermal channel is repeated rather than colourized, preserving the thermal
    signal while allowing reuse of pretrained weights.
    """
    torch, *_ = _torchvision_imports()
    array = normalize_detector_frame(frame, mean=mean, std=std)
    return torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)


class TorchvisionCocoDataset:
    """Lazy COCO dataset yielding torchvision detection samples."""

    def __init__(
        self,
        dataset: CocoDataset,
        category_ids: Sequence[int] | None = None,
        *,
        allow_uint8: bool = False,
        horizontal_flip: bool = False,
        cache_dir: Path | None = None,
        input_mean: float | None = None,
        input_std: float | None = None,
    ) -> None:
        self.dataset = dataset
        self.allow_uint8 = allow_uint8
        self.horizontal_flip = horizontal_flip
        self.input_mean = input_mean
        self.input_std = input_std
        self.cache = (
            FrameCache(cache_dir, dataset, allow_uint8=allow_uint8)
            if cache_dir is not None
            else None
        )
        self.category_ids = tuple(
            int(category_id)
            for category_id in (category_ids or dataset.used_category_ids)
        )
        self._labels = {
            category_id: index + 1
            for index, category_id in enumerate(self.category_ids)
        }
        unknown = set(dataset.used_category_ids) - set(self._labels)
        if unknown:
            raise ValueError(
                f"Dataset contains categories outside the model map: {unknown}"
            )

    def __len__(self) -> int:
        return len(self.dataset.images)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        torch, *_ = _torchvision_imports()
        image = self.dataset.images[index]
        frame = (
            self.cache.load(index)
            if self.cache is not None
            else load_frame(image.path, allow_uint8=self.allow_uint8)
        )
        input_mean = self.input_mean
        input_std = self.input_std
        if (
            input_mean is not None
            and self.allow_uint8
            and frame_dtype(image.path) == np.dtype(np.uint8)
        ):
            input_mean, input_std = normalization_for_dtype(
                np.dtype(np.uint8),
                mean=input_mean,
                std=input_std,
                allow_uint8=self.allow_uint8,
            )
        tensor = frame_to_tensor(frame, mean=input_mean, std=input_std)
        annotations = [
            annotation
            for annotation in self.dataset.annotations_for_image(image.image_id)
            if not annotation.iscrowd
        ]
        boxes = []
        labels = []
        areas = []
        for annotation in annotations:
            x, y, width, height = annotation.bbox
            boxes.append([x, y, x + width, y + height])
            labels.append(self._labels[annotation.category_id])
            areas.append(annotation.area)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image.image_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(annotations), dtype=torch.int64),
        }
        if self.horizontal_flip and bool(torch.rand(1).item() < 0.5):
            tensor = torch.flip(tensor, dims=(2,))
            boxes = target["boxes"]
            if boxes.numel():
                flipped = boxes.clone()
                flipped[:, 0] = image.width - boxes[:, 2]
                flipped[:, 2] = image.width - boxes[:, 0]
                target["boxes"] = flipped
        return tensor, target


def collate_detection_batch(batch: Sequence[tuple[Any, dict[str, Any]]]) -> tuple:
    """Keep variable-length detection targets grouped by sample."""
    images, targets = zip(*batch)
    return images, targets


def create_fasterrcnn_detector(
    category_ids: Sequence[int],
    pretrained: bool = True,
    *,
    model_variant: str = DEFAULT_MODEL_VARIANT,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    anchor_sizes: Sequence[int] = DEFAULT_ANCHOR_SIZES,
    input_normalization: str = NORMALIZATION_IMAGENET,
    trainable_backbone_layers: int | None = None,
) -> Any:
    """Create a Faster R-CNN model configured for FLIR object scales."""
    (
        _torch,
        weights_320,
        constructor_320,
        weights_high_res,
        constructor_high_res,
        weights_resnet,
        constructor_resnet,
        anchor_generator_type,
        predictor_type,
    ) = _torchvision_imports()
    if model_variant == MODEL_MOBILENET_320:
        weights_enum = weights_320
        constructor = constructor_320
    elif model_variant == MODEL_MOBILENET_HIGH_RES:
        weights_enum = weights_high_res
        constructor = constructor_high_res
        max_trainable_layers = 6
    elif model_variant == MODEL_RESNET50_FPN_V2:
        weights_enum = weights_resnet
        constructor = constructor_resnet
        max_trainable_layers = 5
    else:
        raise ValueError(f"Unsupported detector model variant: {model_variant!r}")
    if model_variant == MODEL_MOBILENET_320:
        max_trainable_layers = 6
    if min_size < 1 or max_size < min_size:
        raise ValueError("Detector sizes must satisfy 1 <= min_size <= max_size")
    if input_normalization not in {NORMALIZATION_IMAGENET, NORMALIZATION_THERMAL}:
        raise ValueError(
            f"Unsupported detector input normalization: {input_normalization!r}"
        )
    if trainable_backbone_layers is None:
        trainable_backbone_layers = max_trainable_layers
    if not 0 <= trainable_backbone_layers <= max_trainable_layers:
        raise ValueError(
            f"trainable_backbone_layers must be in [0, {max_trainable_layers}] "
            f"for {model_variant}"
        )

    anchor_sizes = tuple(int(size) for size in anchor_sizes)
    if len(anchor_sizes) != 5 or any(size < 1 for size in anchor_sizes):
        raise ValueError("anchor_sizes must contain five positive sizes")

    if pretrained:
        model = constructor(
            weights=weights_enum.DEFAULT,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        model = constructor(
            weights=None,
            weights_backbone=None,
        )
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    feature_levels = len(model.rpn.anchor_generator.sizes)
    aspect_ratios = model.rpn.anchor_generator.aspect_ratios
    if model_variant == MODEL_RESNET50_FPN_V2:
        configured_sizes = tuple((size,) for size in anchor_sizes)
    else:
        configured_sizes = (anchor_sizes,) * feature_levels
    if len(configured_sizes) != feature_levels:
        raise ValueError(
            f"{model_variant} exposes {feature_levels} FPN levels, but received "
            f"{len(configured_sizes)} anchor groups"
        )
    model.rpn.anchor_generator = anchor_generator_type(
        sizes=configured_sizes,
        aspect_ratios=aspect_ratios,
    )
    if input_normalization == NORMALIZATION_THERMAL:
        model.transform.image_mean = [0.0, 0.0, 0.0]
        model.transform.image_std = [1.0, 1.0, 1.0]
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = predictor_type(
        input_features, len(tuple(category_ids)) + 1
    )
    return model


class TorchvisionDetector:
    """Frozen-or-trainable wrapper around the torchvision detector."""

    def __init__(
        self,
        model: Any,
        category_ids: Sequence[int],
        device: str,
        *,
        input_normalization: str = NORMALIZATION_IMAGENET,
        input_mean: float | None = None,
        input_std: float | None = None,
    ) -> None:
        torch, *_ = _torchvision_imports()
        self.model = model.to(device)
        self.category_ids = tuple(int(category_id) for category_id in category_ids)
        self.device = torch.device(device)
        self.input_normalization = input_normalization
        self.input_mean = input_mean
        self.input_std = input_std

    @classmethod
    def create(
        cls,
        category_ids: Sequence[int],
        device: str,
        pretrained: bool = True,
        *,
        model_variant: str = DEFAULT_MODEL_VARIANT,
        min_size: int = DEFAULT_MIN_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        anchor_sizes: Sequence[int] = DEFAULT_ANCHOR_SIZES,
        input_normalization: str = DEFAULT_NORMALIZATION,
        input_mean: float | None = None,
        input_std: float | None = None,
        trainable_backbone_layers: int | None = None,
    ) -> TorchvisionDetector:
        """Create a detector with a fresh classification head."""
        model = create_fasterrcnn_detector(
            category_ids,
            pretrained=pretrained,
            model_variant=model_variant,
            min_size=min_size,
            max_size=max_size,
            anchor_sizes=anchor_sizes,
            input_normalization=input_normalization,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        return cls(
            model,
            category_ids,
            device,
            input_normalization=input_normalization,
            input_mean=input_mean,
            input_std=input_std,
        )

    @classmethod
    def from_checkpoint(cls, path: Path, device: str) -> TorchvisionDetector:
        """Load a checkpoint produced by ``scripts/train_detector.py``."""
        torch, *_ = _torchvision_imports()
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        category_ids = tuple(int(value) for value in checkpoint["category_ids"])
        model_config = checkpoint.get("model_config") or {
            "model_variant": LEGACY_MODEL_VARIANT,
            "min_size": LEGACY_MIN_SIZE,
            "max_size": LEGACY_MAX_SIZE,
            "anchor_sizes": LEGACY_ANCHOR_SIZES,
            "input_normalization": NORMALIZATION_IMAGENET,
            "input_mean": None,
            "input_std": None,
            "trainable_backbone_layers": 3,
        }
        input_normalization = str(
            model_config.get("input_normalization", NORMALIZATION_IMAGENET)
        )
        input_mean = model_config.get("input_mean")
        input_std = model_config.get("input_std")
        detector = cls(
            create_fasterrcnn_detector(
                category_ids,
                pretrained=False,
                model_variant=str(model_config["model_variant"]),
                min_size=int(model_config["min_size"]),
                max_size=int(model_config["max_size"]),
                anchor_sizes=tuple(int(size) for size in model_config["anchor_sizes"]),
                input_normalization=input_normalization,
                trainable_backbone_layers=int(
                    model_config.get("trainable_backbone_layers", 3)
                ),
            ),
            category_ids,
            device,
            input_normalization=input_normalization,
            input_mean=float(input_mean) if input_mean is not None else None,
            input_std=float(input_std) if input_std is not None else None,
        )
        detector.model.load_state_dict(checkpoint["model_state"])
        detector.model.eval()
        return detector

    def training_losses(
        self,
        images: Sequence[Any],
        targets: Sequence[dict],
    ) -> dict[str, Any]:
        """Return named torchvision training losses for one batch."""
        non_blocking = self.device.type == "cuda"
        images = [image.to(self.device, non_blocking=non_blocking) for image in images]
        targets = [
            {
                key: value.to(self.device, non_blocking=non_blocking)
                for key, value in target.items()
            }
            for target in targets
        ]
        return self.model(images, targets)

    def training_loss(self, images: Sequence[Any], targets: Sequence[dict]) -> Any:
        """Return the summed torchvision training loss for one batch."""
        return sum(self.training_losses(images, targets).values())

    def predict(
        self,
        images: Sequence[Any],
        image_ids: Sequence[int],
        score_threshold: float = 0.001,
    ) -> list[Detection]:
        """Predict backend-neutral detections for a batch of tensors."""
        torch, *_ = _torchvision_imports()
        if len(images) != len(image_ids):
            raise ValueError("images and image_ids must have the same length")
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        self.model.eval()
        non_blocking = self.device.type == "cuda"
        previous_threshold = self.model.roi_heads.score_thresh
        self.model.roi_heads.score_thresh = min(previous_threshold, score_threshold)
        try:
            with torch.inference_mode():
                outputs = self.model(
                    [
                        image.to(self.device, non_blocking=non_blocking)
                        for image in images
                    ]
                )
        finally:
            self.model.roi_heads.score_thresh = previous_threshold

        detections = []
        for image_id, output in zip(image_ids, outputs):
            for box, label, score in zip(
                output["boxes"].cpu().tolist(),
                output["labels"].cpu().tolist(),
                output["scores"].cpu().tolist(),
            ):
                if score < score_threshold or not 1 <= label <= len(self.category_ids):
                    continue
                x1, y1, x2, y2 = box
                detections.append(
                    Detection(
                        image_id=int(image_id),
                        category_id=self.category_ids[label - 1],
                        bbox=(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
                        score=float(score),
                    )
                )
        return detections
