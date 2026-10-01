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
from thermal_restore.data.loader import load_frame
from thermal_restore.detection.types import Detection


def _torchvision_imports() -> tuple[Any, Any, Any, Any]:
    """Import optional detector dependencies with an actionable error."""
    try:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as error:
        raise RuntimeError(
            "The torchvision detector requires the 'detector' extra. "
            "Run `uv sync --extra detector` before training or inference."
        ) from error
    return (
        torch,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FastRCNNPredictor,
    )


def frame_to_tensor(frame: np.ndarray) -> Any:
    """Convert a 0–255 thermal frame to a three-channel model tensor.

    The pretrained torchvision backbone expects three channels. The normalized
    thermal channel is repeated rather than colourized, preserving the thermal
    signal while allowing reuse of pretrained weights.
    """
    torch, *_ = _torchvision_imports()
    array = np.asarray(frame, dtype=np.float32) / 255.0
    if array.ndim != 2:
        raise ValueError(f"Expected one thermal frame with shape (H, W), got {array.shape}")
    return torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)


class TorchvisionCocoDataset:
    """Lazy COCO dataset yielding torchvision detection samples."""

    def __init__(
        self, dataset: CocoDataset, category_ids: Sequence[int] | None = None
    ) -> None:
        self.dataset = dataset
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
            raise ValueError(f"Dataset contains categories outside the model map: {unknown}")

    def __len__(self) -> int:
        return len(self.dataset.images)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        torch, *_ = _torchvision_imports()
        image = self.dataset.images[index]
        frame = load_frame(image.path)
        tensor = frame_to_tensor(frame)
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
        return tensor, target


def collate_detection_batch(batch: Sequence[tuple[Any, dict[str, Any]]]) -> tuple:
    """Keep variable-length detection targets grouped by sample."""
    images, targets = zip(*batch)
    return images, targets


def create_fasterrcnn_mobilenet(
    category_ids: Sequence[int],
    pretrained: bool = True,
) -> Any:
    """Create a small Faster R-CNN model with a thermal-compatible head."""
    (
        _torch,
        weights_enum,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        predictor_type,
    ) = _torchvision_imports()
    weights = weights_enum.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = predictor_type(
        input_features, len(tuple(category_ids)) + 1
    )
    return model


class TorchvisionDetector:
    """Frozen-or-trainable wrapper around the torchvision detector."""

    def __init__(self, model: Any, category_ids: Sequence[int], device: str) -> None:
        torch, *_ = _torchvision_imports()
        self.model = model.to(device)
        self.category_ids = tuple(int(category_id) for category_id in category_ids)
        self.device = torch.device(device)

    @classmethod
    def create(
        cls,
        category_ids: Sequence[int],
        device: str,
        pretrained: bool = True,
    ) -> TorchvisionDetector:
        """Create a detector with a fresh classification head."""
        model = create_fasterrcnn_mobilenet(category_ids, pretrained=pretrained)
        return cls(model, category_ids, device)

    @classmethod
    def from_checkpoint(cls, path: Path, device: str) -> TorchvisionDetector:
        """Load a checkpoint produced by ``scripts/train_detector.py``."""
        torch, *_ = _torchvision_imports()
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        category_ids = tuple(int(value) for value in checkpoint["category_ids"])
        detector = cls(
            create_fasterrcnn_mobilenet(category_ids, pretrained=False),
            category_ids,
            device,
        )
        detector.model.load_state_dict(checkpoint["model_state"])
        detector.model.eval()
        return detector

    def training_loss(self, images: Sequence[Any], targets: Sequence[dict]) -> Any:
        """Return the summed torchvision training loss for one batch."""
        images = [image.to(self.device) for image in images]
        targets = [
            {key: value.to(self.device) for key, value in target.items()}
            for target in targets
        ]
        loss_dict = self.model(images, targets)
        return sum(loss_dict.values())

    def predict(
        self,
        images: Sequence[Any],
        image_ids: Sequence[int],
        score_threshold: float = 0.25,
    ) -> list[Detection]:
        """Predict backend-neutral detections for a batch of tensors."""
        torch, *_ = _torchvision_imports()
        if len(images) != len(image_ids):
            raise ValueError("images and image_ids must have the same length")
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model([image.to(self.device) for image in images])

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
