"""YOLO inference implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AbstractInference
from ..registry import register_inference


@register_inference("yolo")
class YoloInference(AbstractInference):
    def __init__(self, model_path: Path, **kwargs: Any):
        super().__init__(model_path)
        # TODO: load YOLO model

    def predict(self, image_path: Path, **kwargs: Any):
        print(f"[YoloInference] Predicting on {image_path}")

    def predict_batch(self, image_dir: Path, **kwargs: Any):
        print(f"[YoloInference] Batch predicting in {image_dir}")
