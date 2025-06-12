"""PyTorch inference implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AbstractInference
from ..registry import register_inference


@register_inference("pytorch")
class TorchInference(AbstractInference):
    def __init__(self, model_path: Path, **kwargs: Any):
        super().__init__(model_path)
        import torch

        self.torch = torch
        # TODO: load model

    def predict(self, image_path: Path, **kwargs: Any):
        print(f"[TorchInference] Predicting on {image_path}")
        # TODO: perform inference

    def predict_batch(self, image_dir: Path, **kwargs: Any):
        print(f"[TorchInference] Batch predicting in {image_dir}")
        # TODO: batch inference
