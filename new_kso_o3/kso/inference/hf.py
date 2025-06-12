"""Hugging Face inference implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AbstractInference
from ..registry import register_inference


@register_inference("hf")
class HFInference(AbstractInference):
    def __init__(self, model_path: Path, **kwargs: Any):
        super().__init__(model_path)
        # TODO: load pretrained model

    def predict(self, image_path: Path, **kwargs: Any):
        print(f"[HFInference] Predicting on {image_path}")

    def predict_batch(self, image_dir: Path, **kwargs: Any):
        print(f"[HFInference] Batch predicting in {image_dir}")
