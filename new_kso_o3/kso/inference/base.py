"""Base inference interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class AbstractInference(ABC):
    def __init__(self, model_path: Path, **kwargs: Any):
        self.model_path = model_path

    @abstractmethod
    def predict(self, image_path: Path, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, image_dir: Path, **kwargs: Any):
        raise NotImplementedError
