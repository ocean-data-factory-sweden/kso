"""Abstract base Trainer class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class AbstractTrainer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    # ---------------------------------------------------------------------
    @abstractmethod
    def train(self, dataset_path: Path):
        """Run training."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, dataset_path: Path):
        """Evaluate on validation/test set."""
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: Path) -> str:
        """Save model weights & return model_id/path."""
        raise NotImplementedError
