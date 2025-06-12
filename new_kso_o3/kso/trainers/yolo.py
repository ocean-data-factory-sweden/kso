"""Skeleton YOLOv5/8 trainer (placeholder)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import AbstractTrainer
from ..registry import register_trainer


@register_trainer("yolo")
class YoloTrainer(AbstractTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from ultralytics import YOLO

        self.YOLO = YOLO
        self.model_name = config.get("arch", "yolov8n.pt")  # Can be pre-trained checkpoint
        self.epochs = int(config.get("epochs", 50))

        # Instantiate model (downloads weights if not present)
        self.model = YOLO(self.model_name)

    def train(self, dataset_path: Path):
        print(f"[YoloTrainer] Training on {dataset_path} with cfg: {self.cfg}")

        data_yaml = dataset_path / "yolo_data.yaml"  # assume user prepared YAML
        if not data_yaml.exists():
            # quick generate minimal data.yaml referencing train/val dirs
            (dataset_path / "images").mkdir(exist_ok=True)
            # For demo we just write simple yaml; real usage expects proper spec
            data_yaml.write_text(
                f"path: {dataset_path}\ntrain: images/train\nval: images/validation\n"
            )

        from ..utils import metrics as metrics

        with metrics.start_run():
            metrics.log_params(self.cfg)

            self.model.train(data=str(data_yaml), epochs=self.epochs, imgsz=640, device=0)

            # ultralytics logs metrics internally; capture summary
            metrics.log_metrics({"epochs": self.epochs})

    def evaluate(self, dataset_path: Path):
        print(f"[YoloTrainer] Evaluating on {dataset_path}")
        metrics = self.model.val()
        print(metrics)
        return metrics.get("metrics/mAP50-95", 0.0)

    def save(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "yolo_model.pt"
        print(f"[YoloTrainer] Saving model to {model_path}")
        return str(model_path)
