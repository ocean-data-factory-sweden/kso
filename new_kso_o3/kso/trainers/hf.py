"""Skeleton Hugging Face vision trainer (e.g., DETR, ViT)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import AbstractTrainer
from ..registry import register_trainer


@register_trainer("hf")
class HFTrainer(AbstractTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Lazy import heavy deps on demand
        from transformers import (
            AutoModelForImageClassification,
            AutoImageProcessor,
            TrainingArguments,
            Trainer,
        )

        self.Trainer = Trainer
        self.AutoModelCls = AutoModelForImageClassification
        self.Preprocess = AutoImageProcessor
        self.TrainingArguments = TrainingArguments

        model_name = config.get("model_name", "google/vit-base-patch16-224")
        num_labels = int(config.get("num_classes", 2))

        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def train(self, dataset_path: Path):
        print(f"[HFTrainer] Training on {dataset_path} with cfg: {self.cfg}")

        from datasets import load_from_disk, load_dataset

        # Expect dataset_path contains HF Arrow dataset or raw images in train/val/test
        if (dataset_path / "dataset_info.json").exists():
            dataset = load_from_disk(str(dataset_path))
        else:
            dataset = load_dataset(
                "imagefolder",
                data_dir=str(dataset_path),
            )

        def transform(example):
            image = example["image"]
            inputs = self.processor(image.convert("RGB"), return_tensors="pt")
            example["pixel_values"] = inputs["pixel_values"][0]
            return example

        dataset = dataset.with_transform(transform)

        train_ds = dataset["train"]
        val_ds = dataset.get("validation", None) or dataset["test"]

        args = self.TrainingArguments(
            output_dir="./hf_runs",
            per_device_train_batch_size=int(self.cfg.get("batch_size", 8)),
            per_device_eval_batch_size=int(self.cfg.get("batch_size", 8)),
            num_train_epochs=int(self.cfg.get("epochs", 3)),
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
        )

        from ..utils import metrics as metrics

        with metrics.start_run():
            metrics.log_params(self.cfg)

            trainer = self.Trainer(
                model=self.model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
            )

            trainer.train()

            eval_metrics = trainer.evaluate()
            metrics.log_metrics(eval_metrics)

    def evaluate(self, dataset_path: Path):
        print(f"[HFTrainer] Evaluating on {dataset_path}")
        from datasets import load_from_disk, load_dataset

        if (dataset_path / "dataset_info.json").exists():
            test_ds = load_from_disk(str(dataset_path))["test"]
        else:
            test_ds = load_dataset("imagefolder", data_dir=str(dataset_path))["test"]

        def transform(example):
            image = example["image"]
            inputs = self.processor(image.convert("RGB"), return_tensors="pt")
            example["pixel_values"] = inputs["pixel_values"][0]
            return example

        test_ds = test_ds.with_transform(transform)

        trainer = self.Trainer(model=self.model)
        metrics = trainer.evaluate(test_ds)
        print(metrics)
        return metrics.get("eval_accuracy", 0.0)

    def save(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "hf_model"
        print(f"[HFTrainer] Saving model to {model_path}")
        # TODO: self.model.save_pretrained(model_path)
        return str(model_path)
