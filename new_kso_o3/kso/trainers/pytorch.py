"""Skeleton PyTorch trainer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import AbstractTrainer
from ..registry import register_trainer


@register_trainer("pytorch")
class TorchTrainer(AbstractTrainer):
    def __init__(self, config: Dict[str, Any]):
        """Instantiate trainer.

        Expected config keys:
        - arch: torchvision model name (e.g. "resnet18")
        - num_classes: int
        - batch_size: int
        - learning_rate: float
        - epochs: int
        """
        super().__init__(config)

        # Lazy imports to keep startup light when not in torch workflows
        import torch
        import torchvision

        self.torch = torch
        self.tv = torchvision

        arch = config.get("arch", "resnet18")
        num_classes = int(config.get("num_classes", 2))

        # Dynamically fetch model constructor from torchvision.models
        model_fn = getattr(self.tv.models, arch)
        self.model = model_fn(weights=None)

        # Patch final layer for classification
        if hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features  # type: ignore[attr-defined]
            self.model.fc = self.torch.nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]

        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = self.torch.nn.CrossEntropyLoss()
        self.optimizer = self.torch.optim.Adam(
            self.model.parameters(), lr=float(config.get("learning_rate", 1e-3))
        )

    # ------------------------------------------------------------------
    def train(self, dataset_path: Path):
        print(f"[TorchTrainer] Training on {dataset_path} with cfg: {self.cfg}")

        batch_size = int(self.cfg.get("batch_size", 16))
        epochs = int(self.cfg.get("epochs", 10))

        transform = self.tv.transforms.Compose([
            self.tv.transforms.Resize((224, 224)),
            self.tv.transforms.ToTensor(),
        ])

        train_dir = dataset_path / "train"
        val_dir = dataset_path / "validation"

        train_ds = self.tv.datasets.ImageFolder(train_dir, transform=transform)
        val_ds = self.tv.datasets.ImageFolder(val_dir, transform=transform)

        train_loader = self.torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = self.torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=4
        )

        from ..utils import metrics as metrics  # local import to prevent circular

        with metrics.start_run():
            metrics.log_params(self.cfg)

            for epoch in range(1, epochs + 1):
                self.model.train()
                running_loss = 0.0
                for images, targets in train_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * images.size(0)

                epoch_loss = running_loss / len(train_loader.dataset)

                # Evaluate
                acc = self._evaluate_loader(val_loader)

                print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - val_acc: {acc:.4f}")

                metrics.log_metrics({"loss": epoch_loss, "val_acc": acc, "epoch": epoch})

    def evaluate(self, dataset_path: Path):
        print(f"[TorchTrainer] Evaluating on {dataset_path}")
        transform = self.tv.transforms.Compose([
            self.tv.transforms.Resize((224, 224)),
            self.tv.transforms.ToTensor(),
        ])

        test_ds = self.tv.datasets.ImageFolder(dataset_path / "test", transform=transform)
        test_loader = self.torch.utils.data.DataLoader(
            test_ds, batch_size=int(self.cfg.get("batch_size", 16)), shuffle=False, num_workers=4
        )

        acc = self._evaluate_loader(test_loader)
        print(f"Test accuracy: {acc:.4f}")
        return acc

    def save(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pt"
        print(f"[TorchTrainer] Saving model to {model_path}")

        from ..storage import get_storage

        self.torch.save(self.model.state_dict(), model_path)

        # Optional remote upload
        storage = get_storage()
        if hasattr(storage, "upload_file"):
            storage.upload_file(model_path, f"{self.cfg.get('project_name', 'model')}/model.pt")

        # Write metadata
        meta_path = output_dir / "metadata.json"
        import json
        import subprocess

        meta = {
            "hyper_params": self.cfg,
            "git_commit": subprocess.getoutput("git rev-parse --short HEAD"),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return str(model_path)

    # ------------------------------------------------------------------
    def _evaluate_loader(self, loader):
        """Compute accuracy on a dataloader."""
        self.model.eval()
        correct = 0
        total = 0
        with self.torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)
                _, preds = self.torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return correct / total if total else 0.0
