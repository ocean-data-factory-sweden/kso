"""Unit tests for trainer registry and lightweight instantiation.

These tests avoid heavyweight training by mocking external libraries and network
calls. The aim is to guarantee that each registered trainer class can be
imported and instantiated with a minimal config without raising exceptions.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kso.registry import get_trainer

# Ensure trainer modules are imported so decorators run and registry populated
importlib.import_module("kso.trainers.pytorch")
importlib.import_module("kso.trainers.yolo")
importlib.import_module("kso.trainers.hf")

@pytest.mark.parametrize("name", ["pytorch", "yolo", "hf"])
def test_trainer_registry_resolution(name):
    """Registry should resolve trainer names to classes."""
    cls = get_trainer(name)
    assert cls is not None
    assert cls.__name__.lower().endswith("trainer")


def test_pytorch_trainer_init(tmp_path):
    """TorchTrainer should instantiate with minimal config."""
    TorchTrainer = get_trainer("pytorch")

    trainer = TorchTrainer({"epochs": 0, "num_classes": 2})

    assert hasattr(trainer, "train") and callable(trainer.train)
    assert hasattr(trainer, "evaluate") and callable(trainer.evaluate)
    assert hasattr(trainer, "save")


def test_yolo_trainer_init(monkeypatch):
    """YoloTrainer should instantiate using mocked ultralytics to avoid downloads."""

    # Build dummy ultralytics module
    dummy_ultra = types.ModuleType("ultralytics")

    class DummyYOLO:  # noqa: WPS110
        def __init__(self, *_, **__):
            self.trained = False

        def train(self, *_, **__):  # noqa: D401
            self.trained = True

        def val(self, *_, **__):  # noqa: D401
            return {"metrics/mAP50-95": 0.42}

    dummy_ultra.YOLO = DummyYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", dummy_ultra)

    # Reload trainer module to pick up patched ultralytics
    yolo_mod = importlib.reload(importlib.import_module("kso.trainers.yolo"))
    YoloTrainer = yolo_mod.YoloTrainer

    trainer = YoloTrainer({"epochs": 0})
    assert isinstance(trainer.model, DummyYOLO)


def test_hf_trainer_init(monkeypatch, tmp_path):
    """HFTrainer should instantiate with mocked transformers stack."""

    dummy_transformers = types.ModuleType("transformers")

    # Dummy model & processor
    class DummyModel:  # noqa: WPS110
        def save_pretrained(self, *_, **__):
            pass

    dummy_transformers.AutoModelForImageClassification = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: DummyModel()
    )
    dummy_transformers.AutoImageProcessor = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: types.SimpleNamespace(
            __call__=lambda self, img, return_tensors=None: {"pixel_values": [[0]]}
        )
    )
    dummy_transformers.TrainingArguments = MagicMock
    dummy_transformers.Trainer = MagicMock

    monkeypatch.setitem(sys.modules, "transformers", dummy_transformers)

    # datasets module minimal stub to satisfy imports
    dummy_datasets = types.ModuleType("datasets")
    dummy_datasets.load_from_disk = MagicMock(return_value={"train": [], "test": []})
    dummy_datasets.load_dataset = MagicMock(return_value={"train": [], "test": []})
    monkeypatch.setitem(sys.modules, "datasets", dummy_datasets)

    hf_mod = importlib.reload(importlib.import_module("kso.trainers.hf"))
    HFTrainer = hf_mod.HFTrainer

    trainer = HFTrainer({"epochs": 0, "num_classes": 2})
    assert isinstance(trainer.model, DummyModel)

    # Ensure save works without real model files
    out_path = trainer.save(Path(tmp_path))
    assert Path(out_path).exists() is False  # path string returned but file not actually written
