"""Central registry mapping model names to Trainer / Inference classes.

This allows users to request a back-end by name (e.g. "pytorch", "yolo", "hf")
and resolves it to the correct implementation without conditional imports in the
caller code.
"""
from __future__ import annotations

from typing import Callable, Dict, Type

_trainers: Dict[str, Type] = {}
_inferences: Dict[str, Type] = {}


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def register_trainer(name: str) -> Callable[[Type], Type]:
    """Decorator to register a Trainer implementation."""

    def _inner(cls: Type) -> Type:  # noqa: D401
        _trainers[name.lower()] = cls
        return cls

    return _inner


def register_inference(name: str) -> Callable[[Type], Type]:
    """Decorator to register an Inference implementation."""

    def _inner(cls: Type) -> Type:  # noqa: D401
        _inferences[name.lower()] = cls
        return cls

    return _inner


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_trainer(name: str):
    key = name.lower()
    if key not in _trainers:
        # Attempt lazy import of corresponding module
        import importlib
        try:
            importlib.import_module(f"kso.trainers.{key}")
        except ModuleNotFoundError:
            pass  # keep original error handling

    try:
        return _trainers[key]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown trainer: {name}") from exc


def get_inference(name: str):
    try:
        return _inferences[name.lower()]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown inference back-end: {name}") from exc
