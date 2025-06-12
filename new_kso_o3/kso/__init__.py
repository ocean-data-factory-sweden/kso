"""Top-level package for new_kso_o3.

Exposes high-level helpers and lazy-loads sub-modules to keep import cost low.
"""

from importlib import import_module
from types import ModuleType

__all__ = [
    "load_project_config",
    "get_storage",
    "get_trainer",
    "get_inference",
]

# Lazy proxies ---------------------------------------------------------------

def _lazy_import(name: str) -> ModuleType:
    return import_module(name, package=__name__)


def load_project_config(path: str | None = None):
    """Convenience wrapper around kso.config.load_project_config."""

    config = _lazy_import("kso.config")
    return config.load_project_config(path)


def get_storage():
    storage_mod = _lazy_import("kso.storage")
    return storage_mod.get_storage()


def get_trainer(model: str):
    registry = _lazy_import("kso.registry")
    return registry.get_trainer(model)


def get_inference(model: str):
    registry = _lazy_import("kso.registry")
    return registry.get_inference(model)
