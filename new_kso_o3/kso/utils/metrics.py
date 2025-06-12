"""Metrics & Experiment Tracking Abstraction.

Provides unified API with pluggable back-ends: MLflow or Weights & Biases.
Detection of active back-end is based on environment variables:
- `KSO_TRACKING=mlflow`  -> use MLflow
- `KSO_TRACKING=wandb`   -> use Weights & Biases
If unset, falls back to a no-op logger.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict


class _NoopTracker:  # noqa: D101
    def log_params(self, *_args: Any, **_kwargs: Any):
        pass

    def log_metrics(self, *_args: Any, **_kwargs: Any):
        pass

    def log_artifact(self, *_args: Any, **_kwargs: Any):
        pass

    @contextmanager
    def start_run(self, *_args: Any, **_kwargs: Any):  # noqa: D401
        yield self


class _MLflowTracker:  # noqa: D101
    def __init__(self):
        import mlflow

        self.mlflow = mlflow

    # Proxy common methods
    def __getattr__(self, item):  # noqa: D401
        return getattr(self.mlflow, item)


class _WandBTracker:  # noqa: D101
    def __init__(self):
        import wandb

        self.wandb = wandb

    def log_params(self, params: Dict[str, Any]):
        self.wandb.config.update(params)

    def log_metrics(self, metrics: Dict[str, Any]):
        self.wandb.log(metrics)

    def log_artifact(self, path: str, name: str):
        self.wandb.save(path, name=name)

    @contextmanager
    def start_run(self, *_args: Any, **_kwargs: Any):  # noqa: D401
        run = self.wandb.init()
        try:
            yield run
        finally:
            run.finish()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_backend = os.getenv("KSO_TRACKING", "none").lower()

if _backend == "mlflow":
    tracker: Any = _MLflowTracker()
elif _backend == "wandb":
    tracker = _WandBTracker()
else:
    tracker = _NoopTracker()


# Re-export common methods ---------------------------------------------------
log_params = tracker.log_params  # type: ignore[attr-defined]
log_metrics = tracker.log_metrics  # type: ignore[attr-defined]
log_artifact = tracker.log_artifact  # type: ignore[attr-defined]
start_run = tracker.start_run  # type: ignore[attr-defined]
