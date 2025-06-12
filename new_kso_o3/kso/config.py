"""Unified configuration loader/validator for new_kso_o3.

Merges `configs/project.yaml`, `configs/storage.yaml`, and a model-specific
`configs/models/<model>.yaml` into a single `ProjectConfig` dataclass.
"""
from __future__ import annotations

from pathlib import Path
import os
import yaml
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict

CONFIG_DIR = Path(
    os.getenv("KSO_CONFIG_DIR", Path(__file__).resolve().parent.parent / "configs")
)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class StorageRemote(BaseModel):
    provider: str = Field(..., description="e.g., s3")
    endpoint: str
    bucket: str
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    region: str | None = None


class StorageConfig(BaseModel):
    local_root: Path = Field(Path("./data"))
    remote: StorageRemote | None = None


class ProjectConfig(BaseModel):
    project_name: str = "new_kso_o3_demo"
    model: str = "pytorch"
    output_dir: Path = Path("outputs")
    storage: StorageConfig
    model_params: Dict[str, Any]

    @field_validator("output_dir", mode="before")
    def _expand_output(cls, v):  # noqa: N805
        return Path(v)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_project_config(path: str | Path | None = None) -> ProjectConfig:
    """Load & merge YAML configs, optionally from custom directory."""
    base_dir = Path(path).expanduser().resolve() if path else CONFIG_DIR

    project_cfg = _load_yaml(base_dir / "project.yaml")
    storage_cfg = _load_yaml(base_dir / "storage.yaml")

    model_name = project_cfg.get("model", "pytorch")
    model_cfg = _load_yaml(base_dir / "models" / f"{model_name}.yaml")

    merged = {
        **project_cfg,
        "storage": storage_cfg,
        "model_params": model_cfg,
    }
    return ProjectConfig.model_validate(merged)
