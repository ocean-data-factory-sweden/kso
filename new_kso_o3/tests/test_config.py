"""Unit tests for configuration loader."""
from pathlib import Path

from kso.config import load_project_config


def _write_yaml(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_load_project_config(tmp_path):
    # Arrange: create minimal YAML hierarchy in temp dir
    _write_yaml(
        tmp_path / "project.yaml",
        """
project_name: test_project
model: pytorch
output_dir: outputs
""",
    )

    _write_yaml(
        tmp_path / "storage.yaml",
        """
local_root: ./data
""",
    )

    _write_yaml(
        tmp_path / "models" / "pytorch.yaml",
        """
arch: resnet18
num_classes: 2
""",
    )

    # Act
    cfg = load_project_config(tmp_path)

    # Assert
    assert cfg.project_name == "test_project"
    assert cfg.model == "pytorch"
    assert cfg.output_dir.name == "outputs"
    assert cfg.model_params["arch"] == "resnet18"
    assert cfg.storage.local_root == Path("./data")
