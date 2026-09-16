"""Tests for ProjectManager.record and the flat project-YAML collections.

Save this file as tests/test_project_record.py in the kso repository.

It follows the existing tests/test_project.py pattern: a module-level
ProjectManager, and projects created under a relative "kso" path so that the
make_relative_path / make_abs_path chain behaves the same way it does in the
repository. Each test gets its own uniquely named project and removes it
afterwards, so the file can be run repeatedly without failing on
FileExistsError.
"""

import shutil
import uuid
from pathlib import Path

import pytest
import yaml

from kso import Project, ProjectManager

proj = ProjectManager()

COLLECTIONS = ("datasets", "models", "inferences", "analyses", "publications")
PRESERVED_KEYS = (
    "project_name",
    "Config_file_path",
    "data_path",
    "tracking",
    "metadata",
)


@pytest.fixture
def project():
    """A fresh project on disk, removed again when the test finishes."""
    created = proj.create_project(
        project_name=f"record test {uuid.uuid4().hex[:8]}",
        project_path="kso",
    )
    yield created
    shutil.rmtree(Path(created.Config_file_path).parent, ignore_errors=True)


def read_yaml(project):
    with open(project.Config_file_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(project, data):
    with open(project.Config_file_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False)


def test_create_project_writes_the_five_empty_collections(project):
    data = read_yaml(project)
    for collection in COLLECTIONS:
        assert collection in data, f"{collection} is missing from the project file"
        assert data[collection] == [], f"{collection} should start as an empty list"


def test_record_adds_an_entry(project):
    proj.record(project, "datasets", {"name": "skane", "task": "segment"})
    assert read_yaml(project)["datasets"] == [{"name": "skane", "task": "segment"}]


def test_record_replaces_by_identity_rather_than_appending(project):
    proj.record(project, "datasets", {"name": "skane", "task": "segment"})
    proj.record(project, "datasets", {"name": "skane", "task": "detect"})
    datasets = read_yaml(project)["datasets"]
    assert len(datasets) == 1, "re-recording the same name must not append"
    assert datasets[0]["task"] == "detect"


def test_record_replaces_the_complete_entry(project):
    proj.record(
        project, "datasets", {"name": "skane", "task": "segment", "note": "old"}
    )
    proj.record(project, "datasets", {"name": "skane", "task": "segment"})
    assert (
        "note" not in read_yaml(project)["datasets"][0]
    ), "replacement must be complete, not a partial merge"


def test_record_keeps_other_entries_in_the_same_collection(project):
    proj.record(project, "datasets", {"name": "skane"})
    proj.record(project, "datasets", {"name": "pascal"})
    proj.record(project, "datasets", {"name": "skane", "task": "segment"})
    names = [d["name"] for d in read_yaml(project)["datasets"]]
    assert names == ["skane", "pascal"]


def test_record_preserves_unrelated_keys(project):
    before = read_yaml(project)
    proj.record(project, "analyses", {"name": "cover summary"})
    after = read_yaml(project)
    for key in PRESERVED_KEYS:
        assert after[key] == before[key], f"{key} changed"
    assert after["models"] == []


def test_record_handles_a_legacy_file_without_the_collection(project):
    data = read_yaml(project)
    del data["inferences"]
    write_yaml(project, data)
    proj.record(project, "inferences", {"name": "run 1"})
    assert read_yaml(project)["inferences"] == [{"name": "run 1"}]


def test_record_round_trips_through_load_project(project):
    """The real loader round trip, not a stub."""
    proj.record(
        project,
        "models",
        {
            "model_name": "skane_yolo11n_seg",
            "model_path": "runs/segment/skane_yolo11n_seg/weights/best.pt",
        },
    )
    reloaded = proj.load_project(yaml_path=project.Config_file_path)
    assert isinstance(reloaded, Project)
    assert reloaded.model_name == "skane_yolo11n_seg"


def test_record_after_add_data_keeps_the_dataset_path(project, tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [a]\n", encoding="utf-8")
    proj.add_data(project, data_path=str(data_yaml))
    before = read_yaml(project)["data_path"]
    proj.record(project, "datasets", {"name": "skane"})
    assert read_yaml(project)["data_path"] == before


@pytest.mark.parametrize(
    "collection, entry",
    [
        ("deployments", {"name": "x"}),
        ("datasets", {"task": "segment"}),
        ("datasets", {"name": ""}),
        ("datasets", {"name": 3}),
        ("datasets", {}),
        ("models", {"model_name": "m"}),
        ("models", {"model_path": "p"}),
    ],
    ids=[
        "unknown collection",
        "no identifier",
        "empty identifier",
        "non-string identifier",
        "empty entry",
        "model without model_path",
        "model without model_name",
    ],
)
def test_record_rejects_invalid_input(project, collection, entry):
    with pytest.raises((ValueError, TypeError)):
        proj.record(project, collection, entry)


def test_record_rejects_an_ambiguous_duplicate(project):
    data = read_yaml(project)
    data["datasets"] = [{"name": "skane"}, {"name": "skane"}]
    write_yaml(project, data)
    with pytest.raises(ValueError):
        proj.record(project, "datasets", {"name": "skane"})


def test_record_rejects_a_collection_that_is_not_a_list(project):
    data = read_yaml(project)
    data["analyses"] = {"name": "not a list"}
    write_yaml(project, data)
    with pytest.raises(TypeError):
        proj.record(project, "analyses", {"name": "x"})


def test_a_failed_write_leaves_the_existing_file_intact(project):
    """The whole point of the atomic write."""

    class Unserialisable:
        pass

    before = Path(project.Config_file_path).read_text(encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        proj.record(project, "datasets", {"name": "skane", "bad": Unserialisable()})
    after = Path(project.Config_file_path).read_text(encoding="utf-8")
    assert after == before, "a failed save must not damage the project file"


def test_a_failed_write_leaves_no_temporary_file_behind(project):
    class Unserialisable:
        pass

    folder = Path(project.Config_file_path).parent
    with pytest.raises(yaml.YAMLError):
        proj.record(project, "datasets", {"name": "skane", "bad": Unserialisable()})
    leftovers = list(folder.glob("*.tmp"))
    assert leftovers == [], f"temporary files left behind: {leftovers}"
