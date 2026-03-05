from pathlib import Path
from kso import Project, TrainingManager
import pytest

proj = TrainingManager()


def test_init_create_project():
    """
    test project config function creat_project
    """
    project_test = proj.create_project(
        project_name="Test project 1",
        project_path="kso",
        model_name="best model",
        weights_path="yolov8n.pt",
    )
    assert isinstance(
        project_test, Project
    ), f"{project_test} is not set correctly, they should be an instance of {Project} "


def test_load_project():
    """
    test the loading of an existing project using the function creat_project
    """

    loaded_project = proj.load_project(
        yaml_path="kso/test_project_1/test_project_1.project.yaml"
    )
    assert isinstance(
        loaded_project, Project
    ), f"{loaded_project} is not set correctly, they should be an instance of {Project} "
