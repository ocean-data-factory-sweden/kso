from pathlib import Path
from kso2 import create_project, Project, load_project
import pytest


def test_init_create_project():
    """
    test project config function creat_project
    """
    project_test = create_project(
        project_name="Test project 1",
        project_path="kso/proj",
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

    loaded_project = load_project(
        yaml_path="kso/proj/test_project_1/test_project_1.project.yaml"
    )
    assert isinstance(
        loaded_project, Project
    ), f"{loaded_project} is not set correctly, they should be an instance of {Project} "
