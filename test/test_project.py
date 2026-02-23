from pathlib import Path
from kso2 import create_project, Project
import pytest


def test_init_create_project():
    """
    test project config function creat_project
    """
    project_test = create_project(
        project_name="Test project 1",
        project_path="~/desktop/",
        model_name="best model",
        weights_path="yolov8n.pt",
    )
    assert isinstance(
        project_test, Project
    ), f"{project_test} is not set correctly, they should be an instance of {Project} "
