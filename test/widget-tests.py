# -*- coding: utf-8 -*-
"""

Single-widget test suite. 

"""
## Import Python packages
import os
import sys

if "kso_utils" not in sys.modules:
    # for when we are running it on git, the yaml file will install the docker image. then we do need to go to the correct directory
    os.chdir("tutorials")
    sys.path.append("..")
    import kso_utils.kso_utils

    sys.modules["kso_utils"] = kso_utils.kso_utils

import kso_utils.widgets as kso_widgets
import pytest
import pandas as pd
import kso_utils.project_utils as p_utils
from kso_utils.project import ProjectProcessor, MLProjectProcessor

# -----------------Initiating of the project structure -------------------------
# Initiate project's database
project = p_utils.find_project(project_name="Template project")
pp = ProjectProcessor(project)
mlp = MLProjectProcessor(pp, test=True)
# Create a folder for temporary test output
os.makedirs("../test/test_output", exist_ok=True)


def test_choose_project():
    widget = kso_widgets.choose_project()
    assert widget.value == "Template project"


def test_gpu_select():
    widget = kso_widgets.gpu_select().children[0]
    assert widget.value == "No GPU"


def test_select_movie():
    widget = kso_widgets.select_movie(
        available_movies_df=pd.DataFrame(columns=["filename"])
    )
    assert widget.value is None


def test_choose_species():
    widget = kso_widgets.choose_species(project=project)
    assert widget.value == ("Banded weedfish",)


def test_choose_folder():
    widget = kso_widgets.choose_species(
        df=DataFrame(columns=["label"], data=[["BLUECOD"], ["SNAPPER"]])
    )
    assert widget.value == ("BLUECOD",)


def test_choose_footage():
    widget = kso_widgets.choose_footage(
        start_path="../test/test_output",
        project=project,
        server_connection=pp.server_connection,
        db_connection=pp.db_connection,
        folder_type="test_title",
    )
    assert widget.default_path == "../test/test_output"


def test_choose_agg_parameters():
    widget = kso_widgets.choose_agg_parameters(subject_type="frame")
    assert len(widget) == 5
    widget = kso_widgets.choose_agg_parameters(subject_type="clip")
    assert len(widget) == 2


def test_choose_w_version():
    widget = kso_widgets.choose_w_version(
        workflows_df=pd.DataFrame(
            columns=["display_name", "version"], data=[["test", "0.0"], ["test", "1.0"]]
        ),
        workflow_id="test",
    )
    assert widget[0].value == 0.0


def test_choose_workflows():
    widget = kso_widgets.choose_workflows(
        workflows_df=pd.DataFrame(
            columns=["display_name", "version"], data=[["test", "0.0"], ["test", "1.0"]]
        )
    )
    assert widget[0].value == "test"


def test_choose_movie_review():
    widget = kso_widgets.choose_movie_review()
    assert (
        widget.value
        == "Basic: Checks for available movies and empty cells in the movies.csv"
    )


def test_choose_new_videos_to_upload():
    widget = kso_widgets.choose_new_videos_to_upload()
    assert widget == []


def test_select_clip_length():
    widget = kso_widgets.select_clip_length()
    assert widget.value == 10


def test_select_modification():
    widget = kso_widgets.select_modification()
    assert (
        str(widget.value["filter"]).replace(" ", "")
        == ".filter('curves','0/00.396/0.671/1','0/00.525/0.4511/1','0/00.459/0.5171/1')"
    )
