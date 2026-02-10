from pathlib import Path

import pytest
from kso_utils.MLProjectProcessor import MLProjectProcessor
import kso_utils.project_utils as project_utils

template = project_utils.find_project("Template project")


def test_init_MLProjectProcessor():
    mlp = MLProjectProcessor(template)
