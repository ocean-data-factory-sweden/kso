from pathlib import Path

import pytest
import kso_utils.project as p
import kso_utils.project_utils as project_utils

template = project_utils.find_project("Template project")


def test_init_ProjectProcessor():
    """
    The project_utils.Project has its own tests.
    connect_to_server and setup_db only call upon db_utils,
    which functions are tested in test_db_utils.
    So to initiate the ProjectProcessor, we only have the
    _map_init_csv() left that is untested, and via that one the
    _load_meta(). Since they change properties from the
    ProjectProcessor class, we test them by running the initiation
    of the ProjectProcessor
    """
    pp = p.ProjectProcessor(template)
    template_csv_paths = {
        "local_movies_csv": str(Path("db_csv_info", "movies_example.csv")),
        "local_sites_csv": str(Path("db_csv_info", "sites_example.csv")),
        "local_species_csv": str(Path("db_csv_info", "species_example.csv")),
    }
    assert (
        pp.csv_paths == template_csv_paths
    ), f"The self.csv_paths are not set correctly in _load_meta. They should be {template_csv_paths}, but got {pp.csv_paths}."
    # Also pp.local_sites_csv is set and this is the sites df itself, but I don't know what to test for that.


def test_init_MLProjectProcessor():
    mlp = p.MLProjectProcessor(template)
