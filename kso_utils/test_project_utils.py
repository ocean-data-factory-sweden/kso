import os
import pytest
import kso_utils.project_utils as project_utils


def test_find_project():
    template = project_utils.find_project("Template project")
    assert isinstance(
        template, project_utils.Project
    ), f"The function does not return an instance of the class {project_utils.Project}, instead it returns a {type(template)}"
    assert (
        template.Project_name == "Template project"
    ), f"The Project_name is incorrect. Should be 'Template project', but got {template.Project_name}"
    assert (
        template.Zooniverse_number == 9754
    ), f"The Zooniverse_number is incorrect. Should be 9754, but got {template.Zooniverse_number}"
    assert (
        template.movie_folder == "None"
    ), f"The movie_folder is incorrect. Should be None, but got {template.movie_folder}"

    with pytest.raises(AttributeError):
        project_utils.find_project("Does not exist")
