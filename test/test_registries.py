import importlib
import pathlib
import pkgutil
import pytest
import kso_utils.registries
from pathlib import Path

import kso_utils.project_utils as p_utils
from kso_utils.MLProjectProcessor import MLProjectProcessor


def get_registry_modules():
    """
    A function to get all python files that are stored inside this registry folder.
    """
    package = kso_utils.registries
    prefix = package.__name__ + "."
    for _, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
        if not ispkg and not modname.split(".")[-1].startswith("test_"):
            yield importlib.import_module(modname)


@pytest.fixture
def mlp(needs_wandb):
    project_name = "Template project"
    project = p_utils.find_project(project_name=project_name)
    # Initialise mlp
    mlp = MLProjectProcessor(project)

    yield mlp

    mlp.db_connection.close()
    Path(mlp.project.db_path).unlink()


@pytest.mark.parametrize("registry_module", list(get_registry_modules()))
def test_validate_registry(registry_module):
    """
    We want the code to be more modular, so that we can swap out different model training packages,
    or training tracking services for each other. To have all these modular components, all of these will
    have their own utils file, which needs to contain a specified set of functions.
    This function will check a utils file if it contains all the required functions.
    """
    required_functions = [
        "init",
        "start_run",
        "close_run",
        "show_available_models",
        "get_model",
        "get_dataset",
    ]
    # Validate they exist in the module
    missing = [
        func for func in required_functions if not hasattr(registry_module, func)
    ]
    if missing:
        raise ImportError(
            f"Module '{registry_module}' is missing required functions: {missing}"
        )


@pytest.mark.parametrize("registry_module", list(get_registry_modules()))
def test_get_model(registry_module, mlp):
    model_download_dir = (
        "."  # WandB allows this to be a nonexisting path and creates it automatically
    )

    # Test if it runs, returns a string and if the file is downloaded
    model_name = "yolov8m-base-model"
    model_path = registry_module.get_model(
        mlp, model_name, model_download_dir, baseline=True
    )
    assert isinstance(
        model_path, str
    ), f"The get_model function should return a string with the path on the local computer where the model is downloaded too. But got a {type(model_path)}."
    assert pathlib.Path(
        model_path
    ).is_file(), (
        f"The model is not downloaded, since the file at {model_path} does not exist"
    )

    # Test if an error is returned for the wrong model name
    with pytest.raises(AttributeError, match="Error when trying to retrieve"):
        registry_module.get_model(
            mlp, "not_existing", model_download_dir, baseline=True
        )


@pytest.mark.parametrize("registry_module", list(get_registry_modules()))
def test_show_available_models(registry_module, mlp):
    # Test if it runs, returns a string and if the file is downloaded
    models = registry_module.show_available_models(mlp, baseline=True)
    assert isinstance(
        models, list
    ), f"The show_available_models function should return a list of the available models. But got a {type(models)}."


@pytest.mark.parametrize("registry_module", list(get_registry_modules()))
def test_get_dataset(registry_module, mlp):
    # Test if it runs, returns a tuple with strings and if the file is downloaded
    # The model name used here is just a run in the template project
    dirs_data = registry_module.get_dataset(mlp, "run_cdjzvb7h_model")
    assert (
        isinstance(dirs_data, tuple)
        and isinstance(dirs_data[0], str)
        and isinstance(dirs_data[1], str)
    ), f"The get_dataset function should return a tuple with two strings. But got a {type(dirs_data)} containing a {type(dirs_data[0])} and {type(dirs_data[1])}."
    # TODO: this needs an example for a model where it actually has data logged on WandB.
    # Right now it does not test anything yet.

    # Test if it returns empty strings for externally trained models
    dirs_data = registry_module.get_dataset(mlp, "yolov8m-base-model")
    assert (
        isinstance(dirs_data, tuple)
        and isinstance(dirs_data[0], str)
        and isinstance(dirs_data[1], str)
    ), f"The get_dataset function should return a tuple with two strings. But got a {type(dirs_data)} containing a {type(dirs_data[0])} and {type(dirs_data[1])}."
    assert (
        dirs_data[0] == "" and dirs_data[1] == ""
    ), f"Externally trained models do not contain any data and should return an empty string, instead got {dirs_data}"
