import importlib
import pkgutil
import pytest
import kso_utils.registries


def get_registry_modules():
    """
    A function to get all python files that are stored inside this registry folder.
    """
    package = kso_utils.registries
    prefix = package.__name__ + "."
    for _, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
        if not ispkg and not modname.split(".")[-1].startswith("test_"):
            yield importlib.import_module(modname)


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
        "choose_baseline_model",
        "choose_model",
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
