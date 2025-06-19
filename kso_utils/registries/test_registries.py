def validate_registry(utils, required_functions):
    """
    We want the code to be more modular, so that we can swap out different model training packages,
    or training tracking services for each other. To have all these modular components, all of these will
    have their own utils file, which needs to contain a specified set of functions.
    This function will check a utils file if it contains all the required functions.
    """
    # Validate they exist in the module
    missing = [func for func in required_functions if not hasattr(utils, func)]
    if missing:
        raise ImportError(
            f"Module 'kso_utils.{utils}_utils' is missing required functions: {missing}"
        )


validate_registry(
    self.registry_utils,
    [
        "init",
        "start_run",
        "close_run",
        "choose_baseline_model",
        "choose_model",
        "get_model",
        "get_dataset",
    ],
)
