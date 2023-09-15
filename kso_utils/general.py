# base imports
import logging
import multiprocessing

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def import_model_modules(module_names):
    """
    This function imports specified modules and returns them as a dictionary.

    :param module_names: A list of strings representing the names of the modules to be imported. The
    order of the names in the list should correspond to the order of the modules in the returned
    dictionary. In this case, the expected order is ["train", "detect", "val"]
    :return: a dictionary containing the imported modules with keys "train", "detect", and "val". If a
    module fails to import, an error message is logged and that module is not included in the returned
    dictionary.
    """
    importlib = __import__("importlib")
    modules = {}
    for module_name, module_full in zip(["train", "detect", "val"], module_names):
        try:
            modules[module_name] = importlib.import_module(module_full)
        except ModuleNotFoundError:
            logging.error(f"Module {module_name} could not be imported.")
    return modules


def import_modules(module_names, utils: bool = True, models: bool = False):
    """
    The function imports specified modules and returns them as a dictionary.

    :param module_names: A list of strings representing the names of the modules to be imported
    :param utils: A boolean parameter that specifies whether the module names provided are for utility
    modules located in the "kso_utils" package or not. If True, the function will prepend "kso_utils."
    to each module name before attempting to import it. If False, the function will attempt to import
    the module directly, defaults to True
    :type utils: bool (optional)
    :param models: A boolean parameter that determines whether the imported modules are related to
    models or not. If it is set to True, the function will assume that the module names correspond to
    the three model presets: "train", "detect", and "val". If it is set to False, the function will
    import the, defaults to False
    :type models: bool (optional)
    :return: a dictionary of imported modules, where the keys are the names of the modules and the
    values are the imported module objects.
    """
    importlib = __import__("importlib")
    modules = {}
    model_presets = ["train", "detect", "val"]
    for i, module_name in enumerate(module_names):
        if utils:
            module_full = "kso_utils." + module_name
        else:
            module_full = module_name
        try:
            if models:
                module_name = model_presets[i]
            modules[module_name] = importlib.import_module(module_full)
        except ModuleNotFoundError:
            logging.error(f"Module {module_name} could not be imported.")
    return modules


def parallel_map(func, iterable, args=()):
    """
    The function `parallel_map` uses multiprocessing to apply a given function to each element of an
    iterable in parallel.

    :param func: The function to be applied to each element of the iterable
    :param iterable: The iterable is a sequence of elements that can be iterated over, such as a list,
    tuple, or range object. The function `func` will be applied to each element of the iterable in
    parallel using multiple processes
    :param args: args is a tuple of additional arguments that can be passed to the function being mapped
    in parallel. These arguments will be unpacked and passed to the function along with the
    corresponding element from the iterable. If no additional arguments are needed, the default value of
    an empty tuple can be used
    :return: The function `parallel_map` returns a list of results obtained by applying the function
    `func` to each element of the `iterable` in parallel using multiple processes. The `args` parameter
    is optional and can be used to pass additional arguments to the function `func`.
    """
    with multiprocessing.Pool() as pool:
        results = pool.starmap(func, zip(iterable, *args))
    return results
