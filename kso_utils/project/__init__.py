"""kso_utils.project: Project configuration utilities.

This subpackage currently exposes a single helper – :func:`create_project` –
which generates a YAML project specification on disk.  The YAML file makes it
simple for downstream tooling and notebooks to discover a project's structure
and metadata in a uniform way.

The design follows the global development rules provided by the KSO codebase:
• Modular & readable code with minimal dependencies (only PyYAML and std-lib).
• Reproducible – the output is a deterministic YAML file.
• Security & error-handling – user input is validated and the function refuses
  to overwrite an existing YAML file unless explicitly requested.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

__all__ = ["create_project"]


def _validate_tracking(tracking: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate the *tracking* configuration.

    The *tracking* dictionary **must** contain a top-level key ``type`` whose
    value is either ``"wandb"`` or ``"mlflow"``.  Additional key/value pairs are
    preserved and written verbatim to the YAML output to allow flexible
    per-backend configuration.

    Parameters
    ----------
    tracking
        Dictionary describing the experiment-tracking configuration or ``None``.

    Returns
    -------
    dict | None
        The validated tracking dictionary or ``None`` if *tracking* is
        ``None``.

    Raises
    ------
    ValueError
        If the ``type`` key is missing or has an unsupported value.
    """
    if tracking is None:
        return None

    if not isinstance(tracking, dict):
        raise ValueError("'tracking' must be a dictionary if provided.")

    _type = tracking.get("type")
    if _type not in {"wandb", "mlflow"}:
        raise ValueError("'tracking.type' must be 'wandb' or 'mlflow'.")

    return tracking


def create_project(
    Project_name: str,
    project_path: str | Path = Path(__file__).resolve().parent,
    *,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    users: Optional[Dict[str, Any]] = None,
    tracking: Optional[Dict[str, Any]] = None,
    db_path: str = None,
    server: str = None,
    csv_folder: str = None,
    movie_folder: str = None,
    photo_folder: str = None,
    ml_folder: str = None,
    utils_path: str = None,
    Utils_path: str = None,
    overwrite: bool = False,
) -> Path:
    """Create a YAML file describing a KSO project.

    The resulting file is saved to ``<project_path>/<name>.project.yaml`` and
    contains the following top-level keys:

    ``name``
        The *name* parameter exactly as provided.
    ``input_data`` / ``output_data``
        Arbitrary dictionaries that minimally **should** contain a string key
        ``path`` and can optionally include an open-schema ``schema`` sub-key.
    ``users``
        Description of users – expected shape::

            {
                "username": "alice",
                "roles": ["admin", "annotator"]
            }
    ``tracking``
        Experiment-tracking backend configuration (see
        :func:`_validate_tracking`).

    Parameters
    ----------
    name
        Project name.  Must be non-empty.
    project_path
        Destination directory for the YAML file.  Defaults to current working
        directory (``'.'``).  The directory is *created* when it does not yet
        exist.
    input_data, output_data, users, tracking
        Optional dictionaries carrying project metadata.
    overwrite
        If *True*, an existing YAML file will be overwritten.  By default we
        refuse to clobber existing files to prevent accidental data loss.

    Returns
    -------
    pathlib.Path
        Absolute path to the written YAML file.

    Raises
    ------
    ValueError
        If *name* is empty or any of the optional arguments violate the
        expected schema.
    FileExistsError
        If the YAML file already exists and *overwrite* is *False*.
    """
    # Basic validation — keep it lightweight but strict enough to avoid common
    # user mistakes.
    if not Project_name or not isinstance(Project_name, str):
        raise ValueError("'Project_name' must be a non-empty string.")

    tracking = _validate_tracking(tracking)

    # Ensure the target directory exists.
    project_path = Path(project_path).expanduser().resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    yaml_path = project_path / f"{Project_name}.project.yaml"
    if yaml_path.exists() and not overwrite:
        raise FileExistsError(
            f"{yaml_path} already exists.  Pass overwrite=True to replace it."
        )

    # Assemble the YAML structure.  We keep the original order for readability.
    yaml_dict: Dict[str, Any] = {
        "Project_name": Project_name,
        "input_data": input_data,
        "output_data": output_data,
        "users": users,
        "tracking": tracking,
        "db_path": db_path,
        "server": server,
        "csv_folder": csv_folder,
        "movie_folder": movie_folder,
        "photo_folder": photo_folder,
        "ml_folder": ml_folder,
        "utils_path": utils_path,
        "Utils_path": Utils_path,
    }

    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(yaml_dict, fh, sort_keys=False, default_flow_style=False)

    logging.info(f"Project YAML created at {yaml_path}")
    return yaml_path


# update_users,
# update_input_data,
# update_output_data,
# update_tracking

# each should take in dictionary, which will overwrite the parts which are there and keep the rest as is
new_user_data = {"username": "alice", "roles": ["admin", "annotator"]}


def update_users(
    project_path: str | Path,
    new_user_data: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """
    ``users``
        Description of users – expected shape::

            {
                "username": "alice",
                "roles": ["admin", "annotator"]
            }
    """
    if not new_user_data or not isinstance(new_user_data, dict):
        raise TypeError("new_user_data must be a dictionary")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileExistsError(
            "Please provide the correct path to your project. Pass overwrite=True to replace it."
        )

    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["users"] = new_user_data
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML users data updated at {project_path}")
    return project_path


def update_input_data(
    project_path: str | Path,
    input_data: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """
    ``input_data`` Arbitrary dictionaries that minimally **should** contain a string key
    ``path`` and can optionally include an open-schema ``schema`` sub-key.
    """
    if not isinstance(input_data, dict):
        raise TypeError("input data must be a dictionary")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileNotFoundError(
            f"{project_path} not found. Pass overwrite=True to replace it."
        )
    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["input_data"] = input_data
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info("Project YAML input_data updated at {project_path}")
    return project_path


def update_output_data(
    project_path: str | Path,
    output_data: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """
    ``output_data`` Arbitrary dictionaries that minimally **should** contain a string key
    """
    if not isinstance(output_data, dict):
        raise TypeError("out data must be a dictionary")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileNotFoundError(f"{project_path} not found")
    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["output_data"] = output_data
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML output_data updated at {project_path}")
    return project_path


def update_tracking(
    project_path: str | Path,
    tracking: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """
    ``tracking``
        Experiment-tracking backend configuration (see
        :func:`_validate_tracking`).
    """
    if not isinstance(tracking, dict):
        raise TypeError("tracking must be a dictionary")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileNotFoundError(
            f"{project_path} not found. Pass overwrite=True to replace it."
        )
    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["tracking"] = tracking
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML tracking updated at {project_path}")
    return project_path


def update_db_path(
    project_path: str | Path, db_path: str = None, overwrite: bool = False
) -> Path:
    """
    this fonction update the db_path in the yaml config file.
    :param project_path path of the config file
    :param db_path databases for the projects
    """
    if not isinstance(db_path, str):
        raise TypeError("db_path must be a string")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileNotFoundError(
            f"{project_path} not found. Pass overwrite=True to replace it."
        )
    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["db_path"] = db_path
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML db_path updated at {project_path}")
    return project_path


# creat fonction to update Server
def update_server(
    project_path: str | Path, server: str = None, overwrite: bool = False
) -> Path:
    """
    this fonction update the server in the yaml config file.
    :param project_path path of the config file
    :param server which server we are using
    """
    if not isinstance(server, str):
        raise TypeError("server must be a string")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileNotFoundError(
            f"{project_path} not found. Pass overwrite=True to replace it."
        )
    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["server"] = server
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML server updated at {project_path}")
    return project_path


def update_csv_folder(
    project_path: str | Path, csv_folder: str = None, overwrite: bool = False
) -> Path:
    """
    this fonction update the csv_folder in the yaml config file.
    :param project_path path of the config file
    :param csv_folder project argument
    """
    if not isinstance(csv_folder, str):
        raise TypeError("csv_folder must be a string")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileNotFoundError(
            f"{project_path} not found. Pass overwrite=True to replace it."
        )
    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["csv_folder"] = csv_folder
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML csv_folder updated at {project_path}")
    return project_path


def update_movie_folder(
    project_path: str | Path, movie_folder: str = None, overwrite: bool = False
) -> Path:
    """
    this fonction update the movie_folder in the yaml config file.
    :param project_path path of the config file
    :param movie_folder project argument
    """
    if not isinstance(movie_folder, str):
        raise TypeError("movie_folder must be a string")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileNotFoundError(
            f"{project_path} not found. Pass overwrite=True to replace it."
        )
    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["movie_folder"] = movie_folder
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML movie_folder updated at {project_path}")
    return project_path


def update_photo_folder(
    project_path: str | Path, photo_folder: str = None, overwrite: bool = False
) -> Path:
    """
    this fonction update the photo_folder in the yaml config file.
    :param project_path path of the config file
    :param photo_folder project arguments
    """
    if not isinstance(photo_folder, str):
        raise TypeError("photo_folder must be a string")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileNotFoundError(
            f"{project_path} not found. Pass overwrite=True to replace it."
        )
    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["photo_folder"] = photo_folder
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML photo_folder updated at {project_path}")
    return project_path


def update_ml_folder(
    project_path: str | Path, ml_folder: str = None, overwrite: bool = False
) -> Path:
    """
    this fonction update the ml_folder in the yaml config file.
    :param project_path path of the config file
    :param ml_folder project arguments
    """
    if not isinstance(ml_folder, str):
        raise TypeError("ml_folder must be a string")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileNotFoundError(
            f"{project_path} not found. Pass overwrite=True to replace it."
        )
    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["ml_folder"] = ml_folder
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML ml_folder updated at {project_path}")
    return project_path


def update_utils_path(
    project_path: str | Path, utils_path: str = None, overwrite: bool = False
) -> Path:
    """
    this fonction update the utils_path in the yaml config file.
    :param project_path path of the config file
    :param utils_path utils specific to a project
    """
    if not isinstance(utils_path, str):
        raise TypeError("utils_path must be a string")
    project_path = Path(project_path).expanduser().resolve()
    if not project_path.exists() or not overwrite:
        raise FileNotFoundError(
            f"{project_path} not found. Pass overwrite=True to replace it."
        )
    with open(project_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["utils_path"] = utils_path
    with open(project_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML utils_path updated at {project_path}")
    return project_path


# TODO creat fonction for team_name (WandB)


def read_project(path_project: str | Path):
    """
    Find project information using
    project path
    """
    # tut_path = Path.cwd()
    # abspath = Path(__file__).resolve()
    # dname = abspath.parent
    # os.chdir(dname)
    project_path = Path(path_project).expanduser().resolve()
    if not project_path.exists():
        raise FileNotFoundError(f"{project_path} not found.")
    with open(project_path, "r", newline="", encoding="utf-8") as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    logging.info(f"Project YAML at {project_path}")

    return data
