# base imports
import os
import csv
from pathlib import Path
import logging
from dataclasses import dataclass
import pandas as pd
import yaml
# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


@dataclass
class Project:
    # This is defining a data class called `Project` with several attributes.
    # The `@dataclass` decorator is used to automatically generate several special
    # methods such as `__init__`, `__repr__`, and `__eq__` for the class. This makes it
    # easier to create and work with instances of the `Project` class.
    Project_name: str
    Zooniverse_number: int = 0
    db_path: str = None
    server: str = None
    bucket: str = None
    key: str = None
    csv_folder: str = None
    movie_folder: str = None
    photo_folder: str = None
    ml_folder: str = None
    utils_path: str = None


def get_cdn_user():
    # The code below makes sure to use the cloudina project file when working on Cloudina
    full_username = os.environ.get("USER")
    # Check if the username starts with 'jupyter-' and extract the part after it
    if full_username and full_username.startswith("jupyter-"):
        username = full_username.split("jupyter-", 1)[1]
    else:
        username = (
            None  # Handle the case where the username does not start with 'jupyter-'
        )
    cdn_user = f"/cache/album/cache/{username}"
    return cdn_user


def _get_projects_yaml_file():
    cdn_user = get_cdn_user()
    if Path(cdn_user, "bucket").exists():
        # If this path exists, we are on cloudina and use the cloudina csv
        project_csv = "kso_utils/db_starter/cdn_projects_list.csv"
    else:
        # We are not on cloudina, so use the normal csv
        # Get the directory of this utils file
        base_dir = Path(__file__).resolve().parent
        # Build path to the data file
        #project_csv = base_dir / "db_starter" / "projects_list.csv"
        project_yaml = base_dir / "db_starter" / "projects_list.yaml"

    # Check if the csv exists, otherwise retrieve it from github
    if not Path(project_yaml).exists():
        logging.info(
            f"The yaml {project_yaml} did not exist yet, so it is retrieved from gitlab."
        )
        if Path(cdn_user, "bucket").exists():
            # We are on cloudina
            github_path = "https://github.com/ocean-data-factory-sweden/kso_utils/blob/dev/kso_utils/db_starter/cdn_projects_list.csv?raw=true"
        else:
            github_path = "https://github.com/ocean-data-factory-sweden/kso_utils/blob/dev/kso_utils/db_starter/projects_list.yaml?raw=true"
        read_file = pd.read_csv(github_path)
        with open(project_yaml, "w", encoding="utf-8") as d:
                yaml.safe_dump(read_file.to_dict(orient='records'), d, sort_keys=False, default_flow_style=False)

    if not Path(project_yaml).exists():
        raise FileNotFoundError(
            f"The yaml {project_yaml} does not exist and could not be retrieved from Gitlab."
        )
    return project_yaml


def find_project(project_name: str = ""):
    """Find project information using project csv path and project name"""
    # project_csv = _get_projects_csv_file()
    project_yaml = _get_projects_yaml_file()

    cdn_user = get_cdn_user()  # is used if on cloudina, otherwise not used

    # with open(project_csv, mode="r", newline="", encoding="utf-8") as file:
    with open(project_yaml, mode="r", newline="", encoding="utf-8") as file:

        # reader = csv.DictReader(file)  # Reads rows as dictionaries
        reader = yaml.load(file,Loader=yaml.SafeLoader)  # Reads rows as dictionaries
        for row in reader:
            if row["Project_name"] == project_name:
                logging.info(f"{project_name} loaded successfully")

                if "bucket" in row.get("csv_folder"):
                    row["csv_folder"] = cdn_user + row.get("csv_folder")
                    row["movie_folder"] = cdn_user + row.get("movie_folder")

                # Convert CSV row (dict) into a Project instance
                project = Project(
                    Project_name=row["Project_name"],
                    Zooniverse_number=int(row.get("Zooniverse_number")),
                    db_path=row.get("db_path"),
                    server=row.get("server"),
                    bucket=row.get("bucket"),
                    key=row.get("key"),
                    csv_folder=row.get("csv_folder"),
                    movie_folder=row.get("movie_folder"),
                    photo_folder=row.get("photo_folder"),
                    ml_folder=row.get("ml_folder"),
                    utils_path=row.get("utils_path"),
                )
                return project

    raise AttributeError(
        f"Project {project_name} is not found in yaml {project_yaml}. Please select another project or add the information to the yaml."
    )
