# base imports
import os
from pathlib import Path
import logging
import pandas as pd
from dataclasses import dataclass
from dataclass_csv import DataclassReader, exceptions

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


@dataclass
class Project:
    # This is defining a data class called `Project` with several attributes including `Project_name`,
    # `Zooniverse_number`, `db_path`, `server`, `bucket`, `key`, `csv_folder`, `movie_folder`,
    # `photo_folder`, and `ml_folder`. The `@dataclass` decorator is used to automatically generate
    # several special methods such as `__init__`, `__repr__`, and `__eq__` for the class. This makes it
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


def find_project(
    project_name: str = "", project_csv: str = "db_starter/projects_list.csv"
):
    """Find project information using
    project csv path and project name"""
    # Specify the path to the list of projects
    tut_path = Path.cwd()
    abspath = Path(__file__).resolve()
    dname = abspath.parent
    os.chdir(dname)

    # Switch to cdn project list (temporary fix)
    if Path("/buckets").exists():
        project_csv = "db_starter/cdn_projects_list.csv"

    # Check path to the list of projects is a csv
    if Path(project_csv).exists() and not project_csv.endswith(".csv"):
        logging.error("A csv file was not selected. Please try again.")

    # If list of projects doesn't exist retrieve it from github
    elif not Path(project_csv).exists():
        if Path("/buckets").exists():
            github_path = "https://github.com/ocean-data-factory-sweden/kso_utils/blob/main/kso_utils/db_starter/cdn_projects_list.csv?raw=true"
        else:
            github_path = "https://github.com/ocean-data-factory-sweden/kso_utils/blob/main/kso_utils/db_starter/projects_list.csv?raw=true"
        read_file = pd.read_csv(github_path)
        read_file.to_csv(project_csv, index=None)

    with open(project_csv) as csv:
        reader = DataclassReader(csv, Project)
        try:
            for row in reader:
                if row.Project_name == project_name:
                    logging.info(f"{project_name} loaded succesfully")
                    os.chdir(tut_path)
                    return row
        except exceptions.CsvValueError:
            logging.error(
                f"This project {project_name} does not contain any csv information. Please select another."
            )
    os.chdir(tut_path)
    return


# def add_project(project_info: dict = {}):
#     """Add new project information to
#     project csv using a project_info dictionary
#     """
#     tut_path = os.getcwd()
#     abspath = os.path.abspath(__file__)
#     dname = os.path.dirname(abspath)
#     os.chdir(dname)
#     # Specify standard project list location
#     project_path = "db_starter/projects_list.csv"
#     # Specify volume allocated by SNIC
#     snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"

#     if not os.path.exists(project_path) and os.path.exists(snic_path):
#         project_path = os.path.join(snic_path, "db_starter/projects_list.csv")
#     with open(project_path, "a") as f:
#         project = [Project(*list(project_info.values()))]
#         w = DataclassWriter(f, project, Project)
#         w.write(skip_header=True)
#     os.chdir(tut_path)
