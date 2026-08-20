from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Any, Optional
from .data_preprocessing import (
    preprocess_biigle_csv,
    resolve_up,
    make_abs_path,
    make_relative_path,
)
import yaml
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pprint
from ultralytics import settings
from PIL import Image
import random
import pandas as pd
from collections import defaultdict
import json
import shutil

# settings.reset()
os.environ["MLFLOW_TRACKING_URI"] = "http://your-server:5000"
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class Project:
    project_name: str
    project_path: str | Path | None = None
    Config_file_path: str | Path | None = None
    data_path: Optional[Dict[str, Any]] = None
    tracking: Optional[Dict[str, Any]] = None
    model_path: str = None
    model_name: str = None
    metadata: str = None
    # Mlflow: Optional[Dict[str, Any]] = None


class ProjectManager:

    def __init__(self):
        self.lumi = self.is_lumi()

    def sanitized_name(self, project_name: str):
        sanitized = "".join(
            c.lower() if c.isalnum() else "_" for c in project_name
        ).strip("_")
        return sanitized

    def create_project(
        self,
        project_name: str,
        project_path: str | Path = None,
        ultralytics_data: Optional[Dict[str, Any]] = None,
        tracking: Optional[Dict[str, Any]] = None,
        weights_path: str = None,
        model_name: str = None,
        metadata: str = None,
    ) -> Project:
        """Create a YAML file describing a KSO project."""
        settings.reset()
        # user mistakes.
        if not project_name or not isinstance(project_name, str):
            raise ValueError(f"{project_name} must be a non-empty string.")

        sanitized = self.sanitized_name(project_name)
        base_dir = Path(__file__).resolve().parents[3]
        if project_path:
            project_path = Path(project_path).expanduser()
            if not project_path.is_absolute():
                project_path = resolve_up(relative_path=project_path)
            project = project_path / sanitized
        else:
            project_path = base_dir / "projects"
            project = project_path / sanitized
            # print(f"project:{project} and project_path : {project_path}")
            # print(f"base_dir : {base_dir}")

        if project.exists():
            raise FileExistsError(f"the project {str(project)} already exist.")

        else:
            project.mkdir(parents=True, exist_ok=True)

            yaml_path = project / f"{sanitized}.project.yaml"
            mlflow_path = project / "mlflow.db"

            mlflow_relative_path = make_relative_path(abs_path=mlflow_path)

            yaml_path_relative = make_relative_path(abs_path=yaml_path)
            if weights_path:
                weights_path = Path(weights_path).expanduser()
                if not weights_path.is_absolute():
                    if weights_path.name == str(weights_path):
                        weights_path = base_dir / "models" / weights_path
                    weights_path = resolve_up(relative_path=weights_path)

            if ultralytics_data:

                ultralytics_data = Path(ultralytics_data).expanduser()
                if not ultralytics_data.is_absolute():
                    ultralytics_data = resolve_up(relative_path=ultralytics_data)

            # Assemble the YAML structure.
            yaml_dict: Dict[str, Any] = {
                "project_name": sanitized,
                "Config_file_path": str(yaml_path_relative),
                "data_path": {
                    "ultralytics_data_path": str(ultralytics_data),
                },
                "models": [],
                "tracking": {
                    "mlflow": {
                        "experiment_name": None,
                        "mlflow.db": str(mlflow_relative_path),
                    },
                },
                "metadata": metadata,
            }
            yaml_dict = self.yaml_data_dump(yaml_path, yaml_dict)

        runs_dir = str(project / "runs")
        datasets_dir = str(project)
        # Update multiple settings
        settings.update({"datasets_dir": datasets_dir, "runs_dir": runs_dir})

        logging.info(f"Project YAML created at {str(project)}")
        # Convert yaml into a project instance
        project = Project(
            project_name=sanitized,
            project_path=str(project_path),
            Config_file_path=str(yaml_path),
            data_path=yaml_dict["data_path"],
            tracking=str(mlflow_path),
            model_path=str(weights_path),
            model_name=model_name,
        )
        pprint.pp(yaml_dict)
        return project

    def load_project(
        self,
        yaml_path: str | Path,
        model_name: str = None,
        model_path: str = None,
    ):
        """load an existing project"""
        if not yaml_path or not isinstance(yaml_path, (str, Path)):
            raise ValueError(f"{yaml_path} must be non-empty string or Path")
        if model_name and not isinstance(model_name, str):
            raise ValueError("'model_name' must be a non-empty string.")
        if model_path and not isinstance(model_path, str):
            raise ValueError("'model_path' must be a non-empty string.")

        yaml_path = Path(yaml_path).expanduser()

        if not yaml_path.is_absolute():
            yaml_path = resolve_up(relative_path=yaml_path)
        project_abs_path = yaml_path.parents[1]  # project cfg file path
        yaml_dict = self.yaml_data_retrieve(yaml_path=yaml_path)

        """index the last model added if none is provided"""
        index = -1
        if yaml_dict["models"]:

            if model_name or model_path:
                model_paths = [m["model_path"] for m in yaml_dict["models"]]
                model_names = [m["model_name"] for m in yaml_dict["models"]]
                if model_name in model_names:
                    index = model_names.index(model_name)
                elif model_path in model_paths:
                    index = model_paths.index(model_path)

            model_path = yaml_dict["models"][index]["model_path"]
            model_name = yaml_dict["models"][index]["model_name"]
        else:
            model_path = None
            model_name = None
        logging.info(f"{yaml_path} loaded successfully")

        """get the absolute path from the cfg yaml file"""

        mlflow_db_path = yaml_dict["tracking"]["mlflow"]["mlflow.db"]
        data_path = yaml_dict["data_path"]

        mlflow_abs_path = make_abs_path(relative_path=mlflow_db_path)

        # Convert yaml into a project instance
        project = Project(
            project_name=yaml_dict["project_name"],
            project_path=str(project_abs_path),
            Config_file_path=str(yaml_path),
            data_path=data_path,
            tracking=mlflow_abs_path,
            model_path=model_path,
            model_name=model_name,
        )
        pprint.pp(yaml_dict)
        return project

    def yaml_data_retrieve(self, yaml_path: str | Path, data: str = None):
        """
        retreive data from the yaml config file
        if a data column was provided retreive it else return all the data
        """
        if not yaml_path or not isinstance(yaml_path, (str, Path)):
            raise TypeError(f"{yaml_path} has to be a non empty string")
        yaml_path = Path(yaml_path).expanduser()
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
        if data:
            data = yaml_data.get(data)
            logging.info("data was rtreived successfully")
            return data
        elif data and not isinstance(data, str):
            raise TypeError(f"{data} should be a non-empty string")
        else:
            logging.info("data was retreived successfully")
            return yaml_data

    def yaml_data_dump(self, yaml_path: str | Path, data: str = None):
        """
        dump data to the yaml config file
        """
        if not yaml_path or not isinstance(yaml_path, (str, Path)):
            raise TypeError(f"{yaml_path} has to be a non empty string")
        if not data or not isinstance(data, Dict):
            raise TypeError(f"{data} has to be a non empty Dictionary")

        yaml_path = Path(yaml_path).expanduser()
        with open(yaml_path, "w", encoding="utf-8") as d:
            yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
        logging.info("data was dumped successfully")
        return data

    def add_data(
        self,
        project: Project,
        data_path: str = None,
    ):

        if not project or not isinstance(project, Project):
            raise ValueError("'Project_path' must be a project instance.")
        if data_path and not isinstance(data_path, str):
            raise ValueError("'data_path' must be a non-empty string.")

        Config_file_path = project.Config_file_path
        yaml_path = Path(Config_file_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"{yaml_path} not found.")

        data = self.yaml_data_retrieve(yaml_path=yaml_path)

        if data_path:
            data_path = Path(data_path).expanduser()
            if not data_path.is_absolute():
                data_path = resolve_up(relative_path=data_path)

            settings.update({"datasets_dir": str(data_path.parent)})
        else:
            generated_yolo_data = resolve_up("kso/src/kso/default_dataset/coco8.yaml")
            data_path = Path(generated_yolo_data).expanduser()

        data["data_path"]["ultralytics_data_path"] = str(data_path)
        project.data_path["ultralytics_data_path"] = str(data_path)

        self.yaml_data_dump(yaml_path=yaml_path, data=data)
        logging.info(f"Project YAML data path updated at {yaml_path}")
        pprint.pp(data)

    def is_lumi(self):
        return os.environ.get("SLURM_CLUSTER_NAME") == "lumi"

    def home_path_synthesizer(self):
        if self.is_lumi():
            self.home_path = (
                Path("/scratch") / os.environ["PROJECT"] / os.environ["USER"]
            )
        else:
            self.home_path = Path(__file__).resolve().parents[3]
        return self.home_path

    def preprocess_Biigle(self, images_root, data_path, dataset_dir=None):

        if not images_root or not isinstance(images_root, str):
            raise ValueError(f"{images_root} must be a non empty string")
        if not data_path or not isinstance(data_path, str):
            raise ValueError(f"{data_path} must be a non empty string")
        if not dataset_dir:
            home_path = self.home_path_synthesizer()
            dataset_dir = home_path / "datasets"

            idx = 0
            while True:
                suffix = "" if idx == 0 else f"_{idx}"
                new_dir = dataset_dir / f"ifremer_sled_2026{suffix}"
                if not new_dir.exists():
                    new_dir.mkdir(parents=True)
                    break
                idx += 1
        else:
            new_dir = Path(dataset_dir).expanduser()

        biigle_yaml_path = preprocess_biigle_csv(
            biigle_csv_path=data_path,
            images_root=images_root,
            dataset_dir=str(new_dir),
        )
        return biigle_yaml_path

    def add_model(
        self, project: Project, model_path: str = None, model_name: str = None
    ):
        """
        Update the project's YAML with a model path and/or model name.

        Rules for `model`:
        - Absolute path ending with '.pt': accepted if it exists.
        """
        if not project or not isinstance(project, Project):
            raise ValueError("'Project_path' must be a Project instance.")
        if not isinstance(model_path, str):
            raise ValueError("'model' must be non-empty string")
        if not isinstance(model_name, str):
            raise ValueError("'model_name' must be a non-empty string")
        project_name = project.project_name
        yaml_path = Path(project.Config_file_path)
        project_path = Path(project.project_path)
        home_dir = self.home_path_synthesizer()
        models_dir = home_dir / "models"
        # print(f"models_dir:{models_dir}")
        os.makedirs(models_dir, exist_ok=True)
        # Get the yaml path
        if not yaml_path.exists():
            raise FileExistsError(f"{yaml_path} does not exist.")

        index = -1

        data = self.yaml_data_retrieve(yaml_path)
        model_paths = [m["model_path"] for m in data["models"]]
        if model_path and model_path.endswith(".pt"):
            candidate = Path(model_path).expanduser()

            if not candidate.is_absolute():
                if candidate.name == str(candidate):
                    model_trail = (models_dir / candidate).resolve()
                else:
                    model_trail = (home_dir / candidate).resolve()
            else:
                model_trail = candidate
            """update project instance with provided model or last added model"""
            project.model_path = str(model_trail)

            """CHECK IF THE MODEL ALREADY ADDED"""
            if str(model_trail) in model_paths:
                index = model_paths.index(str(model_trail))
                """update project instance with provided model or last added model"""
                project.model_name = data["models"][index]["model_name"]
                logging.info(f"model {str(model_trail)} already exists")
            else:
                # filter None values out before appending or before saving to config
                data["models"] = [
                    model
                    for model in data["models"]
                    if model["model_name"] is not None
                    and model["model_path"] is not None
                ]
                data["models"].append(
                    {"model_name": model_name, "model_path": str(model_trail)}
                )
                """update project instance with provided model name or last added model name"""
                project.model_name = model_name

        elif model_path and not model_path.endswith(".pt"):
            raise ValueError("model is not valid, must end with '.pt'")
        elif not model_path:
            project.model_name = data["models"][index]["model_name"]
            project.model_path = data["models"][index]["model_path"]
        self.yaml_data_dump(yaml_path=yaml_path, data=data)
        logging.info(f"Project YAML model name updated at {yaml_path}")

        pprint.pp(data)
