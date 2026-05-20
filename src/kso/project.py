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

settings.reset()
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
        pass

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
        biigle_path: Optional[Dict[str, Any]] = None,
        tracking: Optional[Dict[str, Any]] = None,
        weights_path: str = None,
        model_name: str = None,
        metadata: str = None,
    ) -> Project:
        """Create a YAML file describing a KSO project."""
        # user mistakes.
        if not project_name or not isinstance(project_name, str):
            raise ValueError(f"{project_name} must be a non-empty string.")

        sanitized = self.sanitized_name(project_name)

        if project_path:
            project_path = Path(project_path).expanduser()
            if not project_path.is_absolute():
                project_path = resolve_up(relative_path=project_path)
            project = project_path / sanitized
        else:
            base_dir = Path(__file__).resolve().parents[2]
            project_path = base_dir / "projects"
            project = project_path / sanitized

        """index the last model added if none is provided"""
        index = -1

        if project.exists():
            raise FileExistsError(f"the project {str(project)} already exist.")

        else:
            project.mkdir(parents=True, exist_ok=True)

            yaml_path = project / f"{sanitized}.project.yaml"
            mlflow_path = project / "mlflow.db"

            # get the relative paths
            ultralytics_relative_data = None
            biigle_relative_path = None
            weights_relative_path = None

            yaml_relative_path = make_relative_path(
                abs_path=yaml_path, startPoint=project_path
            )
            mlflow_relative_path = make_relative_path(
                abs_path=mlflow_path, startPoint=project_path
            )
            if weights_path and not Path(weights_path).is_absolute():
                weights_path = Path(weights_path).expanduser()
                weights_relative_path = project / weights_path
                weights_relative_path = make_relative_path(
                    abs_path=weights_relative_path, startPoint=project_path
                )
            else:
                weights_relative_path = weights_path

            ultralytics_data = (
                str(ultralytics_data) if ultralytics_data else ultralytics_data
            )
            biigle_path = str(biigle_path) if biigle_path else biigle_path

            # Assemble the YAML structure.
            yaml_dict: Dict[str, Any] = {
                "project_name": sanitized,
                "Config_file_path": str(yaml_relative_path),
                "data_path": {
                    "ultralytics_data_path": ultralytics_data,
                    "biigle_path": biigle_path,
                },
                "models": [
                    {"model_path": str(weights_relative_path), "model_name": model_name}
                ],
                "tracking": {
                    "mlflow": {
                        "path": None,
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
            model_path=str(weights_relative_path),
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

        if yaml_path.exists():
            yaml_dict = self.yaml_data_retrieve(yaml_path=yaml_path)

            """index the last model added if none is provided"""
            index = -1

            if model_name or model_path:
                model_paths = [m["model_path"] for m in yaml_dict["models"]]
                model_names = [m["model_name"] for m in yaml_dict["models"]]
                if model_name in model_names:
                    index = model_names.index(model_name)
                elif model_path in model_paths:
                    index = model_paths.index(model_path)

            logging.info(f"{yaml_path} loaded successfully")
        else:
            raise FileNotFoundError(f"project {yaml_path} was not found")
        """get the absolute path from the cfg yaml file"""
        # project_path=make_abs_path(relative_path=yaml_path,startPoint=project_abs_path)
        Config_file_abs_path = make_abs_path(
            relative_path=yaml_dict["Config_file_path"], startPoint=project_abs_path
        )
        mlflow_db_path = yaml_dict["tracking"]["mlflow"]["mlflow.db"]
        data_path = {
            u: (
                make_abs_path(relative_path=v, startPoint=project_abs_path)
                if v is not None
                else None
            )
            for u, v in yaml_dict["data_path"].items()
        }
        mlflow_db_abs_path = make_abs_path(
            relative_path=mlflow_db_path, startPoint=project_abs_path
        )

        model_path = yaml_dict["models"][index]["model_path"]

        if model_path and not Path(model_path).is_absolute():
            model_path = make_abs_path(
                relative_path=model_path, startPoint=project_abs_path
            )

        # Convert yaml into a project instance
        project = Project(
            project_name=yaml_dict["project_name"],
            project_path=str(project_abs_path),
            Config_file_path=Config_file_abs_path,
            data_path=data_path,
            tracking=mlflow_db_abs_path,
            model_path=model_path,
            model_name=yaml_dict["models"][index]["model_name"],
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
        data_type: str,
        data_path: str = None,
        images_root: str = None,
        dataset_dir: str | None = None,
    ):

        if not project or not isinstance(project, Project):
            raise ValueError("'Project_path' must be a project instance.")
        if data_path and not isinstance(data_path, str):
            raise ValueError("'data_path' must be a non-empty string.")
        if not data_type or not isinstance(data_type, str):
            raise ValueError("'data_type' must be a non-empty string .")
        project_path = Path(project.project_path)
        Config_file_path = project.Config_file_path
        yaml_path = Path(Config_file_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"{yaml_path} not found.")

        data = self.yaml_data_retrieve(yaml_path=yaml_path)

        if dataset_dir and not isinstance(dataset_dir, str):
            raise ValueError(f"dataset_dir must be a non empty string")
        if dataset_dir:
            dataset_dir = Path(dataset_dir).expanduser()
            if not dataset_dir.is_absolute():
                dataset_dir = resolve_up(relative_path=dataset_dir)
                print(f"dataset_dir:{dataset_dir}")
            if not dataset_dir.exists():
                raise FileNotFoundError(f"{dataset_dir} not found")
        if not dataset_dir:
            dataset_dir = project_path
            # dataset_dir.mkdir(parents=True, exist_ok=True)

        if data_type == "yolo_dataset":

            if data_path:

                data_path = Path(data_path).expanduser()
                if not data_path.is_absolute():
                    data_path = make_abs_path(
                        relative_path=data_path, startPoint=project_path
                    )
                    data_path = Path(data_path)
                # Update multiple settings
                settings.update({"datasets_dir": str(data_path.parent)})
            else:
                generated_yolo_data = resolve_up(
                    "kso/src/kso/default_dataset/coco8.yaml"
                )
                data_path = Path(generated_yolo_data).expanduser()
            if data_path.exists():
                data_relative_path = make_relative_path(
                    abs_path=data_path, startPoint=project_path
                )

            else:
                raise FileNotFoundError(f"Dataset file not found: {data_path}")
            data["data_path"]["ultralytics_data_path"] = str(data_relative_path)
            project.data_path["ultralytics_data_path"] = str(data_path)

        elif data_type == "Biigle_dataset":
            if not images_root or not isinstance(images_root, str):
                raise ValueError(f"images_root must be a non empty string")

            biigle_yaml_path = preprocess_biigle_csv(
                biigle_csv_path=data_path,
                images_root=images_root,
                dataset_dir=str(dataset_dir),
            )

            biigle_yaml_relative_path = make_relative_path(
                abs_path=biigle_yaml_path, startPoint=project_path
            )
            # data["data_path"] = {"biigle_path":str(biigle_yaml_path)}
            data["data_path"].update({"biigle_path": str(biigle_yaml_relative_path)})
            project.data_path["biigle_path"] = str(biigle_yaml_path)

        self.yaml_data_dump(yaml_path=yaml_path, data=data)
        logging.info(f"Project YAML data path updated at {yaml_path}")
        pprint.pp(data)

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
        # Get the yaml path
        if not yaml_path.exists():
            raise FileExistsError(f"{yaml_path} does not exist.")

        index = -1

        data = self.yaml_data_retrieve(yaml_path)
        model_paths = [m["model_path"] for m in data["models"]]
        if model_path and model_path.endswith(".pt"):
            candidate = Path(model_path).expanduser()
            if candidate.is_absolute():
                model_trail = make_relative_path(
                    abs_path=candidate, startPoint=project_path
                )
                """update project instance with provided model or last added model"""
                project.model_path = model_trail
            else:
                if candidate.name == str(candidate):
                    model_trail_path = (
                        project_path / project_name / candidate
                    ).resolve()
                    model_trail = make_relative_path(
                        abs_path=model_trail_path, startPoint=project_path
                    )
                else:
                    model_trail = candidate
                """update project instance with provided model or last added model"""
                project.model_path = model_trail

            """CHECK IF THE MODEL ALREADY ADDED"""
            if str(model_trail) in model_paths:
                index = model_paths.index(str(model_trail))
                """update project instance with provided model or last added model"""
                project.model_name = data["models"][index]["model_name"]
                logging.info(f"model {str(model_trail)} already exists")
            else:
                data["models"].append(
                    {"model_name": model_name, "model_path": str(model_trail)}
                )

        elif model_path and not model_path.endswith(".pt"):
            raise ValueError("model is not valid, must end with '.pt'")
        elif not model_path:
            project.model_name = data["models"][index]["model_name"]
            project.model_path = data["models"][index]["model_path"]
        self.yaml_data_dump(yaml_path=yaml_path, data=data)
        logging.info(f"Project YAML model name updated at {yaml_path}")

        pprint.pp(data)
