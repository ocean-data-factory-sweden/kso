from __future__ import annotations
from pathlib import Path
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pprint
from ultralytics import settings


@dataclass
class Project:
    Project_name: str
    project_path: str | Path | None = None
    ultralytics_data: Optional[Dict[str, Any]] = None
    tracking: Optional[Dict[str, Any]] = None
    model_path: str = None
    model_name: str = None
    metadata: str = None
    Mlflow: Optional[Dict[str, Any]] = None


def create_project(
    Project_name: str,
    ultralytics_data: Optional[Dict[str, Any]] = None,
    tracking: Optional[Dict[str, Any]] = None,
    weights_path: str = None,
    model_name: str = None,
    metadata: str = None,
) -> Project:
    """Create a YAML file describing a KSO project."""
    # user mistakes.
    if not Project_name or not isinstance(Project_name, str):
        raise ValueError("'Project_name' must be a non-empty string.")

    sanitized = "".join(c.lower() if c.isalnum() else "_" for c in Project_name).strip(
        "_"
    )

    project_path = Path.cwd().parent / "projects"
    yaml_path = project_path / sanitized / f"{sanitized}.project.yaml"
    if yaml_path.exists():
        with open(yaml_path, mode="r", newline="", encoding="utf-8") as file:
            yaml_dict = yaml.load(file, Loader=yaml.SafeLoader)

        logging.info(f"{Project_name} loaded successfully")
    else:

        project = project_path / sanitized
        project.mkdir(parents=True, exist_ok=True)

        yaml_path = project / f"{sanitized}.project.yaml"
        mlflow_path = project_path / "mlflow.db"
        # Assemble the YAML structure.
        yaml_dict: Dict[str, Any] = {
            "Project_name": sanitized,
            "Project_path": str(project),
            "ultralytics_data": {"path": str(ultralytics_data)},
            "model": {"model_path": weights_path, "model_name": model_name},
            "tracking": tracking,
            "metadata": metadata,
            "Mlflow": {
                "path": None,
                "experiment_name": None,
                "mlflow.db": str(mlflow_path),
            },
        }

        with yaml_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(yaml_dict, fh, sort_keys=False, default_flow_style=False)
    runs_dir = str(project_path / sanitized / "runs")
    datasets_dir = str(project_path / sanitized)
    # print(runs_dir,datasets_dir)
    # Update multiple settings
    settings.update({"datasets_dir": datasets_dir, "runs_dir": runs_dir})

    logging.info(f"Project YAML created at {yaml_path}")
    # Convert yaml into a project instance
    project = Project(
        Project_name=yaml_dict["Project_name"],
        project_path=yaml_dict["Project_path"],
        ultralytics_data=yaml_dict["ultralytics_data"]["path"],
        tracking=yaml_dict["tracking"],
        model_path=yaml_dict["model"]["model_path"],
        model_name=yaml_dict["model"]["model_name"],
        metadata=yaml_dict["metadata"],
    )
    pprint.pp(yaml_dict)
    return project


def add_data(project_path: Project, data: str = None) -> Dict:

    if not project_path or not isinstance(project_path, Project):
        raise ValueError("'Project_path' must be a project instance.")
    if data and not isinstance(data, str):
        raise ValueError("'Ultralytics data path' must be a non-empty string.")
    if data:
        data_path = Path(data).expanduser().resolve()

    project_name = project_path.Project_name
    base_dir = Path.cwd().parent
    project = base_dir / "projects" / project_name

    yaml_path = project / f"{project_name}.project.yaml"

    if not data:
        data_path = add_ultralytics_dataset_yaml(str(project / "coco8.yaml"))

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["ultralytics_data"]["path"] = str(data_path)
    with open(yaml_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)

    logging.info(f"Project YAML data path updated at {yaml_path}")
    return pprint.pp(data)


def add_model(project_path: Project, model: str = None, model_name: str = None) -> Dict:

    if not project_path or not isinstance(project_path, Project):
        raise ValueError("'Project_path' must be a Project instance.")
    project_name = project_path.Project_name

    base_dir = Path.cwd().parent
    project = base_dir / "projects" / project_name
    yaml_path = project / f"{project_name}.project.yaml"
    if not yaml_path.exists():
        raise FileExistsError(f"{yaml_path} does not exist.")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    if model:
        if Path(model).expanduser().resolve().exists():
            model_path = Path(model).expanduser().resolve()
        else:
            model_path = project / model
        data["model"]["model_path"] = str(model_path)
    else:
        if not data[model]["model_path"]:
            raise ValueError("'model' was not provided.")

    with open(yaml_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML model weights path updated at {yaml_path}")

    if model_name:
        if not isinstance(model_name, str):
            raise ValueError("'model_name' must be a non-empty string.")

        data["model"]["model_name"] = model_name

        with open(yaml_path, "w", encoding="utf-8") as d:
            yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)

        logging.info(f"Project YAML model name updated at {yaml_path}")
    return pprint.pp(data)


def add_ultralytics_dataset_yaml(data_path: str) -> str:
    path = Path(data_path).expanduser().resolve()
    if path.exists():
        logging.info(f"Ultralytics data yaml {path} exist")
    else:
        data = {
            "path": "coco8",
            "train": "images/train",
            "val": "images/val",
            "test": "",
            "names": {
                "0": "person",
                "1": "bicycle",
                "2": "car",
                "3": "motorcycle",
                "4": "airplane",
                "5": "bus",
                "6": "train",
                "7": "truck",
                "8": "boat",
                "9": "traffic light",
                "10": "fire hydrant",
                "11": "stop sign",
                "12": "parking meter",
                "13": "bench",
                "14": "bird",
                "15": "cat",
                "16": "dog",
                "17": "horse",
                "18": "sheep",
                "19": "cow",
                "20": "elephant",
                "21": "bear",
                "22": "zebra",
                "23": "giraffe",
                "24": "backpack",
                "25": "umbrella",
                "26": "handbag",
                "27": "tie",
                "28": "suitcase",
                "29": "frisbee",
                "30": "skis",
                "31": "snowboard",
                "32": "sports ball",
                "33": "kite",
                "34": "baseball bat",
                "35": "baseball glove",
                "36": "skateboard",
                "37": "surfboard",
                "38": "tennis racket",
                "39": "bottle",
                "40": "wine glass",
                "41": "cup",
                "42": "fork",
                "43": "knife",
                "44": "spoon",
                "45": "bowl",
                "46": "banana",
                "47": "apple",
                "48": "sandwich",
                "49": "orange",
                "50": "broccoli",
                "51": "carrot",
                "52": "hot dog",
                "53": "pizza",
                "54": "donut",
                "55": "cake",
                "56": "chair",
                "57": "couch",
                "58": "potted plant",
                "59": "bed",
                "60": "dining table",
                "61": "toilet",
                "62": "tv",
                "63": "laptop",
                "64": "mouse",
                "65": "remote",
                "66": "keyboard",
                "67": "cell phone",
                "68": "microwave",
                "69": "oven",
                "70": "toaster",
                "71": "sink",
                "72": "refrigerator",
                "73": "book",
                "74": "clock",
                "75": "vase",
                "76": "scissors",
                "77": "teddy bear",
                "78": "hair drier",
                "79": "toothbrush",
            },
            "download": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
        }

        with open(path, "w", encoding="utf-8") as d:
            yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    return str(path)
