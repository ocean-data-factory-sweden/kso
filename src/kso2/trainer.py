from __future__ import annotations
from pathlib import Path
import logging
from typing import Any, Dict, Optional, List, Union
import yaml
from dataclasses import asdict
import os
import sys
from ultralytics import YOLO
import re
from .project import Project
import psutil

from mlflow_export_import.bulk.export_experiments import export_experiments
from mlflow_export_import.bulk.import_experiments import import_experiments


import json
import pandas as pd
import shutil
from collections import defaultdict
import random
from PIL import Image
from mlflow.pyfunc import PythonModel
import numpy as np
from ultralytics import settings
import mlflow

# Logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YOLOv11MLflowModel(PythonModel):
    def __init__(self):
        # Don't keep any state in the constructor
        super().__init__()

    def load_context(self, context):
        # Get the model path from artifacts
        model_path = context.artifacts["weights"]

        logger.info(f"Loading YOLO weights from: {model_path}")
        logger.info(f"Context: {context}")

        self.model = YOLO(model_path)

    def predict(
        self,
        model_input: List[Union[pd.DataFrame, np.ndarray, List[Any], Dict[str, Any]]],
    ):
        # Import libraries when needed to avoid serialization issues
        image = model_input.get("image")
        if isinstance(image, list):
            image = np.array(image, dtype=np.uint8)

        # Run prediction
        results = self.model.predict(image)

        # Convert to JSON string
        output = []
        for result in results:
            result_dict = {
                "boxes": (
                    result.boxes.xyxy.cpu().numpy().tolist()
                    if result.boxes is not None
                    else []
                ),
                "scores": (
                    result.boxes.conf.cpu().numpy().tolist()
                    if result.boxes is not None
                    else []
                ),
                "classes": (
                    result.boxes.cls.cpu().numpy().astype(int).tolist()
                    if result.boxes is not None
                    else []
                ),
                "names": result.names,
                "shape": list(result.orig_shape),
            }
            output.append(result_dict)
        logger.info(output)
        # Return as JSON string
        return output


def set_group_writable_umask(mask=0o002):
    old_umask = os.umask(mask)
    logging.info(f"Changed umask from {oct(old_umask)} to {oct(mask)}")


def training_model(
    project: Project, epochs: int = 100, imgsz: int = 640, change_umask=False
):
    if change_umask:
        set_group_writable_umask()

    project_name = project.Project_name

    base_dir = Path(__file__).resolve().parents[2]
    project_path = base_dir / "projects" / project_name
    yaml_path = project_path / f"{project_name}.project.yaml"

    if not yaml_path.exists():
        raise FileExistsError(f"{yaml_path} does not exist.")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data_path = data["data_path"]["Biigle_path"]
    # data_path = data["data_path"]["ultralytics_data_path"]

    if not isinstance(project, Project):
        raise ValueError("'model' must be a Project instance.")

    model_source = data["model"]["model_path"] or project.model_name
    if not model_source or not isinstance(model_source, str):
        raise ValueError("'model' must be a non-empty string.")

    artifact_root = (
        Path(__file__).resolve().parents[2] / "projects" / project_name / "mlruns"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflowdb_path = Path(__file__).resolve().parents[2] / "projects" / "mlflow.db"
    experiment_name = project_name

    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{mlflowdb_path}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    os.environ["MLFLOW_ARTIFACT_URI"] = artifact_root.as_uri()

    # Check if experiment exists
    mlflow.set_tracking_uri(f"sqlite:///{mlflowdb_path}")
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # create the experiment with a project location
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_root.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    model = YOLO(model_source)

    def dict_mapper(yolo_result):

        return re.sub(r"[(B)]", "", yolo_result)

    with mlflow.start_run():
        results = model.train(data=data_path, epochs=epochs, imgsz=imgsz)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metrics({dict_mapper(k): v for k, v in results.results_dict.items()})

        best_weight_path = Path(results.save_dir) / "weights" / "best.pt"

        mlflow.pyfunc.log_model(
            name="model",
            python_model=YOLOv11MLflowModel(),
            artifacts={"weights": str(best_weight_path)},
        )

    mlflow.end_run()

    data["Mlflow"] = {
        "path": str(mlflowdb_path),
        "experiment_name": project_name,
        "mlflow.db": str(mlflowdb_path),
    }

    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)

    # Examine the deleted experiment details.
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Name: {experiment.name}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Last Updated timestamp: {experiment.last_update_time}")
    return results


def export_experiment(
    project: Project, port=8080, notebook_formats=None, use_threads=False
):

    project_name = project.Project_name
    experiment_name = project_name
    # 1) Tracking URI points to your local mlruns folder
    mlflow.set_tracking_uri(f"http://localhost:{port}")
    # Check if experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    artifact_root = (
        Path(__file__).resolve().parents[2] / "projects" / project_name / "output"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = mlflow.MlflowClient()

    # 2) Export experiment(s)
    export_experiments(
        mlflow_client=client,
        experiments=[experiment.experiment_id],
        output_dir=str(artifact_root),
        notebook_formats=None,
        use_threads=False,
    )


def import_experiment(project: Project, input_dir: str, port: int = 8080):

    project_name = project.Project_name
    experiment_name = project_name
    # 1) Tracking URI points to your local mlruns folder
    mlflow.set_tracking_uri(f"http://localhost:{port}")

    client = mlflow.MlflowClient()

    import_experiments(
        mlflow_client=client,
        input_dir=input_dir,
        use_src_user_id=False,
        use_threads=False,
    )
