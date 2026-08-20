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
from .project import Project, ProjectManager
from .data_preprocessing import resolve_up
import psutil
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature
from mlflow_export_import.bulk.export_experiments import export_experiments
from mlflow_export_import.bulk.import_experiments import import_experiments


import json
import pandas as pd
import shutil
from collections import defaultdict
import random
from PIL import Image
from mlflow.pyfunc import PythonModel, PyFuncModel
import numpy as np
from ultralytics import settings
import mlflow
import torch
import torch.nn as nn


# Logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("mlflow.store").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)
logging.getLogger("git.cmd").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


proj = ProjectManager()


class YOLOUltralyticsMLflowModel(PythonModel):
    def __init__(self):
        # Don't keep any state in the constructor
        super().__init__()

    def load_context(self, context):
        # Get the model path from artifacts
        model_path = context.artifacts["weights"]

        logger.info(f"Loading YOLO weights from: {model_path}")
        # logger.info(f"Context: {context}")

        self.yolo_model = YOLO(model_path)

    def nn_model(self):
        """get the pytorch model from mlflow.pyfunc"""
        return self.yolo_model.model

    def predict(
        self,
        context,
        model_input: List[Union[pd.DataFrame, np.ndarray, List[Any], Dict[str, Any]]],
        params: dict[str, Any] | None = None,
    ):
        params = (
            dict(params) if params else {}
        )  # if params is None it defaults to empty dict and stride default to 1.
        vid_stride = params.pop("vid_stride", 1)
        output = []
        for item in model_input:

            image = item.get("image")

            if isinstance(image, list) and all(isinstance(p, str) for p in image):
                for p in image:
                    if not Path(p).exists():
                        raise ValueError(f"Path does not exist: {p}")

            elif isinstance(image, str):
                if not Path(image).exists():
                    raise FileNotFoundError(f"the provided file {image} was not found")

            # Run prediction

            results = self.yolo_model.predict(image, vid_stride=vid_stride, **params)

            # logger.info(f"predict vid_stride: {vid_stride}")
            # logger.info(f"predict results: {len(results)}")
            # logger.info(f"predict paramter: {params}")

            # Convert to JSON string

            for idx, result in enumerate(results):
                result_dict = {
                    "frame_number": idx * vid_stride,
                    "file_name": result.path or "",
                    "boxes": (
                        result.boxes.xyxy.cpu().numpy().tolist()
                        if result.boxes is not None and result.boxes.xyxy is not None
                        else []
                    ),
                    "plot": (result.plot() if result.plot() is not None else []),
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


class TrainingManager:
    def __init__(self):
        pass

    def train_model(
        self,
        project: Project,
        epochs: int = 100,
        imgsz: int = 640,
        change_umask=False,
        **kwargs,
    ):
        if not isinstance(project, Project):
            raise ValueError("'model' must be a Project instance.")

        if change_umask:
            set_group_writable_umask()

        # ensure Ultralytics MLflow Callbacks are False
        settings.update({"mlflow": False})

        project_name = project.project_name
        model_name = project.model_name
        model_source = project.model_path
        project_path = Path(project.project_path)
        yaml_path = Path(project.Config_file_path)
        mlflowdb_path = project.tracking

        if not model_name:
            model_name = f"{Path(project.model_path).stem}"

        if not yaml_path.exists():
            raise FileExistsError(f"{yaml_path} does not exist.")

        data = proj.yaml_data_retrieve(yaml_path=yaml_path)
        data_path = project.data_path.get("biigle_path") or project.data_path.get(
            "ultralytics_data_path"
        )

        if not data_path:
            raise ValueError("No valid data path found in project configuration.")

        if not model_source or not isinstance(model_source, (Path, str)):
            raise ValueError("'model' must be a non-empty string.")

        artifact_root = project_path / project_name / "mlruns"
        artifact_root.mkdir(parents=True, exist_ok=True)

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

        yolo_model = YOLO(model_source)

        def dict_mapper(yolo_result):

            return re.sub(r"[(B)]", "", yolo_result)

        with mlflow.start_run(run_name=model_name):
            results = yolo_model.train(
                data=data_path, name=model_name, epochs=epochs, imgsz=imgsz, **kwargs
            )
            mlflow.log_param("epochs", epochs)
            mlflow.log_metrics(
                {dict_mapper(k): v for k, v in results.results_dict.items()}
            )

            mlflow.log_artifacts(results.save_dir, artifact_path="yolo")

            best_weight_path = Path(results.save_dir) / "weights" / "best.pt"

            # Mock input, output, and parameters
            sample_input = [{"image": "path/to/sample_video_or_image.mp4"}]

            # This matches the exact format your predict() method returns
            sample_output = [
                {
                    "frame_number": 0,
                    "file_name": "name",
                    "boxes": [[0.0, 0.0, 100.0, 100.0]],
                    "plot": [],
                    "scores": [0.95],
                    "classes": [0],
                    "names": {0: "person"},
                    "shape": [640, 640],
                }
            ]
            sample_params = {"vid_stride": 10, "device": "cpu"}

            signature = infer_signature(
                model_input=sample_input,
                model_output=sample_output,
                params=sample_params,
            )

            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=YOLOUltralyticsMLflowModel(),
                artifacts={"weights": str(best_weight_path)},
                registered_model_name=model_name,
                input_example=sample_input,
                signature=signature,
            )
            # copy model into run artifacts
            local_model_dir = mlflow.artifacts.download_artifacts(
                artifact_uri=model_info.model_uri
            )

            mlflow.log_artifacts(local_model_dir, artifact_path="model")
        mlflow.end_run()

        # Examine the deleted experiment details.
        experiment = mlflow.get_experiment(experiment_id)
        print(f"Name: {experiment.name}")
        print(f"Artifact Location: {experiment.artifact_location}")
        print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
        print(f"Last Updated timestamp: {experiment.last_update_time}")
        return results

    def export_experiment(
        self, project: Project, port=8080, notebook_formats=None, use_threads=False
    ):

        project_name = project.project_name
        experiment_name = project_name
        # 1) Tracking URI points to your local mlruns folder
        mlflow.set_tracking_uri(f"http://localhost:{port}")
        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        artifact_root = Path(project.project_path) / project_name / "output"
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

    def import_experiment(self, project: Project, input_dir: str, port: int = 8080):

        project_name = project.project_name
        experiment_name = project_name
        input_dir = Path(input_dir).expanduser()
        if not input_dir.is_absolute():
            input_dir = resolve_up(input_dir)
        # 1) Tracking URI points to your local mlruns folder
        mlflow.set_tracking_uri(f"http://localhost:{port}")

        client = mlflow.MlflowClient()

        import_experiments(
            mlflow_client=client,
            input_dir=str(input_dir),
            use_src_user_id=False,
            use_threads=False,
        )

    def loading_model(self, project: Project, model_name: str, version: int):
        """load pyfuncModel from registred Mlflow models and versions"""

        mlflowdb = project.tracking
        tracking_uri = f"sqlite:///{str(mlflowdb)}"

        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        return model

    def model_inference(self, model: PyFuncModel, data_path: str, vid_stride: int):
        if not isinstance(model, PyFuncModel):
            raise TypeError(f"model {model} must be an mlflow pyfunc model.")
        if not isinstance(data_path, str):
            raise TypeError(f"data path {data_path} must be a non-empty string.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"current device: {device}")
        data_path = Path(data_path).expanduser()
        data_path = resolve_up(data_path)
        result = model.predict(
            [{"image": str(data_path)}],
            params={"device": str(device), "vid_stride": vid_stride},
        )
        return result

    def internal_model(self, model: PyFuncModel):
        """load the internal pytorch model of type torch.nn.Module"""
        if not isinstance(model, PyFuncModel):
            raise TypeError(f"model {model} must be a mlflow pyfuncModel")
        internal_model = model._model_impl.python_model
        pytorch_model = internal_model.nn_model()
        if not isinstance(pytorch_model, nn.Module):
            raise TypeError(f"model {pytorch_model} is not a pytorch model")
        return pytorch_model

    def model_validation(
        self,
        project: Project,
        model: PyFuncModel = None,
        model_name: str = None,
        version: str | int = None,
        data_path: str = None,
        split: str = "test",
        imgsz=640,
        batch=8,
    ):
        """evaluate the model with selected data"""
        mlflowdb = project.tracking
        tracking_uri = f"sqlite:///{str(mlflowdb)}"
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        model_name = project.model_name

        if not model and model_name:
            model = self.loading_model(
                project=project, model_name=model_name, version=int(version)
            )
        elif not model and not model_name:
            logged_models = client.get_latest_versions(name=model_name)
            if not logged_models:
                raise ValueError(f"No registered model found with name: {model_name}")

            version = logged_models[0].version
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.pyfunc.load_model(model_uri)
        if not data_path:
            # data_path = project.data_path["ultralytics_data_path"]
            data_path = project.data_path.get("biigle_path") or project.data_path.get(
                "ultralytics_data_path"
            )
        data_path = Path(data_path).expanduser()
        data_path = resolve_up(data_path)
        uri = model.metadata.artifact_path
        run_id = model.metadata.run_id
        # Get the run object
        run = client.get_run(model.metadata.run_id)
        # Get the experiment ID
        experiment_id = run.info.experiment_id
        run_name = run.info.run_name
        val_run_name = f"validation_{run_name}"

        uri_path = Path(urlparse(uri).path)
        model_path = uri_path / "artifacts/best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        yolo_model = YOLO(model_path)

        with mlflow.start_run(experiment_id=experiment_id, run_name=val_run_name):

            results = yolo_model.val(
                data=str(data_path),
                split=split,
                imgsz=imgsz,
                batch=batch,
            )

            metrics = results.results_dict
            for k, v in metrics.items():
                clean_name = f"test_{k.replace('metrics/', '').replace('(', '').replace(')', '').replace(' ', '_')}"
                mlflow.log_metric(clean_name, float(v))
        return metrics
