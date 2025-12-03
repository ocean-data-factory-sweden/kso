from __future__ import annotations
from pathlib import Path
import logging
from typing import Any, Dict, Optional
import yaml
from dataclasses import asdict
import os
import sys
from ultralytics import YOLO
from project import Project, create_project
import mlflow
from project import Project
from mlflow_export_import.bulk.export_experiments import export_experiments
from mlflow_export_import.bulk.import_experiments import import_experiments


def training_model(project_path: Project, epochs: int = 100, imgsz: int = 640):

    project_name = project_path.Project_name
    # project = create_project(project_path=project_name)
    # artifact_root=Path.cwd().parent/"projects"/project_name

    base_dir = Path.cwd().parent
    project = base_dir / "projects" / project_name
    yaml_path = project / f"{project_name}.project.yaml"
    if not yaml_path.exists():
        raise FileExistsError(f"{yaml_path} does not exist.")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data_path = data["ultralytics_data"]["path"]

    if not isinstance(project_path, Project):
        raise ValueError("'model' must be a Project instance.")

    model_source = data["model"]["model_path"] or project_path.model_name
    if not model_source or not isinstance(model_source, str):
        raise ValueError("'model' must be a non-empty string.")

    artifact_root = Path.cwd().parent / "projects" / project_name / "mlruns"
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflowdb_path = Path.cwd().parent / "projects" / "mlflow.db"
    experiment_name = f"{project_name}_yolov8"

    # Check if experiment exists
    mlflow.set_tracking_uri(f"sqlite:///{mlflowdb_path}")
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # create the experiment with a project location
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=f"file://{artifact_root}"
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    model = YOLO(model_source)
    with mlflow.start_run():
        results = model.train(data=data_path, epochs=epochs, imgsz=imgsz)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("accuracy", 0.87)
    mlflow.end_run()

    data["Mlflow"] = {
        "path": str(mlflowdb_path),
        "experiment_name": f"yolov8_with_ep={epochs}",
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


def export_experiment(project_path: Project, notebook_formats=None, use_threads=False):

    project_name = project_path.Project_name
    experiment_name = f"{project_name}_yolov8"
    # 1) Tracking URI points to your local mlruns folder
    mlflow.set_tracking_uri("http://localhost:8080")
    # Check if experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(experiment)
    artifact_root = Path.cwd().parent / "projects" / project_name / "output"
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = mlflow.MlflowClient()

    # 2) Export experiment(s)
    export_experiments(
        client=client,
        experiments=[experiment.experiment_id],
        output_dir=str(artifact_root),
        notebook_formats=None,
        use_threads=False,
    )


# def import_experiment(input_dir:str, experiment_rename_file:str):

#     import_experiments(
#     input_dir = input_dir,
#     experiment_renames = experiment_rename_file
# )
#     pass
