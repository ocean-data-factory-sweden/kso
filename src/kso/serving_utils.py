import subprocess
import mlflow
import mlflow.models
import random
import os
import argparse
import logging
import webbrowser
import time
import signal
import psutil
from pathlib import Path
import cv2
from typing import Any, Dict, Optional, List, Union
import requests
import numpy as np
from .project import Project
from .data_preprocessing import resolve_up
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)


class MLflowServerManager:
    def __init__(self):
        pass

    def find_process_by_port(self, port):
        """
        Find the process based on the port number

        Args:
            port: port number

        Returns:
            list: List of processes using this port
        """
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        processes.append(proc)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return processes

    def find_mlflow_processes(self):
        """
        Find all MLflow-related processes

        Returns:
            list: MLflow process list
        """
        mlflow_processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and any("mlflow" in cmd for cmd in cmdline):
                    # Check if it is the MLflow server process
                    if "server" in cmdline:
                        mlflow_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return mlflow_processes

    def stop_mlflow_server(self, port=8080, force=False):
        """
        Stop MLflow server

        Args:
            port: The service port number to be stopped
            force: Force terminate the process

        Returns:
            bool: Was the service successfully stopped?
        """
        logger.info(f"Looking for port {port} MLflow service on...")

        # First, find the process on the specific port.
        port_processes = self.find_process_by_port(port)

        # Then search for all MLflow processes
        mlflow_processes = self.find_mlflow_processes()

        # Merge process list, remove duplicates
        all_processes = []
        process_pids = set()

        for proc in port_processes + mlflow_processes:
            if proc.pid not in process_pids:
                all_processes.append(proc)
                process_pids.add(proc.pid)

        if not all_processes:
            logger.info("No running MLflow service found")
            return True

        # No running MLflow service found
        logger.info(f"turn up {len(all_processes)} MLflow related processes:")
        for proc in all_processes:
            try:
                cmdline = " ".join(proc.cmdline())
                logger.info(f"  PID: {proc.pid}, Order: {cmdline[:100]}...")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.info(f"  PID: {proc.pid}, Order: <Unable to obtain>")

        # Stop process
        stopped_count = 0
        for proc in all_processes:
            try:
                logger.info(f"Stopping process {proc.pid}...")

                if force:
                    # Forced termination
                    proc.kill()
                    logger.info(f"The process has been forcibly terminated. {proc.pid}")
                else:
                    # graceful stop
                    proc.terminate()

                    # Wait for process to end
                    try:
                        proc.wait(timeout=10)
                        logger.info(f"process {proc.pid} Stopped gracefully")
                    except psutil.TimeoutExpired:
                        logger.warning(
                            f"process {proc.pid} If no response is received within 10 seconds, the connection will be forcibly terminated..."
                        )
                        proc.kill()
                        logger.info(
                            f"The process has been forcibly terminated. {proc.pid}"
                        )

                stopped_count += 1

            except psutil.NoSuchProcess:
                logger.info(f"process {proc.pid} no longer exists")
                stopped_count += 1
            except psutil.AccessDenied:
                logger.error(f"No permission to terminate the process {proc.pid}")
            except Exception as e:
                logger.error(f"Stop process {proc.pid} error: {str(e)}")

        if stopped_count > 0:
            logger.info(f"successfully stopped {stopped_count} processes")

            # Double-check that the port has been released.
            time.sleep(2)
            remaining_processes = self.find_process_by_port(port)
            if not remaining_processes:
                logger.info(f"port {port} released")
                return True
            else:
                logger.warning(f"port {port} There are still processes running on it.")
                return False
        else:
            logger.error("No process was successfully stopped.")
            return False

    def check_port_available(self, port):
        """
        Check if the port is available

        Args:
            port: port number

        Returns:
            bool: Is the port available?
        """
        processes = self.find_process_by_port(port)
        return len(processes) == 0

    def start_mlflow_server(
        self, project: Project, host="127.0.0.1", port=8080, auto_open=True
    ):
        """
        Start the MLflow server

        Args:
            host: Server host address
            port: Server port number
            auto_open: Does the browser open automatically?
        """

        # Check if the port is already in use.
        if not self.check_port_available(port):
            logger.error(f"Port {port} is already in use!")
            logger.info(
                f"You can run the following command to stop the existing service:"
            )
            logger.info(f"  python {__file__} --stop --port {port}")
            return False

        # check if mlflow.db exists.
        proj_dir = Path(project.project_path) / project.project_name
        mlflowdb_path = Path(project.tracking)
        artifact_path = proj_dir / "mlruns" / "artifact"
        artifact_path.mkdir(parents=True, exist_ok=True)

        if not mlflowdb_path.exists():
            logger.warning("The MLflowdb was not found.")

        # Start the MLflow server
        logger.info(f"MLflow server, address {host}:{port}...")

        log_dir = proj_dir / "mlflow_logs"

        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "mlflow_stdout.log"

        try:
            process = subprocess.Popen(
                [
                    "mlflow",
                    "server",
                    "--backend-store-uri",
                    f"sqlite:///{mlflowdb_path}",
                    "--default-artifact-root",
                    f"file://{artifact_path}",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                stdout=open(log_file, "a"),
                stderr=subprocess.STDOUT,
            )

            # Wait for server to start
            time.sleep(2)

            # Check if the process is still running
            if process.poll() is not None:
                logger.error("MLflow server failed to start")
                return False

            ui_url = f"http://{host}:{port}"
            logger.info(f"The MLflow server is running. Please visit: {ui_url}")

            # Automatically open browser
            if auto_open:
                webbrowser.open(ui_url)
                logger.info("The browser has opened automatically.")

            # Wait for user interruption
            print("\nexecute stop_mlflow_server function to stop the MLflow server...")

            return True
        except Exception as e:
            logger.error(f"MLflow server failed to start: {str(e)}")
            return False

    def get_free_port_in_range(self, start=8080, end=9000) -> int:
        for _ in range(100):  # limit attempts
            port = random.randint(start, end)
            result = self.check_port_available(port)
            if result:
                return port
        print("no free port was found")

    def mlflow_serving(
        self, image_path: str, url: str = "http://127.0.0.1:5001/invocations"
    ) -> Dict:
        # Load a real image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (640, 640))

        # Prepare payload
        # MLflow "inputs" format
        payload = {"inputs": {"image": img.tolist()}}

        # Post to server
        response = requests.post(url, json=payload)

        if response.status_code == 200:

            print(response.json())
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    def deploy_mlflow_registered_model(
        self, model_name: str, version: str, port=5000, auto_open=True
    ):
        """
        deploy your model locally and returns base URL.

        Uses:
        mlflow models serve -m models:/<name>/<stage_or_version> -h <host> -p <port> --no-conda
        """

        # Check if the port is already in use.
        if not self.check_port_available(port):
            logger.error(f"Port {port} is already in use!")
            logger.info(
                f"You can run the following command to stop the existing service:"
            )
            return False

        # # check if mlflow.db exists.
        # dir_path = Path(__file__).resolve().parents[2]
        # mlruns_path = dir_path / "projects"

        # if not mlflowdb_path.exists():
        #     logger.warning("The MLflowdb was not found.")

        # # Start the MLflow server
        # logger.info(f"MLflow server, address :{port}...")

        # dir = Path(__file__).resolve().parents[2]

        # log_dir = dir / "projects" / "mlflow_logs"

        # log_dir.mkdir(exist_ok=True)
        # log_file = log_dir / "mlflow_stdout.log"

        os.environ["MLFLOW_TRACKING_URI"] = (
            "sqlite:////Users/ghaith/Desktop/kso/kso/projects/mlflow.db"
        )
        os.environ["MLFLOW_ARTIFACT_URI"] = (
            "file:///Users/ghaith/Desktop/kso/kso/projects/test_project_1/mlruns"
        )
        try:
            process = subprocess.Popen(
                [
                    "mlflow",
                    "models",
                    "serve",
                    "-m",
                    f"models:/{model_name}/{version}",
                    "-p",
                    f"{port}",
                    "--no-conda",
                ],
                # stdout=open(log_file, "a"),
                # stderr=subprocess.STDOUT,
                cwd="/Users/ghaith/Desktop/kso/kso/projects/test_project_1",
            )

            # Wait for server to start
            time.sleep(2)

            # Check if the process is still running
            if process.poll() is not None:
                logger.error("MLflow server failed to start")
                return False

            # ui_url = f"http://{host}:{port}"
            # logger.info(f"The MLflow server is running. Please visit: {ui_url}")

            # # Automatically open browser
            # if auto_open:
            #     webbrowser.open(ui_url)
            #     logger.info("The browser has opened automatically.")

            # # Wait for user interruption
            # print("\nexecute stop_mlflow_server function to stop the MLflow server...")

            return True
        except Exception as e:
            logger.error(f"MLflow server failed to start: {str(e)}")
            return False

    def plot_bboxes(self, results, image_path: str):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (640, 640))

        pred = results["predictions"][0]
        names = pred["names"]
        scores = pred["scores"]
        classes = pred["classes"]
        boxes = np.array(pred["boxes"], dtype=int)

        for score, cls, bbox in zip(scores, classes, boxes):
            x1, y1, x2, y2 = bbox
            label = f"{names[str(cls)]}: {score:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return img

    def save_predictions(
        self, project: Project, predictions: List, save_dir: str = None
    ) -> Path:
        """
        save the predictions created by the model inference
        """

        if save_dir:
            save_dir = Path(save_dir).expanduser()
            if not save_dir.is_absolute():
                save_dir = resolve_up(relative_path=save_dir)
        else:
            project_name = project.Project_name
            base_dir = Path(project.project_path).expanduser()
            save_dir = base_dir / project_name

        if not save_dir.exists():
            raise FileNotFoundError(f"dir {save_dir} was not found")

        idx = 0
        while True:
            suffix = "" if idx == 0 else f"_{idx}"
            new_dir = save_dir / f"inference_results{suffix}"
            if not new_dir.exists():
                new_dir.mkdir(parents=True)
                break
            idx += 1

        for i, pred in enumerate(predictions):
            cv2.imwrite(f"{new_dir}/annotated_{i}.jpg", pred["plot"])
        logging.info(f"Saving inference results to: {new_dir}")

    def get_registered_models(self, project: Project) -> pd.DataFrame:
        """
        List all registered models in MLflow.

        Returns:
            DataFrame with registered model information
        """
        mlflowdb = project.tracking

        tracking_uri = f"sqlite:///{str(mlflowdb)}"
        client = mlflow.tracking.MlflowClient(tracking_uri)  # type: ignore[attr-defined]
        models = client.search_registered_models()

        model_info = []
        for model in models:
            if model.latest_versions:
                for version in model.latest_versions:
                    model_info.append(
                        {
                            "name": model.name,
                            "version": version.version,
                            "stage": version.current_stage,
                            "run_id": version.run_id,
                            "status": version.status,
                        }
                    )

        return pd.DataFrame(model_info)
