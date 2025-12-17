import subprocess
import mlflow
import mlflow.models
import config

import os
import argparse
import logging
import webbrowser
import time
import signal
import psutil
from pathlib import Path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_process_by_port(port):
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


def find_mlflow_processes():
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
                # 检查是否是MLflow server进程
                if "server" in cmdline:
                    mlflow_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return mlflow_processes


def stop_mlflow_server(port=5000, force=False):
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
    port_processes = find_process_by_port(port)

    # Then search for all MLflow processes
    mlflow_processes = find_mlflow_processes()

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
                    logger.info(f"The process has been forcibly terminated. {proc.pid}")

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
        remaining_processes = find_process_by_port(port)
        if not remaining_processes:
            logger.info(f"port {port} released")
            return True
        else:
            logger.warning(f"port {port} There are still processes running on it.")
            return False
    else:
        logger.error("No process was successfully stopped.")
        return False


def check_port_available(port):
    """
    Check if the port is available

    Args:
        port: port number

    Returns:
        bool: Is the port available?
    """
    processes = find_process_by_port(port)
    return len(processes) == 0


def start_mlflow_server(host="127.0.0.1", port=8080, auto_open=True):
    """
    Start the MLflow server

    Args:
        host: Server host address
        port: Server port number
        auto_open: Does the browser open automatically?
    """

    # Check if the port is already in use.
    if not check_port_available(port):
        logger.error(f"Port {port} is already in use!")
        logger.info(f"You can run the following command to stop the existing service:")
        logger.info(f"  python {__file__} --stop --port {port}")
        return False

    # check if mlflow.db exists.
    dir_path = Path.cwd().parent
    mlflowdb_path = dir_path / "projects" / "mlflow.db"
    artifact_path = dir_path / "artifact"

    if not mlflowdb_path.exists():
        logger.warning("The MLflowdb was not found.")

    # Start the MLflow server
    logger.info(f"MLflow server, address {host}:{port}...")

    dir = Path.cwd().parent
    log_dir = dir / "projects" / "mlflow.log"

    log_dir.mkdir(exist_ok=True)
    stdout_log = open(log_dir / "mlflow_stdout.log", "a")

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
            stdout=stdout_log,
            stderr=stdout_log,
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
