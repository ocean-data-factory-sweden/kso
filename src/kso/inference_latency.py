from thop import profile
from ultralytics import YOLO
import torch
import torchvision
import os
import platform
import subprocess
import logging
from torchinfo import summary
from .trainer import TrainingManager
from mlflow.pyfunc import PyFuncModel
import psutil
from pathlib import Path
from .data_preprocessing import resolve_up
import math
import torchvision.transforms as T

train = TrainingManager()


class ModelProfiler:
    def __init__(self):
        self.transform = T.Resize((640, 640))
        self.batch_used = 1

    def model_stats(self, model, image_path, batch_size):

        image_path = Path(image_path)
        if not isinstance(model, PyFuncModel):
            raise TypeError(f"model {model} is not PyFuncModel model")
        # Iterate over the directory and load its contents into tensors
        if image_path.is_dir():
            all_videos, all_frames = self.discover_dir(image_path)
            u = [
                self.transform(torchvision.io.read_image(path=f).float() / 255.0)
                for f in all_frames
            ]
            x = torch.stack(u)
            x = x[:batch_size]
        elif image_path.is_file:

            x = torchvision.io.read_image(path=image_path)
            x = x.float() / 255.0
            x = x.unsqueeze(0).expand(1, -1, -1, -1)  # (batch_size, 3, H, W)
            x = self.transform(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model = train.internal_model(model).to(device).eval()
        if not isinstance(torch_model, torch.nn.Module):
            raise ValueError(f"model {torch_model} is not a nn.Module")
        torch_model.requires_grad_(False)

        x = x.to(device, non_blocking=False)
        self.batch_used = x.shape[0]

        macs, params = profile(torch_model, inputs=(x,), verbose=False)
        print(f"model macs: {macs:.2e}")
        print(f"model flops: {macs * 2:.2e}")
        print(f"model params: {params:.2e}")
        return macs, params

    def image_num_indir(self, path: str):
        VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        # Travers all the branch of a specified path
        files = os.listdir(path)
        return sum([file.endswith(s) for s in VALID_EXTS for file in files])

    def os_system(self):
        "retrieve the information about the platform on which the program is running"
        paltform = platform.uname()
        system = paltform.system
        return system

    def hardware_flops(self):

        TFLOPS = 0.0
        system = self.os_system()
        if system == "Linux" and torch.cuda.is_available():
            cu_output = subprocess.check_output(
                "rocminfo | grep -i 'Compute Unit' | awk '{s+=$3} END{print s}'",
                shell=True,
                text=True,
            )
            CUs = int(cu_output.strip())

            # MI250X official specs
            FP32_PER_GCD_TFLOPS = 47.9
            CUS_PER_GCD = 110

            FP32_PER_CU_TFLOPS = FP32_PER_GCD_TFLOPS / CUS_PER_GCD
            TFLOPS = CUs * FP32_PER_CU_TFLOPS
        elif system == "Darwin":
            # Apple M1/M2 GPU
            try:
                cores_output = subprocess.check_output(
                    "system_profiler SPDisplaysDataType | grep 'Total Number of Cores' | awk '{print $NF}'",
                    shell=True,
                    text=True,
                )
                cores = int(cores_output.strip())
            except Exception as e:
                logging.warning(f"Could not detect GPU cores: {e}")
                cores = 0

            TFLOPS_PER_CORE = 0.325  # M1 GPU approximation
            TFLOPS = cores * TFLOPS_PER_CORE
        # elif system=="Windows":
        #     try:
        #         cores_output = subprocess.check_output(
        #         "powershell -Command \"(Get-CimInstance Win32_Processor).NumberOfCores\"",
        #         shell=True,
        #         text=True
        #         )
        #         cores = int(cores_output.strip())
        #     except Exception as e:
        #         logging.warning(f"Could not detect GPU cores: {e}")
        # logging.info(f"Detected Compute Units: {CUs}")
        # logging.info(f"Estimated FP32 peak TFLOPS of your Slurm allocation: {TFLOPS:.2f}")
        return TFLOPS

    def model_latency_inference(self, model, image_path, batch_size):

        macs, _ = self.model_stats(model, image_path, batch_size)  # model FLOPs
        FLOPs = macs * 2

        device_TFLOPs = self.hardware_flops()  # MI250X allocation
        device_FLOPs = device_TFLOPs * 1e12

        latency_sec = FLOPs / device_FLOPs
        latency_ms = latency_sec * 1e3

        # print(f"Theoretical lower-bound latency: {latency_ms:.4f} ms")
        return latency_ms

    def inference_memory(self, model, batch_size):
        """estimate the model memory used during inference"""
        my_device = "cuda" if torch.cuda.is_available() else "cpu"
        pytorch_model = train.internal_model(model)
        pytorch_model.eval().to(my_device)
        batch_size = min(self.batch_used, batch_size)
        stats = summary(
            pytorch_model,
            input_size=(batch_size, 3, 640, 640),
            col_names=("input_size", "output_size", "num_params", "mult_adds"),
            verbose=0,
            mode="eval",
        )

        MB = 1e6

        inference_bytes = (
            stats.total_input  # input bytes
            + stats.total_output_bytes / 2  # forward activations only
            + stats.total_param_bytes  # model weights
        )

        inference_mb = inference_bytes / MB
        # print(f"Inference Memory (MB): {inference_mb:.2f}")
        return inference_mb

    def memory_estimator(self, model, image_path, batch_size=1):
        """estimate the inference latency and the memory usage
        comenpared to the current slurm allocation"""
        image_path = Path(image_path).expanduser()
        if not image_path.is_absolute():
            image_path = resolve_up(relative_path=image_path)
        # count the number of images and videos in a directory
        if image_path.is_dir():
            all_videos, all_frames = self.discover_dir(image_path)
            num_all_videos = len(all_videos)

        if "SLURM_JOB_ID" in os.environ:
            ram_gb = int(os.environ["SLURM_MEM_PER_NODE"]) / 1024
            cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
        else:
            ram_gb = psutil.virtual_memory().total / (1024**3)

        inference_mb = self.inference_memory(model, batch_size)
        latency_ms = self.model_latency_inference(model, image_path, batch_size)

        # calculate the total_latency of all frames by the batch size
        # total_latency = latency_ms * math.ceil(num_all_frames / batch_size)

        print(
            f"current memory allocation is {ram_gb} and the model inference estimate memory is {inference_mb} mb"
        )
        print(
            f"the model inference latency for the provided data {image_path} is {latency_ms} ms"
        )
        return latency_ms, inference_mb

    def discover_dir(self, dir: str | Path):
        vid_extentions = {".wmv", ".mpg", ".mov", ".avi", ".mp4"}
        pic_extentions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        dir = Path(dir).expanduser()
        all_frames = [
            p
            for p in dir.iterdir()
            if p.is_file() and p.suffix.lower() in pic_extentions
        ]
        all_videos = [
            p
            for p in dir.iterdir()
            if p.is_file() and p.suffix.lower() in vid_extentions
        ]

        return all_videos, all_frames
