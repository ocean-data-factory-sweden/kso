from thop import profile
from ultralytics import YOLO
import torch
import torchvision
import os
import platform
import subprocess
import logging
from torchinfo import summary
from .trainer import internal_model
from mlflow.pyfunc import PyFuncModel


def model_stats(model, image_path, batch_size=16):
    if not isinstance(model, PyFuncModel):
        raise TypeError(f"model {model} is not PyFuncModel model")
    torch_model = internal_model(model)
    if not isinstance(torch_model, torch.nn.Module):
        raise ValueError(f"model {torch_model} is not a nn.model")
    torch_model.eval()

    x = torchvision.io.read_image(path=image_path)
    x = x.float() / 255.0
    x = x.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch_size, 3, H, W)
    x = torch.nn.functional.interpolate(
        x, size=(640, 640), mode="bilinear", align_corners=False
    )
    macs, params = profile(torch_model, inputs=(x,), verbose=False)
    print(f"model macs: {macs:.2e}")
    print(f"model flops: {macs * 2:.2e}")
    print(f"model params: {params:.2e}")
    return macs, params


def image_num_indir(path: str):
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    # Travers all the branch of a specified path
    files = os.listdir(path)
    return sum([file.endswith(s) for s in VALID_EXTS for file in files])


def os_system():
    "retrieve the information about the platform on which the program is running"
    paltform = platform.uname()
    system = paltform.system
    return system


def hardware_flops():
    system = os_system()
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

    logging.info(f"Detected Compute Units: {CUs}")
    logging.info(f"Estimated FP32 peak TFLOPS of your Slurm allocation: {TFLOPS:.2f}")
    return TFLOPS


def model_latency_inference(model, image_path, batch=None):

    torch_model = internal_model(model)
    macs, _ = model_stats(torch_model, image_path, batch)  # model FLOPs
    FLOPs = macs * 2

    device_TFLOPs = hardware_flops()  # MI250X allocation
    device_FLOPs = device_TFLOPs * 1e12

    latency_sec = FLOPs / device_FLOPs
    latency_ms = latency_sec * 1e3

    # print(f"Theoretical lower-bound latency: {latency_ms:.4f} ms")
    return latency_ms


def inference_memory(model, batch_size=16):
    """estimate the model memory used during inference"""

    pytorch_model = internal_model(model)
    pytorch_model.eval().to("cpu")

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


def memory_estimator(model, image_path, batch_size=16):
    """estimate the inference latency and the memory usage
    comenpared to the current slurm allocation"""

    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("No Slurm allocation found")
    ram_gb = int(os.environ["SLURM_MEM_PER_NODE"]) / 1024
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])

    inference_mb = inference_memory(model, batch_size)
    latency_ms = model_latency_inference(model, image_path)
    print(
        f"current memory allocation is {ram_gb} and the model inference estimate memory is {inference_mb}"
    )
    print(
        f"the model inference latency for the provided data {image_path} is {latency_ms}"
    )
    return latency_ms, inference_mb
