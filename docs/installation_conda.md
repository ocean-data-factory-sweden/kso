# Installation Guide (Conda)

This document explains how to set up a **Conda** environment for running the platform notebooks.

---

## 1. Install Miniconda / Anaconda
- Download Miniconda for your OS: <https://docs.conda.io/en/latest/miniconda.html>
- Follow the installer, enable *Add conda to PATH* (optional but convenient).

## 2. Create Environment
```bash
conda create -n platform_env python=3.8 -y
conda activate platform_env
```

## 3. Install Dependencies
```bash
# Core scientific stack
conda install -c conda-forge jupyterlab numpy pandas scikit-learn matplotlib pillow -y

# PyTorch (CUDA 12.1 example; change to match your GPU/driver)
conda install pytorch torchvision torchaudio cudatoolkit=12.1 -c pytorch -c nvidia -y

# Platform SDK & CLI
pip install platform-sdk
```

## 4. Launch JupyterLab
```bash
jupyter lab
```
Open your browser at <http://localhost:8888> and start the notebooks.

---

## Reproducibility
Export the exact environment:
```bash
conda env export --no-builds > environment.yml
```
Others can recreate it via `conda env create -f environment.yml`.

## GPU Check
Run:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

### Tips
- Use `mamba` for faster installs (`conda install mamba -n base -c conda-forge`).
- Keep environments small; remove unused packages.
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256` for large models.
