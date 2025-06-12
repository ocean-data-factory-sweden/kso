# Installation Guide (Conda)

This document describes how to set up **new_kso_o3** in a fresh Conda environment on Linux/macOS/Windows.

---

## 1. Prerequisites

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ≥ 4.10 installed and on your `$PATH`.
* **Optional (GPU users)**: CUDA-compatible NVIDIA driver and Toolkit.

## 2. Clone the repository

```bash
git clone https://github.com/<your-org>/kso.git
cd kso/new_kso_o3
```

## 3. Create & activate the environment

```bash
conda create -n kso python=3.10 -y
conda activate kso
```

## 4. Install dependencies

### Option A – using Rye (recommended)
Rye provides locked dependency management via the existing `pyproject.toml`.

```bash
pip install rye
rye sync --features ml  # install ML extras (torch, transformers, etc.)
```

### Option B – using pip only

```bash
pip install -e .[ml]      # editable install with "ml" extra
```

## 5. Verify

```bash
pytest -q        # should report all tests passing
rye run python -c "import kso, sys; print('kso version', kso.__version__)"
```

## 6. Launch Jupyter

```bash
jupyter lab
```

Open the notebooks under `notebooks/` and follow the workflow (`01_download_data.ipynb` → `03_evaluate_model.ipynb`).

---

### Troubleshooting
| Issue | Resolution |
|-------|------------|
| `cl.exe` / compiler errors on Windows | Install Build Tools or use WSL2 + Ubuntu.
| CUDA mismatch | Ensure driver and PyTorch CUDA wheels match (`torch.version.cuda`).
| Proxy blocks downloads | Set `HTTPS_PROXY` / `HTTP_PROXY` env-vars.
