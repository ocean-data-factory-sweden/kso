# new_kso_o3 – Modular Image ML Toolkit

`new_kso_o3` is a **plug-and-play computer-vision platform** that lets you train, evaluate and serve image models with your choice of back-end:

* **PyTorch** – torchvision models (classification)
* **YOLO** – Ultralytics YOLOv8 detector/segmenter
* **Hugging Face** – Vision Transformers (ViT, etc.) via 🤗 Transformers

Key pillars:

* **Unified YAML configuration** (project/storage/model)  
* **Flexible storage** – local filesystem or S3-compatible remote  
* **Pluggable experiment tracking** – MLflow, Weights & Biases, or noop (set `KSO_TRACKING`)  
* **Reproducible notebooks** – parameter-driven, zero widgets  
* **Extensible** – register new trainers / inference back-ends with one decorator  

---

## 1 » Quick Start

### Clone & install (Conda)
```bash
# prerequisites: git + miniconda
 git clone https://github.com/<your-org>/kso.git
 cd kso/new_kso_o3

 conda create -n kso python=3.10 -y
 conda activate kso

 # use Rye for locked deps
 pip install rye
 rye sync --features ml      # installs torch, transformers, etc.

 pytest -q                   # all tests should pass
```
*See [`docs/installation_conda.md`](docs/installation_conda.md) for details or the Docker guide.*

### Download example data
```bash
mkdir -p data && \
  curl -L https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz | \
  tar -xz --strip-components=1 -C data
```

### Train! (CLI)
```bash
# ensure project.yaml -> model: pytorch
rye run python - <<'PY'
from kso.config import load_project_config
from kso.registry import get_trainer
from pathlib import Path

cfg = load_project_config()
trainer_cls = get_trainer(cfg.model)         # auto-imports backend
trainer = trainer_cls(cfg.model_params)
trainer.train(Path('data'))                  # outputs/ contains model
PY
```
### Or use notebooks
```bash
jupyter lab
# open notebooks/01_download_data.ipynb … 03_evaluate_model.ipynb
```

---

## 2 » Configuration Overview
Three YAML files under `configs/`:

| File | Purpose |
|------|---------|
| `project.yaml` | global settings (`project_name`, `model`, `output_dir`) |
| `storage.yaml` | `local_root` + optional `remote` S3 creds |
| `models/<backend>.yaml` | hyper-parameters for chosen trainer |

See [`docs/config_reference.md`](docs/config_reference.md) for complete schema & examples.

---

## 3 » Storage
* Local paths saved under `output_dir` (default `./outputs`).
* When `storage.remote` is configured with `provider: s3`, trained artifacts are automatically uploaded by each trainer’s `save()`.

---

## 4 » Experiment Tracking
Set env-var **`KSO_TRACKING`** to one of:

* `mlflow` → logs to local/remote MLflow
* `wandb` → logs to Weights & Biases
* unset/other → noop logger

The wrapper lives in `kso/utils/metrics.py`.

---

## 5 » Directory Layout
```
new_kso_o3/
├─ kso/                # core library
│   ├─ trainers/       # pytorch.py, yolo.py, hf.py, base.py
│   ├─ storage.py      # local & S3 abstraction
│   ├─ registry.py     # trainer/inference registry
│   └─ config.py       # YAML loader + Pydantic models
├─ configs/            # sample configs (project/storage/models)
├─ notebooks/          # 01_download_data.ipynb …
├─ docs/               # installation & reference docs
├─ tests/              # pytest suite (moto mock for S3)
└─ review.md           # incremental code-review log
```

---

## 6 » Testing & Linting
```bash
rye run pytest -q                # run unit tests
rye run ruff check kso/ tests/   # static checks (once added)
```

---

## 7 » Contributing
1. Fork → feature branch → PR.  
2. Follow [global rules](../.windsurfrules) for code quality, security & docs.  
3. Update `tests/` & `docs/` as you add features.  
4. Run `pre-commit` before pushing.

---

## 8 » License
MIT 2025 Platform Research Team
