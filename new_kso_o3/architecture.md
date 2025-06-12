# new_kso_o3 Architecture (Draft v0.1)

## Goals
- **Modular** support for multiple model back-ends (PyTorch, YOLOv5/8, Hugging Face transformers)
- **Unified** yet **simple** configuration via YAML
- **Pluggable storage** – local FS & remote (S3-compatible)
- **Reproducible**: Conda/Docker envs, deterministic configs
- **Extensible**: add new model back-ends with minimal code changes
- **Trackable**: unified experiment logging via pluggable tracking back-ends (MLflow, WandB)
- **User-friendly notebooks**: simplified notebooks relying on CLI endpoints to minimise custom code

## High-Level Layout
```
new_kso_o3/
├── configs/            # YAML configuration files
│   ├── project.yaml    # Global project settings
│   ├── storage.yaml    # Local/S3 paths & creds
│   └── models/         # Per-model configs (pytorch.yaml, yolo.yaml, hf.yaml)
├── data/
│   └── ...             # (git-ignored) local/ mounted datasets
├── kso/                # Core library (importable as `kso`)
│   ├── __init__.py
│   ├── config.py       # Unified config loader / schema validator
│   ├── storage.py      # Local & S3 abstractions
│   ├── registry.py     # Model registry / factory pattern
│   ├── datasets.py     # Dataset utils (COCO, YOLO, etc.)
│   ├── trainers/       # Training back-ends
│   │   ├── base.py     # AbstractTrainer
│   │   ├── pytorch.py  # TorchTrainer
│   │   ├── yolo.py     # Yolov5Trainer
│   │   └── hf.py       # HFTrainer
│   ├── inference/      # Inference pipelines
│   │   ├── base.py
│   │   ├── pytorch.py
│   │   ├── yolo.py
│   │   └── hf.py
│   └── utils/
│       ├── logging.py  # Standardised logging
│       └── metrics.py   # Metrics & experiment tracking (pluggable: MLflow, WandB)
├── notebooks/
│   └── *.ipynb         # Example Jupyter workflows
├── cli.py              # `python -m kso` entry-point (Typer)
├── tests/
│   └── ...
└── README.md
```

## Configuration Strategy
- **`project.yaml`** – project-level settings: default model, dataset location, output paths
- **`storage.yaml`** – credentials & buckets for S3/MinIO; local root path
- **`configs/models/*.yaml`** – hyper-parameters per model back-end

### Example project.yaml
```yaml
project_name: new_kso_o3_demo
model: pytorch
output_dir: outputs/
```

### Example storage.yaml
```yaml
local_root: ./data
remote:
  provider: s3
  endpoint: https://s3.yourdns.com
  bucket: datasets
  aws_access_key_id: YOURKEY
  aws_secret_access_key: YOURSECRET
  region: eu-north-1
```

### Example models/pytorch.yaml
```yaml
arch: faster_rcnn_resnet50_fpn
num_classes: 10
batch_size: 16
learning_rate: 0.001
epochs: 50
augmentations:
  horizontal_flip: true
  rotation: 15
```

## Core Components
1. **Config Loader (`kso.config`)**
   - Merges YAMLs, validates via `pydantic` schemas
2. **Storage Layer (`kso.storage`)**
   - `LocalStorage` & `S3Storage` classes implementing common interface
3. **Registry (`kso.registry`)**
   - Maps `model` key to Trainer/Inference classes
4. **AbstractTrainer**
   - `train()`, `evaluate()`, `save()` contract
5. **Inference API**
   - Unified `predict(image|dir)` for any back-end
6. **Metrics & Tracking (`kso.utils.metrics`)**
   - Pluggable adapters for MLflow and Weights & Biases
7. **CLI (`cli.py`)** with Typer commands:
   - `kso train --config configs/models/pytorch.yaml`
   - `kso infer --model-id 123 --image path.jpg`

## Development Workflow (standard_development)
1. **architecture_agent** – maintains this file
2. **python_coder** – implement modules in `kso/`
3. **code_review_agent** – review PRs
4. **tester_agent** – add tests in `tests/`
5. **python_coder** – iterate/fix
6. **documentation_agent** – update docs

---
*End of draft architecture*
