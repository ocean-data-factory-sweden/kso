# Comprehensive Guide: Annotated Image Data with Jupyter Notebooks  
*(Last updated: 12 Jun 2025)*

---

## Table of Contents
1. [Introduction](#introduction)
2. [Quick-Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Installation Options](#installation-options)
5. [Available Notebooks](#available-notebooks)
6. [Workflow Walk-Through](#workflow-walk-through)
   1. [Data Ingestion](#data-ingestion)
   2. [Training](#training)
   3. [Inference](#inference)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Introduction
This guide is aimed at researchers who possess **annotated image datasets** and wish to *ingest, train, and run inference* on the platform using the curated **Jupyter notebooks** provided.

> **Why notebooks?**  
> Notebooks offer an interactive, repeatable, and transparent workflow that aligns with our global principles of *modularity*, *readability*, *reproducibility*, and *security-first* development.

---

## Quick-Start
1. **Clone / download** the project repository.
2. Choose an [installation option](#installation-options) (Conda or Docker).
3. Launch JupyterLab: `jupyter lab` (Conda) **or** `make jupyter` (Docker).
4. Follow the notebooks in numerical order (`01_*` … `07_*`).

---

## Data Preparation
1. Organise data as:
   ```text
   dataset/
   ├─ images/
   │  └─ *.jpg | *.png
   ├─ annotations/
   │  └─ *.json | *.xml | *.txt | *.png
   └─ dataset_metadata.json
   ```
2. Supported annotation formats: **COCO**, **Pascal VOC**, **YOLO**, **Segmentation masks**, **CSV**.
3. Example `dataset_metadata.json`:
   ```json
   {
     "name": "my_dataset",
     "version": "1.0.0",
     "annotation_type": "coco",
     "classes": ["class1", "class2"],
     "split": {"train": 0.7, "validation": 0.15, "test": 0.15}
   }
   ```

---

## Installation Options
Choose **one** of the following methods. Detailed step-by-step instructions are in the sub-documents under [`docs/`](docs/):

| Method | Target Audience | Link |
| ------ | --------------- | ---- |
| **Conda** | Local development on Linux/macOS/Windows with GPU/CPU | [Installation (Conda)](docs/installation_conda.md) |
| **Docker** | Fully reproducible containerised setup | [Installation (Docker)](docs/installation_docker.md) |

Both methods install:
- Python 3.8+ and core ML packages (`torch`, `torchvision`, `scikit-learn`, …)
- `platform_sdk` & CLI tools
- JupyterLab extensions for progress monitoring

---

## Available Notebooks
| Notebook | Purpose |
| -------- | ------- |
| `01_data_exploration.ipynb` | Visualise dataset statistics & sample images |
| `02_data_preprocessing.ipynb` | Clean & augment data |
| `03_data_ingestion.ipynb` | Validate & upload dataset |
| `04_model_training.ipynb` | Configure & start training jobs |
| `05_model_evaluation.ipynb` | Evaluate metrics (mAP, confusion matrix) |
| `06_inference.ipynb` | Batch/single-image predictions & visualisation |
| `07_export_model.ipynb` | Export to ONNX / TorchScript; deploy endpoints |

---

## Workflow Walk-Through
### Data Ingestion
1. Open **`03_data_ingestion.ipynb`**.
2. Set dataset path variables (`DATA_PATH`, `METADATA_FILE`).
3. Run **validation** cell – fix any issues flagged.
4. Execute **upload** cell; note the returned `dataset_id`.

### Training
1. Open **`04_model_training.ipynb`**.
2. Insert `DATASET_ID` from ingestion.
3. Edit `training_config` dict (architecture, hyper-params, augmentations).
4. Run *Create Job* cell – obtain `job.id`.
5. Use provided widgets / dashboard URL to monitor progress.

### Inference
1. Open **`06_inference.ipynb`**.
2. Provide `MODEL_ID` from completed training job.
3. Predict on single image or directory; visualise &/or export results.
4. Optionally deploy as REST endpoint via `ModelDeployment` cell.

---

## Best Practices
- **Version control** datasets & notebooks (`dvc`, `git-lfs`).
- **Record** seeds, hyper-params, and environment hashes for reproducibility.
- **Secure** API keys via environment variables (`export PLATFORM_API_KEY=...`).
- **Monitor** GPU/CPU usage; down-scale idle endpoints.
- **Test** on a subset before full training to validate pipeline.

---

## Troubleshooting
| Symptom | Likely Cause | Fix |
| ------- | ------------ | ---- |
| Validation fails | Wrong annotation paths | Verify `dataset_metadata.json` & folder layout |
| OOM during training | Batch size too high | Reduce `batch_size` or use gradient accumulation |
| Poor accuracy | Class imbalance | Re-sample data or use class weights |
| Empty predictions | Threshold too high | Lower `confidence_threshold` in inference |

For further help:
- Official docs: <https://platform-docs.url>
- Community forum: <https://community.platform.url>
- Support email: support@platform.url

---

© 2025 Platform Research Team. All rights reserved.
