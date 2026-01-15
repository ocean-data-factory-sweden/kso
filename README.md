# KSO System

The KSO System is an open-source machine learning framework for underwater video analysis, developed from the [Koster Seafloor Observatory][koster-url] research initiative and the Swedish Platform for Subsea Image Analysis ([SUBSIM][subsim-url]).

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL License][license-shield]][license-url]

> **📘 New to KSO?** Each notebook contains detailed, step-by-step instructions with clearly marked **EDIT THIS** cells. This README provides an overview—the notebooks will guide you through each stage.

## Overview

KSO is a Python-based toolkit for training object detection models on underwater imagery and video. It supports a full workflow from annotation (primarily Biigle, with optional Roboflow and legacy Zooniverse paths) through YOLO model training, inference, analysis, and publication of models and derived data. The system is optimized for GPU-accelerated HPC environments (especially LUMI) and integrates with Weights & Biases (W&B) and MLflow for experiment tracking.

![KSO System Overview][high-level-overview]

## Documentation

Quick links:

- **Quick Start** – Get running in minutes (see [Quick Start](#quick-start))
- **Notebook Pipeline** – Five-stage workflow, Stages I–II stable (see [Notebooks](#notebooks))
- **Installation Guide** – HPC, Docker, and local setup (see [Installation](#installation))
- **LUMI Setup Guide** – Detailed HPC instructions in [`docs/LUMI_SETUP.md`](./docs/LUMI_SETUP.md)
- **Developer Guide** – Contribution workflow (see [Developer Instructions](#developer-instructions))
- **Citation** – How to reference SUBSIM/KSO (see [Citation](#citation))

## Quick Start

Minimal steps for a local or HPC-backed run.

### 1. Clone the repository

```bash
git clone -b dev https://github.com/ocean-data-factory-sweden/kso.git
cd kso
```

### 2. Install dependencies (local dev)

```bash
pip install -r requirements.txt
```

For LUMI or other HPC systems, follow your site's container or module best practices and see [`docs/LUMI_SETUP.md`](./docs/LUMI_SETUP.md) for a full LUMI recipe.

### 3. Launch Jupyter and run the notebooks in order

```bash
jupyter lab
# or: jupyter notebook
```

**Suggested workflow:**

- **Stage I – Setup** (`01_Project_Setup.ipynb`):
  - Create a KSO2 project (writes a `.project.yaml` with paths, model slot, and tracking config).
  - Attach data by either pointing to an existing YOLO `data.yaml` (Ultralytics/Roboflow style), or converting Biigle CSV → YOLO.
  - *(Optional)* Run offline augmentation on the training split.

- **Stage II – Training & Evaluation** (`02_Train_and_Eval_Models.ipynb`):
  - Re-open the project from Stage I.
  - Choose and register a YOLO model.
  - Train with the selected dataset, track runs with W&B or MLflow, and evaluate on the test set.

- **Stages III–V** – Inference, Analysis, Publication: follow additional notebooks as they become available (see [Notebooks](#notebooks)).

## Notebooks

The pipeline is organized into five stages; **Stages I–II are stable**, later stages are under active development.

### Main workflow

| Stage | Notebook | Description | Status |
|-------|----------|-------------|--------|
| I. Setup | [01_Project_Setup.ipynb](./notebooks/01_Project_Setup.ipynb) | Create a KSO2 project (`*.project.yaml`), attach data (existing YOLO `data.yaml` or Biigle CSV → YOLO), and optionally run offline augmentation on the train split. | ✅ Stable |
| II. Training & Eval | [02_Train_and_Eval_Models.ipynb](./notebooks/02_Train_and_Eval_Models.ipynb) | Re-open a project, choose and register a YOLO model, run training (v8–v11), track runs with W&B/MLflow, compute test metrics, and export artifacts. | ✅ Stable |
| III. Inference | 03_Inference.ipynb | Batch inference on new images or video; export detections (CSV + annotated media). | 🚧 In development |
| IV. Analysis | 04_Analysis.ipynb | Summary statistics, maxN, per-class summaries, and visualizations for ecological analysis. | 🚧 Planned |
| V. Publication | 05_Publish_Models.ipynb | Package models and metadata and publish to Zenodo or Researchdata.se. | 🚧 Planned |

### Available YOLO models

This section stays high-level; the training notebook contains full details.

<details>
<summary><b>YOLO families and sizes</b> (click to expand)</summary>

- **Supported families**: YOLOv8, YOLOv9, YOLOv10, YOLOv11.
- **Sizes**: nano (n), small (s), medium (m), large (l), xlarge (x), plus task-specific variants where available.

Practical guidance:

- Small datasets (~100–250 images): prefer **nano** or **small** models.
- Medium datasets (~250–800 images): use **medium** models.
- Large datasets (800+ images): consider **large** or **xlarge** if resources allow.
- For exploratory work, start with a small model and scale up once the pipeline is stable.

</details>

### Legacy notebooks (Zooniverse workflow)

These notebooks implement the original Zooniverse-centric citizen-science pipeline and remain available but are no longer the recommended path for new projects.

| Task | Notebook | Description | Colab |
|------|----------|-------------|-------|
| Set up | Check_metadata | Check format and contents of footage, sites, media and species CSV files | [![Open In Colab][colab-badge]][colab_tut_1] |
| Classify | Upload_subjects_to_Zooniverse | Prepare original footage and upload short clips to Zooniverse | [![Open In Colab][colab-badge]][colab_tut_3] |
| Classify | Process_classifications | Pull and process classifications from Zooniverse | [![Open In Colab][colab-badge]][colab_tut_8] |
| Analyse | Evaluate_models | Standalone model evaluation (now integrated into Stage II) | [![Open In Colab][colab-badge]][colab_tut_6] |
| Publish | Publish_models | Publish model to a public repository | [![Open In Colab][colab-badge]][colab_tut_7] |
| Publish | Publish_observations | Export observations to GBIF/OBIS | [![Open In Colab][colab-badge]][colab_tut_9] |

For new projects, use the **Biigle → YOLO** pathway in Stage I instead of the Zooniverse workflow.

## Installation

### System requirements

- **Minimum**: Python 3.12, 16 GB RAM, ≈10 GB free disk space.
- **Recommended**: CUDA-capable GPU (≥8 GB VRAM) and access to an HPC system (e.g. LUMI).

### Option 1 – HPC: LUMI (recommended)

KSO is currently tuned for the LUMI supercomputer and is typically run via a Singularity/Apptainer container on GPU nodes. For a full step-by-step guide (interactive Jupyter sessions, batch jobs, storage layout, and troubleshooting), see:

- [`docs/LUMI_SETUP.md`](./docs/LUMI_SETUP.md)

Then launch Jupyter through the LUMI web interface and open the notebooks under `notebooks/` as described in the Quick Start.

### Option 2 – Other HPC systems

For other HPC clusters:

```bash
git clone -b dev https://github.com/ocean-data-factory-sweden/kso.git
cd kso
# Follow your HPC's recommended way to launch Jupyter or batch jobs
```

Use your center's standard GPU modules or containers, and bind project and scratch storage as appropriate.

### Option 3 – Local: Docker (recommended for users without HPC)

```bash
docker pull ghcr.io/ocean-data-factory-sweden/kso:dev
docker run --gpus all -it -p 8888:8888 ghcr.io/ocean-data-factory-sweden/kso:dev
# Then open http://localhost:8888 in your browser
```

This keeps dependencies isolated and makes it easy to reproduce environments.

### Option 4 – Local: pip (for development)

```bash
git clone -b dev https://github.com/ocean-data-factory-sweden/kso.git
cd kso
pip install -r requirements.txt
jupyter lab
```

On local machines without HPC, use smaller models and lower batch sizes to avoid out-of-memory errors and accept longer training times.

## Developer Instructions

We welcome contributions!

1. Work from the `dev` branch; create feature branches off `dev`.
2. Format Python code with Black:
   ```bash
   black filename.py
   ```
3. Use **Conventional Commits** for messages:
   - `feat:` – new features
   - `fix:` – bug fixes
   - `docs:` – documentation changes
   - `refactor:` – code restructuring
   - `test:` – adding or updating tests
4. Keep commit history clean and logical (squash where appropriate) and **rebase** onto `dev` (never merge).
5. Open a Pull Request targeting `dev` and request at least 2 reviewers.

## Citation

If this code or its trained models contribute to your research, please cite:

> Anton V, Germishuys J, Bergström P, Lindegarth M, Obst M (2021). An open-source, citizen science and machine learning approach to analyse subsea movies. *Biodiversity Data Journal* 9: e60548. [https://doi.org/10.3897/BDJ.9.e60548](https://doi.org/10.3897/BDJ.9.e60548)

## Support & Contact

- **Website**: [https://subsim.se](https://subsim.se)
- **Issues**: [GitHub Issues][issues-url]
- **Contact**: matthias.obst[at]marine.gu.se

We are always excited to collaborate with marine scientists. Feel free to reach out with questions or ideas!

External resources:

- [Biigle Annotation Platform](https://biigle.de)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com)
- [LUMI Supercomputer Documentation](https://docs.lumi-supercomputer.eu)

## Roadmap

Planned and ongoing work:

- ✅ Stage I – YAML-based project setup + Biigle → YOLO conversion + optional augmentation.
- ✅ Stage II – Model selection + YOLO training + evaluation + MLflow tracking.
- ⬜ Stage III – Batch inference notebook for new footage.
- ⬜ Stage IV – Analysis notebook for ecological metrics and visualizations.
- ⬜ Stage V – Publishing notebook for models and datasets (e.g. Zenodo).
- ⬜ Updated workflow diagram reflecting the five-stage pipeline.

## License

SUBSIM/KSO is released under the **GPL-3.0 license**. See [LICENSE.txt](./LICENSE.txt) for details.

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/ocean-data-factory-sweden/kso.svg?style=for-the-badge
[contributors-url]: https://github.com/ocean-data-factory-sweden/kso/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ocean-data-factory-sweden/kso.svg?style=for-the-badge
[forks-url]: https://github.com/ocean-data-factory-sweden/kso/network/members
[stars-shield]: https://img.shields.io/github/stars/ocean-data-factory-sweden/kso.svg?style=for-the-badge
[stars-url]: https://github.com/ocean-data-factory-sweden/kso/stargazers
[issues-shield]: https://img.shields.io/github/issues/ocean-data-factory-sweden/kso.svg?style=for-the-badge
[issues-url]: https://github.com/ocean-data-factory-sweden/kso/issues
[license-shield]: https://img.shields.io/github/license/ocean-data-factory-sweden/kso.svg?style=for-the-badge
[license-url]: https://github.com/ocean-data-factory-sweden/kso/blob/main/LICENSE.txt
[high-level-overview]: https://github.com/ocean-data-factory-sweden/kso/blob/main/assets/high-level-overview.png?raw=true
[koster-url]: https://www.zooniverse.org/projects/victorav/the-koster-seafloor-observatory
[subsim-url]: https://subsim.se
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab_tut_1]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/setup/Check_metadata.ipynb
[colab_tut_3]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/classify/Upload_subjects_to_Zooniverse.ipynb
[colab_tut_6]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/analyse/Evaluate_models.ipynb
[colab_tut_7]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/publish/Publish_models.ipynb
[colab_tut_8]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/classify/Process_classifications.ipynb
[colab_tut_9]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/publish/Publish_observations.ipynb
