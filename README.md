# KSO System

The KSO System is an open-source machine learning framework for underwater video analysis, developed from the [Koster Seafloor Observatory][koster-url] research initiative and the Swedish Platform for Subsea Image Analysis ([SUBSIM][subsim-url]). It is optimized for GPU-accelerated HPC environments, particularly LUMI, and integrates with MLflow for experiment tracking.

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL License][license-shield]][license-url]

> **📘 New to KSO?** Each notebook contains step-by-step instructions with clearly marked **EDIT THIS** cells. This README provides an overview — the notebooks will guide you through each stage.

## System Overview

![KSO System Overview][high-level-overview]

## Quick Start

### 1) Choose your environment

See [installation](#installation).

### 2) Run the notebooks

Use the table below to choose the first stage that matches what you already have.

| You have… | Start at |
|---|---|
| Raw footage only **OR** BIIGLE-annotated images but need a YOLO dataset | **00** Data Preparation |
| A YOLO dataset (`data.yaml` + train/val/test splits) | **01** Project Setup |
| A trained model (or weights) and you want to fine-tune on a dataset | **02** Training & Eval |
| A trained model and you want to run inference on images/video | **03** Inference + **04** Analysis |
| A validated model that you want to publish along with your dataset | **05** Publish Model |

> **Note:** Notebooks 00, 03, 04, and 05 are still in development. For the recommended working path today, see the [Standalone notebooks](#standalone-notebooks) section below.

## Notebook workflow
KSO is transitioning to a clear, staged notebook pipeline. Stages **01–02** are stable today; later stages are under active development.

### Official Pipeline (00–05)

| # | Notebook | Description | Status |
|--:|----------|-------------|--------|
| 00 | 00_Data_Preparation.ipynb | Transfer footage to LUMI (optional), extract frames, build your image set for annotation in [BIIGLE](https://biigle.de), convert annotation CSV → YOLO. *Skip if you already have a YOLO dataset.* | 🔜 In development |
| 01 | 01_Project_Setup.ipynb | Create a KSO2 project (`.project.yaml`), attach your YOLO dataset, configure tracking, and optionally run offline augmentation. | ✅ Stable |
| 02 | 02_Train_and_Eval_Models.ipynb | Train or fine-tune a YOLO model, track runs with MLflow, and evaluate on the test set. | ✅ Stable |
| 03 | 03_Inference.ipynb | Run inference or batch inference on new images or video; export detections (CSV + annotated media). | 🔜 In development |
| 04 | 04_Analysis.ipynb | Summary statistics, maxN, per-class summaries, and visualizations. | 🚧 Planned |
| 05 | 05_Publish_Models.ipynb | Package models and metadata; publish to Zenodo or Researchdata.se. | 🚧 Planned |

### Standalone Notebooks

While the official pipeline is being finalized, these notebooks provide a working path for new users — covering dataset preparation in [BIIGLE](https://biigle.de), and end-to-end model training.

| Notebook | Path | Covers |
|----------|------|--------|
| Biigle_to_YOLO.ipynb | [`notebooks/setup/Biigle_to_YOLO.ipynb`](./notebooks/setup/Biigle_to_YOLO.ipynb) | BIIGLE CSV → YOLO conversion (data preparation for BIIGLE users) |
| Train_models.ipynb | [`notebooks/analyse/Train_models.ipynb`](./notebooks/analyse/Train_models.ipynb) | YOLO model training and fine-tuning using Ultralytics |

### Available YOLO models

The training notebook supports several Ultralytics model families, including [YOLO11](https://docs.ultralytics.com/models/yolo11/). See the notebook itself for the authoritative model list and parameters.

<details>
<summary><b>Model sizing guidance</b> (click to expand)</summary>

Practical guidance:

- **Small datasets (~100–250 images):** prefer **nano** or **small** — larger models may overfit.
- **Medium datasets (~250–750 images):** use **medium** for a good balance.
- **Large datasets (750+ images):** consider **large** or **xlarge** if resources allow.

</details>

## Installation

### System requirements

- **Minimum**: Python 3.11, 16 GB RAM, ≈10 GB free disk space.
- **Recommended**: CUDA/ROCm-capable GPU (≥8 GB VRAM) and access to an HPC system (e.g. LUMI).

### Option 1 — LUMI (recommended)

KSO is primarily developed and tested on the LUMI supercomputer, running via a Singularity/Apptainer container on GPU nodes. 

If you're a first time user, start here:
- [`contrib/lumi/README.md`](./contrib/lumi/README.md)

### Option 2 — Other HPC systems

```bash
git clone -b dev https://github.com/ocean-data-factory-sweden/kso.git
cd kso
# Follow your HPC's recommended way to launch Jupyter or batch jobs
```

Use your center's standard GPU modules or containers, and bind project and scratch storage as appropriate.

### Option 3 — Local development

For local use without HPC access. Training without a GPU will be slow; smaller models and lower batch sizes are recommended.

**Docker**

Note: The instructions below run the notebooks inside the container.
Any changes you make will be lost when the container stops unless you save them outside the container
(e.g., using a mounted volume: `-v $(pwd):/opt/kso/notebooks`).

```bash
# Pull kso with a suitable backend
docker pull ghcr.io/ocean-data-factory-sweden/kso:dev-ubuntu24.04             # CPU only
# docker pull ghcr.io/ocean-data-factory-sweden/kso:dev-cuda12.9-ubuntu24.04  # CUDA / NVIDIA GPUs
# docker pull ghcr.io/ocean-data-factory-sweden/kso:dev-rocm6.4-ubuntu24.04   # ROCm / AMD GPUs

# Run the notebooks
docker run -it --rm -p 8888:8888 ghcr.io/ocean-data-factory-sweden/kso:dev-ubuntu24.04 notebooks/
# docker run -it --rm -p 8888:8888 --gpus all ghcr.io/ocean-data-factory-sweden/kso:dev-cuda12.9-ubuntu24.04 notebooks/
# docker run -it --rm -p 8888:8888 --device /dev/kfd --device /dev/dri ghcr.io/ocean-data-factory-sweden/kso:dev-rocm6.4-ubuntu24.04 notebooks/

# Then open http://localhost:8888 in your browser and use the token printed out
```

**pip in venv**

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Fetch the repository
git clone -b dev https://github.com/ocean-data-factory-sweden/kso.git
cd kso

# Install kso with a suitable backend
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cpu        # CPU only
# pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu129    # CUDA / NVIDIA GPUs
# pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/rocm6.4  # ROCm / AMD GPUs

# Run the notebooks
jupyter lab notebooks/
```

## Developer Instructions

We welcome contributions!

1. Work from the `dev` branch; create feature branches off `dev`.
2. Format Python code with Black:
   ```bash
   black filename.py
   ```
3. Use **Conventional Commits** for messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`.
4. Keep commit history clean and logical (squash where appropriate) and **rebase** onto `dev` (never merge).
5. Open a Pull Request targeting `dev` and request at least 2 reviewers.

## Citation

If this code or its trained models contribute to your research, please cite:

> Anton V, Germishuys J, Bergström P, Lindegarth M, Obst M (2021). An open-source, citizen science and machine learning approach to analyse subsea movies. *Biodiversity Data Journal* 9: e60548. [https://doi.org/10.3897/BDJ.9.e60548](https://doi.org/10.3897/BDJ.9.e60548)

## Support & Contact

- **Website**: [https://subsim.se](https://subsim.se)
- **Issues**: [GitHub Issues][issues-url]
- **Contact**: matthias.obst(at)marine.gu.se

We are always excited to collaborate with marine scientists. Feel free to reach out with questions or ideas!

## Legacy Notebooks (Zooniverse workflow)

These notebooks implement the original Zooniverse citizen-science pipeline and are maintained for existing projects. For new work, use the main workflow above.

| Task | Notebook | Description | Colab |
|------|----------|-------------|-------|
| Check Zooniverse metadata | Check_metadata | Check format of footage, sites, media and species CSV files | [![Open In Colab][colab-badge]][colab_tut_1] |
| Classify | Upload_subjects_to_Zooniverse | Prepare footage and upload clips to Zooniverse | [![Open In Colab][colab-badge]][colab_tut_3] |
| Classify | Process_classifications | Pull and process classifications from Zooniverse | [![Open In Colab][colab-badge]][colab_tut_8] |
| Analyse | Evaluate_models | Standalone model evaluation | [![Open In Colab][colab-badge]][colab_tut_6] |
| Publish | Publish_models | Publish model to a public repository | [![Open In Colab][colab-badge]][colab_tut_7] |
| Publish | Publish_observations | Export observations to GBIF/OBIS | [![Open In Colab][colab-badge]][colab_tut_9] |

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
[license-url]: https://github.com/ocean-data-factory-sweden/kso/blob/dev/LICENSE.txt
[high-level-overview]: https://github.com/ocean-data-factory-sweden/kso/blob/dev/assets/high-level-overview.png?raw=true
[koster-url]: https://www.zooniverse.org/projects/victorav/the-koster-seafloor-observatory
[subsim-url]: https://subsim.se
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab_tut_1]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/setup/Check_metadata.ipynb
[colab_tut_3]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/classify/Upload_subjects_to_Zooniverse.ipynb
[colab_tut_6]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/analyse/Evaluate_models.ipynb
[colab_tut_7]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/publish/Publish_models.ipynb
[colab_tut_8]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/classify/Process_classifications.ipynb
[colab_tut_9]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/publish/Publish_observations.ipynb
