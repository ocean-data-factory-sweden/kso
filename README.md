# KSO System

KSO is an open-source machine learning workflow for analysing underwater imagery. It is developed through the [Koster Seafloor Observatory][koster-url] and the Swedish Platform for Subsea Image Analysis ([SUBSIM][subsim-url]), with a focus on reproducible computer-vision workflows for marine research.

KSO is designed for GPU-accelerated environments, particularly LUMI, and uses MLflow to keep detailed training history alongside a lightweight project record that follows the workflow from data to model outputs.

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL License][license-shield]][license-url]

## System Overview

![KSO System Overview][high-level-overview]

> **New to KSO?** Choose your environment and starting point below. Each notebook explains its own steps and marks the values you need to change with **EDIT**.

## Start here

### 1. Choose your environment

**LUMI is the primary environment for the staged KSO workflow.** First-time LUMI users should follow [`contrib/lumi/README.md`](./contrib/lumi/README.md) for the current launcher, container and Jupyter instructions.

For another HPC system or local development, [jump to Installation](#installation).

### 2. Choose your starting point

The new project workflow consists of **Notebooks 1 to 5**. **Notebook 0 is an optional preparation step** for users who do not yet have a YOLO dataset.

| You have | Start here |
|---|---|
| Raw footage or BIIGLE annotations, but no YOLO dataset | **Notebook 0: Prepare Data** when available. For BIIGLE detection data today, use [`Biigle_to_YOLO.ipynb`](./notebooks/setup/Biigle_to_YOLO.ipynb), then continue to Notebook 1. |
| A YOLO **detection** dataset with `data.yaml` and train/val/test splits | **Notebook 1:** [`01_Project_Setup.ipynb`](./notebooks/01_Project_Setup.ipynb), then **Notebook 2:** [`02_Train_and_Eval_Models.ipynb`](./notebooks/02_Train_and_Eval_Models.ipynb). To fine-tune your own weights, start Notebook 2 from them. |
| A trained detection model and new videos to analyse | Use the current standalone [`Model_Inference.ipynb`](./notebooks/analyse/Model_Inference.ipynb) until Notebook 3 is introduced. |
| A segmentation dataset or model | Not covered yet: these notebooks are for object detection. |

> **Pipeline rollout:** Notebooks 1 and 2 are the first notebooks of the new workflow. The later stages are in development, and until they are ready the [standalone notebooks](#current-standalone-notebooks) cover data preparation and inference.

## The notebook workflow

```text
Optional preparation   0  Prepare Data
Project workflow       1  Project Setup
                       2  Train and Evaluate Models
                       3  Model Inference
                       4  Analysis
                       5  Publish
```

The project starts in **Notebook 1**. It creates a project folder and project YAML, attaches an existing YOLO dataset without copying it, checks the dataset, and records the dataset information needed downstream.

**Notebook 2 currently covers object detection.** It trains candidate YOLO models and logs each run to MLflow. Candidate models are compared on the validation split. After you choose a model, the test split is used once for the final performance estimate, and the selected model, training settings and test results are recorded in the project YAML.

**MLflow** keeps the detailed training history, while the **project YAML** keeps the compact record of the dataset and selected outputs that later notebooks need.

### Pipeline status

| Notebook | Purpose | Current status |
|---|---|---|
| **0 - Prepare Data** | Build a YOLO dataset from source imagery and annotations before a KSO project exists. | In development. BIIGLE detection users can currently use [`Biigle_to_YOLO.ipynb`](./notebooks/setup/Biigle_to_YOLO.ipynb). |
| **1 - Project Setup** | Create a project, attach and check a YOLO dataset, then record the dataset in the project YAML. | Available: [`01_Project_Setup.ipynb`](./notebooks/01_Project_Setup.ipynb) |
| **2 - Train and Evaluate Models** | Train and compare YOLO detection models with MLflow, test the selected model once, and record it in the project YAML. | Available: [`02_Train_and_Eval_Models.ipynb`](./notebooks/02_Train_and_Eval_Models.ipynb) |
| **3 - Model Inference** | Run the selected model on new imagery and record the inference outputs and settings. | In development. Use [`Model_Inference.ipynb`](./notebooks/analyse/Model_Inference.ipynb) for video inference today. |
| **4 - Analysis** | Turn inference outputs into summaries, figures and analysis products. | Planned. |
| **5 - Publish** | Publish trained models (Zenodo) and species occurrences (GBIF/OBIS) with their metadata. | Planned. |

## Current standalone notebooks

Until Notebooks 0 and 3 are ready, these notebooks cover their steps. They work on their own, without a project.

| Notebook | Use |
|---|---|
| [`notebooks/setup/Biigle_to_YOLO.ipynb`](./notebooks/setup/Biigle_to_YOLO.ipynb) | Convert BIIGLE detection annotations to a YOLO dataset, then continue at Notebook 1. |
| [`notebooks/analyse/Model_Inference.ipynb`](./notebooks/analyse/Model_Inference.ipynb) | Run inference on a folder of videos with trained weights and write a detections CSV. |

## Installation

### LUMI (recommended)

KSO is primarily developed and tested on LUMI using GPU nodes and the project container. The LUMI guide is the authoritative onboarding route:

- [`contrib/lumi/README.md`](./contrib/lumi/README.md)

Use that guide rather than copying another user's scratch paths, project number or container path.

### Other HPC systems

```bash
git clone -b dev https://github.com/ocean-data-factory-sweden/kso.git
cd kso
# Follow your HPC centre's recommended way to launch Jupyter or batch jobs.
```

Use your centre's supported GPU environment, modules or containers, and make the repository and data storage available inside the job.

### Local development

Local use is suitable for development, inspection and small tests. You need Python 3.11 or later, about 16 GB of RAM and 10 GB of free disk space. Training without a GPU is slow; a CUDA or ROCm GPU with at least 8 GB of memory is recommended.

#### Docker

Pull the image for your hardware, then start Jupyter in it:

```bash
# CPU
docker pull ghcr.io/ocean-data-factory-sweden/kso:dev-cpu-ubuntu24.04
docker run -it --rm -p 8888:8888 ghcr.io/ocean-data-factory-sweden/kso:dev-cpu-ubuntu24.04 notebooks/

# NVIDIA / CUDA
# docker pull ghcr.io/ocean-data-factory-sweden/kso:dev-cuda12.9-ubuntu24.04
# docker run -it --rm -p 8888:8888 --gpus all ghcr.io/ocean-data-factory-sweden/kso:dev-cuda12.9-ubuntu24.04 notebooks/

# AMD / ROCm
# docker pull ghcr.io/ocean-data-factory-sweden/kso:dev-rocm6.4-ubuntu24.04
# docker run -it --rm -p 8888:8888 --device /dev/kfd --device /dev/dri ghcr.io/ocean-data-factory-sweden/kso:dev-rocm6.4-ubuntu24.04 notebooks/
```

Then open http://localhost:8888 and use the token printed in the terminal. Changes made inside the container are lost when it stops, unless you save them to a mounted folder (for example `-v $(pwd):/opt/kso/notebooks`).

#### pip in a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate

git clone -b dev https://github.com/ocean-data-factory-sweden/kso.git
cd kso

# CPU
pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cpu

# NVIDIA / CUDA
# pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu129

# AMD / ROCm
# pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/rocm6.4

jupyter lab notebooks/
```

## Contributing

Contributions are welcome.

1. Create feature branches from `dev` and open pull requests back to `dev`, with at least two reviewers.
2. Format Python and notebooks with Black and commit notebooks without outputs:

   ```bash
   pip install "black[jupyter]~=24.0" nbstripout
   black filename.py notebook.ipynb
   nbstripout notebook.ipynb
   ```

3. Use clear, focused commits with Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, `refactor:` and `test:`.
4. Keep changes narrow enough to review and test independently, and rebase onto `dev` rather than merging `dev` into your branch.

## Legacy notebooks

Older workflows are kept in [`legacy_notebooks/`](./legacy_notebooks/), including the original Zooniverse citizen-science workflow and earlier standalone training notebooks. For new object-detection work, follow the staged workflow above as each notebook becomes available.

## Citation

If KSO or models trained with it contribute to your research, please cite:

> Anton V, Germishuys J, Bergström P, Lindegarth M, Obst M (2021). An open-source, citizen science and machine learning approach to analyse subsea movies. *Biodiversity Data Journal* 9: e60548. [https://doi.org/10.3897/BDJ.9.e60548](https://doi.org/10.3897/BDJ.9.e60548)

## Support and contact

- **Website:** [https://subsim.se](https://subsim.se)
- **Issues:** [GitHub Issues][issues-url]
- **Contact:** matthias.obst(at)marine.gu.se

## License

SUBSIM/KSO is released under the **GPL-3.0 license**. See [LICENSE](./LICENSE) for details.

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
[license-url]: https://github.com/ocean-data-factory-sweden/kso/blob/dev/LICENSE
[koster-url]: https://www.zooniverse.org/projects/victorav/the-koster-seafloor-observatory
[subsim-url]: https://subsim.se
[high-level-overview]: https://github.com/ocean-data-factory-sweden/kso/blob/dev/assets/high-level-overview.png?raw=true
