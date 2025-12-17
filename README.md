# KSO System

The Koster Seafloor Observatory is an open-source, citizen science and machine learning approach to analyse subsea movies.

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL License][license-shield]][license-url]

### KSO overview
The KSO system has been developed to:
* move and process underwater footage and its associated data (e.g. location, date, sampling device). (TBD)
* make this data available to citizen scientists in Zooniverse to annotate the data. (TBD)
* train and evaluate machine learning models (customise [Yolov5][YoloV5] or [Yolov8][YoloV8] models).
  
![koster_info_diag][high-level-overview]

The system is built around a series of easy-to-use [Jupyter Notebooks][Jupyter_site]. Each notebook allows users to perform a specific task of the system (e.g. upload footage to the citizen science platform or analyse the classified data).

Users can run these notebooks locally or on a high-performance computing (HPC) environment.

### Notebooks

| Notebook | Description |
| :--- | :--- |
| **[demo_setup.ipynb](notebooks/demo_setup.ipynb)** | A complete demonstration of the KSO workflow: creating a project, adding data/models, training, and serving with MLflow. |

## Local Installation

### Docker Installation
#### Requirements
* [Docker](https://www.docker.com/products/docker-desktop/)

#### Pull KSO Docker image
```
Bash
docker pull ghcr.io/ocean-data-factory-sweden/kso:dev
```

### Installation
#### Requirements
* [Python >=3.12](https://www.python.org/)
* [uv](https://github.com/astral-sh/uv) (Recommended) or pip

#### Prepare your system

You can install the KSO system locally using either `uv` (recommended for speed and reliability) or standard `pip`.

##### Option 1: Using uv (Recommended)

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync the environment (this creates a virtual environment and installs dependencies):
   ```bash
   uv sync
   ```

3. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

##### Option 2: Using pip

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Developer instructions
If you would like to expand and improve the KSO capabilities, please follow the instructions above to set the project up on your local computer.

When you add any changes, please create your branch on top of the current 'dev' branch. Before submitting a Merge Request, please:
* Run Black on the code you have edited 
```shell
black filename 
```
* Clean up your commit history on your branch, so that every commit represents a logical change. (so squash and edit commits so that it is understandable for others)
* For the commit messages, we ask that you please follow the [conventional commits guidelines](https://www.conventionalcommits.org/en/v1.0.0/) (table below) to facilitate code sharing. Also, please describe the logic behind the commit in the body of the message.
  #### Commit types

| Commit Type | Title                    | Description                                                                                                 | Emoji | 
|:-----------:|--------------------------|-------------------------------------------------------------------------------------------------------------|:-----:|
|   `feat`    | Features                 | A new feature                                                                                               |   ✨   |       
|    `fix`    | Bug Fixes                | A bug Fix                                                                                                   |  🐛   |      
|   `docs`    | Documentation            | Documentation only changes                                                                                  |  📚   |        
|   `style`   | Styles                   | Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)      |  💎   |         
| `refactor`  | Code Refactoring         | A code change that neither fixes a bug nor adds a feature                                                   |  📦   |         
|   `perf`    | Performance Improvements | A code change that improves performance                                                                     |  🚀   |         
|   `test`    | Tests                    | Adding missing tests or correcting existing tests                                                           |  🚨   |         
|   `build`   | Builds                   | Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)         |  🛠   |       
|    `ci`     | Continuous Integrations  | Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs) |  ⚙️   |       
|   `chore`   | Chores                   | Other changes that don't modify src or test files                                                           |  ♻️   |        
|  `revert`   | Reverts                  | Reverts a previous commit                                                                                   |  🗑   |        

* Rebase on top of dev. (never merge, only use rebase)
* Submit a Pull Request and link at least 2 reviewers


## Citation

If you use this code or its models in your research, please cite:

Anton V, Germishuys J, Bergström P, Lindegarth M, Obst M (2021) An open-source, citizen science and machine learning approach to analyse subsea movies. Biodiversity Data Journal 9: e60548. https://doi.org/10.3897/BDJ.9.e60548

## Collaborations/Questions
You can find out more about the project at https://subsim.se.

We are always excited to collaborate and help other marine scientists. Please feel free to contact us (matthias.obst(at)marine.gu.se) with your questions.

## Troubleshooting

If you experience issues importing panoptes_client in Windows, it is a known [issue with the libmagic package](https://github.com/zooniverse/panoptes-python-client/issues/264). [Pmason's suggestions in the Talk board of Zooniverse](https://www.zooniverse.org/talk/18/3283063) can be useful for troubleshooting it.
The following code fixed the issue in a Windows machine:
```bash
pip uninstall panoptescli
pip install panoptescli
pip install python-magic-bin
```

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ocean-data-factory-sweden/kso.svg?style=for-the-badge
[contributors-url]: https://https://github.com/ocean-data-factory-sweden/kso/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ocean-data-factory-sweden/kso.svg?style=for-the-badge
[forks-url]: https://github.com/ocean-data-factory-sweden/kso/network/members
[stars-shield]: https://img.shields.io/github/stars/ocean-data-factory-sweden/kso.svg?style=for-the-badge
[stars-url]: https://github.com/ocean-data-factory-sweden/kso/stargazers
[issues-shield]: https://img.shields.io/github/issues/ocean-data-factory-sweden/kso.svg?style=for-the-badge
[issues-url]: https://github.com/ocean-data-factory-sweden/kso/issues
[license-shield]: https://img.shields.io/github/license/ocean-data-factory-sweden/kso.svg?style=for-the-badge
[license-url]: https://github.com/ocean-data-factory-sweden/kso/blob/main/LICENSE.txt
[high-level-overview]: https://github.com/ocean-data-factory-sweden/kso/blob/main/assets/high-level-overview.png?raw=true
[Jupyter_site]: https://jupyter.org/
[colablogo]: https://colab.research.google.com/assets/colab-badge.svg
[binderlogo]: https://mybinder.org/badge_logo.svg
[colab_tut_1]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/setup/Check_metadata.ipynb
[binder_tut]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso/main
[colab_tut_3]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/classify/Upload_subjects_to_Zooniverse.ipynb
[colab_tut_5]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/analyse/Train_models.ipynb
[colab_tut_6]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/analyse/Evaluate_models.ipynb
[colab_tut_7]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/publish/Publish_models.ipynb
[colab_tut_8]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/classify/Process_classifications.ipynb
[colab_tut_9]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/publish/Publish_observations.ipynb
[Microsoft_C++]: https://visualstudio.microsoft.com/visual-cpp-build-tools/
[pytorch]: https://pytorch.org/
[YoloV5]: https://github.com/ultralytics/yolov5
[YoloV8]: https://github.com/ultralytics/ultralytics
[cdn_bucket]: https://console.cloudina.org/
[cdn_album]: https://album.cloudina.org/
[cdn_vendor]: https://vendor.cloudina.org/
