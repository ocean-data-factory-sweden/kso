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
* move and process underwater footage and its associated data (e.g. location, date, sampling device).
* make this data available to citizen scientists in Zooniverse to annotate the data.
* train and evaluate machine learning models (customise [Yolov5][YoloV5] or [Yolov8][YoloV8] models).
  
![koster_info_diag][high-level-overview]

The system is built around a series of easy-to-use [Jupyter Notebooks][Jupyter_site]. Each notebook allows users to perform a specific task of the system (e.g. upload footage to the citizen science platform or analyse the classified data).

Users can run these notebooks via Google Colab (by clicking on the Colab links in the table below), locally or on a High-Performance Computer environment.

### Notebooks
| Name                                              | Description                                                                                 | Try it!  | 
| ------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------|
| 1. Check footage and metadata                     | Check format and contents of footage and sites, media and species csv files                 | [![Open In Colab][colablogo]][colab_tut_1] [![binder][binderlogo]][binder_tut] | 
| 2. Upload new media to the system*                | Upload new underwater media to the cloud/server and update the csv files                    | WIP | 
| 3. Upload clips to Zooniverse                     | Prepare original footage and upload short clips to Zooniverse                               | [![Open In Colab][colablogo]][colab_tut_3] [![binder][binderlogo]][binder_tut] |
| 4. Upload frames to Zooniverse                    | Extract frames of interest from the original footage and upload them to Zooniverse              | [![Open In Colab][colablogo]][colab_tut_4] [![binder][binderlogo]][binder_tut] |
| 5. Train ML models                                | Prepare the training and test data, set model parameters and train models                   | [![Open In Colab][colablogo]][colab_tut_5] [![binder][binderlogo]][binder_tut] | 
| 6. Evaluate ML models                            | Use ecologically relevant metrics to test the models                                        | [![Open In Colab][colablogo]][colab_tut_6] [![binder][binderlogo]][binder_tut]   |
| 7. Publish ML models                               | Publish the model to a public repository                                                   | [![Open In Colab][colablogo]][colab_tut_7] [![binder][binderlogo]][binder_tut]  | 
| 8. Analyse Zooniverse classifications             | Pull and analyse up-to-date classifications from Zooniverse and export observations to GBIF             | [![Open In Colab][colablogo]][colab_tut_8] [![binder][binderlogo]][binder_tut] |
| 9. Run ML models on footage                      | Automatically classify new footage and export observations to GBIF                                                          | [![Open In Colab][colablogo]][colab_tut_9] [![binder][binderlogo]][binder_tut] |

  
\* Project-specific notebook

## Local Installation

### Docker Installation
#### Requirements
* [Docker](https://www.docker.com/products/docker-desktop/)

#### Pull KSO Docker image
```
Bash
docker pull ghcr.io/ocean-data-factory-sweden/kso:dev
```

### Conda Installation
#### Requirements
* [Python 3.8](https://www.python.org/)
* [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
* [GIT](https://git-scm.com/downloads)

#### Download this repository
Clone this repository using
```python
git clone https://github.com/ocean-data-factory-sweden/kso.git
``` 

#### Prepare your system
Depending on your system (Windows/Linux/MacOS), you might need to install some extra tools. If this is the case, you will get a message about what you need to install in the next steps. 
For example, [Microsoft Build Tools C++][Microsoft_C++] with a version higher than 14.0 is required for Windows systems.

#### Set up the environment with Conda
1. Open the Anaconda Prompt
2. Navigate to the folder where you have cloned the repository or unzipped the manually downloaded repository. Then go into the kso folder.
```
cd kso
```

3. Create an Anaconda environment with Python 3.8. Remember to change the name env.
```
conda create -n <name env> python=3.8
```

4. Enter the environment: 
```
conda activate <name env>
```

5. Specify your GPU details.

  5a. Find out the [pytorch][pytorch] installation you need. Navigate to the system options (example below) and select your device/platform details.
  <div style="text-align: center;">
    <img src="https://github.com/ocean-data-factory-sweden/kso/blob/dev/assets/cuda_gpu_example_requirements.png?raw=true" alt="CUDA Requirements" width="450" height="150">
  </div>
  
  5b. Add the recommended command to the KSO's gpu_requirements_user.txt file.

6. Install all the requirements:
```
pip install -r requirements.txt -r gpu_requirements_user.txt
```

## Cloudina 
Cloudina is a hosted version of KSO (powered by JupyterHub) on NAISS Science Cloud. It allows users to scale and automate larger workflows using a powerful processing backend. This is currently an invitation-only service. To access the platform, please contact jurie.germishuys[at]combine.se.

The current portals are accessible as:
1. Console (object storage) - [storage][cdn_bucket]
2. Album (JupyterHub) - [notebooks][cdn_album]
3. Vendor (MLFlow) - [mlflow][cdn_vendor]


## Starting a new project
To start a new project you will need to:
1. Create initial information for the database: Input the information about the underwater footage files, sites and species of interest. You can use a [template of the csv files](https://drive.google.com/file/d/1PZGRoSY_UpyLfMhRphMUMwDXw4yx1_Fn/view?usp=sharing) and move the directory to the "db_starter" folder.
2. Link your footage to the database: You will need files of underwater footage to run this system. You can [download some samples](https://drive.google.com/drive/folders/1t2ce8euh3SEU2I8uhiZN1Tu-76ZDqB6w?usp=sharing) and move them to `db_starter`. You can also store your own files and specify their directory in the notebooks.

Please remember the format of the underwater media is standardised (typically .mp4 or .jpg) and the associated metadata captured in three CSV files (“movies”, “sites” and “species”) should follow the [Darwin Core standards (DwC)](https://dwc.tdwg.org/simple/). 

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
You can find out more about the project at https://subsim.wnmedia.se.

We are always excited to collaborate and help other marine scientists. Please feel free to contact us (matthias.obst(at)marine.gu.se) with your questions.

## Troubleshooting

If you experience issues with the Panoptes package and/or uploading movies to Zooniverse, it might be related to the libmagic package. In Windows, the following commands might fix the issue:
```python
pip install python-libmagic
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
[colab_tut_1]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/01_Check_and_update_csv_files.ipynb
[binder_tut]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso/main
[colab_tut_2]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/02_Upload_new_footage.ipynb
[colab_tut_3]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/03_Upload_clips_to_Zooniverse.ipynb
[colab_tut_4]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/04_Upload_frames_to_Zooniverse.ipynb
[colab_tut_5]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/05_Train_ML_models.ipynb
[colab_tut_6]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/06_Evaluate_ML_Models.ipynb
[colab_tut_7]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/07_Transfer_ML_Models.ipynb
[colab_tut_8]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/08_Analyse_Aggregate_Zooniverse_Annotations.ipynb
[colab_tut_9]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/notebooks/09_Run_ML_Models_on_footage.ipynb
[Microsoft_C++]: https://visualstudio.microsoft.com/visual-cpp-build-tools/
[pytorch]: https://pytorch.org/
[YoloV5]: https://github.com/ultralytics/yolov5
[YoloV8]: https://github.com/ultralytics/ultralytics
[cdn_bucket]: https://console.cloudina.org/
[cdn_album]: https://album.cloudina.org/
[cdn_vendor]: https://vendor.cloudina.org/
