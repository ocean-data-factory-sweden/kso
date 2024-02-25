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

### KSO Information architecture
The system processes underwater footage and its associated metadata into biologically meaningful information. The format of the underwater media is standardised (typically .mp4 or .jpg) and the associated metadata should be captured in three csv files (“movies”, “sites” and “species”) following  the [Darwin Core standards (DwC)](https://dwc.tdwg.org/simple/). 
![koster_info_diag][high-level-overview2]

## Repository Overview
This repository contains scripts and resources for:
* move and process underwater footage and its associated data (e.g. location, date, sampling device).
* make this data available for citizen science to help you with annotating the data.
* train and evaluate machine learning models. (customise [Yolov5][YoloV5] or [Yolov8][YoloV8] models using Ultralytics.)

![high-level][high-level-overview]

The system is built around a series of easy-to-use Jupyter Notebook tutorials. Each tutorial allows users to perform a specific task of the system (e.g. upload footage to the citizen science platform or analyse the classified data).

Users can run these tutorials via Google Colab (by clicking on the Colab links in the table below), locally or on a High-Performance Computer environment.

### Tutorials
| Name                                              | Description                                                                                 | Try it!  | 
| ------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------|
| 1. Check footage and metadata                     | Check format and contents of footage and sites, media and species csv files                 | [![Open In Colab][colablogo]][colab_tut_1] [![binder][binderlogo]][binder_tut] | 
| 2. Upload new media to the system*                | Upload new underwater media to the cloud/server and update the csv files                    | WIP | 
| 3. Upload clips to Zooniverse                     | Prepare original footage and upload short clips to Zooniverse                               | [![Open In Colab][colablogo]][colab_tut_3] [![binder][binderlogo]][binder_tut] |
| 4. Upload frames to Zooniverse                    | Extract frames of interest from the original footage and upload them to Zooniverse              | [![Open In Colab][colablogo]][colab_tut_4] [![binder][binderlogo]][binder_tut] |
| 5. Train ML models                                | Prepare the training and test data, set model parameters and train models                   | [![Open In Colab][colablogo]][colab_tut_5] [![binder][binderlogo]][binder_tut] | 
| 6. Evaluate ML models                            | Use ecologically relevant metrics to test the models                                        | [![Open In Colab][colablogo]][colab_tut_6] [![binder][binderlogo]][binder_tut]   |
| 7. Publish ML models                               | Publish the model to a public repository                                                   | [![Open In Colab][colablogo]][colab_tut_7] [![binder][binderlogo]][binder_tut]  | 
| 8. Analyse Zooniverse classifications             | Pull up-to-date classifications from Zooniverse and report summary stats/graphs             | [![Open In Colab][colablogo]][colab_tut_8] [![binder][binderlogo]][binder_tut] |
| 9. Run ML models on footage                      | Automatically classify new footage                                                          | [![Open In Colab][colablogo]][colab_tut_9] [![binder][binderlogo]][binder_tut] |

  
\* Project-specific tutorial

## Local Installation
If you want to fully use our system (Binder has computing limitations), you will need to download this repository on your local computer or server. (Or use SNIC or Cloudina, see instructions below)
Note that depending on your choice of infrastructure, you will be limited to either [Yolov5][YoloV5] or [Yolov8][YoloV8]:
* Locally it is possible to either use Yolov5 or Yolov8.
* SNIC: only possible to use Yolov5.
* Cloudina: Only possible to use Yolov8.

The latest developments are only available in combination with Yolov8. However, there is a stable tagged [yolov5] (https://github.com/ocean-data-factory-sweden/kso/yolov5) version if you prefer Yolov5. 

### Local installation with Yolov5
Requirements
* [Python 3.8](https://www.python.org/)
* [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
* [GIT](https://git-scm.com/downloads)


#### Download this repository
Clone this repository using
```python
git clone --recurse-submodules --depth 1 --branch yolov5 https://github.com/ocean-data-factory-sweden/kso.git
``` 

#### Prepare your system
Depending on which system you are using (Windows/Linux/MacOS), you might need to install some extra tools. If this is the case, you will get a message about what you need to install in the next steps. 
For example, on a Windows system, it will request you to install the Microsoft Build Tools C++ with a version higher than 14.0. You can install it from https://visualstudio.microsoft.com/visual-cpp-build-tools/. You only need to select the "Windows <your version> SDK" in the install menu.

#### Set up the environment with Conda
1. Open the Anaconda Prompt
2. Navigate to the folder where you have cloned the repository or unzipped the manually downloaded repository. Then go into the kso folder. (```cd kso ```)
3. Create an Anaconda environment with Python 3.8:

```conda create -n <name env> python=3.8 ```

4. Enter the environment: 

```conda activate <name env>```

5. Install numpy to prevent an error that will otherwise occur in the next step.

```pip install numpy==1.22```

6. Install all the requirements. If you do not have a GPU, run the following:

```pip install -r yolov5_tracker/requirements.txt -r yolov5_tracker/yolov5/requirements.txt -r requirements.txt```

Have a GPU? Find out which pytorch installation you need here (https://pytorch.org/), depending on your device and CUDA version. Add the recommended command to the gpu_requirements_user.txt file in the same way as the current example. Then run:

```pip install -r yolov5_tracker/requirements.txt -r yolov5_tracker/yolov5/requirements.txt -r requirements.txt -r gpu_requirements_user.txt```


#### Set up the environment with another virtual environment package
If using another virtual environment package, install the same requirements inside your fresh environment (Python 3.8).


#### Link your environment to Jupyter notebooks
After installing all the requirements, run the following command in your environment:

```ipython kernel install --user --name=<name env>```

Now open the Jupyter notebook and select/change the kernel to run the notebooks from your environment.

### Local installation with Yolov8
These instructions will be provided once a stable version with Yolov8 is achieved. 

## SNIC Users (VPN required)

**Before using the VPN to connect to SNIC, users should have login credentials and set up the Chalmers VPN on their local computers**

Instructions to [set up the Chalmers VPN](https://www.c3se.chalmers.se/documentation/connecting/#vpn)

To use the Jupyter Notebooks within the Alvis HPC cluster, please visit [Alvis Portal](https://portal.c3se.chalmers.se) and log in using your SNIC credentials. 

Once you have been authorized, click on "Interactive Apps" and then "Jupyter". This will open the server creation options. 

Creating a Jupyter session requires a custom environment file, which is available on our shared drive */mimer/NOBACKUP/groups/snic2022-22-1210/jupter_envs*. Please copy this file (jupyter-kso.sh) to your **Home Directory** to use the custom environment we have created.

Here you can keep the settings as default, apart from the "Number of hours" which you can set to the desired limit. Then choose kso-jupyter.sh from the Runtime dropdown options.

![screenshot_load][screenshot_loading]

This will directly queue a server session using the correct container image, first showing a blue window and then you should see a green window when the session has been successfully started and the button **"Connect to Jupyter"** appears on the screen. Click this to launch into the Jupyter Notebook environment. 


![screenshot_start][screenshot_started]

Important note: The remaining time for the server is shown in the green window as well. If you have finished using the notebook server before the allocated time runs out, please select **"Delete"** so that the resources can be released for use by others within the project. 

## Cloudina 
Instructions will come...


## Starting a new project
If you will work on a new project you will need to:
1. Create initial information for the database: Input the information about the underwater footage files, sites and species of interest. You can use a [template of the csv files](https://drive.google.com/file/d/1PZGRoSY_UpyLfMhRphMUMwDXw4yx1_Fn/view?usp=sharing) and move the directory to the "db_starter" folder.
2. Link your footage to the database: You will need files of underwater footage to run this system. You can [download some samples](https://drive.google.com/drive/folders/1t2ce8euh3SEU2I8uhiZN1Tu-76ZDqB6w?usp=sharing) and move them to `db_starter`. You can also store your own files and specify their directory in the tutorials.


## Developer instructions
If you would like to expand and improve the KSO capabilities, please follow the instructions above to set the project up on your local computer.

When you start adding changes, please create your branch on top of the current 'dev' branch. Before submitting a Merge Request, please:
* Run Black on the code you have edited 
```shell
black filename 
```
* Clean up your commit history on your branch, so that every commit represents a logical change. (so squash and edit commits so that it is understandable for others)
* For the commit messages, we ask that you please follow the [conventional commits guidelines](https://www.conventionalcommits.org/en/v1.0.0/) (table below) to facilitate code sharing. Also, please describe the logic behind the commit in the body of the message.
  ## Commit types

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
You can find out more about the project at https://www.zooniverse.org/projects/victorav/the-koster-seafloor-observatory.

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
[high-level-overview2]: https://github.com/ocean-data-factory-sweden/kso/blob/main/assets/high-level-overview-2.png?raw=true "Overview of the three main modules and the components of the Koster Seafloor Observatory"
[high-level-overview]: https://github.com/ocean-data-factory-sweden/kso/blob/main/assets/high-level-overview.png?raw=true "Overview of the three main modules and the components of the Koster Seafloor Observatory"
[Data_management_module]: https://github.com/ocean-data-factory-sweden/kso/blob/main/assets/Koster_data_management_module.png?raw=true
[object_detection_module]: https://github.com/ocean-data-factory-sweden/kso/blob/main/assets/Koster_object_detection_module.png?raw=true
[koster_utils_repo]: https://github.com/ocean-data-factory-sweden/kso_utils
[colablogo]: https://colab.research.google.com/assets/colab-badge.svg
[binderlogo]: https://mybinder.org/badge_logo.svg
[colab_tut_1]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/tutorials/01_Check_and_update_csv_files.ipynb
[binder_tut]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso/main
[colab_tut_2]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/tutorials/02_Upload_new_footage.ipynb
[colab_tut_3]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/tutorials/03_Upload_clips_to_Zooniverse.ipynb
[colab_tut_4]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/tutorials/04_Upload_frames_to_Zooniverse.ipynb
[colab_tut_5]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/tutorials/05_Train_ML_models.ipynb
[colab_tut_6]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/tutorials/06_Evaluate_ML_Models.ipynb
[colab_tut_7]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/tutorials/07_Transfer_ML_Models.ipynb
[colab_tut_8]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/tutorials/08_Analyse_Aggregate_Zooniverse_Annotations.ipynb
[colab_tut_9]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso/blob/dev/tutorials/09_Run_ML_Models_on_footage.ipynb
[objdecmodule]: https://github.com/ocean-data-factory-sweden/kso
[YoloV5]: https://github.com/ultralytics/yolov5
[YoloV8]: https://github.com/ultralytics/ultralytics
[OBIS-site]: https://www.gbif.org/network/2b7c7b4f-4d4f-40d3-94de-c28b6fa054a6
[Koster_info_diagram]: https://github.com/ocean-data-factory-sweden/kso/blob/main/assets/Koster_information_flow.png?raw=true "Information architecture of the Koster Seafloor Observatory"
[screenshot_loading]: https://github.com/ocean-data-factory-sweden/kso/blob/main/assets/screenshot_loading.png?raw=true
[screenshot_started]: https://github.com/ocean-data-factory-sweden/kso/blob/main/assets/screenshot_started.png?raw=true
