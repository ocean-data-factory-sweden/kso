# KSO - Object Detection

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
[![MIT License][license-shield]][license-url]

### KSO Information architecture
The system processes underwater footage and its associatead metadata into biologically-meaningfull information. The format of the underwater media is standard (.mp4 or .png) and the associated metadata should be captured in three csv files (“movies”, “sites” and “species”) following  the [Darwin Core standards (DwC)](https://dwc.tdwg.org/simple/). 
![koster_info_diag][high-level-overview]

## Module Overview
This Object Detection module contains scripts and resources to train and evaluate object detection models.

![object_detection_module][object_detection_module]

The tutorials enable users to customise [Yolov5][YoloV5] models using Ultralytics. The repository contains both model-specific files (same structure as Ultralytics) as well as specific source files related to Koster pipelines (src folder) and utils (kso_utils). It is not recommended to simply clone this repository as many dependencies are resolved using the supplied Dockerfile. The notebooks rely on the [koster utility functions][koster_utils_repo].

### Tutorials
| Name                                              | Description                                                                                 | Try it!  | 
| ------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------|
| 1. Check footage and metadata                     | Check format and contents of footage and sites, media and species csv files                 | [![Open In Colab][colablogo]][colab_tut_1] [![binder][binderlogo]][binder_tut_1] | 
| 2. Upload new media to the system*                | Upload new underwater media to the cloud/server and update the csv files                    | [![Open In Colab][colablogo]][colab_tut_2] [![binder][binderlogo]][binder_tut_2] | 
| 3. Upload clips to Zooniverse                     | Prepare original footage and upload short clips to Zooniverse                               | [![Open In Colab][colablogo]][colab_tut_3] [![binder][binderlogo]][binder_tut_3] |
| 4. Upload frames to Zooniverse                    | Extract frames of interest from original footage and upload them to Zooniverse              | [![Open In Colab][colablogo]][colab_tut_4] [![binder][binderlogo]][binder_tut_4] |
| 5. Train ML models                                | Prepare the training and test data, set model parameters and train models                   | [![Open In Colab][colablogo]][colab_tut_5] [![binder][binderlogo]][binder_tut_5] | 
| 6. Evaluate ML models                            | Use ecologically-relevant metrics to test the models                                        | [![Open In Colab][colablogo]][colab_tut_6] [![binder][binderlogo]][binder_tut_6]   |
| 7. Publish ML models                               | Publish the model to a public repository                                                   | Coming soon | 
| 8. Analyse Zooniverse classifications             | Pull up-to-date classifications from Zooniverse and report summary stats/graphs             | [![Open In Colab][colablogo]][colab_tut_8] [![binder][binderlogo]][binder_tut_8] |
| 9. Download and format Zooniverse classifications | Pull up-to-date classifications from Zooniverse and format them for further analysis        | Coming soon  | 
| 10. Run ML models on footage                      | Automatically classify new footage                                                          | Coming soon  |

  
\* Project-specific tutorial

## Dev Installation
If you want to fully use our system (Binder has computing limitations), you will need to download this repository on your local computer or server.

### Requirements
* [Python 3.7+](https://www.python.org/)
* [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
* [GIT](https://git-scm.com/downloads)

### Option 1: Local / Cloud Installation
-----------------
#### Download this repository
Clone this repository using
```python
git clone --recurse-submodules https://github.com/ocean-data-factory-sweden/koster_yolov4.git
``` 

#### Install dependecies
Navigate to the folder where you have cloned the repository or unzipped the manually downloaded repository. 
```python
cd koster_yolov4
```

Then install the requirements by running.
```python
pip install -r requirements.txt
```
### Option 2: SNIC Users (VPN required)

-----------------

**Before using Option 2, users should have login credentials and have setup the Chalmers VPN on their local computers**

Information for Windows users: [Click here](https://it.portal.chalmers.se/itportal/NonCDAWindows/VPN)
Information for MAC users: [Click here](https://it.portal.chalmers.se/itportal/NonCDAMac/VPN)

To use the Jupyter Notebooks within the Alvis HPC cluster, please visit [Alvis Portal](https://portal.c3se.chalmers.se) and login using your SNIC credentials. 

Once you have been authorized, click on "Interactive Apps" and then "Jupyter". This open the server creation options. 

Here you can keep the settings as default, apart from the "Number of hours" which you can set to the desired limit. Then choose either **Data Management (Runtime (User specified jupyter1.sh))** or **Machine Learning (Runtime (User specified jupyter2.sh))** from the Runtime dropdown options.

![screenshot_load][screenshot_loading]

This will directly queue a server session using the correct container image, first showing a blue window and then you should see a green window when the session has been successfully started and the button **"Connect to Jupyter"** appears on the screen. Click this to launch into the Jupyter notebook environment. 


![screenshot_start][screenshot_started]

Important note: The remaining time for the server is shown in green window as well. If you have finished using the notebook server before the alloted time runs out, please select **"Delete"** so that the resources can be released for use by others within the project. 


## Citation

If you use this code or its models in your research, please cite:

Anton V, Germishuys J, Bergström P, Lindegarth M, Obst M (2021) An open-source, citizen science and machine learning approach to analyse subsea movies. Biodiversity Data Journal 9: e60548. https://doi.org/10.3897/BDJ.9.e60548

## Collaborations/questions
You can find out more about the project at https://www.zooniverse.org/projects/victorav/the-koster-seafloor-observatory.

We are always excited to collaborate and help other marine scientists. Please feel free to [contact us](matthias.obst@marine.gu.se) with your questions.

## Dev instructions

- Installing conda 
- Create new environment (e.g. "new environment")
- Install git and pip (with conda)
- Clone kso repo
- pip install ipykernel
- python -m ipykernel install --user --name="new_environment"
- from the jupyter notebook select kernel/change kernel


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ocean-data-factory-sweden/koster_data_management.svg?style=for-the-badge
[contributors-url]: https://https://github.com/ocean-data-factory-sweden/koster_data_management/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ocean-data-factory-sweden/koster_data_management.svg?style=for-the-badge
[forks-url]: https://github.com/ocean-data-factory-sweden/koster_data_management/network/members
[stars-shield]: https://img.shields.io/github/stars/ocean-data-factory-sweden/koster_data_management.svg?style=for-the-badge
[stars-url]: https://github.com/ocean-data-factory-sweden/koster_data_management/stargazers
[issues-shield]: https://img.shields.io/github/issues/ocean-data-factory-sweden/koster_data_management.svg?style=for-the-badge
[issues-url]: https://github.com/ocean-data-factory-sweden/koster_data_management/issues
[license-shield]: https://img.shields.io/github/license/ocean-data-factory-sweden/koster_data_management.svg?style=for-the-badge
[license-url]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/LICENSE.txt
[high-level-overview]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/high-level-overview-2.png?raw=true "Overview of the three main modules and the components of the Koster Seafloor Observatory"
[Data_management_module]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/Koster_data_management_module.png?raw=true
[object_detection_module]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/Koster_object_detection_module.png?raw=true
[koster_utils_repo]: https://github.com/ocean-data-factory-sweden/kso_utils
[colablogo]: https://colab.research.google.com/assets/colab-badge.svg
[binderlogo]: https://mybinder.org/badge_logo.svg
[colab_tut_1]: https://colab.research.google.com/github/ocean-data-factory-sweden/koster_data_management/blob/main/tutorials/01_Check_and_update_csv_files.ipynb
[binder_tut_1]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/koster_data_management/main
[colab_tut_2]: https://colab.research.google.com/github/ocean-data-factory-sweden/koster_data_management/blob/main/tutorials/02_Upload_new_footage.ipynb
[binder_tut_2]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/koster_data_management/main
[colab_tut_3]: https://colab.research.google.com/github/ocean-data-factory-sweden/koster_data_management/blob/main/tutorials/03_Upload_clips_to_Zooniverse.ipynb
[binder_tut_3]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/koster_data_management/main
[colab_tut_4]: https://colab.research.google.com/github/ocean-data-factory-sweden/koster_data_management/blob/main/tutorials/04_Upload_frames_to_Zooniverse.ipynb
[binder_tut_4]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/koster_data_management/main
[colab_tut_5]: https://colab.research.google.com/github/ocean-data-factory-sweden/koster_yolov4/blob/master/tutorials/5_Train_YOLO_models.ipynb
[binder_tut_5]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/koster_data_management/main
[colab_tut_6]: https://colab.research.google.com/github/ocean-data-factory-sweden/koster_yolov4/blob/master/tutorials/6_Evaluate_ML_Models.ipynb
[binder_tut_6]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/koster_data_management/main
[colab_tut_8]: https://colab.research.google.com/github/ocean-data-factory-sweden/koster_data_management/blob/main/tutorials/08_Analyse_Aggregate_Zooniverse_Annotations.ipynb
[binder_tut_8]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/koster_data_management/main
[colab_tut_11]: https://colab.research.google.com/github/ocean-data-factory-sweden/koster_data_management/blob/main/tutorials/11_Concatenate_videos_from_AWS.ipynb
[binder_tut_11]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/koster_data_management/main
[colab_tut_12]: https://colab.research.google.com/github/ocean-data-factory-sweden/koster_data_management/blob/main/tutorials/12_Display_movies_available_on_the_server.ipynb
[binder_tut_12]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/koster_data_management/main
[objdecmodule]: https://github.com/ocean-data-factory-sweden/koster_yolov4
[YoloV5]: https://github.com/ultralytics/yolov5
[OBIS-site]: https://www.gbif.org/network/2b7c7b4f-4d4f-40d3-94de-c28b6fa054a6
[Koster_info_diagram]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/Koster_information_flow.png?raw=true "Information architecture of the Koster Seafloor Observatory"
[screenshot_loading]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/screenshot_loading.png?raw=true
[screenshot_started]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/screenshot_started.png?raw=true
