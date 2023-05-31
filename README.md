# KSO - Machine Learning models

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
The system processes underwater footage and its associatead metadata into biologically-meaningfull information. The format of the underwater media is standard (.mp4 or .png) and the associated metadata should be captured in three csv files (“movies”, “sites” and “species”) following  the [Darwin Core standards (DwC)](https://dwc.tdwg.org/simple/). 
![koster_info_diag][high-level-overview]

## Module Overview
This Object Detection module contains scripts and resources to train and evaluate object detection models.

![object_detection_module][object_detection_module]

The tutorials enable users to customise [Yolov5][YoloV5] models using Ultralytics. The repository contains both model-specific files (same structure as Ultralytics) as well as specific source files related to Koster pipelines (src folder) and utils (kso_utils). The notebooks rely on the [koster utility functions][koster_utils_repo].

Users can run these tutorials via Google Colab (by clicking on the colab links in the table below), locally  or on SNIC.

### Tutorials
| Name                                              | Description                                                                                 | Try it!  | 
| ------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------|
| 1. Check footage and metadata                     | Check format and contents of footage and sites, media and species csv files                 | [![Open In Colab][colablogo]][colab_tut_1] [![binder][binderlogo]][binder_tut_1] | 
| 2. Upload new media to the system*                | Upload new underwater media to the cloud/server and update the csv files                    | [![Open In Colab][colablogo]][colab_tut_2] [![binder][binderlogo]][binder_tut_2] | 
| 3. Upload clips to Zooniverse                     | Prepare original footage and upload short clips to Zooniverse                               | [![Open In Colab][colablogo]][colab_tut_3] [![binder][binderlogo]][binder_tut_3] |
| 4. Upload frames to Zooniverse                    | Extract frames of interest from original footage and upload them to Zooniverse              | [![Open In Colab][colablogo]][colab_tut_4] [![binder][binderlogo]][binder_tut_4] |
| 5. Train ML models                                | Prepare the training and test data, set model parameters and train models                   | [![Open In Colab][colablogo]][colab_tut_5] [![binder][binderlogo]][binder_tut_5] | 
| 6. Evaluate ML models                            | Use ecologically-relevant metrics to test the models                                        | [![Open In Colab][colablogo]][colab_tut_6] [![binder][binderlogo]][binder_tut_6]   |
| 7. Publish ML models                               | Publish the model to a public repository                                                   | [![Open In Colab][colablogo]][colab_tut_7] [![binder][binderlogo]][binder_tut_7]  | 
| 8. Analyse Zooniverse classifications             | Pull up-to-date classifications from Zooniverse and report summary stats/graphs             | [![Open In Colab][colablogo]][colab_tut_8] [![binder][binderlogo]][binder_tut_8] |
| 9. Download and format Zooniverse classifications | Pull up-to-date classifications from Zooniverse and format them for further analysis        | Coming soon  | 
| 10. Run ML models on footage                      | Automatically classify new footage                                                          | Coming soon  |

  
\* Project-specific tutorial

## Local Installation
If you want to fully use our system (Binder has computing limitations), you will need to download this repository on your local computer or server.

### Requirements
* [Python 3.8](https://www.python.org/)
* [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
* [GIT](https://git-scm.com/downloads)


#### Download this repository
Clone this repository using
```python
git clone --recurse-submodules https://github.com/ocean-data-factory-sweden/kso-object-detection.git
``` 
#### Prepare your system
* Windows: If you will run it on a windows computer, you need to install the microsoft build tools C++ with a version that is higher than 14.0. You can install it from https://visualstudio.microsoft.com/visual-cpp-build-tools/. In the install menu, you only need to select the "Windows <your version> SDK".
* Linux: ...upcoming...
* Apple: ...upcominb...

#### Set up the environment with conda
1. Open the anaconda prompt
2. Navigate to the folder where you have cloned the repository or unzipped the manually downloaded repository. Then go into the kso-object-detection folder. (```cd kso-object-detection ```)
3. Create an anaconda environment with python 3.8:

```conda create -n <name env> python=3.8 ```

4. Enter the environment: 

```conda activate <name env>```

5. Install numpy to prevent an error that will otherwise occur in the next step.

```pip install numpy==1.22```

6. Install all the requirements. Do you NOT have a GPU, run the following:

```pip install -r yolov5_tracker/requirements.txt -r yolov5_tracker/yolov5/requirements.txt -r kso_utils/requirements.txt```

Do you have a GPU? Find here (https://pytorch.org/) which pytorch installation you need depending on your device and cuda version. Add the recommended command to the gpu_requirements_user.txt in the same way as the current example is added. Then run:

```pip install -r yolov5_tracker/requirements.txt -r yolov5_tracker/yolov5/requirements.txt -r kso_utils/requirements.txt -r gpu_requirements_user.txt```


#### Set up the environment with another virtual environment package
If you are using another virtual environment package, install the same requirements inside your fresh environment (Python 3.8).


#### Link your environment to jupyter notebooks
After installing all the requirements, run from your environment the following command:

```ipython kernel install --user --name=<name env>```

Now open the jupyter notebook and select/change kernel to run the notebooks from your environment.

## SNIC Users (VPN required)

**Before using Option 2, users should have login credentials and have setup the Chalmers VPN on their local computers**

Information for Windows users: [Click here](https://it.portal.chalmers.se/itportal/NonCDAWindows/VPN)
Information for MAC users: [Click here](https://it.portal.chalmers.se/itportal/NonCDAMac/VPN)

To use the Jupyter Notebooks within the Alvis HPC cluster, please visit [Alvis Portal](https://portal.c3se.chalmers.se) and login using your SNIC credentials. 

Once you have been authorized, click on "Interactive Apps" and then "Jupyter". This open the server creation options. 

Creating a Jupyter session requires a custom environment file, which is available on our shared drive */mimer/NOBACKUP/groups/snic2022-22-1210/jupter_envs*. Please copy these files to your **Home Directory** in order to use the custom environments we have created.

Here you can keep the settings as default, apart from the "Number of hours" which you can set to the desired limit. Then choose either **Data Management (Runtime (User specified jupyter1.sh))** or **Machine Learning (Runtime (User specified jupyter2.sh))** from the Runtime dropdown options.

![screenshot_load][screenshot_loading]

This will directly queue a server session using the correct container image, first showing a blue window and then you should see a green window when the session has been successfully started and the button **"Connect to Jupyter"** appears on the screen. Click this to launch into the Jupyter notebook environment. 


![screenshot_start][screenshot_started]

Important note: The remaining time for the server is shown in green window as well. If you have finished using the notebook server before the alloted time runs out, please select **"Delete"** so that the resources can be released for use by others within the project. 

## Starting a new project
If you will work in a new project you will need to:
1. Create initial information for the database: Input the information about the underwater footage files, sites and species of interest. You can use a [template of the csv files](https://drive.google.com/file/d/1PZGRoSY_UpyLfMhRphMUMwDXw4yx1_Fn/view?usp=sharing) and move the directory to the "db_starter" folder.
2. Link your footage to the database: You will need files of underwater footage to run this system. You can [download some samples](https://drive.google.com/drive/folders/1t2ce8euh3SEU2I8uhiZN1Tu-76ZDqB6w?usp=sharing) and move them to `db_starter`. You can also store your own files and specify their directory in the tutorials.


## Developer instructions
If you would like to expand and improve the KSO cpabilities, pleas follow the instructions above to set the project up on your own local computer.

When you start adding changes, please create your own branch on top of dev. Before submitting a Merge Request, please:
* run Black on the code you have edited 
```shell
black filename 
```
* update the timestamp on the notebook
```shell
python update_ts.py filename 
```
Remember to follow the [conventional commits guidelines](https://www.conventionalcommits.org/en/v1.0.0/) to facilitate code sharing. 


## Citation

If you use this code or its models in your research, please cite:

Anton V, Germishuys J, Bergström P, Lindegarth M, Obst M (2021) An open-source, citizen science and machine learning approach to analyse subsea movies. Biodiversity Data Journal 9: e60548. https://doi.org/10.3897/BDJ.9.e60548

## Collaborations/questions
You can find out more about the project at https://www.zooniverse.org/projects/victorav/the-koster-seafloor-observatory.

We are always excited to collaborate and help other marine scientists. Please feel free to [contact us](matthias.obst@marine.gu.se) with your questions.

## Troubleshooting

If you experience issues uploading movies to Zooniverse, it might be related to the libmagic package. In Windows, the following commands seem to work:
```python
pip install python-libmagic
pip install python-magic-bin
```

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ocean-data-factory-sweden/kso-object-detection.svg?style=for-the-badge
[contributors-url]: https://https://github.com/ocean-data-factory-sweden/kso-object-detection/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ocean-data-factory-sweden/kso-object-detection.svg?style=for-the-badge
[forks-url]: https://github.com/ocean-data-factory-sweden/kso-object-detection/network/members
[stars-shield]: https://img.shields.io/github/stars/ocean-data-factory-sweden/kso-object-detection.svg?style=for-the-badge
[stars-url]: https://github.com/ocean-data-factory-sweden/kso-object-detection/stargazers
[issues-shield]: https://img.shields.io/github/issues/ocean-data-factory-sweden/kso-object-detection.svg?style=for-the-badge
[issues-url]: https://github.com/ocean-data-factory-sweden/kso-object-detection/issues
[license-shield]: https://img.shields.io/github/license/ocean-data-factory-sweden/kso-object-detection.svg?style=for-the-badge
[license-url]: https://github.com/ocean-data-factory-sweden/kso-object-detection/blob/main/LICENSE.txt
[high-level-overview]: https://github.com/ocean-data-factory-sweden/kso-data-management/blob/main/images/high-level-overview-2.png?raw=true "Overview of the three main modules and the components of the Koster Seafloor Observatory"
[Data_management_module]: https://github.com/ocean-data-factory-sweden/kso-data-management/blob/main/images/Koster_data_management_module.png?raw=true
[object_detection_module]: https://github.com/ocean-data-factory-sweden/kso-data-management/blob/main/images/Koster_object_detection_module.png?raw=true
[koster_utils_repo]: https://github.com/ocean-data-factory-sweden/kso_utils
[colablogo]: https://colab.research.google.com/assets/colab-badge.svg
[binderlogo]: https://mybinder.org/badge_logo.svg
[colab_tut_1]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso-data-management/blob/main/tutorials/01_Check_and_update_csv_files.ipynb
[binder_tut_1]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso-data-management/main
[colab_tut_2]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso-data-management/blob/main/tutorials/02_Upload_new_footage.ipynb
[binder_tut_2]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso-data-management/main
[colab_tut_3]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso-data-management/blob/main/tutorials/03_Upload_clips_to_Zooniverse.ipynb
[binder_tut_3]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso-data-management/main
[colab_tut_4]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso-data-management/blob/main/tutorials/04_Upload_frames_to_Zooniverse.ipynb
[binder_tut_4]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso-data-management/main
[colab_tut_5]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso-object-detection/blob/master/tutorials/05_Train_ML_models.ipynb
[binder_tut_5]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso-data-management/main
[colab_tut_6]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso-object-detection/blob/master/tutorials/06_Evaluate_ML_Models.ipynb
[binder_tut_6]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso-data-management/main
[colab_tut_7]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso-object-detection/blob/master/tutorials/07_Transfer_ML_Models.ipynb
[binder_tut_7]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso-data-management/main
[colab_tut_8]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso-data-management/blob/main/tutorials/08_Analyse_Aggregate_Zooniverse_Annotations.ipynb
[binder_tut_8]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso-data-management/main
[colab_tut_11]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso-data-management/blob/main/tutorials/11_Concatenate_videos_from_AWS.ipynb
[binder_tut_11]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso-data-management/main
[colab_tut_12]: https://colab.research.google.com/github/ocean-data-factory-sweden/kso-data-management/blob/main/tutorials/12_Display_movies_available_on_the_server.ipynb
[binder_tut_12]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/kso-data-management/main
[objdecmodule]: https://github.com/ocean-data-factory-sweden/kso-object-detection
[YoloV5]: https://github.com/ultralytics/yolov5
[OBIS-site]: https://www.gbif.org/network/2b7c7b4f-4d4f-40d3-94de-c28b6fa054a6
[Koster_info_diagram]: https://github.com/ocean-data-factory-sweden/kso-data-management/blob/main/images/Koster_information_flow.png?raw=true "Information architecture of the Koster Seafloor Observatory"
[screenshot_loading]: https://github.com/ocean-data-factory-sweden/kso-data-management/blob/main/images/screenshot_loading.png?raw=true
[screenshot_started]: https://github.com/ocean-data-factory-sweden/kso-data-management/blob/main/images/screenshot_started.png?raw=true
