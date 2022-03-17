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

![high-level][high-level-overview]

## Module Overview
This Object Detection module contains scripts and resources to train and evaluate Object Detection models. 

![object_detection_module][object_detection_module]
 
 
The tutorials enable users to customise [Yolov5][YoloV5] models using Ultralytics. The repository contains both model-specific files (same structure as Ultralytics) as well as specific source files related to Koster pipelines (src folder) and utils (tutorial_utils).

## Installation

### Requirements
* [Python 3.7+](https://www.python.org/)
* [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
* [GIT](https://git-scm.com/downloads)

#### Download this repository
Clone this repository using
```python
git clone --recurse-submodules https://github.com/ocean-data-factory-sweden/koster_data_management.git
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

WIP

## Citation

If you use this code or its models in your research, please cite:

Anton V, Germishuys J, Bergström P, Lindegarth M, Obst M (2021) An open-source, citizen science and machine learning approach to analyse subsea movies. Biodiversity Data Journal 9: e60548. https://doi.org/10.3897/BDJ.9.e60548

## Collaborations/questions
You can find out more about the project at https://www.zooniverse.org/projects/victorav/the-koster-seafloor-observatory.

We are always excited to collaborate and help other marine scientists. Please feel free to [contact us](matthias.obst@marine.gu.se) with your questions.




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ocean-data-factory-sweden/koster_yolov4.svg?style=for-the-badge
[contributors-url]: https://https://github.com/ocean-data-factory-sweden/koster_yolov4/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ocean-data-factory-sweden/koster_yolov4.svg?style=for-the-badge
[forks-url]: https://github.com/ocean-data-factory-sweden/koster_yolov4/network/members
[stars-shield]: https://img.shields.io/github/stars/ocean-data-factory-sweden/koster_yolov4.svg?style=for-the-badge
[stars-url]: https://github.com/ocean-data-factory-sweden/koster_yolov4/stargazers
[issues-shield]: https://img.shields.io/github/issues/ocean-data-factory-sweden/koster_yolov4.svg?style=for-the-badge
[issues-url]: https://github.com/ocean-data-factory-sweden/koster_yolov4/issues
[license-shield]: https://img.shields.io/github/license/ocean-data-factory-sweden/koster_yolov4.svg?style=for-the-badge
[license-url]: https://github.com/ocean-data-factory-sweden/koster_yolov4/blob/main/LICENSE.txt
[high-level-overview]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/high-level-overview.png?raw=true "Overview of the three main modules and the components of the Koster Seafloor Observatory"
[YoloV5]: https://github.com/ultralytics/yolov5
[object_detection_module]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/Koster_object_detection_module.png?raw=true
[objdecmodule]: https://github.com/ocean-data-factory-sweden/koster_yolov4
[OBIS-site]: https://www.gbif.org/network/2b7c7b4f-4d4f-40d3-94de-c28b6fa054a6
[Database_diagram]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/Database_diagram.png?raw=true "Entity relationship diagram of the SQLite database of the Koster Seafloor Observatory"
[binderlogo]: https://mybinder.org/badge_logo.svg
[binderlink]: https://mybinder.org/v2/gh/ocean-data-factory-sweden/koster_data_management/main
[screenshot_loading]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/screenshot_loading.png?raw=true
[screenshot_started]: https://github.com/ocean-data-factory-sweden/koster_data_management/blob/main/images/screenshot_started.png?raw=true

