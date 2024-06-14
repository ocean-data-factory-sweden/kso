This test directory contains the following directories:

a. requirements.txt, to start: use the following command in your terminal: 
      cd /path/to/directory/where/requirements.txt/is/stored
      pip (or pip3) install -r requirements.txt

1. models: a dir to store models
2. notebooks: This is where i stored both the notebooks and the scripts that i worked on. 
                - image_compression.py: script to compress images if u want to upload them to zooniverse
                - Object_detection.ipynb: notebook using roboflow to perform object detection
                - seg_gui.py: python script that has to be copied into vscode for it to work locally. 
                - Segmentation.ipynb: notebook using roboflow to perform segmentation + coverage calculation + mapping out on worldmap
                - yolov8-seg.ipynb: Try-out notebook, currently not functional, tried to use the yolo commands to train, validate and test models. 
3. predictions: This directory stores the results from both the segmentation and object detection workflow
4. runs: in the segment directory, results from running the yolov8-seg.ipynb notebook are stored.
5. Seafloor_footage-1: Dir made after pulling the data from roboflow as a result of running the yolov8-seg.ipynb notebook.
6. wandb: weights and biases directory in which wandb data is stored after running the yolov8-seg.ipynb notebook.
