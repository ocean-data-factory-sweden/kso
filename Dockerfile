# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.05-py3 AS build

RUN apt update
RUN apt install -y zip htop screen libgl1-mesa-glx
RUN apt-get autoremove
RUN apt-get clean

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

ADD https://api.github.com/repos/ocean-data-factory-sweden/koster_yolov4/git/refs/heads/master version.json
RUN git clone --recurse-submodules https://github.com/ocean-data-factory-sweden/koster_yolov4.git
# Copy files with minor changes from main repository
RUN cp /usr/src/app/koster_yolov4/src/multi_tracker_zoo.py /usr/src/app/koster_yolov4/yolov5_tracker/trackers/multi_tracker_zoo.py

FROM nvcr.io/nvidia/pytorch:21.05-py3
COPY --from=build /usr/src/app /usr/src/app

RUN python -m pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN python -m pip install --no-cache -r /usr/src/app/requirements.txt coremltools onnx gsutil notebook
RUN python -m pip install --no-cache -r /usr/src/app/koster_yolov4/yolov5_tracker/requirements.txt
RUN python -m pip install --no-cache -r /usr/src/app/koster_yolov4/yolov5_tracker/yolov5/requirements.txt
RUN pip uninstall -y torch torchvision
RUN python -m pip install --no-cache torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install SNIC requirements
RUN jupyter nbextension install --user --py widgetsnbextension
RUN jupyter nbextension install --user --py jupyter_bbox_widget

# Set environment variables
ENV HOME=/usr/src/app/koster_yolov4
ENV WANDB_DIR=/mimer/NOBACKUP/groups/snic2021-6-9/
ENV WANDB_CACHE_DIR=/mimer/NOBACKUP/groups/snic2021-6-9/

## Binder setup

# Create user
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown ${NB_USER} -R ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}

# Ensure widget extensions are activated
RUN jupyter nbextension enable --user --py widgetsnbextension
RUN jupyter nbextension enable --user --py jupyter_bbox_widget
