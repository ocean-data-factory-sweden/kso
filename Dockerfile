# Build the ffmpeg with CUDA support from source.
# We need ffmpeg on the system that works with the GPU.
# Only having the python package is not enough. ---
# To build from source we need the devel cuda image.
FROM nvcr.io/nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04 as builder
# So that we are not asked for user input during the build
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -y \
        automake \
        autoconf \
        build-essential \
        git \
        libc6-dev \
        libtool \
        libxcb1-dev \
        libxau-dev \
        libxdmcp-dev \
        pkg-config \
        yasm && \
    apt-get clean

# Build ffmpeg with CUDA support from source
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install && \
    cd .. && \
    git clone https://git.ffmpeg.org/ffmpeg.git --depth 1 ffmpeg/ && \
    cd ffmpeg && \
    ./configure \
        --enable-nonfree \
        --enable-cuda-nvcc \
        --enable-libnpp \
        --disable-doc \
        --disable-ffplay \
        --disable-ffprobe \
        --extra-cflags=-I/usr/local/cuda/include \
        --extra-ldflags=-L/usr/local/cuda/lib64 && \
    make -j 8 && \
    make install

# Start over from the docker image with cuda 12.0
# since we only want the final result from the previous run and we copy that.
# Now we can use the runtime cuda image, since we do not need to build anything
# from scratch. This is better, since the runtime image is smaller
FROM nvcr.io/nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04
COPY --from=builder /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
# So that we are not asked for user input during the build
ARG DEBIAN_FRONTEND=noninteractive

# Create a working directory
WORKDIR /usr/src/app

COPY . ./kso
# Install everything that is needed
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        libc6 \
        libgl1 \
        libxau6 \
        libxcb1 \
        libxdmcp6 && \
    # Install python and git and upgrade pip
    apt-get install --no-install-recommends -y \
        python3.8 \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        vim && \
    apt-get clean && \
    cp \
        /usr/src/app/kso/src/multi_tracker_zoo.py \
        /usr/src/app/kso/yolov5_tracker/trackers/multi_tracker_zoo.py && \
    # Install all python packages, numpy needs to be installed
    # first to avoid the lap build error
    python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 -m pip --no-cache-dir install numpy && \
    python3 -m pip --no-cache-dir install \
        jupyter_contrib_nbextensions==0.7.0 jupyter==1.0.0 av==8.1.0 more-itertools==9.1.0 notebook==6.5.4\
        -r /usr/src/app/kso/yolov5_tracker/requirements.txt \
        -r /usr/src/app/kso/yolov5_tracker/yolov5/requirements.txt \
        -r /usr/src/app/kso/kso_utils/requirements.txt && \
    apt-get remove --autoremove -y git python3-dev build-essential

# Set environment variables
ENV WANDB_DIR=/mimer/NOBACKUP/groups/snic2021-6-9/ \
    WANDB_CACHE_DIR=/mimer/NOBACKUP/groups/snic2021-6-9/ \
    PYTHONPATH=$PYTHONPATH:/usr/src/app/kso

# Set everything up to work with the jupyter notebooks
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} \
    NB_UID=${NB_UID} \
    HOME=/home/${NB_USER}
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} && \
    # Ensure widget extensions are activated
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable --user --py widgetsnbextension && \
    jupyter nbextension enable --user --py jupyter_bbox_widget

USER ${NB_USER}
