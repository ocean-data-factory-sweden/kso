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
        libssl-dev \
        libtool \
        # The next package is needed to support -libx246 for ffmpeg
        libx264-dev \
        libxcb1-dev \
        libxau-dev \
        libxdmcp-dev \
        pkg-config \
        yasm && \
    apt-get clean

# --- Build ffmpeg with CUDA support from source ---
RUN git clone --depth 1 --branch n12.0.16.0 https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install && \
    cd .. && \
    git clone https://git.ffmpeg.org/ffmpeg.git --depth 1 ffmpeg/ && \
    cd ffmpeg && \
    ./configure \
        --enable-nonfree \
        --enable-cuda-nvcc \
        --enable-libnpp \
        --enable-openssl \
        --disable-doc \
        --disable-ffplay \
        # The libx246 encoder is used in the project, therefore we need to enable libx246 and gpl
        --enable-libx264 \
        --enable-gpl \
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
# Update the package lists and install dependencies for OpenCV and others
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        libc6 \
        libmagic1 \
        libgl1 \
        libglib2.0-0 \
        libx264-155 \
        libxau6 \
        libxcb1 \
        libxdmcp6 \
        openssl && \
    apt-get install --no-install-recommends -y wget git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -afy

# Create a conda environment with Python 3.12
RUN /opt/conda/bin/conda create -n myenv python=3.12 -y && \
    /opt/conda/bin/conda clean -afy

# Copy requirements and install packages
COPY requirements.txt /usr/src/app/
RUN /opt/conda/bin/conda run -n myenv pip install --no-cache-dir -r /usr/src/app/requirements.txt && \
    /opt/conda/bin/conda run -n myenv pip uninstall -y opencv-python opencv-contrib-python && \  
    /opt/conda/bin/conda run -n myenv conda install -y -c conda-forge libstdcxx-ng opencv 

# Copy over custom autobackend file
# RUN cp /usr/src/app/kso/src/autobackend.py /opt/conda/envs/myenv/lib/python3.8/site-packages/ultralytics/nn/
## IS THIS NEEDED?

# Clean up unnecessary packages
RUN apt-get remove --autoremove -y wget git && apt-get clean && rm -rf /var/lib/apt/lists/*



# Set environment variables
ENV WANDB_DIR=/mimer/NOBACKUP/groups/snic2021-6-9/ \
    WANDB_CACHE_DIR=/mimer/NOBACKUP/groups/snic2021-6-9/ \
    WANDB_DATA_DIR=/mimer/NOBACKUP/groups/snic2021-6-9/ \
    DATA_DIR=/tmp \
    ARTIFACT_DIR=/tmp \
    PYTHONPATH=$PYTHONPATH:/usr/src/app/kso \
    PATH=/opt/conda/bin:$PATH \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/myenv/lib/

# Set everything up to work with the jupyter notebooks
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} \
    NB_UID=${NB_UID} \
    HOME=/home/${NB_USER}
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
USER ${NB_USER}

# Make sure conda is activated in the entry point
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate myenv && exec \"$@\"", "--"]
