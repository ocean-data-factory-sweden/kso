# ----------------------------------------------------------------------
#  Stage for building ffmpeg
# ----------------------------------------------------------------------
FROM nvcr.io/nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04 AS ffmpeg-build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
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
        yasm \
        nasm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

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
    make install && \
    make clean


# ----------------------------------------------------------------------
#  Final image
# ----------------------------------------------------------------------
FROM nvcr.io/nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

# Install everything that is needed
# Update the package lists and install dependencies for OpenCV and others
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        python3.12 \
        python3-pip \
        python3.12-venv \
        libc6 \
        libmagic1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
        libx264-164 \
        libxau6 \
        libxcb1 \
        libxdmcp6 \
        openssl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg-build /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg

# Install pip packages directly so that the user can create
# their own venv with --system-site-packages to use the packages in the container
ARG PIP_BREAK_SYSTEM_PACKAGES=1
ARG PIP_ROOT_USER_ACTION=ignore
ARG PIP_NO_CACHE_DIR=1

# Install Python packages
COPY requirements.txt /opt/kso/
COPY requirements_cuda12.9.txt /opt/kso/
RUN pip install -r /opt/kso/requirements.txt -r /opt/kso/requirements_cuda12.9.txt
RUN pip install papermill
