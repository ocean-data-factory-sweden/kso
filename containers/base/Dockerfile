ARG BASE_IMAGE=docker.io/ubuntu:latest
ARG FFMPEG_BASE_IMAGE=docker.io/ubuntu:latest

# ----------------------------------------------------------------------
#  Stage for building ffmpeg
# ----------------------------------------------------------------------
FROM ${FFMPEG_BASE_IMAGE} AS ffmpeg-build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        automake \
        autoconf \
        build-essential \
        ca-certificates \
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
        nasm \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Build ffmpeg with CUDA support from source
# TODO: Add ROCm support
ARG CUDA_VERSION=false
RUN mkdir -p /opt/ffmpeg && \
    ffmpeg_config_flags="\
        --enable-openssl --enable-version3 \
        --disable-doc \
        --disable-ffplay \
        --enable-libx264 \
        --enable-gpl \
    "; \
    if [ "$CUDA_VERSION" != "false" ] ; then \
      # Install nv codec headers if CUDA is enabled
      git clone -b n12.0.16.1 --depth 1 https://github.com/FFmpeg/nv-codec-headers.git && \
      cd nv-codec-headers && \
      make install && \
      cd .. && \
      rm -rf nv-codec-headers && \
      ffmpeg_config_flags="$ffmpeg_config_flags \
          --enable-nonfree \
          --enable-cuda-nvcc \
          --enable-libnpp \
          --extra-cflags=-I/usr/local/cuda/include \
          --extra-ldflags=-L/usr/local/cuda/lib64 \
      "; \
    fi && \
    git clone -b n8.0.1 --depth 1 https://github.com/FFmpeg/FFmpeg.git ffmpeg && \
    cd ffmpeg && \
    echo "ffmpeg configure flags: $ffmpeg_config_flags" && \
    ./configure $ffmpeg_config_flags && \
    make -j $(nproc) && \
    make install && \
    make clean && \
    cd .. && \
    rm -rf ffmpeg


# ----------------------------------------------------------------------
#  Final image
# ----------------------------------------------------------------------
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        # Python
        python3.12 \
        python3-pip \
        python3.12-venv \
        # General utilities
        adduser \
        git \
        ca-certificates \
        wget \
        # For ffmpeg
        openssl \
        libc6 \
        libx264-164 \
        libxau6 \
        libxcb1 \
        libxdmcp6 \
        # For OpenCV
        libgl1 \
        libglib2.0-0 \
        # Are these needed?
        # libmagic1 \
        # libsm6 \
        # libxrender1 \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg-build /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg

# Install pip packages directly so that the user can create
# their own venv with --system-site-packages to use the packages in the container
ARG PIP_BREAK_SYSTEM_PACKAGES=1
ARG PIP_ROOT_USER_ACTION=ignore
ARG PIP_NO_CACHE_DIR=1

# Install Python packages
ARG PYTHON_REQUIREMENTS=requirements.txt
COPY requirements.txt /opt/kso/
#COPY . /opt/kso
WORKDIR /opt/kso

RUN pip install --break-system-packages -r /opt/kso/$PYTHON_REQUIREMENTS
RUN pip install papermill

# Install MIOpen kdb files with ROCm PyTorch wheels, see
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#using-a-wheels-package
# TODO: Include more AMD architectures when needed
ARG ROCM_VERSION=false
RUN if [ "$ROCM_VERSION" != "false" ] ; then \
      cd /opt && \
      wget https://raw.githubusercontent.com/wiki/ROCm/pytorch/files/install_kdb_files_for_pytorch_wheels.sh && \
      export GFX_ARCH=gfx90a && \
      export ROCM_VERSION=$ROCM_VERSION && \
      bash install_kdb_files_for_pytorch_wheels.sh && \
      rm install_kdb_files_for_pytorch_wheels.sh ; \
    fi
