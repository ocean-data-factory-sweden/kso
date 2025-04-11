ARG ROCM_VERSION
FROM docker.io/rocm/dev-ubuntu-24.04:$ROCM_VERSION

ENV DEBIAN_FRONTEND=noninteractive

# Install ffmpeg
# TODO: GPU acceleration
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        ffmpeg \
        wget \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

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

# Install pip packages directly so that the user can create
# their own venv with --system-site-packages to use the packages in the container
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1

# Install Python packages
WORKDIR /usr/src/app
COPY requirements.txt ./
COPY requirements_rocm6.4.txt ./
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt -r /usr/src/app/requirements_rocm6.4.txt
RUN pip install papermill

# Install MIOpen kdb files with ROCm PyTorch wheels, see
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#using-a-wheels-package
ARG ROCM_VERSION
RUN cd /opt && \
    wget https://raw.githubusercontent.com/wiki/ROCm/pytorch/files/install_kdb_files_for_pytorch_wheels.sh && \
    export GFX_ARCH=gfx90a && \
    export ROCM_VERSION=$ROCM_VERSION && \
    bash install_kdb_files_for_pytorch_wheels.sh && \
    rm install_kdb_files_for_pytorch_wheels.sh
