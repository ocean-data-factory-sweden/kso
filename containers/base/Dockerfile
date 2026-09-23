ARG BASE_IMAGE=docker.io/ubuntu:latest
ARG FFMPEG_BASE_IMAGE=docker.io/ubuntu:latest

# ----------------------------------------------------------------------
#  Stage for building ffmpeg
# ----------------------------------------------------------------------
FROM ${FFMPEG_BASE_IMAGE} AS ffmpeg-build

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG FFMPEG_DIR=/opt/ffmpeg

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

# Build ffmpeg with CUDA and ROCm support from source
ARG CUDA_VERSION=false
ARG ROCM_VERSION=false
RUN mkdir -p /opt/src && \
    cd /opt/src && \
    ffmpeg_config_flags="\
        --prefix="${FFMPEG_DIR}" \
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
    if [ "$ROCM_VERSION" != "false" ] ; then \
      # Install vaapi if ROCm is enabled
      apt-get update -qy && \
      apt-get install --no-install-recommends -y \
          libva-amdgpu-dev \
          libdrm-dev \
          mesa-amdgpu-va-drivers \
          && \
      apt-get clean && rm -rf /var/lib/apt/lists/* && \
      export PKG_CONFIG_PATH=/opt/amdgpu/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH} && \
      ffmpeg_config_flags="$ffmpeg_config_flags \
          --enable-vaapi \
          --enable-libdrm \
      "; \
    fi && \
    git clone -b n8.0.1 --depth 1 https://github.com/FFmpeg/FFmpeg.git ffmpeg && \
    cd ffmpeg && \
    echo "ffmpeg configure flags: $ffmpeg_config_flags" && \
    ./configure $ffmpeg_config_flags && \
    make -j $(nproc) && \
    make install


# ----------------------------------------------------------------------
#  Stage for building python
# ----------------------------------------------------------------------
FROM ${BASE_IMAGE} AS python-build

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12.14
ARG PYTHON_DIR=/opt/python

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        dpkg-dev \
        libbz2-dev \
        libdb-dev \
        libffi-dev \
        libgdbm-dev \
        liblzma-dev \
        libncurses-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        tk-dev \
        uuid-dev \
        wget \
        xz-utils \
        zlib1g-dev \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /opt/src && \
    cd /opt/src && \
    wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" && \
    tar xzf "Python-${PYTHON_VERSION}.tgz" && \
    cd "Python-${PYTHON_VERSION}" && \
    ./configure \
        --prefix="${PYTHON_DIR}" \
        --enable-optimizations \
        --enable-shared \
        --with-lto \
        LDFLAGS="-Wl,-rpath,${PYTHON_DIR}/lib" \
        && \
    make -j $(nproc) && \
    make install

RUN ln -s python3 "${PYTHON_DIR}/bin/python" && \
    ln -s pip3 "${PYTHON_DIR}/bin/pip"


# ----------------------------------------------------------------------
#  Final image
# ----------------------------------------------------------------------
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
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
        # For python
        libbz2-1.0 \
        libdb5.3t64 \
        libffi8 \
        libgdbm6t64 \
        liblzma5 \
        libncursesw6 \
        libreadline8t64 \
        libsqlite3-0 \
        libssl3t64 \
        libtinfo6 \
        tk \
        uuid-runtime \
        zlib1g \
        # Are these needed?
        # libmagic1 \
        # libsm6 \
        # libxrender1 \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install platform-specific apt dependencies
ARG ROCM_VERSION=false
RUN if [ "$ROCM_VERSION" != "false" ] ; then \
        apt-get update && \
        apt-get install --no-install-recommends -y \
            # For ffmpeg
            libva2-amdgpu \
            libva-amdgpu-drm2 \
            mesa-amdgpu-va-drivers \
            vainfo \
            && \
        apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# Copy ffmpeg
ARG FFMPEG_DIR=/opt/ffmpeg
COPY --from=ffmpeg-build $FFMPEG_DIR/bin $FFMPEG_DIR/bin
ENV PATH=$FFMPEG_DIR/bin:$PATH

# Copy python
ARG PYTHON_DIR=/opt/python
COPY --from=python-build $PYTHON_DIR $PYTHON_DIR
ENV PATH=$PYTHON_DIR/bin:$PATH

# Check that all runtime dependencies are installed
RUN python3 -c "import bz2, ctypes, lzma, readline, sqlite3, ssl, uuid, zlib; print('Runtime OK')"

# Install Python packages
ARG PIP_ROOT_USER_ACTION=ignore
ARG PIP_NO_CACHE_DIR=1
ARG PYTHON_REQUIREMENTS=requirements.txt
COPY containers/requirements.txt /opt/
COPY containers/$PYTHON_REQUIREMENTS /opt/
RUN pip install -r /opt/requirements.txt -r "/opt/$PYTHON_REQUIREMENTS"

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
