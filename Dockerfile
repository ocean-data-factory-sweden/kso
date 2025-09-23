# Build the ffmpeg with CUDA support from source.
# We need ffmpeg on the system that works with the GPU.
# Only having the python package is not enough. ---
# To build from source we need the devel cuda image.
FROM nvcr.io/nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04 as builder
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

# Start over from the docker image with cuda 12.0
# since we only want the final result from the previous run and we copy that.
# Now we can use the runtime cuda image, since we do not need to build anything
# from scratch. This is better, since the runtime image is smaller
FROM nvcr.io/nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04
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

# Create a virtual environment and install Python packages
COPY requirements.txt /usr/src/app/
RUN python3.12 -m venv /opt/venv && \
    # Activate the venv in this RUN step
    /bin/bash -c "source /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /usr/src/app/requirements.txt"
# Set environment variables
ENV PYTHONPATH=/opt/venv/lib/python3.12/site-packages:$PYTHONPATH:/usr/src/app/kso
ENV PATH="/opt/venv/bin:$PATH"

# Set the user
ARG NB_USER=jovyan
ARG NB_UID=1500
# Random number higher than 1000,
# since 1000 is already in use in the base image.
ENV USER=${NB_USER} \
    NB_UID=${NB_UID} \
    HOME=/home/${NB_USER}
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
USER ${NB_USER}

# Make sure we use the environment as entry point
ENTRYPOINT ["/bin/bash", "-c", "source /opt/venv/bin/activate && exec \"$@\"", "--"]
