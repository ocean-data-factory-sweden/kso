ARG BASE_IMAGE=docker.io/inseefrlab/onyxia-jupyter-python:py3.12.12
FROM ${BASE_IMAGE}

USER root

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        # For OpenCV
        libgl1 \
        libglib2.0-0 \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
ARG PIP_NO_CACHE_DIR=1
ARG PYTHON_REQUIREMENTS=requirements.txt
COPY requirements.txt /opt/kso/
COPY $PYTHON_REQUIREMENTS /opt/kso/
RUN pip install -r /opt/kso/requirements.txt -r "/opt/kso/$PYTHON_REQUIREMENTS"

# Copy KSO
COPY . /opt/kso

USER 1000
