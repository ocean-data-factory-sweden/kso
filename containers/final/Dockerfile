ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Copy KSO
COPY . /opt/kso

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

WORKDIR /opt/kso
