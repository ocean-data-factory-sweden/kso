ARG BASE_IMAGE=ghcr.io/ocean-data-factory-sweden/kso-base:latest
FROM ${BASE_IMAGE}

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

# Copy KSO
COPY --chown=${NB_UID}:${NB_UID} . /opt/kso
RUN chmod +x /opt/kso/containers/entrypoint.sh

# Install KSO (all dependencies should be satisfied already)
ARG PIP_BREAK_SYSTEM_PACKAGES=1
ARG PIP_ROOT_USER_ACTION=ignore
ARG PIP_NO_CACHE_DIR=1
RUN pip install --no-deps /opt/kso[dev]

# Check that all dependencies are satisfied
RUN pip check || { \
    echo ""; \
    echo "ERROR: Installed dependencies failed validation."; \
    echo ""; \
    echo "This means that requirements.txt used for containers is inconsistent with pyproject.toml."; \
    echo ""; \
    echo "If you changed dependencies in pyproject.toml, update requirements.txt too."; \
    echo ""; \
    echo "Dependency issues reported by 'pip check':"; \
    pip check; \
    exit 1; \
}

# Set run environment
USER ${NB_USER}
WORKDIR /opt/kso
ENTRYPOINT ["/opt/kso/containers/entrypoint.sh"]
CMD []
