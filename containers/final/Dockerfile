ARG BASE_IMAGE=ghcr.io/ocean-data-factory-sweden/kso-base:latest
FROM ${BASE_IMAGE}

USER root

# Create user and group if not existing already
# The variable names are consistent with onyxia-base image
# used for edito image
ARG USERNAME
ENV USERNAME=${USERNAME:-jovyan}
ARG UID
ENV UID=${UID:-1500}
ARG GROUPNAME
ENV GROUPNAME=${GROUPNAME:-users}
ARG GID
ENV GID=${GID:-100}
ENV HOME=/home/${USERNAME}
RUN if ! getent passwd "${USERNAME}" > /dev/null 2>&1 ; then \
      getent group "${GROUPNAME}" > /dev/null 2>&1 || addgroup --gid "${GID}" "${GROUPNAME}" ; \
      adduser --disabled-password --gecos "Default user" \
        --uid "${UID}" --gid "${GID}" --home "${HOME}" "${USERNAME}" ; \
    fi

# Copy KSO
COPY --chown=${UID}:${GID} . /opt/kso

# Install KSO (all dependencies should be satisfied already)
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
COPY containers/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh
USER ${USERNAME}
ENV WORKSPACE_DIR=${WORKSPACE_DIR:-/opt/kso}
WORKDIR ${WORKSPACE_DIR}
EXPOSE 8888
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD []
