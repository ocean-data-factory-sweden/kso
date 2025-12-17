IMAGE_ROOT ?= ghcr.io/ocean-data-factory-sweden
IMAGE_NAME := kso-lumi
IMAGE_VERSION := 0.3.0
ROCM_VERSION := 6.4.4
ROCM_MAJOR_MINOR_VERSION := $(word 1,$(subst ., ,$(ROCM_VERSION))).$(word 2,$(subst ., ,$(ROCM_VERSION)))

SIF_FILE := $(IMAGE_NAME)_$(IMAGE_VERSION).sif
TAR_FILE := $(IMAGE_NAME)_$(IMAGE_VERSION).tar

.PHONY: build push singularity

build: Dockerfile requirements.txt requirements_rocm$(ROCM_MAJOR_MINOR_VERSION).txt
	docker buildx build --platform linux/amd64 \
		--label "org.opencontainers.image.source=https://github.com/ocean-data-factory-sweden/kso" \
		--label "org.opencontainers.image.description=KSO environment for LUMI-G" \
		--build-arg ROCM_VERSION=$(ROCM_VERSION) \
		-t $(IMAGE_ROOT)/$(IMAGE_NAME):$(IMAGE_VERSION) \
		.

push:
	docker push $(IMAGE_ROOT)/$(IMAGE_NAME):$(IMAGE_VERSION)

singularity:
	rm -f $(SIF_FILE) $(TAR_FILE)
	docker save $(IMAGE_ROOT)/$(IMAGE_NAME):$(IMAGE_VERSION) -o $(TAR_FILE)
	singularity build $(SIF_FILE) docker-archive://$(TAR_FILE)
	rm -f $(TAR_FILE)

%.txt: ../../../%.txt
	cp $< ./
