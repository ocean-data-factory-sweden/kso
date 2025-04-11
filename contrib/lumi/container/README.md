# Building containers for LUMI-G

This directory has two files:
- `Dockerfile`: This is the recipe for building a container image
- `Makefile`: This is a helper file for simplifying building the image (so that one can say `make build` instead of `docker buildx build --platform ... --build-arg ...`)


## Prerequisites

Docker/Podman is needed for building the image. Follow these steps on Ubuntu:

1. Install docker or podman:

       sudo apt install podman-docker

2. Add the following environment variable to `~/.bashrc` or redefine it always before running build commands:

       export BUILDAH_FORMAT=docker


## Workflow for building, publishing, and pulling images

1. Updating and building a new image on a local computer

   1. Update `IMAGE_VERSION` variable in `Makefile`. If this is not done, an existing image with the same version gets overwritten, which is problematic for reproducibility.

   2. Update `Dockerfile` and/or `Makefile` as needed.

   3. Build a new image:

          make build

2. Pushing image to ghcr.io

    1. Login to GitHub container registry.
       Use GitHub username and Personal Access Token with scope 'write:packages' as username and password, respectively.
       See [these instructions for creating a token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token#creating-a-personal-access-token-classic).

           docker login ghcr.io

    2. Push the image to ghcr.io:

           make push

3. Pulling image to LUMI and converting it to singularity

    1. Execute the following on LUMI (if image is private, login is needed with a token with scope 'read:packages'):

           singularity pull --disable-cache --docker-login docker://ghcr.io/ocean-data-factory-sweden/kso-lumi:IMAGE_VERSION


## Development workflow without publishing images

1. Updating and building a new image on a local computer

   - Follow the same steps as above

2. Converting the image to singularity on a local computer

   1. Convert the image to singularity (results in a .sif file):

           make singularity

3. Transferring the image to supercomputer

   1. Use scp:

           scp file.sif user@lumi.csc.fi:/scratch/project_xxx/user/

