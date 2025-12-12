# Using KSO on LUMI

Note! Replace the `project_...` with the correct project code in the instructions below.

## First steps

- Login to LUMI using SSH or via Login node shell in https://www.lumi.csc.fi/ and run the following commands in that shell.

- Clone the KSO repository to your personal working directory:

      git clone https://github.com/ocean-data-factory-sweden/kso.git /scratch/project_.../$USER/kso

- Use the following steps to checkout other branch:

      cd /scratch/project_.../$USER/kso
      git checkout BRANCH_NAME

- **This step is not needed in DTO-BioFlow project, an image is pulled already!**
  Pull container image (change the image version 'dev-rocm6.4-ubuntu24.04' if needed):

      cd /scratch/project_.../$USER
      singularity pull --disable-cache docker://ghcr.io/ocean-data-factory-sweden/kso:dev-rocm6.4-ubuntu24.04

## Batch job execution

- Login to LUMI using SSH or via Login node shell in https://www.lumi.csc.fi/

- Go to the your working directory

      cd /scratch/project_.../$USER/kso/contrib/lumi

- Submit a job running a notebook non-interactively (modify the script as needed), set `CONTAINER` to point to the path to the container image sif file

      sbatch -A project_... scripts/submit.lumi.sh CONTAINER

- The output will go to directory `/scratch/project_.../$USER/kso/contrib/lumi/outputs/JOBID/`, where `JOBID` is the unique id of the submitted job

## Interactive notebook execution

- Go to https://www.lumi.csc.fi/ and login
- Select Jupyter app ([direct link](https://www.lumi.csc.fi/pun/sys/dashboard/batch_connect/sys/ood-base-jupyter/session_contexts/new))
- Choose the following settings:
  - Project: project_...
  - Partition: small-g
  - Number of CPU cores: 1 (Note: use values below 8 if using only a single GPU; see [billing](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/#gpu-billing))
  - Memory (GB): 30 (Note: this is CPU memory; use values below 64 GB if using only a single GPU; see [billing](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/#gpu-billing))
  - Number of GPUs (MI250 GCDs): 1
  - Time: 2:00:00 (Note: adjust as needed)
  - Working directory: /scratch/$PROJECT
  - Under 'Advanced'
    - Custom Python type: Script
    - Script or path to script: Copy-paste and edit the following lines:

          CONTAINER="..."  # Put the correct path to container here!
          # The following lines need no changes
          export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl"
          export python="singularity exec $CONTAINER python3"
          export PYTHONUSERBASE="/scratch/$PROJECT/$USER/venv"

- Click Launch
- Wait for the Jupyter session to be queued and launched
- Click 'Connect to Jupyter' once the button appears
- Navigate in Jupyter to the notebooks under **your working directory** (the directory named as your LUMI username)
