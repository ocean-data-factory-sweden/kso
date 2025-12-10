#!/bin/bash -l
#SBATCH -J subsim
#SBATCH -o %x-%j.out
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 --cpus-per-task=7 --gpus-per-node=1 --mem=30G
#SBATCH --time=0:30:00

# Check if at least one argument is given
if [ $# -eq 0 ]; then
    echo "Error: No container path provided."
    exit 1
fi

CONTAINER="$1"
echo "Using $CONTAINER"

export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl"
export PYTHONUSERBASE="/scratch/$PROJECT/$USER/venv"

export MIOPEN_USER_DB_PATH="/tmp/$USER/miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

KSO_PATH=$(readlink -f ../../)
export PYTHONPATH=$KSO_PATH${PYTHONPATH:+:$PYTHONPATH}

notebook=$KSO_PATH/notebooks/analyse/Train_models.ipynb
dpath="outputs/$SLURM_JOB_ID"
python="singularity exec $CONTAINER python3"

notebook=$(readlink -f $notebook)

mkdir -p $dpath
cd $dpath
$python -m papermill $notebook $(basename $notebook) -p epochs 2

