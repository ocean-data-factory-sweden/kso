# Using KSO on LUMI

This guide covers how to run KSO notebooks on the LUMI supercomputer, both interactively via Jupyter and as batch jobs. For uploading files (footage, datasets, model weights), see [`TRANSFERS.md`](./TRANSFERS.md).

> Make sure you replace `project_...` with your actual project code throughout.

---

## Prerequisites

- LUMI account with project allocation
- SSH access: `ssh <username>@lumi.csc.fi`

---

## First Steps

Log in to LUMI via SSH or the [LUMI web interface](https://www.lumi.csc.fi/) and clone the KSO repository to your scratch workspace:

```bash
git clone -b dev https://github.com/ocean-data-factory-sweden/kso.git \
  /scratch/project_.../$USER/kso
```

To switch to a different branch:

```bash
cd /scratch/project_.../$USER/kso
git checkout BRANCH_NAME
```

### Container

The KSO Singularity container has all dependencies pre-installed. It is available at:

```
/projappl/project_.../containers/kso_dev-rocm6.4-ubuntu24.04.sif
```

> **DTO-BioFlow project users:** the container is already pulled for you — skip any container pull step.

If you need to pull a fresh copy yourself (change the tag if needed):

```bash
cd /scratch/project_.../$USER
singularity pull --disable-cache \
  docker://ghcr.io/ocean-data-factory-sweden/kso:dev-rocm6.4-ubuntu24.04
```

---

## Interactive Jupyter Session

### 1. Launch Jupyter

Go to [https://www.lumi.csc.fi/](https://www.lumi.csc.fi/) and select **Interactive Apps → Jupyter** ([direct link](https://www.lumi.csc.fi/pun/sys/dashboard/batch_connect/sys/ood-base-jupyter/session_contexts/new)).

Configure the following settings:

| Setting | Value |
|---------|-------|
| Project | `project_...` |
| Partition | `small-g` |
| CPU cores | `1` (≤8 when using a single GPU — see [billing](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/#gpu-billing)) |
| Memory (GB) | `32` (≤64 GB when using a single GPU) |
| GPUs (MI250 GCDs) | `1` |
| Time | `2:00:00` (adjust as needed) |
| Working directory | `/scratch/project_...` |

### 2. Configure the container

Under **Advanced → Custom Python**, select **Script** and paste:

```bash
# Auto-clone KSO if not already present
[[ -e "/scratch/$PROJECT/$USER/kso" ]] || \
  git clone -b dev https://github.com/ocean-data-factory-sweden/kso.git \
  "/scratch/$PROJECT/$USER/kso"

# Set up Singularity container
CONTAINER="/projappl/project_.../containers/kso_dev-rocm6.4-ubuntu24.04.sif"
export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl"
export python="singularity exec $CONTAINER python3"
export PYTHONUSERBASE="/scratch/$PROJECT/$USER/venv"
```

### 3. Connect and navigate

Click **Launch**, wait for the session to start, then click **Connect to Jupyter**.

Navigate to your KSO notebooks:
```
/scratch/project_.../<your_username>/kso/notebooks/
```

Then run the notebooks following the pipeline.

---

## Batch Job Execution

For long training runs, submitting a batch job is more reliable than an interactive session.

```bash
cd /scratch/project_.../$USER/kso/contrib/lumi
```

Submit a notebook to run non-interactively (modify the script as needed). Set `CONTAINER` to the path of the container `.sif` file:

```bash
sbatch -A project_... scripts/submit.lumi.sh CONTAINER
```

Output will be saved to:
```
/scratch/project_.../$USER/kso/contrib/lumi/outputs/JOBID/
```

where `JOBID` is the unique ID assigned to your job.

---

## Storage on LUMI

| Storage Area | Path | Quota | Retention | Use For |
|--------------|------|-------|-----------|---------|
| **Scratch** | `/scratch/project_<ID>` | 50 TB | Project lifetime | Datasets, training outputs, active work |
| **Project** | `/projappl/project_<ID>` | 50 GB | Project lifetime | Containers, shared tools |
| **Home** | `/users/$USER` | 20 GB | User lifetime | Config files only |

> For current data retention policies, see the [LUMI documentation](https://docs.lumi-supercomputer.eu/storage/).

---

## Additional Resources

- [LUMI Documentation](https://docs.lumi-supercomputer.eu)
- [LUMI User Support](https://www.lumi-supercomputer.eu/user-support/)
