# LUMI Setup Guide

This guide shows how to run KSO notebooks on the LUMI supercomputer using interactive Jupyter sessions.

## Prerequisites

- LUMI account with project allocation (e.g., `project_465002425`)
- SSH access: `ssh <username>@lumi.csc.fi`

---

## Interactive Jupyter Setup

### 1. Log in to LUMI Web Interface

Go to [https://www.lumi.csc.fi/](https://www.lumi.csc.fi/) and log in with your credentials.

### 2. Launch Jupyter

1. Select **Interactive Apps → Jupyter**
2. Configure the following settings:
   - **Project**: `project_<YOUR_PROJECT_ID>` (e.g., `project_465002425`)
   - **Partition**: `small-g` (GPU nodes)
   - **CPU cores**: `1` (≤8 if using 1 GPU)
   - **Memory (GB)**: `30` (≤64 GB if using 1 GPU)
   - **GPUs (MI250 GCDs)**: `1`
   - **Time**: `2:00:00` (adjust as needed)
   - **Working directory**: `/scratch/project_<YOUR_PROJECT_ID>`

3. Under **Advanced → Custom Python**, select **Script** and paste:

```bash
# Auto-clone KSO if not already present
[[ -e "/scratch/$PROJECT/$USER/kso" ]] || \
  git clone -b dev https://github.com/ocean-data-factory-sweden/kso.git \
  "/scratch/$PROJECT/$USER/kso"

# Set up Singularity container
CONTAINER="/projappl/project_465002425/containers/kso-lumi_0.3.0.sif"
export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl"
export python="singularity exec $CONTAINER python3"
export PYTHONUSERBASE="/scratch/$PROJECT/$USER/venv"
```

> **Note**: The Singularity container (`kso-lumi_0.3.0.sif`) contains all KSO dependencies pre-installed. You don't need to install anything manually.

4. Click **Launch** → wait for the job to start → click **Connect to Jupyter**

5. Navigate to your KSO folder: `/scratch/project_<YOUR_PROJECT_ID>/<your_username>/kso/notebooks/`

6. Open `01_Project_Setup.ipynb` to begin

---

## Storage on LUMI

LUMI has different storage areas with different purposes:

| Storage Area | Path | Quota | Retention | Use For |
|--------------|------|-------|-----------|---------|
| **Scratch** | `/scratch/project_<ID>` | 50 TB | **90 days** | Datasets, training outputs, active work |
| **Project** | `/project/project_<ID>` | 50 GB | Project lifetime | Containers, final models to keep |
| **Home** | `/users/$USER` | 20 GB | User lifetime | Config files only (too small for data) |

### Important: Scratch is temporary!

Files in `/scratch/` are **automatically deleted after 90 days** of no access. If you have trained models or results you want to keep long-term, copy them to `/project/` or download them to your local machine.

```bash
# Example: copy a trained model to project storage
cp /scratch/project_465002425/$USER/kso/runs/best_model.pt \
   /project/project_465002425/saved_models/
```

---

## Troubleshooting

### Container not found

If you see an error about `kso-lumi_0.3.0.sif` not found, contact your project admin. The container should be at:
```
/projappl/project_465002425/containers/kso-lumi_0.3.0.sif
```

### Out of memory errors

Reduce batch size in the training notebook:
```python
batch_size = 4  # or even 2 for very large images
```

### Moving large files between projects

If you need to move large files (e.g., video footage) between LUMI projects, use SSH in the command line rather than Jupyter. For help, contact [LUMI User Support](https://www.lumi-supercomputer.eu/user-support/).

---

## Additional Resources

- [LUMI Documentation](https://docs.lumi-supercomputer.eu)
- [LUMI User Support](https://www.lumi-supercomputer.eu/user-support/)
