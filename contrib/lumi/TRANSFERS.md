# Uploading Files to LUMI

This guide explains how to upload large files (video footage, datasets, pretrained models) to LUMI using SSH and rsync. It covers Windows (via WSL), macOS, and Linux.

For Jupyter session setup and workspace layout, see [`LUMI_SETUP.md`](./LUMI_SETUP.md).

---

## Prerequisites

- A LUMI account with SSH access
- Your LUMI username and project number
- Access to [MyAccessID](https://mms.myaccessid.org/) for SSH key registration
- Internet access on port 22 (SSH)

---

## Destination paths on LUMI

Most uploads go to your scratch workspace:

```
/scratch/project_PROJECT_NUMBER/YOUR_USERNAME/
```

| What | Destination |
|------|-------------|
| YOLO datasets | `datasets/your_dataset_name/` (must contain `data.yaml`) |
| Pretrained model weights | `models/your_model.pt` |
| Raw video footage | `raw_footage/` or a folder of your choosing |

If the directories don't exist yet, create them first via the LUMI web interface or SSH.

---

## Step 1 — Install SSH and rsync (one-time)

### Windows (via WSL)

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

Reboot if prompted. An Ubuntu terminal will open automatically. Inside it, run:

```bash
sudo apt update && sudo apt install -y rsync openssh-client
```

### macOS / Linux

`ssh` and `rsync` are pre-installed. Verify with:

```bash
ssh -V && rsync --version
```

---

## Step 2 — Create an SSH key (one-time)

On **Windows**, run this inside the WSL/Ubuntu terminal. On macOS/Linux, use your regular terminal.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/lumi_id_ed25519
```

Choose a passphrase when prompted — do not leave it empty.

Display your public key:

```bash
cat ~/.ssh/lumi_id_ed25519.pub
```

Copy the full output line (starts with `ssh-ed25519`).

---

## Step 3 — Register the key in MyAccessID

1. Go to [https://mms.myaccessid.org/](https://mms.myaccessid.org/)
2. Find the **SSH keys** section
3. Add a new key and paste your public key
4. Save

> Key synchronization may take up to a few hours.

---

## Step 4 — Test the SSH connection

```bash
ssh -i ~/.ssh/lumi_id_ed25519 YOUR_USERNAME@lumi.csc.fi
```

If you see the LUMI welcome banner, SSH is working. Type `exit` to disconnect.

---

## Step 5 — Locate your local files

### Windows (WSL)

Windows drives are mounted under `/mnt`:

| Windows path | WSL path |
|-------------|----------|
| `C:\` | `/mnt/c/` |
| `D:\` | `/mnt/d/` |
| `E:\` (external drive) | `/mnt/e/` |

Example: if your files are at `D:\BRUV_VIDEOS\`, the WSL path is `/mnt/d/BRUV_VIDEOS/`.

### macOS / Linux

Use the normal path, e.g. `/Volumes/ExternalDrive/BRUV_VIDEOS/`.

---

## Step 6 — Upload with rsync

Always do a dry run first to preview what will be transferred:

```bash
# Dry run
rsync -avP --dry-run \
  -e "ssh -i ~/.ssh/lumi_id_ed25519" \
  /path/to/local/files/ \
  YOUR_USERNAME@lumi.csc.fi:/scratch/project_PROJECT_NUMBER/YOUR_USERNAME/DESTINATION/

# Real transfer
rsync -avP \
  -e "ssh -i ~/.ssh/lumi_id_ed25519" \
  /path/to/local/files/ \
  YOUR_USERNAME@lumi.csc.fi:/scratch/project_PROJECT_NUMBER/YOUR_USERNAME/DESTINATION/
```

Flag reference: `-a` archive mode (preserves structure), `-v` verbose, `-P` shows progress and resumes on interruption.

**If the transfer is interrupted** (network drops, laptop sleeps), just run the same command again — rsync will pick up where it left off.

To confirm the transfer is complete, re-run the same command. If nothing transfers, you're done.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied (publickey)` | Key not yet synced — wait a few hours after registering in MyAccessID, then retry |
| Connection timeout | Check VPN/firewall; SSH requires port 22 |
| Spaces in file paths | Wrap paths in quotes: `"/mnt/d/My Videos/"` |
| Laptop sleeps during transfer | Disable sleep/hibernate while rsync is running |
| Transfer seems stuck | Large files take time — check that bytes are still moving with `-P` |

---

## Quick reference

```bash
# Test SSH connection
ssh -i ~/.ssh/lumi_id_ed25519 YOUR_USERNAME@lumi.csc.fi

# Dry run
rsync -avP --dry-run \
  -e "ssh -i ~/.ssh/lumi_id_ed25519" \
  /path/to/local/files/ \
  YOUR_USERNAME@lumi.csc.fi:/scratch/project_PROJECT_NUMBER/YOUR_USERNAME/DESTINATION/

# Real transfer
rsync -avP \
  -e "ssh -i ~/.ssh/lumi_id_ed25519" \
  /path/to/local/files/ \
  YOUR_USERNAME@lumi.csc.fi:/scratch/project_PROJECT_NUMBER/YOUR_USERNAME/DESTINATION/
```

Replace `YOUR_USERNAME`, `PROJECT_NUMBER`, and `DESTINATION` with your actual values.
