# Installation Guide (Docker)

This document walks you through a fully reproducible **Docker** setup for the platform notebooks.

---

## 1. Prerequisites
- Install Docker Desktop (Win/Mac) or Docker Engine (Linux)
- Ensure Docker daemon is running (`docker info`)

## 2. Clone Repository
```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
```

## 3. Build Image
A ready-made `Dockerfile` is provided:
```bash
docker build -t platform-notebooks:latest .
```

### Optional: Use Pre-built Image
```bash
docker pull ghcr.io/<org>/platform-notebooks:latest
```

## 4. Run Container
```bash
docker run --gpus all \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  --env PLATFORM_API_KEY=$PLATFORM_API_KEY \
  platform-notebooks:latest
```
The notebook token / URL will be printed to the console.

### Makefile Shortcut
A convenience `Makefile` is included:
```bash
make jupyter  # builds (if needed) then runs container
```

## 5. Persisted Storage
Mount host data or project directories with `-v /host/path:/container/path`.

## 6. Updating the Image
```bash
docker pull ghcr.io/<org>/platform-notebooks:latest && \
  make jupyter
```

---

## GPU Support
- **NVIDIA**: Install latest *NVIDIA Container Toolkit* (<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>)
- Verify with `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

---

## Security Notes
- Do **NOT** hardcode API keys in the Dockerfile.
- Pass them at runtime via `--env` or Docker secrets.
- Limit network if working with sensitive data (`--network none`).

---

### Troubleshooting
| Issue | Cause | Remedy |
| ----- | ----- | ------ |
| Port 8888 busy | Another app using it | Change `-p 8888:8888` to `-p 8890:8888` |
| No GPUs detected | NVIDIA toolkit missing | Follow toolkit install docs |
| Slow builds | Re-use layers | Ensure proper `.dockerignore` |

---

© 2025 Platform Research Team
