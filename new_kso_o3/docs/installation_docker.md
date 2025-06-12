# Installation Guide (Docker)

Run **new_kso_o3** fully containerised—no local Python install required.

---

## 1. Prerequisites

* Docker ≥ 20.10
* (Optional) NVIDIA Container Toolkit for GPU support

## 2. Build the image

```bash
git clone https://github.com/<your-org>/kso.git
cd kso/new_kso_o3
docker build -t kso:latest .
```

> **Tip**: Use `--build-arg PYTHON_VERSION=3.10` or `--platform=linux/amd64` as needed.

## 3. Start a Jupyter session

```bash
docker run --rm -it -p 8888:8888 \
  -v $(pwd):/workspace \
  kso:latest jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

Open <http://localhost:8888> in your browser with the provided token.

## 4. Persist data & models

Mount a host directory:

```bash
mkdir $HOME/kso_data

docker run --rm -it -p 8888:8888 \
  -v $(pwd):/workspace \
  -v $HOME/kso_data:/workspace/outputs \
  kso:latest jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

Outputs and checkpoints will be saved under `$HOME/kso_data`.

---

### GPU Support

```bash
docker run --gpus all -it --rm -p 8888:8888 \
  -v $(pwd):/workspace \
  kso:latest
```

Ensure the host has matching CUDA driver versions.

---

### Troubleshooting
| Symptom | Fix |
|---------|-----|
| Slow build | Add Docker build cache (`--build-arg BUILDKIT_INLINE_CACHE=1`). |
| Permission denied on volume | Use `--user $(id -u):$(id -g)` when mounting volumes. |
