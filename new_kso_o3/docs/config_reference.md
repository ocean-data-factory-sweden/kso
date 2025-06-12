# Configuration Reference

This document describes every field accepted in the YAML configuration files used by **new_kso_o3**.

---

## project.yaml
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `project_name` | `str` | `new_kso_o3_demo` | Human-readable experiment/project identifier. |
| `model` | `str` | `pytorch` | Trainer back-end: `pytorch`, `yolo`, or `hf`. |
| `output_dir` | `path` | `outputs` | Where checkpoints and logs are saved. |

## storage.yaml
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `local_root` | `path` | `data` | Base directory for local datasets & artifacts. |
| `remote` | `dict | null` | `null` | When set, enables remote sync. |

### `remote` sub-keys (S3 example)
| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `provider` | `str` | ✅ | Must be `s3`. |
| `endpoint` | `str` | ✅ | S3 endpoint URL (e.g. `https://s3.amazonaws.com`). |
| `bucket` | `str` | ✅ | Target bucket. |
| `aws_access_key_id` | `str` | ⬜︎ | Access key (env-var subst. supported). |
| `aws_secret_access_key` | `str` | ⬜︎ | Secret key (env-var subst. supported). |
| `region` | `str` | ⬜︎ | AWS region. |

## models/<backend>.yaml
Each trainer defines its own hyper-parameters.

### PyTorch (`models/pytorch.yaml`)
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `arch` | `str` | `resnet18` | torchvision model architecture. |
| `num_classes` | `int` | `2` | Output classes. |
| `batch_size` | `int` | `32` | Training batch size. |
| `learning_rate` | `float` | `3e-4` | Adam LR. |
| `epochs` | `int` | `5` | Training epochs. |

### YOLO (`models/yolo.yaml`)
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `arch` | `str` | `yolov8n.pt` | Pretrained checkpoint. |
| `imgsz` | `int` | `640` | Input resolution. |
| `epochs` | `int` | `50` | Training epochs. |
| `batch` | `int` | `16` | Batch size. |

### Hugging Face (`models/hf.yaml`)
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model_name` | `str` | `google/vit-base-patch16-224` | Model hub id. |
| `num_classes` | `int` | `2` | Labels. |
| `batch_size` | `int` | `8` | Train/Eval batch size. |
| `epochs` | `int` | `3` | Training epochs. |

---

## Environment Variables
| Name | Purpose |
|------|---------|
| `KSO_TRACKING` | Selects tracking backend (`mlflow`, `wandb`, or unset). |
| `KSO_CONFIG_DIR` | Override path to config directory. |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Used by S3 remote storage (optional). |

---

For questions or improvements please open an issue or PR 🚀.
