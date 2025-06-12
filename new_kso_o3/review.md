# Code Review – Sprint 1 (config, storage, trainers, tests, notebooks)

_Date: 2025-06-12_

---

## 1. General Observations

* Project structure follows the designed architecture; separation of `configs/`, core library `kso/`, `tests/`, and `notebooks/` is clear.
* Automated tests now cover configuration loader, storage abstraction, and trainer registry instantiation. 👍  
* Basic training / evaluation loops are in place for PyTorch, YOLO (v8), and Hugging Face back-ends.
* Sample YAML configs and demo notebooks provide a runnable E2E path.

## 2. Detailed Comments & Recommendations

### 2.1 Configuration (`kso/config.py`)
| Item | Comment |
|------|---------|
| **Pydantic v2 deprecations** | Using `@validator` and `parse_obj` triggers warnings. Switch to `@field_validator` and `model_validate` to future-proof. |
| **Dynamic CONFIG_DIR** | Allow env-var override (e.g. `KSO_CONFIG_DIR`) for flexibility in multi-project setups. |

### 2.2 Storage (`kso/storage.py`)
| Item | Comment |
|------|---------|
| **Transfer progress** | Consider adding tqdm progress for large uploads/downloads. |
| **Thread-safety** | `boto3` client is created per instance—OK. Document that `S3Storage` is not thread-safe without its own session. |
| **Automatic upload on `save()`** | Currently manual. See §2.5 for suggestion to integrate into trainer `save()` methods. |

### 2.3 Registry (`kso/registry.py`)
* Works, but races with import order caused test failures.  
  ➜  Suggest adding **lazy import helper**: `get_trainer` could attempt `importlib.import_module(f"kso.trainers.{name}")` if key missing, then retry.

### 2.4 Trainers
#### TorchTrainer
* 👍  Functional minimal loop.
* **Training accuracy** not logged—consider adding.
* **Transforms** hard-coded to `224×224`; expose via config.
* Commented-out `torch.save()`—enable with `strict=False` to avoid missing GPU tensors.
* Might leak GPU memory; wrap evaluation in `torch.inference_mode()`.

#### YoloTrainer
* Generates provisional `yolo_data.yaml`; paths (`images/train`) won’t exist by default.  
  ➜ Document required folder layout or auto-convert ImageFolder to YOLO labels.
* `device=0` will fail on CPU-only machines—make configurable (`"cpu"` fallback).
* Return of `train()` currently unused; could capture metrics dict.

#### HFTrainer
* Transformation returns **numpy** array but Trainer expects torch tensors; empirical check needed.
* Use `remove_columns` in HF datasets to drop original image objects for speed.
* Evaluation creates new `Trainer` without same args/logging; reuse existing to preserve callbacks.
* Save still TODO; call `model.save_pretrained(out_dir)` and `processor.save_pretrained(out_dir)`.

#### Common
* Consider a `BaseVisionTrainer.save_artifact(loader)` helper to unify saving + optional remote upload.

### 2.5 Model Saving & Remote Sync
* Implement inside each `save()`:
  ```python
  artifact = ...            # path or dir
  storage = get_storage()
  remote_key = f"{cfg.project_name}/{Path(artifact).name}"
  storage.upload_file(artifact, remote_key)
  ```
* Emit JSON side-car with hyper-params + git commit hash for reproducibility.

### 2.6 Tests
* Good mocking of heavy deps.
* Missing **S3Storage tests** (use moto’s `mock_s3`).
* Add integration test to run tiny TorchTrainer training for 1 epoch on few images (mark as slow).

### 2.7 Notebooks
* Clean, parameterised.
* Optionally inject `%load_ext autoreload` for iterative dev.
* Add cell to set `KSO_TRACKING` env var for quick MLflow/WandB selection.

### 2.8 `pyproject.toml`
* Duplicate `pytest` in both core deps and `[project.optional] dev`. Keep only in `dev`.
* Add `ruff` or `flake8` + `black` extras for linting.

## 3. Security & Secrets
* Good: credentials supplied via YAML + env-vars.
* Reminder: never log access keys; ensure boto3 client uses env-vars if values start with `${`.

## 4. Action Items (Priority)
1. Migrate to Pydantic v2 API.
2. Enable actual saving in `save()` and implement automatic remote upload when `storage.remote` present.
3. Add trainer import fallback in registry to avoid manual imports.
4. Write tests for S3Storage with moto.
5. Clean up `pyproject.toml` duplicates and add lint/test extras.
6. Improve YOLO data YAML handling or provide helper to convert ImageFolder to YOLO labels.

---
Reviewed by **Code Review Agent**

---
# Code Review – Sprint 2 (post-update)

_Date: 2025-06-12_

_Additions relative to Sprint 1 appended; unchanged sections omitted for brevity._

## Addressed Items

| Section | Status |
|---------|--------|
| **Pydantic v2 migration** | ✅  `kso/config.py` now uses `field_validator` + `model_validate`; warnings gone. |
| **Dynamic CONFIG_DIR** | ✅  `KSO_CONFIG_DIR` env-var supported. |
| **Registry lazy import** | ✅  `get_trainer` attempts `importlib.import_module`. |
| **TorchTrainer save()** | ✅  Saves weights, uploads via storage, writes `metadata.json`. |
| **S3 tests** | ✅  Added `tests/test_storage_s3.py` with moto mock. |

## New Observations

### TorchTrainer
* Good: real saving + metadata.
* Upload path hard-codes key suffix `model.pt`; consider including timestamp or epoch for multiple checkpoints.
* `storage.upload_file` called unconditionally; but LocalStorage also implements it, so OK.

### Config / ENV
* `CONFIG_DIR` constructed via `Path(os.getenv(...))`—if env var is absolute string it’s fine; when relative, join with cwd? Document.

### Dependencies
* Still duplicates of `pytest` in `[project] dependencies` vs `[project.optional.dev]`. Remove from core.
* Suggest adding `moto` to `test` extras.

### Remaining TODO
1. Mirror `save()` improvements to **YoloTrainer** and **HFTrainer**.
2. Add JSON metadata to all trainers.
3. Optionally include training metrics history in metadata.
4. Consider common helper in `AbstractTrainer` for save+upload.

## Action Items (Sprint 3)
1. Refactor common save logic to base class; implement for YOLO/HF.  
2. Remove duplicate deps; create `[tool.black]` & `[tool.ruff]` config and extras.
3. Integration test for TorchTrainer 1-epoch smoke run (mark slow).  
4. YOLO dataset YAML handling helper.

---
_Reviewed by **Code Review Agent**_
