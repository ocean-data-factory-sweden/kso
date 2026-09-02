from __future__ import annotations
from pathlib import Path
import yaml
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Any, Optional
import pandas as pd
from collections import defaultdict
import json
from PIL import Image
import os
import shutil
import logging
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import random
import re
import numpy as np

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
AUGMENT_OPS = ["hflip", "vflip", "rot90", "rot180", "rot270"]


def resolve_up(relative_path: str | Path) -> Path:
    """
    Resolve a relative path by searching upward from base.
    Returns the first existing match.
    """
    base = Path.cwd()
    relative_path = Path(relative_path)

    if relative_path.is_absolute():
        return relative_path

    for parent in [base] + list(base.parents):
        candidate = parent / relative_path
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve '{relative_path}' from '{base}' upward."
    )


def make_relative_path(abs_path: Path | str = None):
    """turn absolut path to relative path based on a start point"""
    if not abs_path or not isinstance(abs_path, (Path, str)):
        raise TypeError(f"{abs_path} must be non-empty path or string.")

    startPoint = Path(__file__).resolve().parents[3]
    abs_path = Path(abs_path).expanduser()
    base_dir = startPoint / "projects"
    relative_path = os.path.relpath(abs_path, start=base_dir)
    return relative_path


def make_abs_path(relative_path: str | Path):
    """turn relative path to absolut path based on a start point"""
    if not relative_path or not isinstance(relative_path, (Path, str)):
        raise TypeError(f"{relative_path} must be non-empty path or string")

    startPoint = Path(__file__).resolve().parents[3]
    base_dir = startPoint / "projects"
    relative_path = Path(relative_path).expanduser()
    abs_path = (base_dir / relative_path).resolve()
    return abs_path


def _transform_bbox(xc, yc, w, h, op):
    """Transform YOLO bbox coordinates for geometric operations."""
    if op == "hflip":
        return (1 - xc, yc, w, h)
    if op == "vflip":
        return (xc, 1 - yc, w, h)
    if op == "rot180":
        return (1 - xc, 1 - yc, w, h)
    if op == "rot90":
        return (yc, 1 - xc, h, w)
    if op == "rot270":
        return (1 - yc, xc, h, w)
    raise ValueError(f"Unknown op: {op}")


def _apply_transform(img, op):
    """Apply geometric transform to PIL Image."""
    if op == "hflip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)  # pylint: disable=no-member
    if op == "vflip":
        return img.transpose(Image.FLIP_TOP_BOTTOM)  # pylint: disable=no-member
    if op == "rot180":
        return img.rotate(180, expand=True)
    if op == "rot90":
        return img.rotate(-90, expand=True)
    if op == "rot270":
        return img.rotate(90, expand=True)
    raise ValueError(f"Unknown op: {op}")


def run_augmentation(data_yaml_path: str, augment_factor=0.5, random_seed=42):
    """
    Offline geometric augmentation for YOLO datasets.

    data_yaml_path: Path or str to data.yaml (Biigle output, Roboflow, or other YOLO)
    augment_factor: 0.5 -> create ~50% as many augmented images as annotated originals
    random_seed:    RNG seed for reproducibility
    """
    if not data_yaml_path or not isinstance(data_yaml_path, str):
        raise ValueError(f"{data_yaml_path} must be a non-empty string.")

    data_yaml_path = resolve_up(relative_path=data_yaml_path)
    cfg = yaml.safe_load(data_yaml_path.read_text())

    # If 'path' missing (e.g. Roboflow), fall back to directory of data.yaml
    base = Path(cfg.get("path") or data_yaml_path.parent)

    train_rel = cfg.get("train", "images/train")
    if isinstance(train_rel, list):
        train_rel = train_rel

    train_img_dir = base / train_rel
    train_lbl_dir = base / "labels" / "train"

    print(f"Using train images: {train_img_dir}")
    print(f"Using train labels: {train_lbl_dir}")

    if not train_img_dir.exists():
        raise FileNotFoundError(f"Train images not found: {train_img_dir}")
    if not train_lbl_dir.exists():
        raise FileNotFoundError(f"Train labels not found: {train_lbl_dir}")

    # Find annotated images
    train_imgs = sorted(
        p for p in train_img_dir.iterdir() if p.suffix.lower() in VALID_EXTS
    )
    annotated = []
    for img in train_imgs:
        lbl = train_lbl_dir / f"{img.stem}.txt"
        if lbl.exists() and lbl.read_text().strip():
            annotated.append(img)

    n_orig = len(train_imgs)
    n_annot = len(annotated)
    n_target = int(round(n_annot * augment_factor))

    if n_target <= 0 or n_annot == 0:
        print("Augmentation: skipped (no annotated images or factor too low)")
        print(f"  Train images: {n_orig} total, {n_annot} with annotations")
        return

    rng = random.Random(random_seed)
    selected = rng.sample(annotated, k=min(n_target, n_annot))
    created = 0

    for img_path in selected:
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        lines = [ln.strip().split() for ln in lbl_path.read_text().strip().splitlines()]

        op = rng.choice(AUGMENT_OPS)
        new_labels = []

        for parts in lines:
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            xc, yc, w, h = _transform_bbox(xc, yc, w, h, op)
            # Clamp to [0, 1]
            xc = max(0, min(1, xc))
            yc = max(0, min(1, yc))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            new_labels.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not new_labels:
            continue

        img = Image.open(img_path)
        aug_img = _apply_transform(img, op)
        suffix = f"_aug_{op}"

        out_img = train_img_dir / f"{img_path.stem}{suffix}{img_path.suffix}"
        out_lbl = train_lbl_dir / f"{img_path.stem}{suffix}.txt"

        aug_img.save(out_img)
        out_lbl.write_text("\n".join(new_labels))
        created += 1

    logging.info("✓ Augmentation complete")
    logging.info(f"  Created {created} new images")
    logging.info(f"  Train set: {n_orig} → {n_orig + created} images")


def preprocess_biigle_csv(
    biigle_csv_path: str,
    images_root: str,
    dataset_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    augment_factor: float = 0.5,
    random_seed=42,
    augment_train=True,
) -> Path:
    """
    Create a YOLO dataset from a biigle CSV file.
    Returns:
        Path to the written data.yaml.
    """
    # ----------------------------
    # Inline config (unchanged defaults)
    # ----------------------------

    AUGMENT_OPS = ["hflip", "vflip", "rot180"]

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    REQUIRED_COLS = ["filename", "label_name", "shape_name", "points", "attributes"]
    SUPPORTED_SHAPES = ["Rectangle", "Polygon"]

    # ----------------------------
    # Normalize paths
    # ----------------------------
    csv_path = Path(biigle_csv_path).expanduser()
    images_root = Path(images_root).expanduser()
    dataset_dir = Path(dataset_dir).expanduser()
    if not csv_path.is_absolute():
        csv_path = resolve_up(relative_path=csv_path)
    if not images_root.is_absolute():
        images_root = resolve_up(relative_path=images_root)
    if not dataset_dir.is_absolute():
        dataset_dir = resolve_up(relative_path=dataset_dir)

    # ----------------------------
    # Nested helpers (private)
    # ----------------------------
    def _validate_paths(csv_path: Path, images_root: Path, dataset_dir: Path) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} does not exist")
        if not images_root.exists():
            raise FileNotFoundError(f"{images_root} does not exist")
        if not dataset_dir.exists():
            raise FileNotFoundError(f"{dataset_dir} does not exist")

    def _read_biigle_csv(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} annotations from CSV")
        return df

    def _validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    def _filter_supported_shapes(
        df: pd.DataFrame, supported: Sequence[str]
    ) -> pd.DataFrame:
        df = df[df["shape_name"].isin(supported)].reset_index(drop=True)
        print(f"Supported annotations (Rectangle/Polygon): {len(df)}")
        return df

    def _parse_image_dimensions(df: pd.DataFrame) -> pd.DataFrame:
        attrs = df["attributes"].apply(json.loads)
        df["img_w"] = attrs.apply(lambda d: d.get("width"))
        df["img_h"] = attrs.apply(lambda d: d.get("height"))
        df = df.dropna(subset=["img_w", "img_h"]).reset_index(drop=True)
        return df

    def _build_class_mapping(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        class_names = sorted(df["label_name"].unique())
        df["class_id"] = df["label_name"].map(
            {name: i for i, name in enumerate(class_names)}
        )
        print(f"\nDetected {len(class_names)} classes:")
        for cid, name in enumerate(class_names):
            print(f"  {cid}: {name} ({(df['class_id'] == cid).sum()} annotations)")
        return class_names, df

    def _bbox_from_points(points_str: str) -> Tuple[float, float, float, float]:
        pts = json.loads(points_str)
        if not isinstance(pts, (list, tuple)) or len(pts) < 4 or len(pts) % 2 != 0:
            raise ValueError(
                f"Invalid 'points' payload (must be flat [x1,y1,...]): {points_str}"
            )
        xs = pts[0::2]
        ys = pts[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    def _convert_annotations_to_yolo(df: pd.DataFrame) -> pd.DataFrame:
        df[["xmin", "ymin", "xmax", "ymax"]] = df["points"].apply(
            lambda s: pd.Series(_bbox_from_points(s))
        )
        df["x_center"] = ((df["xmin"] + df["xmax"]) / 2) / df["img_w"]
        df["y_center"] = ((df["ymin"] + df["ymax"]) / 2) / df["img_h"]
        df["w_norm"] = (df["xmax"] - df["xmin"]) / df["img_w"]
        df["h_norm"] = (df["ymax"] - df["ymin"]) / df["img_h"]
        print(f"\nConverted {len(df)} annotations to YOLO format")
        return df

    def _index_images(images_root: Path, valid_exts: Iterable[str]) -> Dict[str, Path]:
        valid_exts_lower = {e.lower() for e in valid_exts}
        return {
            p.name: p
            for p in images_root.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_exts_lower
        }

    def _split_dataset(
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> Dict[str, str]:
        rng = random.Random(seed)

        img_to_labels = defaultdict(set)
        for row in df.itertuples():
            img_to_labels[row.filename].add(int(row.class_id))

        images = list(img_to_labels.keys())
        n_images = len(images)
        n_train = int(round(train_ratio * n_images))
        n_val = int(round(val_ratio * n_images))
        # test is remainder

        split_for_image: Dict[str, str] = {}
        split_counts = {"train": 0, "val": 0, "test": 0}

        def assign(img: str, split: str) -> bool:
            if img not in split_for_image:
                split_for_image[img] = split
                split_counts[split] += 1
                return True
            return split_for_image[img] == split

        # Ensure class coverage across splits (where possible)
        for cls in sorted(df["class_id"].unique()):
            imgs = [img for img, labels in img_to_labels.items() if cls in labels]
            rng.shuffle(imgs)
            if imgs:
                assign(imgs[0], "train")
            if len(imgs) >= 2:
                for img in imgs[1:]:
                    if assign(img, "val"):
                        break
            if len(imgs) >= 3:
                for img in imgs[2:]:
                    if assign(img, "test"):
                        break

        # Fill remaining to meet target ratios
        remaining = [img for img in images if img not in split_for_image]
        rng.shuffle(remaining)
        for img in remaining:
            needs = {
                "train": n_train - split_counts["train"],
                "val": n_val - split_counts["val"],
                "test": (n_images - n_train - n_val) - split_counts["test"],
            }
            max_need = max(needs.values())
            if max_need <= 0:
                assign(img, "train")
            else:
                assign(img, rng.choice([s for s, n in needs.items() if n == max_need]))

        print(
            f"[split] Images: {n_images} | train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
        )
        return split_for_image

    def _ensure_split_dirs(dataset_dir: Path) -> None:
        split_map = {"train": "train", "val": "valid", "test": "test"}
        for folder in split_map.values():
            (dataset_dir / folder / "images").mkdir(parents=True, exist_ok=True)
            (dataset_dir / folder / "labels").mkdir(parents=True, exist_ok=True)

    def _write_split_files(
        df: pd.DataFrame,
        split_for_image: Mapping[str, str],
        image_index: Mapping[str, Path],
        dataset_dir: Path,
    ) -> Dict[str, Dict[str, int]]:
        _ensure_split_dirs(dataset_dir)
        split_map = {"train": "train", "val": "valid", "test": "test"}
        grouped = {fn: g for fn, g in df.groupby("filename")}
        stats = {split_map[s]: {"images": 0, "annotations": 0} for s in split_map}

        for filename, split in split_for_image.items():
            folder = split_map[split]
            src_img = image_index.get(filename)
            if src_img is None:
                continue

            shutil.copy2(src_img, dataset_dir / folder / "images" / filename)

            label_lines: List[str] = []
            if filename in grouped:
                for row in grouped[filename].itertuples():
                    label_lines.append(
                        f"{int(row.class_id)} {row.x_center:.6f} {row.y_center:.6f} "
                        f"{row.w_norm:.6f} {row.h_norm:.6f}"
                    )

            (dataset_dir / folder / "labels" / f"{Path(filename).stem}.txt").write_text(
                "\n".join(label_lines)
            )

            stats[folder]["images"] += 1
            stats[folder]["annotations"] += len(label_lines)

        print(f"Dataset created")
        for folder, s in stats.items():
            print(f"  {folder}: {s['images']} images, {s['annotations']} annotations")
        return stats

    def _transform_bbox(
        xc: float, yc: float, w: float, h: float, op: str
    ) -> Tuple[float, float, float, float]:
        if op == "hflip":
            return 1 - xc, yc, w, h
        if op == "vflip":
            return xc, 1 - yc, w, h
        if op == "rot180":
            return 1 - xc, 1 - yc, w, h
        raise ValueError(f"Unknown op: {op}")

    def _apply_image_op(img: Image.Image, op: str) -> Image.Image:
        if op == "hflip":
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if op == "vflip":
            return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if op == "rot180":
            return img.rotate(180, expand=True)
        raise ValueError(f"Unknown op: {op}")

    def _augment_training_set(
        dataset_dir: Path,
        valid_exts: Iterable[str],
        augment_ops: Sequence[str],
        augment_factor: float,
        seed: int,
    ) -> Tuple[int, int]:
        if augment_factor <= 0:
            print("Augmentation disabled.")
            return 0, 0

        train_img_dir = dataset_dir / "train" / "images"
        train_lbl_dir = dataset_dir / "train" / "labels"
        valid_exts_lower = {e.lower() for e in valid_exts}

        train_imgs = sorted(
            p
            for p in train_img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts_lower
        )
        n_original = len(train_imgs)
        n_target = int(round(n_original * augment_factor))

        if n_target == 0 or n_original == 0:
            print("No augmentation needed.")
            return 0, n_original

        rng = random.Random(seed + 1)
        selected = rng.sample(train_imgs, k=min(n_target, n_original))

        created = 0
        for img_path in selected:
            label_path = train_lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            raw = label_path.read_text().strip()
            if not raw:
                continue

            op = rng.choice(list(augment_ops))
            new_labels: List[str] = []
            for line in raw.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                xc, yc, w, h = map(float, parts[1:5])
                xc, yc, w, h = _transform_bbox(xc, yc, w, h, op)

                # Clip to [0,1]
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                new_labels.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if not new_labels:
                continue

            img = Image.open(img_path)
            img_aug = _apply_image_op(img, op)
            suffix = f"_aug_{op}"
            img_aug.save(train_img_dir / f"{img_path.stem}{suffix}{img_path.suffix}")
            (train_lbl_dir / f"{img_path.stem}{suffix}.txt").write_text(
                "\n".join(new_labels)
            )
            created += 1

        print(f"Created {created} augmented images")
        print(f"  Train set: {n_original} -> {n_original + created} images")
        return created, n_original

    def _write_data_yaml(dataset_dir: Path, class_names: Sequence[str]) -> Path:
        data_yaml_path = dataset_dir / "data.yaml"
        data_cfg = {
            "path": str(dataset_dir),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": list(class_names),
        }
        with open(data_yaml_path, "w") as f:
            yaml.safe_dump(data_cfg, f, sort_keys=False)
        return data_yaml_path

    # ----------------------------
    # Orchestration (calls the nested helpers)
    # ----------------------------
    _validate_paths(csv_path, images_root, dataset_dir)
    df = _read_biigle_csv(csv_path)
    _validate_required_columns(df, REQUIRED_COLS)

    df = _filter_supported_shapes(df, SUPPORTED_SHAPES)
    df = _parse_image_dimensions(df)
    class_names, df = _build_class_mapping(df)

    df = _convert_annotations_to_yolo(df)
    image_index = _index_images(images_root, valid_exts)
    split_for_image = _split_dataset(
        df, train_ratio, val_ratio, test_ratio, random_seed
    )

    _write_split_files(df, split_for_image, image_index, dataset_dir)

    if augment_train:
        _augment_training_set(
            dataset_dir=dataset_dir,
            valid_exts=valid_exts,
            augment_ops=AUGMENT_OPS,
            augment_factor=augment_factor,
            seed=random_seed,
        )
    else:
        print("Augmentation disabled.")

    return _write_data_yaml(dataset_dir, class_names)


def biigle_yolo_detection(
    biigle_csv_path: str,
    images_root: str,
    dataset_dir: str,
):

    BIIGLE_CSV_PATH = Path(biigle_csv_path).expanduser()
    IMAGES_ROOT = Path(images_root).expanduser()
    OUTPUT_ROOT = Path(dataset_dir).expanduser()
    # Class source: 'label_name' is the exact label (usually the species). Use 'label_hierarchy'
    # to fold species up into parent classes instead.
    CLASS_COLUMN = "label_name"
    # Labels to drop (calibration points etc.). Case-insensitive.
    EXCLUDE_LABELS = {"Laser point", "laser point", "Laser Point"}
    # Split ratios (must sum to 1).
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.2, 0.1
    # Rare-class filter: a class is kept only with at least this many annotations AND images,
    # because image-level splits cannot place a one-image class in all three splits.
    MIN_ANNOTATIONS_PER_CLASS = 3
    MIN_IMAGES_PER_CLASS = 3
    # Background (annotation-free) frames. True + None = include all available.
    INCLUDE_NEGATIVE_IMAGES = True
    NEGATIVE_IMAGE_RATIO = (
        None  # e.g. 0.5 = ~half as many negatives as positives; 0 = none
    )
    # Group-aware split: keep all frames of one source video in the same split, so near-identical
    # frames do not leak across train/test. The group key is the filename before '_frame_'.
    # Falls back to image-level automatically when fewer than 3 groups are found (e.g. a single
    # compilation clip), so it is safe to leave on.
    GROUP_AWARE_SPLIT = True
    RANDOM_SEED = 42
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    DATASET_DIR = OUTPUT_ROOT
    assert (
        abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 0.01
    ), "Split ratios must sum to 1.0"
    assert BIIGLE_CSV_PATH.is_file(), f"CSV not found: {BIIGLE_CSV_PATH}"
    assert IMAGES_ROOT.is_dir(), f"Images dir not found: {IMAGES_ROOT}"
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"Warning: output dir not empty, may overwrite: {DATASET_DIR}")
    else:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CSV:     {BIIGLE_CSV_PATH}")
    print(f"Images:  {IMAGES_ROOT}")
    print(f"Output:  {DATASET_DIR}")
    print(
        f"Classes: {CLASS_COLUMN} | Splits: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO} | "
        f"Group-aware: {GROUP_AWARE_SPLIT}"
    )
    ## Phase 2: Load and convert to boxes
    df = pd.read_csv(BIIGLE_CSV_PATH)
    print(f"Loaded {len(df)} annotations")
    audit = {"loaded": int(len(df))}

    required = [
        "filename",
        "label_name",
        "shape_name",
        "points",
        "attributes",
        CLASS_COLUMN,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Exclude non-training labels (e.g. laser calibration points)
    if EXCLUDE_LABELS:
        exclude_lower = {s.lower() for s in EXCLUDE_LABELS}
        n = len(df)
        df = df[
            ~df["label_name"].astype(str).str.lower().isin(exclude_lower)
        ].reset_index(drop=True)
        if n - len(df):
            print(f"Excluded {n - len(df)} annotations from EXCLUDE_LABELS")
        audit["excluded_labels"] = int(n - len(df))

    # Keep box-able shapes only
    df = df[df["shape_name"].isin(["Rectangle", "Polygon"])].reset_index(drop=True)
    print(f"Rectangle/Polygon annotations: {len(df)}")
    audit["unsupported_shape"] = int(
        audit["loaded"] - audit.get("excluded_labels", 0) - len(df)
    )
    if df.empty:
        raise ValueError("No Rectangle/Polygon annotations remain after filtering.")

    # Image dimensions from the attributes JSON (tolerant of malformed cells)
    def parse_attrs(s):
        try:
            return json.loads(s)
        except Exception:
            return {}

    attrs = df["attributes"].apply(parse_attrs)
    df["img_w"] = [d.get("width") for d in attrs]
    df["img_h"] = [d.get("height") for d in attrs]
    n = len(df)
    df = df.dropna(subset=["img_w", "img_h"]).reset_index(drop=True)
    if n - len(df):
        print(f"Dropped {n - len(df)} annotations missing image width/height")
    audit["missing_dimensions"] = int(n - len(df))

    # Points -> bounding box. Returns None for truncated/unparseable payloads (BIIGLE's 32767-char
    # CSV field limit can cut long polygon points), which are then dropped rather than crashing.
    def bbox_from_points(points_str):
        try:
            pts = json.loads(points_str)
        except Exception:
            return None
        if not isinstance(pts, (list, tuple)) or len(pts) < 4 or len(pts) % 2 != 0:
            return None
        xs, ys = pts[0::2], pts[1::2]
        return (min(xs), min(ys), max(xs), max(ys))

    parsed = df["points"].apply(bbox_from_points)
    n_bad = parsed.isna().sum()
    if n_bad:
        print(f"Dropped {n_bad} annotations with unparseable/truncated points")
    audit["unparseable_points"] = int(n_bad)
    df = df[parsed.notna()].reset_index(drop=True)
    parsed = parsed[parsed.notna()].reset_index(drop=True)
    df[["xmin_raw", "ymin_raw", "xmax_raw", "ymax_raw"]] = pd.DataFrame(parsed.tolist())

    # Clip boxes to the image. Off-screen parts are trimmed to the edge, so a half-visible animal
    # keeps a valid box around its visible portion. Boxes that fall entirely outside collapse and
    # are dropped next.
    oob = (
        (df["xmin_raw"] < 0)
        | (df["ymin_raw"] < 0)
        | (df["xmax_raw"] > df["img_w"])
        | (df["ymax_raw"] > df["img_h"])
    ).sum()
    if oob:
        print(f"Clipped {oob} boxes extending past the image boundary")
    audit["clipped_boxes"] = int(oob)
    df["xmin"] = df["xmin_raw"].clip(lower=0)
    df["ymin"] = df["ymin_raw"].clip(lower=0)
    df["xmax"] = df[["xmax_raw", "img_w"]].min(axis=1)
    df["ymax"] = df[["ymax_raw", "img_h"]].min(axis=1)
    n = len(df)
    df = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])].reset_index(
        drop=True
    )
    if n - len(df):
        print(f"Dropped {n - len(df)} degenerate boxes after clipping")
    audit["degenerate_boxes"] = int(n - len(df))

    # Rare-class filter (needs enough annotations AND enough distinct images)
    df["class_label"] = df[CLASS_COLUMN].astype(str)
    summary = df.groupby("class_label").agg(
        annotations=("class_label", "size"), images=("filename", "nunique")
    )
    rare = summary[
        (summary["annotations"] < MIN_ANNOTATIONS_PER_CLASS)
        | (summary["images"] < MIN_IMAGES_PER_CLASS)
    ]
    if not rare.empty:
        print(
            f"\nRemoving {len(rare)} rare class(es) (< {MIN_ANNOTATIONS_PER_CLASS} annotations "
            f"or < {MIN_IMAGES_PER_CLASS} images):"
        )
        for name, row in rare.iterrows():
            print(
                f"  - {name}: {int(row['annotations'])} annotations, {int(row['images'])} images"
            )
        n_before_rare = len(df)
        df = df[~df["class_label"].isin(rare.index)].reset_index(drop=True)
        audit["rare_classes_removed"] = [str(x) for x in rare.index]
        audit["rare_class_annotations_dropped"] = int(n_before_rare - len(df))
    if df.empty:
        raise ValueError("No annotations remain after rare-class filtering.")

    # Contiguous class ids after filtering
    class_names = sorted(df["class_label"].unique())
    df["class_id"] = df["class_label"].map({n: i for i, n in enumerate(class_names)})
    print(f"\nRetained {len(class_names)} classes:")
    kept = (
        df.groupby("class_label")
        .agg(annotations=("class_label", "size"), images=("filename", "nunique"))
        .loc[class_names]
    )
    for cid, name in enumerate(class_names):
        r = kept.loc[name]
        print(
            f"  {cid}: {name} ({int(r['annotations'])} annotations, {int(r['images'])} images)"
        )

    # Normalized YOLO coordinates + safety check
    df["x_center"] = ((df["xmin"] + df["xmax"]) / 2) / df["img_w"]
    df["y_center"] = ((df["ymin"] + df["ymax"]) / 2) / df["img_h"]
    df["w_norm"] = (df["xmax"] - df["xmin"]) / df["img_w"]
    df["h_norm"] = (df["ymax"] - df["ymin"]) / df["img_h"]
    cols = ["x_center", "y_center", "w_norm", "h_norm"]
    if not df[cols].apply(lambda s: s.between(0, 1).all()).all():
        raise ValueError("Found YOLO coordinates outside [0, 1] after clipping")
    print(f"\nConverted {len(df)} annotations to YOLO boxes")

    audit.setdefault("excluded_labels", 0)
    audit.setdefault("rare_classes_removed", [])
    audit.setdefault("rare_class_annotations_dropped", 0)
    audit["retained_annotations"] = int(len(df))

    ## Phase 3: Split and build the dataset
    rng = random.Random(RANDOM_SEED)

    # Index images on disk
    image_index = {
        p.name: p
        for p in IMAGES_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }

    csv_filenames = set(df["filename"].unique())
    unmatched = csv_filenames - set(image_index)
    if unmatched:
        print(
            f"Warning: {len(unmatched)} CSV filenames not found in {IMAGES_ROOT.name}/"
        )
        for fn in sorted(unmatched)[:10]:
            print(f"  - {fn}")
        if len(unmatched) == len(csv_filenames):
            raise FileNotFoundError(
                "None of the CSV filenames were found on disk. Check IMAGES_ROOT."
            )

    df_matched = df[df["filename"].isin(image_index)].copy().reset_index(drop=True)
    if df_matched.empty:
        raise ValueError("No annotations left after matching filenames to images.")

    img_to_labels = defaultdict(set)
    class_to_imgs = defaultdict(set)
    for row in df_matched.itertuples():
        img_to_labels[row.filename].add(int(row.class_id))
        class_to_imgs[int(row.class_id)].add(row.filename)
    positive_images = sorted(img_to_labels)

    # True negatives: frames never annotated in BIIGLE at all. Frames whose only annotations were
    # filtered out (rare classes, excluded labels) are NOT treated as negatives, to avoid fake blanks.
    all_annotated = set(pd.read_csv(BIIGLE_CSV_PATH)["filename"].unique())
    neg_candidates = sorted(set(image_index) - all_annotated)
    negatives = []
    if INCLUDE_NEGATIVE_IMAGES and neg_candidates:
        rng.shuffle(neg_candidates)
        if NEGATIVE_IMAGE_RATIO is None:
            n_neg = len(neg_candidates)
        else:
            n_neg = min(
                int(round(len(positive_images) * NEGATIVE_IMAGE_RATIO)),
                len(neg_candidates),
            )
        negatives = neg_candidates[:n_neg]
        print(
            f"Including {len(negatives)} background frames (of {len(neg_candidates)} available)"
        )
    else:
        print(f"No background frames included ({len(neg_candidates)} available)")

    images = sorted(positive_images + negatives)
    n_images = len(images)
    n_train = round(TRAIN_RATIO * n_images)
    n_val = round(VAL_RATIO * n_images)
    n_test = n_images - n_train - n_val

    split_for_image = {}
    counts = {"train": 0, "val": 0, "test": 0}

    def assign(img, split):
        if img in split_for_image:
            return split_for_image[img] == split
        split_for_image[img] = split
        counts[split] += 1
        return True

    # Source video = the filename before '_frame_', matching our frame-extraction convention
    # (<video>_frame_<number>.jpg). A filename that does not follow it tells us nothing about which
    # video it came from, so rather than treating it as a video of its own we fall back to the
    # image-level split for the whole dataset.
    def group_of(filename):
        m = re.match(r"(.+?)_frame_\d+$", Path(filename).stem)
        return m.group(1) if m else None

    group_of_image = {img: group_of(img) for img in images}
    unrecognized = sorted(img for img, g in group_of_image.items() if g is None)
    groups = defaultdict(list)
    for img, g in group_of_image.items():
        if g is not None:
            groups[g].append(img)

    use_group = GROUP_AWARE_SPLIT and not unrecognized and len(groups) >= 3
    if GROUP_AWARE_SPLIT and unrecognized:
        print(
            f"{len(unrecognized)} filename(s) do not follow <video>_frame_<number>, so the source "
            f"video cannot be inferred. Using an image-level split. Examples:"
        )
        for fn in unrecognized[:5]:
            print(f"  - {fn}")

    if use_group:
        # Assign whole videos. Seed one video into each split first (smallest target first) so no
        # split can end up empty, then fill by largest RELATIVE shortfall. Absolute shortfall with
        # equal-sized groups ties toward train and can leave test with zero images.
        print(f"Group-aware split across {len(groups)} source videos")
        targets = {"train": n_train, "val": n_val, "test": n_test}
        positive_set = set(positive_images)
        # Seed from videos that actually contain annotations, so no split ends up holding only
        # background frames (a test split with no ground truth cannot produce a meaningful mAP).
        with_pos = sorted(
            (g for g in groups if any(i in positive_set for i in groups[g])),
            key=lambda g: sum(i in positive_set for i in groups[g]),
            reverse=True,
        )
        seeded = with_pos[: len(targets)]
        rest = [g for g in groups if g not in set(seeded)]
        seed_order = sorted(targets, key=lambda s: targets[s])
        for split, g in zip(seed_order, seeded):
            for img in groups[g]:
                assign(img, split)
        for g in sorted(rest, key=lambda g: len(groups[g]), reverse=True):
            shortfall = {
                s: (targets[s] - counts[s]) / max(targets[s], 1) for s in targets
            }
            split = max(shortfall, key=shortfall.get)
            for img in groups[g]:
                assign(img, split)
    else:
        if GROUP_AWARE_SPLIT and not unrecognized:
            print(
                f"Only {len(groups)} source video group(s) found: using an image-level split"
            )

        # Coverage-first: try to place each class in every split, then fill to ratios.
        def choose(cls):
            # sorted() on both the candidate list and the tie-break key: class_to_imgs holds sets,
            # whose iteration order changes between Python processes, so without a total ordering
            # the same seed produces a different split on every run.
            cand = sorted(
                i for i in class_to_imgs.get(cls, set()) if i not in split_for_image
            )
            if not cand:
                return None
            cand.sort(key=lambda i: (len(img_to_labels[i]), i))
            return cand[0]

        for split in ["train", "val", "test"]:
            for cls in sorted(class_to_imgs):
                if any(
                    split_for_image.get(i) == split for i in sorted(class_to_imgs[cls])
                ):
                    continue
                img = choose(cls)
                if img is not None:
                    assign(img, split)
        remaining = [i for i in images if i not in split_for_image]
        rng.shuffle(remaining)
        for img in remaining:
            needs = {
                "train": n_train - counts["train"],
                "val": n_val - counts["val"],
                "test": n_test - counts["test"],
            }
            if max(needs.values()) <= 0:
                assign(img, "train")
            else:
                top = max(needs.values())
                assign(img, rng.choice([s for s, v in needs.items() if v == top]))

    empty = [s for s, c in counts.items() if c == 0]
    if empty:
        raise ValueError(
            f"Split(s) {empty} ended up empty. A dataset without a test split cannot give an "
            f"honest final metric. Add more source videos, adjust the ratios, or set "
            f"GROUP_AWARE_SPLIT = False."
        )

    # Write images + labels
    SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}
    for folder in SPLIT_MAP.values():
        (DATASET_DIR / folder / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / folder / "labels").mkdir(parents=True, exist_ok=True)

    grouped = {fn: g for fn, g in df_matched.groupby("filename")}
    positive_set = set(positive_images)
    stats = {
        f: {"images": 0, "annotations": 0, "positive": 0, "negative": 0}
        for f in SPLIT_MAP.values()
    }

    for filename, split in split_for_image.items():
        folder = SPLIT_MAP[split]
        src = image_index.get(filename)
        if src is None:
            continue
        shutil.copy2(src, DATASET_DIR / folder / "images" / filename)
        lines = []
        if filename in grouped:
            for row in grouped[filename].itertuples():
                lines.append(
                    f"{int(row.class_id)} {row.x_center:.6f} {row.y_center:.6f} "
                    f"{row.w_norm:.6f} {row.h_norm:.6f}"
                )
        (DATASET_DIR / folder / "labels" / f"{Path(filename).stem}.txt").write_text(
            "\n".join(lines)
        )
        stats[folder]["images"] += 1
        stats[folder]["annotations"] += len(lines)
        stats[folder]["positive" if filename in positive_set else "negative"] += 1

    print(
        f"\nDataset created from {n_images} images "
        f"({len(positive_images)} positive, {len(negatives)} background):"
    )
    for folder, s in stats.items():
        print(
            f"  {folder}: {s['images']} images ({s['positive']} pos, {s['negative']} bg), "
            f"{s['annotations']} annotations"
        )

    # Per-class image coverage across splits
    blank = [f for f in ("train", "test") if stats[f]["annotations"] == 0]
    if blank:
        raise ValueError(
            f"Split(s) {blank} contain images but no annotations, so they cannot train or "
            f"evaluate anything. This usually means the videos holding your annotated frames "
            f"all landed in one split. Adjust the ratios or set GROUP_AWARE_SPLIT = False."
        )

    # Split provenance, recorded so it can be written into the project YAML later
    split_method = "video" if use_group else "image"
    source_groups = len(groups) if use_group else 0
    print(
        f"\nSplit method: {split_method}"
        + (f" ({source_groups} source videos)" if use_group else "")
    )

    coverage = {}
    df_matched["split"] = df_matched["filename"].map(split_for_image)
    print("\nPer-class image coverage:")
    print(f"  {'Class':<32s} {'train':>6s} {'valid':>6s} {'test':>6s} {'status':>8s}")
    warn = []
    for cid, name in enumerate(class_names):
        rows = df_matched[df_matched["class_id"] == cid][
            ["filename", "split"]
        ].drop_duplicates()
        c = rows["split"].value_counts()
        tr, va, te = int(c.get("train", 0)), int(c.get("val", 0)), int(c.get("test", 0))
        ok = tr > 0 and va > 0 and te > 0
        if not ok:
            warn.append(name)
        coverage[name] = {
            "train": tr,
            "valid": va,
            "test": te,
            "status": "ok" if ok else "check",
        }
        print(f"  {name:<32s} {tr:>6d} {va:>6d} {te:>6d} {'ok' if ok else 'check':>8s}")
    if warn:
        print(
            "\nNote: these classes are not in every split (expected under group-aware "
            "splitting when a class lives in only one video):"
        )
        for name in warn:
            print(f"  - {name}")
    else:
        print("\nEvery retained class appears in train, valid, and test.")
    ## Phase 4: Write data.yaml
    data_yaml_path = DATASET_DIR / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(
            {
                "path": str(DATASET_DIR),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": len(class_names),
                "names": class_names,
            },
            f,
            sort_keys=False,
        )

    print(f"Wrote {data_yaml_path}\n")
    print(data_yaml_path.read_text())

    # Structured summary of everything this converter did. Printed above for the user, and returned
    # here as data so NB01/NB05 can record the dataset composition in the project YAML without
    # recomputing it. This is the object the backend converter function should return.
    converter_result = {
        "data_yaml": str(data_yaml_path),
        "task": "detect",
        "classes": list(class_names),
        "n_classes": len(class_names),
        "split_method": split_method,
        "source_groups": source_groups,
        "splits": {folder: dict(s) for folder, s in stats.items()},
        "audit": dict(audit),
        "coverage": coverage,
    }
    print(
        f"\nConverter summary stored in `converter_result` "
        f"(keys: {', '.join(converter_result)}).\n"
        f"Run `converter_result` in a cell to inspect it, or pass it on to the project setup."
    )
    return data_yaml_path


def biigle_yolo_segmentation(
    biigle_csv_path: str,
    images_root: str,
    dataset_dir: str,
):
    # Paths (absolute, or ~). Point CSV at the BIIGLE export, IMAGES_ROOT at the raw frames.
    BIIGLE_CSV_PATH = Path(biigle_csv_path).expanduser()
    IMAGES_ROOT = Path(images_root).expanduser()
    OUTPUT_ROOT = Path(dataset_dir).expanduser()

    # Class source: 'label_name' is the exact label (usually the species). Use 'label_hierarchy'
    # to fold species up into parent classes instead.
    CLASS_COLUMN = "label_name"

    # Labels to drop (calibration points etc.). Case-insensitive.
    EXCLUDE_LABELS = {"Laser point", "laser point", "Laser Point"}

    # Split ratios (must sum to 1).
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.2, 0.1

    # Rare-class filter: keep a class only with at least this many annotations AND images.
    MIN_ANNOTATIONS_PER_CLASS = 3
    MIN_IMAGES_PER_CLASS = 3

    # Background (annotation-free) frames. True + None = include all available. Written as empty masks.
    INCLUDE_NEGATIVE_IMAGES = True
    NEGATIVE_IMAGE_RATIO = None

    # Group-aware split by source video (filename before '_frame_'); auto-falls back to image-level
    # when fewer than 3 groups are found.
    GROUP_AWARE_SPLIT = True

    # Polygon handling. MIN_VERTICES drops degenerate shapes; SIMPLIFY_TOLERANCE (pixels) thins dense
    # polygons via cv2.approxPolyDP (0 = keep every vertex; 1.0-2.0 good for 100+ vertex outlines).
    MIN_VERTICES = 3
    SIMPLIFY_TOLERANCE = 1.5

    RANDOM_SEED = 42
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    DATASET_DIR = OUTPUT_ROOT
    assert (
        abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 0.01
    ), "Split ratios must sum to 1.0"
    assert BIIGLE_CSV_PATH.is_file(), f"CSV not found: {BIIGLE_CSV_PATH}"
    assert IMAGES_ROOT.is_dir(), f"Images dir not found: {IMAGES_ROOT}"
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"Warning: output dir not empty, may overwrite: {DATASET_DIR}")
    else:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CSV:     {BIIGLE_CSV_PATH}")
    print(f"Images:  {IMAGES_ROOT}")
    print(f"Output:  {DATASET_DIR}")
    print(
        f"Classes: {CLASS_COLUMN} | Splits: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO} | "
        f"Group-aware: {GROUP_AWARE_SPLIT} | Simplify: {SIMPLIFY_TOLERANCE}px"
    )
    ## Phase 2: Load and convert to polygons
    df = pd.read_csv(BIIGLE_CSV_PATH)
    print(f"Loaded {len(df)} annotations")
    audit = {"loaded": int(len(df))}

    required = [
        "filename",
        "label_name",
        "shape_name",
        "points",
        "attributes",
        CLASS_COLUMN,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Exclude non-training labels
    if EXCLUDE_LABELS:
        exclude_lower = {s.lower() for s in EXCLUDE_LABELS}
        n = len(df)
        df = df[
            ~df["label_name"].astype(str).str.lower().isin(exclude_lower)
        ].reset_index(drop=True)
        if n - len(df):
            print(f"Excluded {n - len(df)} annotations from EXCLUDE_LABELS")
        audit["excluded_labels"] = int(n - len(df))

    # Polygons only
    shape_counts = df["shape_name"].value_counts().to_dict()
    df = df[df["shape_name"] == "Polygon"].reset_index(drop=True)
    print(
        f"Polygon annotations: {len(df)} (other shapes skipped: "
        f"{ {k: v for k, v in shape_counts.items() if k != 'Polygon'} })"
    )
    if df.empty:
        raise ValueError(
            "No Polygon annotations found. For boxes, use the detection converter."
        )

    # Image dimensions from attributes (tolerant)
    def parse_attrs(s):
        try:
            return json.loads(s)
        except Exception:
            return {}

    attrs = df["attributes"].apply(parse_attrs)
    df["img_w"] = [d.get("width") for d in attrs]
    df["img_h"] = [d.get("height") for d in attrs]
    n = len(df)
    df = df.dropna(subset=["img_w", "img_h"]).reset_index(drop=True)
    if n - len(df):
        print(f"Dropped {n - len(df)} annotations missing image width/height")
    audit["missing_dimensions"] = int(n - len(df))

    # Points -> normalized polygon. Tolerant of BIIGLE's 32767-char field truncation (returns None).
    def normalize_polygon(points_str, img_w, img_h, tol):
        try:
            pts = json.loads(points_str)
        except Exception:
            return None
        if len(pts) % 2 != 0:
            pts = pts[:-1]
        xs, ys = pts[0::2], pts[1::2]
        if len(xs) > 1 and xs[0] == xs[-1] and ys[0] == ys[-1]:
            xs, ys = xs[:-1], ys[:-1]  # drop BIIGLE's repeated closing vertex
        if len(xs) < MIN_VERTICES:
            return None
        if tol > 0 and len(xs) > MIN_VERTICES:
            contour = np.array(list(zip(xs, ys)), dtype=np.float32).reshape((-1, 1, 2))
            approx = cv2.approxPolyDP(contour, tol, closed=True).reshape(-1, 2)
            if len(approx) >= MIN_VERTICES:
                xs, ys = approx[:, 0].tolist(), approx[:, 1].tolist()
        norm = []
        for x, y in zip(xs, ys):
            norm.append(max(0.0, min(1.0, x / img_w)))
            norm.append(max(0.0, min(1.0, y / img_h)))
        return norm

    df["seg_coords"] = df.apply(
        lambda r: normalize_polygon(
            r["points"], r["img_w"], r["img_h"], SIMPLIFY_TOLERANCE
        ),
        axis=1,
    )
    n_bad = df["seg_coords"].isna().sum()
    if n_bad:
        print(f"Dropped {n_bad} annotations with truncated/degenerate polygons")
    audit["truncated_polygons"] = int(n_bad)
    df = df[df["seg_coords"].notna()].reset_index(drop=True)

    # Rare-class filter (annotations AND distinct images)
    df["class_label"] = df[CLASS_COLUMN].astype(str)
    summary = df.groupby("class_label").agg(
        annotations=("class_label", "size"), images=("filename", "nunique")
    )
    rare = summary[
        (summary["annotations"] < MIN_ANNOTATIONS_PER_CLASS)
        | (summary["images"] < MIN_IMAGES_PER_CLASS)
    ]
    if not rare.empty:
        print(f"\nRemoving {len(rare)} rare class(es):")
        for name, row in rare.iterrows():
            print(
                f"  - {name}: {int(row['annotations'])} annotations, {int(row['images'])} images"
            )
        n_before_rare = len(df)
        df = df[~df["class_label"].isin(rare.index)].reset_index(drop=True)
        audit["rare_classes_removed"] = [str(x) for x in rare.index]
        audit["rare_class_annotations_dropped"] = int(n_before_rare - len(df))
    if df.empty:
        raise ValueError("No annotations remain after rare-class filtering.")

    class_names = sorted(df["class_label"].unique())
    df["class_id"] = df["class_label"].map({n: i for i, n in enumerate(class_names)})
    print(f"\nRetained {len(class_names)} classes:")
    kept = (
        df.groupby("class_label")
        .agg(annotations=("class_label", "size"), images=("filename", "nunique"))
        .loc[class_names]
    )
    for cid, name in enumerate(class_names):
        r = kept.loc[name]
        print(
            f"  {cid}: {name} ({int(r['annotations'])} annotations, {int(r['images'])} images)"
        )
    print(f"\nConverted {len(df)} annotations to YOLO polygons")

    audit.setdefault("excluded_labels", 0)
    audit.setdefault("rare_classes_removed", [])
    audit.setdefault("rare_class_annotations_dropped", 0)
    audit["retained_annotations"] = int(len(df))
    ## Phase 3: Split and build the dataset
    rng = random.Random(RANDOM_SEED)

    image_index = {
        p.name: p
        for p in IMAGES_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }

    csv_filenames = set(df["filename"].unique())
    unmatched = csv_filenames - set(image_index)
    if unmatched:
        print(
            f"Warning: {len(unmatched)} CSV filenames not found in {IMAGES_ROOT.name}/"
        )
        for fn in sorted(unmatched)[:10]:
            print(f"  - {fn}")
        if len(unmatched) == len(csv_filenames):
            raise FileNotFoundError(
                "None of the CSV filenames were found on disk. Check IMAGES_ROOT."
            )

    df_matched = df[df["filename"].isin(image_index)].copy().reset_index(drop=True)
    if df_matched.empty:
        raise ValueError("No annotations left after matching filenames to images.")

    img_to_labels = defaultdict(set)
    class_to_imgs = defaultdict(set)
    for row in df_matched.itertuples():
        img_to_labels[row.filename].add(int(row.class_id))
        class_to_imgs[int(row.class_id)].add(row.filename)
    positive_images = sorted(img_to_labels)

    all_annotated = set(pd.read_csv(BIIGLE_CSV_PATH)["filename"].unique())
    neg_candidates = sorted(set(image_index) - all_annotated)
    negatives = []
    if INCLUDE_NEGATIVE_IMAGES and neg_candidates:
        rng.shuffle(neg_candidates)
        if NEGATIVE_IMAGE_RATIO is None:
            n_neg = len(neg_candidates)
        else:
            n_neg = min(
                int(round(len(positive_images) * NEGATIVE_IMAGE_RATIO)),
                len(neg_candidates),
            )
        negatives = neg_candidates[:n_neg]
        print(
            f"Including {len(negatives)} background frames (of {len(neg_candidates)} available)"
        )
    else:
        print(f"No background frames included ({len(neg_candidates)} available)")

    images = sorted(positive_images + negatives)
    n_images = len(images)
    n_train = round(TRAIN_RATIO * n_images)
    n_val = round(VAL_RATIO * n_images)
    n_test = n_images - n_train - n_val

    split_for_image = {}
    counts = {"train": 0, "val": 0, "test": 0}

    def assign(img, split):
        if img in split_for_image:
            return split_for_image[img] == split
        split_for_image[img] = split
        counts[split] += 1
        return True

    # Source video = the filename before '_frame_', matching our frame-extraction convention
    # (<video>_frame_<number>.jpg). A filename that does not follow it tells us nothing about which
    # video it came from, so rather than treating it as a video of its own we fall back to the
    # image-level split for the whole dataset.
    def group_of(filename):
        m = re.match(r"(.+?)_frame_\d+$", Path(filename).stem)
        return m.group(1) if m else None

    group_of_image = {img: group_of(img) for img in images}
    unrecognized = sorted(img for img, g in group_of_image.items() if g is None)
    groups = defaultdict(list)
    for img, g in group_of_image.items():
        if g is not None:
            groups[g].append(img)

    use_group = GROUP_AWARE_SPLIT and not unrecognized and len(groups) >= 3
    if GROUP_AWARE_SPLIT and unrecognized:
        print(
            f"{len(unrecognized)} filename(s) do not follow <video>_frame_<number>, so the source "
            f"video cannot be inferred. Using an image-level split. Examples:"
        )
        for fn in unrecognized[:5]:
            print(f"  - {fn}")

    if use_group:
        # Assign whole videos. Seed one video into each split first (smallest target first) so no
        # split can end up empty, then fill by largest RELATIVE shortfall. Absolute shortfall with
        # equal-sized groups ties toward train and can leave test with zero images.
        print(f"Group-aware split across {len(groups)} source videos")
        targets = {"train": n_train, "val": n_val, "test": n_test}
        positive_set = set(positive_images)
        # Seed from videos that actually contain annotations, so no split ends up holding only
        # background frames (a test split with no ground truth cannot produce a meaningful mAP).
        with_pos = sorted(
            (g for g in groups if any(i in positive_set for i in groups[g])),
            key=lambda g: sum(i in positive_set for i in groups[g]),
            reverse=True,
        )
        seeded = with_pos[: len(targets)]
        rest = [g for g in groups if g not in set(seeded)]
        seed_order = sorted(targets, key=lambda s: targets[s])
        for split, g in zip(seed_order, seeded):
            for img in groups[g]:
                assign(img, split)
        for g in sorted(rest, key=lambda g: len(groups[g]), reverse=True):
            shortfall = {
                s: (targets[s] - counts[s]) / max(targets[s], 1) for s in targets
            }
            split = max(shortfall, key=shortfall.get)
            for img in groups[g]:
                assign(img, split)
    else:
        if GROUP_AWARE_SPLIT and not unrecognized:
            print(
                f"Only {len(groups)} source video group(s) found: using an image-level split"
            )

        def choose(cls):
            # sorted() on both the candidate list and the tie-break key: class_to_imgs holds sets,
            # whose iteration order changes between Python processes, so without a total ordering
            # the same seed produces a different split on every run.
            cand = sorted(
                i for i in class_to_imgs.get(cls, set()) if i not in split_for_image
            )
            if not cand:
                return None
            cand.sort(key=lambda i: (len(img_to_labels[i]), i))
            return cand[0]

        for split in ["train", "val", "test"]:
            for cls in sorted(class_to_imgs):
                if any(
                    split_for_image.get(i) == split for i in sorted(class_to_imgs[cls])
                ):
                    continue
                img = choose(cls)
                if img is not None:
                    assign(img, split)
        remaining = [i for i in images if i not in split_for_image]
        rng.shuffle(remaining)
        for img in remaining:
            needs = {
                "train": n_train - counts["train"],
                "val": n_val - counts["val"],
                "test": n_test - counts["test"],
            }
            if max(needs.values()) <= 0:
                assign(img, "train")
            else:
                top = max(needs.values())
                assign(img, rng.choice([s for s, v in needs.items() if v == top]))

    empty = [s for s, c in counts.items() if c == 0]
    if empty:
        raise ValueError(
            f"Split(s) {empty} ended up empty. A dataset without a test split cannot give an "
            f"honest final metric. Add more source videos, adjust the ratios, or set "
            f"GROUP_AWARE_SPLIT = False."
        )

    SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}
    for folder in SPLIT_MAP.values():
        (DATASET_DIR / folder / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / folder / "labels").mkdir(parents=True, exist_ok=True)

    grouped = {fn: g for fn, g in df_matched.groupby("filename")}
    positive_set = set(positive_images)
    stats = {
        f: {"images": 0, "annotations": 0, "positive": 0, "negative": 0}
        for f in SPLIT_MAP.values()
    }

    for filename, split in split_for_image.items():
        folder = SPLIT_MAP[split]
        src = image_index.get(filename)
        if src is None:
            continue
        shutil.copy2(src, DATASET_DIR / folder / "images" / filename)
        lines = []
        if filename in grouped:
            for row in grouped[filename].itertuples():
                coords = " ".join(f"{c:.6f}" for c in row.seg_coords)
                lines.append(f"{int(row.class_id)} {coords}")
        (DATASET_DIR / folder / "labels" / f"{Path(filename).stem}.txt").write_text(
            "\n".join(lines)
        )
        stats[folder]["images"] += 1
        stats[folder]["annotations"] += len(lines)
        stats[folder]["positive" if filename in positive_set else "negative"] += 1

    print(
        f"\nDataset created from {n_images} images "
        f"({len(positive_images)} positive, {len(negatives)} background):"
    )
    for folder, s in stats.items():
        print(
            f"  {folder}: {s['images']} images ({s['positive']} pos, {s['negative']} bg), "
            f"{s['annotations']} annotations"
        )

    blank = [f for f in ("train", "test") if stats[f]["annotations"] == 0]
    if blank:
        raise ValueError(
            f"Split(s) {blank} contain images but no annotations, so they cannot train or "
            f"evaluate anything. This usually means the videos holding your annotated frames "
            f"all landed in one split. Adjust the ratios or set GROUP_AWARE_SPLIT = False."
        )

    # Split provenance, recorded so it can be written into the project YAML later
    split_method = "video" if use_group else "image"
    source_groups = len(groups) if use_group else 0
    print(
        f"\nSplit method: {split_method}"
        + (f" ({source_groups} source videos)" if use_group else "")
    )

    coverage = {}
    df_matched["split"] = df_matched["filename"].map(split_for_image)
    print("\nPer-class image coverage:")
    print(f"  {'Class':<32s} {'train':>6s} {'valid':>6s} {'test':>6s} {'status':>8s}")
    warn = []
    for cid, name in enumerate(class_names):
        rows = df_matched[df_matched["class_id"] == cid][
            ["filename", "split"]
        ].drop_duplicates()
        c = rows["split"].value_counts()
        tr, va, te = int(c.get("train", 0)), int(c.get("val", 0)), int(c.get("test", 0))
        ok = tr > 0 and va > 0 and te > 0
        if not ok:
            warn.append(name)
        coverage[name] = {
            "train": tr,
            "valid": va,
            "test": te,
            "status": "ok" if ok else "check",
        }
        print(f"  {name:<32s} {tr:>6d} {va:>6d} {te:>6d} {'ok' if ok else 'check':>8s}")
    if warn:
        print(
            "\nNote: these classes are not in every split (expected under group-aware "
            "splitting when a class lives in only one video):"
        )
        for name in warn:
            print(f"  - {name}")
    else:
        print("\nEvery retained class appears in train, valid, and test.")

    ## Phase 4: Write data.yaml
    data_yaml_path = DATASET_DIR / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(
            {
                "path": str(DATASET_DIR),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": len(class_names),
                "names": class_names,
            },
            f,
            sort_keys=False,
        )

    print(f"Wrote {data_yaml_path}\n")
    print(data_yaml_path.read_text())

    # Structured summary of everything this converter did. Printed above for the user, and returned
    # here as data so NB01/NB05 can record the dataset composition in the project YAML without
    # recomputing it. This is the object the backend converter function should return.
    converter_result = {
        "data_yaml": str(data_yaml_path),
        "task": "segment",
        "classes": list(class_names),
        "n_classes": len(class_names),
        "split_method": split_method,
        "source_groups": source_groups,
        "splits": {folder: dict(s) for folder, s in stats.items()},
        "audit": dict(audit),
        "coverage": coverage,
    }
    print(
        f"\nConverter summary stored in `converter_result` "
        f"(keys: {', '.join(converter_result)}).\n"
        f"Run `converter_result` in a cell to inspect it, or pass it on to the project setup."
    )
    return data_yaml_path


class auto_dataset_generator:
    def __init__(self):

        pass

    def _normalise_extensions(self, extensions: Iterable[str]) -> set:
        out = set()
        for ext in extensions:
            ext = ext.lower().strip()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            out.add(ext)
        return out or set(DEFAULT_EXTENSIONS)

    def list_frames(self, folder: Path, extensions: Sequence[str]) -> List[Path]:
        """Return all matching image files in a folder, sorted by name.

        Skips dotfiles, including macOS AppleDouble metadata files (._foo.jpg)
        that appear on non-native filesystems like exFAT external drives.
        """
        allowed = self._normalise_extensions(extensions)
        return sorted(
            f
            for f in folder.iterdir()
            if f.is_file()
            and f.suffix.lower() in allowed
            and not f.name.startswith(".")
        )

    def sample_stratified(self, items: List[Path], n: int) -> List[Path]:
        """Pick n items spaced evenly through the sequence.

        For training data this is almost always preferable to pure random sampling
        because it guarantees temporal coverage of the full deployment instead of
        risking clusters of nearby frames.
        """
        if n <= 0 or not items:
            return []
        if n >= len(items):
            return list(items)
        step = len(items) / n
        return [items[int(i * step)] for i in range(n)]

    def sample_random(
        self, items: List[Path], n: int, rng: random.Random
    ) -> List[Path]:
        """Pick n items at random from the sequence."""
        if n <= 0 or not items:
            return []
        if n >= len(items):
            return list(items)
        return rng.sample(items, n)

    def _resolve_target_path(
        self, output_dir: Path, folder_name: str, source_path: Path
    ) -> Path:
        """Return a unique destination path, prefixed with the source folder name
        so collisions across folders are impossible and provenance is preserved."""
        target = output_dir / f"{folder_name}_{source_path.name}"
        if not target.exists():
            return target
        # Append -1, -2, ... if a name collision still happens
        stem = target.stem
        suffix = source_path.suffix
        counter = 1
        while True:
            candidate = output_dir / f"{stem}-{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def build_dataset(
        self,
        base_path: Path,
        output_path: Path | None = None,
        total_frames: int | None = None,
        per_folder: int | None = None,
        strategy: str = "stratified",
        seed: int = 42,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        log=print,
    ) -> Path:
        """Build a balanced dataset by sampling from each subfolder of base_path.

        Exactly one of `total_frames` or `per_folder` must be provided.

        `output_path` should typically be a sibling of `base_path`, not a child,
        so that re-running the script doesn't accidentally treat previous output
        as new source data.

        Returns the path to the created output folder.
        """
        if (total_frames is None) == (per_folder is None):
            raise ValueError("Specify exactly one of total_frames or per_folder")
        if strategy not in ("stratified", "random"):
            raise ValueError("strategy must be 'stratified' or 'random'")

        base_path = Path(base_path).expanduser()
        if not base_path.is_absolute():
            base_path = resolve_up(relative_path=base_path)

        if output_path:
            output_path = Path(output_path).expanduser()
            if not output_path.is_absolute():
                output_path = resolve_up(relative_path=output_path)
        if not output_path:
            output_path = base_path / f"{base_path.name}_AutoDataset"
        rng = random.Random(seed)

        # Catalog all subfolders that contain frames. Defensive: skip any folder
        # that resolves to the output path (in case the user placed output inside
        # base_path).
        log("Scanning folders...")
        catalog = {}
        for folder in sorted(base_path.iterdir(), key=lambda p: p.name):
            if not folder.is_dir():
                continue
            if folder.name.startswith("."):  # skip macOS .Trashes, .Spotlight, etc.
                continue
            if folder.resolve() == output_path:
                continue
            frames = self.list_frames(folder, extensions)
            if frames:
                catalog[folder] = frames

        if not catalog:
            raise FileNotFoundError(f"No frames found in any subfolder of {base_path}")

        log(
            f"Found {len(catalog)} folders containing frames "
            f"({sum(len(f) for f in catalog.values())} frames total)"
        )

        # Compute the per-folder quota
        folders = list(catalog.keys())
        if per_folder is not None:
            if per_folder <= 0:
                raise ValueError("per_folder must be > 0")
            quotas = {f: per_folder for f in folders}
        else:
            if total_frames <= 0:
                raise ValueError("total_frames must be > 0")
            total_available = sum(len(catalog[f]) for f in folders)
            if total_frames > total_available:
                raise ValueError(
                    f"Requested {total_frames} frames but only {total_available} available"
                )
            base_quota, remainder = divmod(total_frames, len(folders))
            quotas = {
                f: base_quota + (1 if i < remainder else 0)
                for i, f in enumerate(folders)
            }

        # Sanity-check quotas against availability
        short_folders = [
            (f.name, q, len(catalog[f]))
            for f, q in quotas.items()
            if q > len(catalog[f])
        ]
        if short_folders:
            msg_lines = [
                "Some folders don't have enough frames for the requested quota:"
            ]
            for name, q, avail in short_folders:
                msg_lines.append(f"  '{name}' needs {q} but only has {avail}")
            if per_folder is not None:
                msg_lines.append(
                    f"Try a smaller --per-folder value (max safe: "
                    f"{min(len(catalog[f]) for f in folders)})"
                )
            else:
                msg_lines.append("Try a smaller --total value")
            raise ValueError("\n".join(msg_lines))

        # Create the output folder
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        if existing:
            log(
                f"\u26a0\ufe0f  Output folder already contains {len(existing)} files. "
                f"New samples will be added alongside them — delete the folder first "
                f"if you want a clean run."
            )

        log(f"Sampling strategy: {strategy}")
        log(f"Output: {output_dir}")

        total_to_copy = sum(quotas.values())
        copied = 0
        with tqdm(total=total_to_copy, desc="Copying frames", unit="frame") as pbar:
            for folder in folders:
                quota = quotas[folder]
                if quota == 0:
                    continue
                frames = catalog[folder]
                if strategy == "stratified":
                    selected = self.sample_stratified(frames, quota)
                else:
                    selected = self.sample_random(frames, quota, rng)

                for frame in selected:
                    target = self._resolve_target_path(output_dir, folder.name, frame)
                    shutil.copy2(frame, target)
                    copied += 1
                    pbar.update(1)

        log(f"\n\u2705 Dataset created: {copied} frames in {output_dir}")
        return output_dir


class video_frame_extractor:
    def __init__(self):
        self.VIDEO_EXTS = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".m4v",
            ".mpg",
            ".mpeg",
        }

    def save_frame_batch(
        self, frames_data, output_dir, video_basename, start_idx, jpeg_quality=85
    ):
        """Save a batch of frames to disk. Raises IOError on first write failure."""
        for i, frame in enumerate(frames_data):
            frame_filename = os.path.join(
                output_dir, f"{video_basename}_frame_{start_idx + i:04d}.jpg"
            )
            ok = cv2.imwrite(
                frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not ok:
                raise IOError(
                    f"Failed to write frame to {frame_filename}. "
                    f"Check that the path is valid and the disk is not full."
                )
        return len(frames_data)

    def extract_frames_optimized(
        self,
        video_path,
        output_dir,
        skip_start=0,
        skip_end=0,
        every_nth_frame=10,
        batch_size=50,
        num_workers=4,
        jpeg_quality=85,
        log=print,
        progress_callback=None,
    ):
        """
        Extract frames from a video using parallel saving.

        Parameters
        ----------
        video_path : str
            Path to your video file.
        output_dir : str
            Directory where your frames will be saved (a subfolder named after the
            video will be created inside).
        skip_start : int
            Seconds to skip at the start of the video.
        skip_end : int
            Seconds to skip at the end of the video.
        every_nth_frame : int
            Extract every nth frame (1 = all frames, 30 = 1 fps for a 30 fps video).
        batch_size : int
            Number of frames to process in each batch.
        num_workers : int
            Number of parallel workers for saving frames.
        jpeg_quality : int
            JPEG compression quality (50-100, higher = better quality).
        log : callable
            Function used to print status messages (defaults to print, the GUI
            replaces this with one that writes into a text widget).
        """
        os.makedirs(output_dir, exist_ok=True)
        video_basename = os.path.splitext(os.path.basename(video_path))[0]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        if not cap.isOpened():
            log("Error: Could not open video.")
            return 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frame_count <= 0:
            log("Error: Could not read video metadata (fps/frame count).")
            cap.release()
            return 0, 0
        duration = total_frame_count / fps

        log(
            f"Video info: {total_frame_count} frames, {fps:.2f} FPS, {duration:.2f} seconds"
        )

        start_frame = int(skip_start * fps)
        end_frame = min(int((duration - skip_end) * fps), total_frame_count)

        if start_frame >= total_frame_count or end_frame <= start_frame:
            log("Error: Invalid skip_start or skip_end values.")
            cap.release()
            return 0, 0

        video_output_dir = os.path.join(output_dir, video_basename)
        os.makedirs(video_output_dir, exist_ok=True)

        # Warn if the target folder already contains frame files from a previous run
        existing = [
            f
            for f in os.listdir(video_output_dir)
            if f.startswith(f"{video_basename}_frame_") and f.endswith(".jpg")
        ]
        if existing:
            log(
                f"\u26a0\ufe0f  Output folder already contains {len(existing)} frame files "
                f"from a previous run. New frames may overwrite some and leave others "
                f"behind. Delete the folder first if you want a clean run."
            )

        total_frames_to_extract = len(range(start_frame, end_frame, every_nth_frame))
        output_fps = fps / every_nth_frame
        log(
            f"Extracting {total_frames_to_extract} frames (effective {output_fps:.1f} fps)"
        )

        frame_buffer = []
        frame_indices = []
        extracted_count = 0
        output_frame_idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            with tqdm(
                total=total_frames_to_extract,
                desc="Extracting frames",
                unit="frame",
            ) as progress:

                while current_frame < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if (current_frame - start_frame) % every_nth_frame == 0:
                        frame_buffer.append(frame)
                        frame_indices.append(output_frame_idx)
                        output_frame_idx += 1

                        if len(frame_buffer) >= batch_size:
                            batch_start_idx = frame_indices[0]
                            future = executor.submit(
                                self.save_frame_batch,
                                frame_buffer.copy(),
                                video_output_dir,
                                video_basename,
                                batch_start_idx,
                                jpeg_quality,
                            )
                            futures.append(future)
                            frame_buffer = []
                            frame_indices = []

                    current_frame += 1

                    # Drain any completed futures so the progress bar advances
                    done_futures = [f for f in futures if f.done()]
                    for f in done_futures:
                        saved = f.result()
                        extracted_count += saved
                        progress.update(saved)
                        if progress_callback:
                            progress_callback(extracted_count, total_frames_to_extract)
                        futures.remove(f)

                # Flush any remaining buffered frames
                if frame_buffer:
                    batch_start_idx = frame_indices[0]
                    futures.append(
                        executor.submit(
                            self.save_frame_batch,
                            frame_buffer,
                            video_output_dir,
                            video_basename,
                            batch_start_idx,
                            jpeg_quality,
                        )
                    )

                for f in as_completed(futures):
                    saved = f.result()
                    extracted_count += saved
                    progress.update(saved)
                    if progress_callback:
                        progress_callback(extracted_count, total_frames_to_extract)

        cap.release()

        # Truthful reporting: distinguish success / partial / failed based on how
        # many frames we actually decoded vs. how many the metadata promised.
        if extracted_count == 0:
            log(
                f"\u274c Extracted 0 frames from {video_basename}. "
                f"The video opened but no frames could be decoded — this usually "
                f"means the file is corrupted or uses an unsupported codec. "
                f"Try opening it in VLC to verify."
            )
        elif (
            total_frames_to_extract > 0
            and extracted_count / total_frames_to_extract < 0.95
        ):
            ratio = extracted_count / total_frames_to_extract
            log(
                f"\u26a0\ufe0f  Extracted {extracted_count}/{total_frames_to_extract} "
                f"expected frames ({ratio:.1%}) to {video_output_dir}. "
                f"The decoder gave up partway through — the source file is likely "
                f"damaged. Output may be incomplete."
            )
        else:
            log(f"\u2705 Extracted {extracted_count} frames to {video_output_dir}")
        return extracted_count, total_frames_to_extract

    def extract_frames_from_directory(
        self,
        input_dir,
        output_dir,
        log=print,
        progress_callback=None,
        video_progress_callback=None,
        **kwargs,
    ):
        """Process every video file in a directory, sequentially.

        Each video gets its own subfolder under output_dir (created by
        extract_frames_optimized). Returns (total_frames_extracted, num_videos).

        progress_callback: called with (current_frame, total_frames) for the
            currently-processing video.
        video_progress_callback: called with (video_index, num_videos, video_name)
            when each new video starts. Lets the GUI show "Video 2/5: foo.mp4".
        """
        input_dir = Path(input_dir).expanduser()
        videos = sorted(
            f
            for f in input_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in self.VIDEO_EXTS
            and not f.name.startswith(
                "."
            )  # skip dotfiles, incl. macOS ._ AppleDouble metadata
        )
        if not videos:
            log(f"\u274c No video files found in {input_dir}")
            log(f"   Looked for extensions: {', '.join(sorted(self.VIDEO_EXTS))}")
            return 0, 0

        log(f"Found {len(videos)} video file(s) in {input_dir.name}")
        log("=" * 50)

        total_extracted = 0
        successful = 0
        partial = 0
        failed = 0
        failed_names = []
        partial_names = []
        for i, video in enumerate(videos, 1):
            log(f"\n[{i}/{len(videos)}] {video.name}")
            if video_progress_callback:
                video_progress_callback(i, len(videos), video.name)
            try:
                count, expected = self.extract_frames_optimized(
                    video_path=str(video),
                    output_dir=str(output_dir),
                    log=log,
                    progress_callback=progress_callback,
                    **kwargs,
                )
                total_extracted += count
                if count == 0:
                    failed += 1
                    failed_names.append(video.name)
                elif expected > 0 and count / expected < 0.95:
                    partial += 1
                    partial_names.append(video.name)
                else:
                    successful += 1
            except Exception as e:
                log(f"\u274c Failed on {video.name}: {e}")
                failed += 1
                failed_names.append(video.name)

        log("\n" + "=" * 50)
        log(f"Batch complete: {total_extracted} frames extracted")
        log(f"  \u2705 {successful} successful")
        if partial:
            log(f"  \u26a0\ufe0f  {partial} partial (likely damaged source files):")
            for name in partial_names:
                log(f"      \u00b7 {name}")
        if failed:
            log(f"  \u274c {failed} failed:")
            for name in failed_names:
                log(f"      \u00b7 {name}")
        return total_extracted, len(videos)

    def video_frames(self, input_path: str, output_dir: str | None = None, **kwags):

        if not input_path or not isinstance(input_path, str):
            raise TypeError(f"{input_path} must be non-empty string")
        if output_dir and not isinstance(output_dir, str):
            raise TypeError(f"{output_dir} must be non-empty string")

        input_path = Path(input_path).expanduser()

        if not input_path.is_absolute():
            input_path = resolve_up(relative_path=input_path)
        if output_dir:
            if not output_dir.is_absolute():
                output_dir = resolve_up(relative_path=output_dir)
        if not output_dir:
            startPoint = Path(__file__).resolve().parents[3]
            output_dir = startPoint / "datasets"
            os.makedirs(output_dir, exist_ok=True)

        common = dict(
            every_nth_frame=10,
            batch_size=50,
            **kwags,
        )
        if input_path.is_dir():
            self.extract_frames_from_directory(
                input_dir=str(input_path),
                output_dir=str(output_dir),
                **common,
            )
        else:
            self.extract_frames_optimized(
                video_path=str(input_path),
                output_dir=str(output_dir),
                **common,
            )
        return f"{output_dir} is the output_dir"
