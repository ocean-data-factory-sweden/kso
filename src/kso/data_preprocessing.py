from pathlib import Path
import yaml
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Any, Optional
import pandas as pd
from collections import defaultdict
import json
from PIL import Image
import random
import os
import shutil
import logging

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


def make_relative_path(abs_path: Path = None, startPoint: Path = None):
    """turn absolut path to relative path based on a start point"""
    if not abs_path or not isinstance(abs_path, Path):
        raise TypeError(f"{abs_path} must be non-empty path")
    if not startPoint or not isinstance(startPoint, Path):
        raise TypeError(f"{startPoint} must be non-empty path")

    if abs_path.is_relative_to(startPoint):
        relative_path = abs_path.relative_to(startPoint.parents[1])
    else:
        relative_path = abs_path

    return relative_path


def make_abs_path(relative_path: str | Path, startPoint: str | Path):
    """turn relative path to absolut path based on a start point"""
    if not relative_path or not isinstance(relative_path, (Path, str)):
        raise TypeError(f"{relative_path} must be non-empty path or string")
    if not startPoint or not isinstance(startPoint, (Path, str)):
        raise TypeError(f"{startPoint} must be non-empty path or string")
    startPoint = Path(startPoint).expanduser()
    relative_path = Path(relative_path).expanduser()
    abs_path = os.path.join(startPoint.parents[1], relative_path)
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
