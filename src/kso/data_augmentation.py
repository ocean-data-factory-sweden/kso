from pathlib import Path
import yaml
from PIL import Image
import random
from .project import Project, resolve_up
import logging

VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
AUGMENT_OPS = ["hflip", "vflip", "rot90", "rot180", "rot270"]


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
    base = Path(cfg.get("path", data_yaml_path.parent))

    train_rel = cfg.get("train", "train/images")
    if isinstance(train_rel, list):
        train_rel = train_rel

    train_img_dir = (base / train_rel).resolve()
    train_lbl_dir = train_img_dir.parent / "labels"

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
