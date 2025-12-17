from __future__ import annotations
from pathlib import Path
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pprint
from ultralytics import settings
from PIL import Image
import random
import pandas as pd
from collections import defaultdict
import json
import shutil


@dataclass
class Project:
    Project_name: str
    project_path: str | Path | None = None
    data_path: Optional[Dict[str, Any]] = None
    tracking: Optional[Dict[str, Any]] = None
    model_path: str = None
    model_name: str = None
    metadata: str = None
    Mlflow: Optional[Dict[str, Any]] = None


def create_project(
    Project_name: str,
    ultralytics_data: Optional[Dict[str, Any]] = None,
    tracking: Optional[Dict[str, Any]] = None,
    weights_path: str = None,
    model_name: str = None,
    metadata: str = None,
) -> Project:
    """Create a YAML file describing a KSO project."""
    # user mistakes.
    if not Project_name or not isinstance(Project_name, str):
        raise ValueError("'Project_name' must be a non-empty string.")

    sanitized = "".join(c.lower() if c.isalnum() else "_" for c in Project_name).strip(
        "_"
    )

    project_path = Path.cwd().parent / "projects"
    yaml_path = project_path / sanitized / f"{sanitized}.project.yaml"
    if yaml_path.exists():
        with open(yaml_path, mode="r", newline="", encoding="utf-8") as file:
            yaml_dict = yaml.load(file, Loader=yaml.SafeLoader)

        logging.info(f"{Project_name} loaded successfully")
    else:

        project = project_path / sanitized
        project.mkdir(parents=True, exist_ok=True)

        yaml_path = project / f"{sanitized}.project.yaml"
        mlflow_path = project_path / "mlflow.db"
        # Assemble the YAML structure.
        yaml_dict: Dict[str, Any] = {
            "Project_name": sanitized,
            "Project_path": str(project),
            "data_path": {"ultralytics_data_path": str(ultralytics_data)},
            "model": {"model_path": weights_path, "model_name": model_name},
            "tracking": tracking,
            "metadata": metadata,
            "Mlflow": {
                "path": None,
                "experiment_name": None,
                "mlflow.db": str(mlflow_path),
            },
        }

        with yaml_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(yaml_dict, fh, sort_keys=False, default_flow_style=False)
    runs_dir = str(project_path / sanitized / "runs")
    datasets_dir = str(project_path / sanitized)
    # print(runs_dir,datasets_dir)
    # Update multiple settings
    settings.update({"datasets_dir": datasets_dir, "runs_dir": runs_dir})

    logging.info(f"Project YAML created at {yaml_path}")
    # Convert yaml into a project instance
    project = Project(
        Project_name=yaml_dict["Project_name"],
        project_path=yaml_dict["Project_path"],
        data_path=yaml_dict["data_path"]["ultralytics_data_path"],
        tracking=yaml_dict["tracking"],
        model_path=yaml_dict["model"]["model_path"],
        model_name=yaml_dict["model"]["model_name"],
        metadata=yaml_dict["metadata"],
    )
    pprint.pp(yaml_dict)
    return project


def add_data(project_path: Project, data: str = None) -> Dict:

    if not project_path or not isinstance(project_path, Project):
        raise ValueError("'Project_path' must be a project instance.")
    if data and not isinstance(data, str):
        raise ValueError("'Ultralytics data path' must be a non-empty string.")

    project_name = project_path.Project_name
    base_dir = Path.cwd().parent
    project = base_dir / "projects" / project_name

    yaml_path = project / f"{project_name}.project.yaml"

    if data:
        data_path = Path(data).expanduser().resolve()
    else:
        data_path = add_ultralytics_dataset_yaml(str(project / "coco8.yaml"))

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data["data_path"]["ultralytics_data_path"] = str(data_path)
    with open(yaml_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)

    logging.info(f"Project YAML data path updated at {yaml_path}")
    return pprint.pp(data)


def preprocess_biigle_csv(BIIGLE_CSV_PATH: str, IMAGES_ROOT: str, DATASET_DIR: str):

    BIIGLE_CSV_PATH = (
        Path(BIIGLE_CSV_PATH).expanduser().resolve()
    )  # e.g. "my_annotations.csv"
    IMAGES_ROOT = (
        Path(IMAGES_ROOT).expanduser().resolve()
    )  # e.g. "my_images" (filenames must match CSV)
    DATASET_DIR = Path(DATASET_DIR).expanduser().resolve()

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    AUGMENT_TRAIN = True
    AUGMENT_FACTOR = 0.5  # 0.5 = +50% images, 1.0 = double
    AUGMENT_OPS = ["hflip", "vflip", "rot180"]

    RANDOM_SEED = 42
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    if not BIIGLE_CSV_PATH.exists():
        raise FileNotFoundError(f"{BIIGLE_CSV_PATH} doent exist")
    if not IMAGES_ROOT.exists():
        raise FileNotFoundError(f"{IMAGES_ROOT} doent exist")
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"{DATASET_DIR} doent exist")

    df = pd.read_csv(BIIGLE_CSV_PATH)
    print(f"Loaded {len(df)} annotations from CSV")

    # Validate columns
    required = ["filename", "label_name", "shape_name", "points", "attributes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Filter to Rectangle/Polygon (converted to bounding boxes)
    supported = ["Rectangle", "Polygon"]
    df = df[df["shape_name"].isin(supported)].reset_index(drop=True)
    print(f"Supported annotations (Rectangle/Polygon): {len(df)}")

    # Build class mapping
    class_names = sorted(df["label_name"].unique())
    df["class_id"] = df["label_name"].map(
        {name: i for i, name in enumerate(class_names)}
    )

    print(f"\nDetected {len(class_names)} classes:")
    for cid, name in enumerate(class_names):
        print(f"  {cid}: {name} ({(df['class_id'] == cid).sum()} annotations)")

    # Parse image dimensions
    attrs = df["attributes"].apply(json.loads)
    df["img_w"] = attrs.apply(lambda d: d.get("width"))
    df["img_h"] = attrs.apply(lambda d: d.get("height"))
    df = df.dropna(subset=["img_w", "img_h"]).reset_index(drop=True)

    # Convert to YOLO format
    def bbox_from_points(points_str):
        pts = json.loads(points_str)
        xs, ys = pts[0::2], pts[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    df[["xmin", "ymin", "xmax", "ymax"]] = df["points"].apply(
        lambda s: pd.Series(bbox_from_points(s))
    )

    df["x_center"] = ((df["xmin"] + df["xmax"]) / 2) / df["img_w"]
    df["y_center"] = ((df["ymin"] + df["ymax"]) / 2) / df["img_h"]
    df["w_norm"] = (df["xmax"] - df["xmin"]) / df["img_w"]
    df["h_norm"] = (df["ymax"] - df["ymin"]) / df["img_h"]

    print(f"\nConverted {len(df)} annotations to YOLO format")

    # Index source images
    image_index = {
        p.name: p
        for p in IMAGES_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }

    # Create splits
    rng = random.Random(RANDOM_SEED)
    img_to_labels = defaultdict(set)
    for row in df.itertuples():
        img_to_labels[row.filename].add(int(row.class_id))

    images = list(img_to_labels.keys())
    n_images = len(images)
    n_train = int(round(TRAIN_RATIO * n_images))
    n_val = int(round(VAL_RATIO * n_images))

    split_for_image = {}
    split_counts = {"train": 0, "val": 0, "test": 0}

    def assign(img, split):
        if img not in split_for_image:
            split_for_image[img] = split
            split_counts[split] += 1
            return True
        return split_for_image[img] == split

    # Ensure class coverage
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

    # Fill remaining
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

    # Build dataset
    SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}
    for folder in SPLIT_MAP.values():
        (DATASET_DIR / folder / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / folder / "labels").mkdir(parents=True, exist_ok=True)

    grouped = {fn: g for fn, g in df.groupby("filename")}
    stats = {f: {"images": 0, "annotations": 0} for f in SPLIT_MAP.values()}

    for filename, split in split_for_image.items():
        folder = SPLIT_MAP[split]
        src_img = image_index.get(filename)
        if src_img is None:
            continue

        shutil.copy2(src_img, DATASET_DIR / folder / "images" / filename)

        label_lines = []
        if filename in grouped:
            for row in grouped[filename].itertuples():
                label_lines.append(
                    f"{int(row.class_id)} {row.x_center:.6f} {row.y_center:.6f} "
                    f"{row.w_norm:.6f} {row.h_norm:.6f}"
                )

        (DATASET_DIR / folder / "labels" / f"{Path(filename).stem}.txt").write_text(
            "\n".join(label_lines)
        )

        stats[folder]["images"] += 1
        stats[folder]["annotations"] += len(label_lines)

    print(f"Dataset created from {n_images} images:")
    for folder, s in stats.items():
        print(f"  {folder}: {s['images']} images, {s['annotations']} annotations")

    if not AUGMENT_TRAIN or AUGMENT_FACTOR <= 0:
        print("Augmentation disabled.")
    else:
        train_img_dir = DATASET_DIR / "train" / "images"
        train_lbl_dir = DATASET_DIR / "train" / "labels"

        train_imgs = sorted(
            p for p in train_img_dir.iterdir() if p.suffix.lower() in VALID_EXTS
        )
        n_original = len(train_imgs)
        n_target = int(round(n_original * AUGMENT_FACTOR))

        if n_target == 0:
            print("No augmentation needed.")
        else:
            rng = random.Random(RANDOM_SEED + 1)
            selected = rng.sample(train_imgs, k=min(n_target, n_original))

            def transform_bbox(xc, yc, w, h, op):
                if op == "hflip":
                    return 1 - xc, yc, w, h
                if op == "vflip":
                    return xc, 1 - yc, w, h
                if op == "rot180":
                    return 1 - xc, 1 - yc, w, h
                raise ValueError(f"Unknown op: {op}")

            def apply_image_op(img, op):
                if op == "hflip":
                    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if op == "vflip":
                    return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                if op == "rot180":
                    return img.rotate(180, expand=True)
                raise ValueError(f"Unknown op: {op}")

            created = 0
            for img_path in selected:
                label_path = train_lbl_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue

                raw = label_path.read_text().strip()
                if not raw:
                    continue

                op = rng.choice(AUGMENT_OPS)
                new_labels = []
                for line in raw.splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = parts[0]
                    xc, yc, w, h = map(float, parts[1:5])
                    xc, yc, w, h = transform_bbox(xc, yc, w, h, op)
                    xc, yc = max(0, min(1, xc)), max(0, min(1, yc))
                    w, h = max(0, min(1, w)), max(0, min(1, h))
                    new_labels.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                if not new_labels:
                    continue

                img = Image.open(img_path)
                img_aug = apply_image_op(img, op)

                suffix = f"_aug_{op}"
                img_aug.save(
                    train_img_dir / f"{img_path.stem}{suffix}{img_path.suffix}"
                )
                (train_lbl_dir / f"{img_path.stem}{suffix}.txt").write_text(
                    "\n".join(new_labels)
                )
                created += 1

            print(f"Created {created} augmented images")
            print(f"  Train set: {n_original} -> {n_original + created} images")

    data_yaml_path = DATASET_DIR / "data.yaml"

    data_cfg = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)
    return data_yaml_path


def add_Biigle_data(
    project_path: Project,
    BIIGLE_CSV_PATH: str,
    IMAGES_ROOT: str,
    DATASET_DIR: str = None,
):

    if not BIIGLE_CSV_PATH or not isinstance(BIIGLE_CSV_PATH, str):
        raise ValueError(f"BIIGLE_CSV_PATH must be a non empty string")
    if not IMAGES_ROOT or not isinstance(IMAGES_ROOT, str):
        raise ValueError(f"IMAGES_ROOT must be a non empty string")
    if DATASET_DIR and not isinstance(DATASET_DIR, str):
        raise ValueError(f"DATASET_DIR must be a non empty string")
    if DATASET_DIR:
        DATASET_DIR_path = Path(DATASET_DIR).expanduser().resolve()
        if not DATASET_DIR_path.exists():
            raise FileNotFoundError(f"{DATASET_DIR_path} not found")
    project_name = project_path.Project_name
    base_dir = Path.cwd().parent
    project = base_dir / "projects" / project_name
    if not DATASET_DIR:
        DATASET_DIR = project / "Dataset"
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    biigle_yaml_path = preprocess_biigle_csv(
        BIIGLE_CSV_PATH=BIIGLE_CSV_PATH,
        IMAGES_ROOT=IMAGES_ROOT,
        DATASET_DIR=str(DATASET_DIR),
    )

    yaml_path = project / f"{project_name}.project.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    # data["data_path"] = {"Biigle_path":str(biigle_yaml_path)}
    data["data_path"].update({"Biigle_path": str(biigle_yaml_path)})

    with open(yaml_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML data path updated at {yaml_path}")
    return pprint.pp(data)


def add_model(project_path: Project, model: str = None, model_name: str = None) -> Dict:

    if not project_path or not isinstance(project_path, Project):
        raise ValueError("'Project_path' must be a Project instance.")
    project_name = project_path.Project_name

    base_dir = Path.cwd().parent
    project = base_dir / "projects" / project_name
    yaml_path = project / f"{project_name}.project.yaml"
    if not yaml_path.exists():
        raise FileExistsError(f"{yaml_path} does not exist.")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    if model:
        if Path(model).expanduser().resolve().exists():
            model_path = Path(model).expanduser().resolve()
        else:
            model_path = project / model
        data["model"]["model_path"] = str(model_path)
    else:
        if not data[model]["model_path"]:
            raise ValueError("'model' was not provided.")

    with open(yaml_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info(f"Project YAML model weights path updated at {yaml_path}")

    if model_name:
        if not isinstance(model_name, str):
            raise ValueError("'model_name' must be a non-empty string.")

        data["model"]["model_name"] = model_name

        with open(yaml_path, "w", encoding="utf-8") as d:
            yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)

        logging.info(f"Project YAML model name updated at {yaml_path}")
    return pprint.pp(data)


def add_ultralytics_dataset_yaml(data_path: str) -> str:
    path = Path(data_path).expanduser().resolve()
    if path.exists():
        logging.info(f"Ultralytics data yaml {path} exist")
    else:
        data = {
            "path": "coco8",
            "train": "images/train",
            "val": "images/val",
            "test": "",
            "names": {
                "0": "person",
                "1": "bicycle",
                "2": "car",
                "3": "motorcycle",
                "4": "airplane",
                "5": "bus",
                "6": "train",
                "7": "truck",
                "8": "boat",
                "9": "traffic light",
                "10": "fire hydrant",
                "11": "stop sign",
                "12": "parking meter",
                "13": "bench",
                "14": "bird",
                "15": "cat",
                "16": "dog",
                "17": "horse",
                "18": "sheep",
                "19": "cow",
                "20": "elephant",
                "21": "bear",
                "22": "zebra",
                "23": "giraffe",
                "24": "backpack",
                "25": "umbrella",
                "26": "handbag",
                "27": "tie",
                "28": "suitcase",
                "29": "frisbee",
                "30": "skis",
                "31": "snowboard",
                "32": "sports ball",
                "33": "kite",
                "34": "baseball bat",
                "35": "baseball glove",
                "36": "skateboard",
                "37": "surfboard",
                "38": "tennis racket",
                "39": "bottle",
                "40": "wine glass",
                "41": "cup",
                "42": "fork",
                "43": "knife",
                "44": "spoon",
                "45": "bowl",
                "46": "banana",
                "47": "apple",
                "48": "sandwich",
                "49": "orange",
                "50": "broccoli",
                "51": "carrot",
                "52": "hot dog",
                "53": "pizza",
                "54": "donut",
                "55": "cake",
                "56": "chair",
                "57": "couch",
                "58": "potted plant",
                "59": "bed",
                "60": "dining table",
                "61": "toilet",
                "62": "tv",
                "63": "laptop",
                "64": "mouse",
                "65": "remote",
                "66": "keyboard",
                "67": "cell phone",
                "68": "microwave",
                "69": "oven",
                "70": "toaster",
                "71": "sink",
                "72": "refrigerator",
                "73": "book",
                "74": "clock",
                "75": "vase",
                "76": "scissors",
                "77": "teddy bear",
                "78": "hair drier",
                "79": "toothbrush",
            },
            "download": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
        }

        with open(path, "w", encoding="utf-8") as d:
            yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    return str(path)
