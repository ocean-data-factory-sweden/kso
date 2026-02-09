from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Any, Optional

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
    project_name: str
    project_path: str | Path | None = None
    Config_file_path: str | Path | None = None
    data_path: Optional[Dict[str, Any]] = None
    tracking: Optional[Dict[str, Any]] = None
    model_path: str = None
    model_name: str = None
    metadata: str = None
    # Mlflow: Optional[Dict[str, Any]] = None


def sanitized_name(project_name: str):
    sanitized = "".join(c.lower() if c.isalnum() else "_" for c in project_name).strip(
        "_"
    )
    return sanitized


def create_project(
    project_name: str,
    project_path: str | Path = None,
    ultralytics_data: Optional[Dict[str, Any]] = None,
    biigle_path: Optional[Dict[str, Any]] = None,
    tracking: Optional[Dict[str, Any]] = None,
    weights_path: str = None,
    model_name: str = None,
    metadata: str = None,
) -> Project:
    """Create a YAML file describing a KSO project."""
    # user mistakes.
    if not project_name or not isinstance(project_name, str):
        raise ValueError(f"{project_name} must be a non-empty string.")

    sanitized = sanitized_name(project_name)

    if project_path:
        project_path = Path(project_path).expanduser()
        project = project_path / sanitized
    else:
        base_dir = Path(__file__).resolve().parents[2]
        project_path = base_dir / "projects"
        project = project_path / sanitized

    """index the last model added if none is provided"""
    index = -1

    if project.exists():
        raise FileExistsError(f"the project {str(project)} already exist.")

    else:
        project.mkdir(parents=True, exist_ok=True)

        yaml_path = project / f"{sanitized}.project.yaml"
        mlflow_path = project / "mlflow.db"
        # Assemble the YAML structure.
        yaml_dict: Dict[str, Any] = {
            "project_name": sanitized,
            "project_path": str(project),
            "Config_file_path": str(yaml_path),
            "data_path": {
                "ultralytics_data_path": ultralytics_data,
                "biigle_path": biigle_path,
            },
            "models": [{"model_path": weights_path, "model_name": model_name}],
            "tracking": {
                "mlflow": {
                    "path": None,
                    "experiment_name": None,
                    "mlflow.db": str(mlflow_path),
                },
            },
            "metadata": metadata,
        }

        yaml_dict = yaml_data_dump(yaml_path, yaml_dict)

    runs_dir = str(project_path / sanitized / "runs")
    datasets_dir = str(project_path / sanitized)
    # Update multiple settings
    settings.update({"datasets_dir": datasets_dir, "runs_dir": runs_dir})

    logging.info(f"Project YAML created at {str(project_path)}")
    # Convert yaml into a project instance
    project = Project(
        project_name=yaml_dict["project_name"],
        project_path=yaml_dict["project_path"],
        Config_file_path=yaml_dict["Config_file_path"],
        data_path=yaml_dict["data_path"],
        tracking=yaml_dict["tracking"],
        model_path=yaml_dict["models"][index]["model_path"],
        model_name=yaml_dict["models"][index]["model_name"],
        metadata=yaml_dict["metadata"],
    )
    pprint.pp(yaml_dict)
    return project


def load_project(
    project_path: str | Path,
    model_name: str = None,
    model_path: str = None,
):
    """load an existing project"""

    if not project_path or not isinstance(project_path, (str, Path)):
        raise ValueError(f"{project_path} must be non-empty string or Path")
    if model_name and not isinstance(model_name, str):
        raise ValueError("'model_name' must be a non-empty string.")
    if model_path and not isinstance(model_path, str):
        raise ValueError("'model_path' must be a non-empty string.")

    yaml_path = Path(project_path).expanduser()
    if yaml_path.exists():
        yaml_dict = yaml_data_retrieve(yaml_path=yaml_path)

        """index the last model added if none is provided"""
        index = -1

        if model_name or model_path:
            model_paths = [m["model_path"] for m in yaml_dict["models"]]
            model_names = [m["model_name"] for m in yaml_dict["models"]]
            if model_name in model_names:
                index = model_names.index(model_name)
            elif model_path in model_paths:
                index = model_paths.index(model_path)

        logging.info(f"{project_path} loaded successfully")
    else:
        raise FileNotFoundError(f"project {yaml_path} was not found")

    # Convert yaml into a project instance
    project = Project(
        project_name=yaml_dict["project_name"],
        project_path=yaml_dict["project_path"],
        Config_file_path=yaml_dict["Config_file_path"],
        data_path=yaml_dict["data_path"],
        tracking=yaml_dict["tracking"],
        model_path=yaml_dict["models"][index]["model_path"],
        model_name=yaml_dict["models"][index]["model_name"],
        metadata=yaml_dict["metadata"],
    )
    pprint.pp(yaml_dict)
    return project


def yaml_data_retrieve(yaml_path: str | Path, data: str = None):
    """
    retreive data from the yaml config file
    if a data column was provided retreive it else return all the data
    """
    if not yaml_path or not isinstance(yaml_path, (str, Path)):
        raise TypeError(f"{yaml_path} has to be a non empty string")
    yaml_path = Path(yaml_path).expanduser()
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
    if data:
        data = yaml_data.get(data)
        logging.info("data was rtreived successfully")
        return data
    elif data and not isinstance(data, str):
        raise TypeError(f"{data} should be a non-empty string")
    else:
        logging.info("data was retreived successfully")
        return yaml_data


def yaml_data_dump(yaml_path: str | Path, data: str = None):
    """
    dump data to the yaml config file
    """
    if not yaml_path or not isinstance(yaml_path, (str, Path)):
        raise TypeError(f"{yaml_path} has to be a non empty string")
    if not data or not isinstance(data, Dict):
        raise TypeError(f"{data} has to be a non empty Dictionary")

    yaml_path = Path(yaml_path).expanduser()
    with open(yaml_path, "w", encoding="utf-8") as d:
        yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    logging.info("data was dumped successfully")
    return data


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
    csv_path = Path(biigle_csv_path).expanduser().resolve()
    images_root = Path(images_root).expanduser().resolve()
    dataset_dir = Path(dataset_dir).expanduser().resolve()

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


def add_data(
    project: Project,
    data_type: str,
    data_path: str = None,
    images_root: str = None,
    dataset_dir: str | None = None,
):

    if not project or not isinstance(project, Project):
        raise ValueError("'Project_path' must be a project instance.")
    if data_path and not isinstance(data_path, str):
        raise ValueError("'data_path' must be a non-empty string.")
    if not data_type or not isinstance(data_type, str):
        raise ValueError("'data_type' must be a non-empty string .")
    project_path = project.project_path
    Config_file_path = project.Config_file_path
    yaml_path = Path(Config_file_path).expanduser()
    if not yaml_path.exists():
        raise FileNotFoundError(f"{yaml_path} not found.")

    data = yaml_data_retrieve(yaml_path=yaml_path)

    if data_type == "yolo_dataset":

        if data_path:

            candidate = Path(data_path).expanduser()
            if candidate.is_absolute():
                data_path = candidate.resolve()
            else:
                data_path = (project_path / candidate).resolve()

        else:
            generated_yolo_data = add_ultralytics_dataset_yaml(
                str(project_path / "coco8.yaml")
            )
            data_path = Path(generated_yolo_data).expanduser().resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        data["data_path"]["ultralytics_data_path"] = str(data_path)
        project.data_path["ultralytics_data_path"] = str(data_path)

    elif data_type == "Biigle_dataset":
        if not images_root or not isinstance(images_root, str):
            raise ValueError(f"images_root must be a non empty string")
        if dataset_dir and not isinstance(dataset_dir, str):
            raise ValueError(f"dataset_dir must be a non empty string")
        if dataset_dir:
            DATASET_DIR_path = Path(dataset_dir).expanduser().resolve()
            if not DATASET_DIR_path.exists():
                raise FileNotFoundError(f"{DATASET_DIR_path} not found")
        project_path = Path(project_path).expanduser()
        if not dataset_dir:
            dataset_dir = project_path / "Dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

        biigle_yaml_path = preprocess_biigle_csv(
            biigle_csv_path=data_path,
            images_root=images_root,
            dataset_dir=str(dataset_dir),
        )

        # data["data_path"] = {"biigle_path":str(biigle_yaml_path)}
        data["data_path"].update({"biigle_path": str(biigle_yaml_path)})
        project.data_path["biigle_path"] = str(biigle_yaml_path)

    yaml_data_dump(yaml_path=yaml_path, data=data)
    logging.info(f"Project YAML data path updated at {yaml_path}")
    return pprint.pp(data)


def add_model(project: Project, model_path: str = None, model_name: str = None):
    """
    Update the project's YAML with a model path and/or model name.

    Rules for `model`:
    - Absolute path ending with '.pt': accepted if it exists.
    """
    if not project or not isinstance(project, Project):
        raise ValueError("'Project_path' must be a Project instance.")
    if not isinstance(model_path, str):
        raise ValueError("'model' must be non-empty string")
    if not isinstance(model_name, str):
        raise ValueError("'model_name' must be a non-empty string")

    Config_file_path = project.Config_file_path
    project_path = project.project_path
    project_path = Path(project_path).expanduser()
    # Get the yaml path
    yaml_path = Path(Config_file_path).expanduser()
    if not yaml_path.exists():
        raise FileExistsError(f"{yaml_path} does not exist.")

    data = yaml_data_retrieve(yaml_path)

    if model_path and model_path.endswith(".pt"):
        candidate = Path(model_path).expanduser()
        if candidate.is_absolute():
            model_trail = candidate
        else:
            model_trail = (project_path / candidate).resolve()

        """CHECK IF THE MODEL ALREADY ADDED"""
        index = -1

        model_paths = [m["model_path"] for m in data["models"]]
        if str(model_trail) in model_paths:
            index = model_paths.index(str(model_trail))
            logging.info(f"model {str(model_trail)} already exists")
        else:
            data["models"].append(
                {"model_name": model_name, "model_path": str(model_trail)}
            )

    elif model_path and not model_path.endswith(".pt"):
        raise ValueError("model is not valid, must end with '.pt'")

    # with open(yaml_path, "w", encoding="utf-8") as d:
    #     yaml.safe_dump(data, d, sort_keys=False, default_flow_style=False)
    yaml_data_dump(yaml_path=yaml_path, data=data)

    """update project instance with provided model or last added model"""
    project.model_path = data["models"][index]["model_path"]
    project.model_name = data["models"][index]["model_name"]

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
