# base imports
import os
import re
import pims
import glob
import shutil
import yaml
import pandas as pd
import numpy as np
import logging
import datetime
import PIL
from pathlib import Path
from functools import partial
from tqdm import tqdm
from PIL import Image

# utils imports
from kso_utils.db_utils import create_connection
from kso_utils.koster_utils import fix_text_encoding
from kso_utils.server_utils import retrieve_movie_info_from_server, get_movie_url
from kso_utils.t4_utils import get_species_ids
import kso_utils.project_utils as project_utils
from src.prepare_input import ProcFrameCuda, ProcFrames
import src.frame_tracker

# Logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# utility functions
def process_frames(frames_path: str, size: tuple = (416, 416)):
    """
    It takes a path to a directory containing frames, and returns a list of processed frames

    :param frames_path: the path to the directory containing the frames
    :param size: The size of the image to be processed
    """
    # Run tests
    gpu_time_0, n_frames = ProcFrames(partial(ProcFrameCuda, size=size), frames_path)
    logging.info(
        f"Processing performance: {n_frames} frames, {gpu_time_0:.2f} ms/frame"
    )


def process_path(path: str):
    """
    Process a single path

    :param path: The path to be processed
    :type path: str
    :return: The processed filename without numerical suffix
    :rtype: str
    """
    # Convert the input string to a Path object
    path_obj = Path(path)

    # Get the filename part without the extension
    filename_without_ext = path_obj.stem

    # Remove numerical suffix if present
    filename_without_suffix = re.split(r"_[0-9]+", filename_without_ext)[0]

    # Remove '_frame' if present
    filename_processed = filename_without_suffix.replace("_frame", "")

    return filename_processed


def clean_species_name(species_name: str):
    """
    Clean species name
    """
    return species_name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def split_frames(data_path: str, perc_test: float):
    """
    Split frames into train and test sets

    :param data_path: The path to the dataset
    :type data_path: str
    :param perc_test: The percentage of frames to allocate to the test set
    :type perc_test: float
    """
    dataset_path = Path(data_path)
    images_path = dataset_path / "images"

    # Create and/or truncate train.txt and test.txt
    with open(dataset_path / "train.txt", "w") as file_train, open(
        dataset_path / "valid.txt", "w"
    ) as file_valid:
        # Populate train.txt and test.txt
        counter = 1
        total_images = len(list(images_path.glob("*.jpg")))
        index_test = int((1 - perc_test) * total_images)
        latest_movie = ""
        for pathAndFilename in glob.iglob(str(images_path / "*.jpg")):
            title, ext = Path(pathAndFilename).stem, Path(pathAndFilename).suffix
            movie_name = title.replace("_frame_*", "")

            if counter >= index_test + 1:
                # Avoid leaking frames into test set
                if movie_name != latest_movie or movie_name == title:
                    file_valid.write(str(Path(pathAndFilename)) + "\n")
                else:
                    file_train.write(str(Path(pathAndFilename)) + "\n")
                counter += 1
            else:
                latest_movie = movie_name
                # if random.uniform(0, 1) <= 0.5:
                #    file_train.write(pathAndFilename + "\n")
                # else:
                file_train.write(str(Path(pathAndFilename)) + "\n")
                counter += 1


def frame_aggregation(
    project: project_utils.Project,
    db_info_dict: dict,
    out_path: str,
    perc_test: float,
    class_list: list,
    img_size: tuple,
    out_format: str = "yolo",
    remove_nulls: bool = True,
    track_frames: bool = True,
    n_tracked_frames: int = 10,
):
    conn = create_connection(db_info_dict["db_path"])

    species_ref = get_species_ids(project, class_list)

    species_sql_query = (
        f"AND species_id IN {tuple(species_ref)}"
        if len(class_list) > 1
        else f"AND species_id='{tuple(species_ref)[0]}'"
    )

    train_rows = pd.read_sql_query(
        f"SELECT b.frame_number, b.movie_id, b.filename, a.species_id, a.x_position, a.y_position, a.width, a.height FROM \
        agg_annotations_frame AS a INNER JOIN subjects AS b ON a.subject_id=b.id WHERE subject_type='frame' {species_sql_query}",
        conn,
    )

    if remove_nulls:
        train_rows = train_rows.dropna(
            subset=["x_position", "y_position", "width", "height"]
        )

    if len(train_rows) == 0:
        logging.error("No frames left. Please adjust aggregation parameters.")

    movie_df = retrieve_movie_info_from_server(
        project=project, db_info_dict=db_info_dict
    )

    Path(out_path).mkdir(parents=True, exist_ok=True)
    img_dir, label_dir = Path(out_path, "images"), Path(out_path, "labels")
    shutil.rmtree(img_dir, ignore_errors=True)
    shutil.rmtree(label_dir, ignore_errors=True)
    img_dir.mkdir(), label_dir.mkdir()

    species_list = [clean_species_name(sp) for sp in class_list]

    data = dict(
        train=str(Path(out_path, "train.txt")),
        val=str(Path(out_path, "valid.txt")),
        nc=len(class_list),
        names=species_list,
    )
    with open(
        Path(
            out_path,
            f"{project.Project_name+'_'+datetime.datetime.now().strftime('%H:%M:%S')}.yaml",
        ),
        "w",
    ) as outfile:
        yaml.dump(data, outfile, default_flow_style=None)

    hyp_data = {
        "lr0": 0.01,
        "lrf": 0.1,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 0.05,
        "cls": 0.5,
        "cls_pw": 1.0,
        "obj": 1.0,
        "obj_pw": 1.0,
        "iou_t": 0.20,
        "anchor_t": 4.0,
        "fl_gamma": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    with open(Path(out_path, "hyp.yaml"), "w") as outfile:
        yaml.dump(hyp_data, outfile, default_flow_style=None)

    species_df = pd.read_sql_query("SELECT id, label FROM species", conn)
    species_df["clean_label"] = species_df.label.apply(clean_species_name)

    sp_id2mod_id = {
        species_df[species_df.clean_label == species_list[i]].id.values[0]: i
        for i in range(len(species_list))
    }

    logging.info(f"There are {len(movie_df)} movies")

    if len(movie_df) > 0:
        train_rows["movie_path"] = train_rows.merge(
            movie_df, left_on="movie_id", right_on="id", how="left"
        )["spath"]
        train_rows["movie_path"] = train_rows["movie_path"].apply(
            lambda x: get_movie_path(project, db_info_dict, x)
        )
        video_dict = {
            i: pims.MoviePyReader(i) if os.path.exists(i) else pims.Video(i)
            for i in tqdm(train_rows["movie_path"].unique())
        }

        train_rows = train_rows[
            [
                "species_id",
                "frame_number",
                "movie_path",
                "x_position",
                "y_position",
                "width",
                "height",
            ]
        ]
        new_rows = []

        for name, group in tqdm(
            train_rows.groupby(["movie_path", "frame_number", "species_id"])
        ):
            movie_path, frame_number, species_id = name[:3]
            named_tuple = tuple([species_id, frame_number, movie_path])

            final_name = (
                name[0] if name[0] in video_dict else fix_text_encoding(name[0])
            )
            if frame_number > len(video_dict[final_name]):
                logging.warning(
                    f"Frame out of range for video of length {len(video_dict[final_name])}"
                )
                frame_number = frame_number // 2

            if final_name in video_dict:
                bboxes, tboxes = [], []
                bboxes.extend(tuple(i[3:]) for i in group.values)
                movie_h, movie_w = (
                    video_dict[final_name][0].shape[1],
                    video_dict[final_name][0].shape[0],
                )

                for box in bboxes:
                    new_rows.append(
                        (species_id, frame_number, movie_path, movie_h, movie_w) + box
                    )

                if track_frames:
                    tboxes.extend(
                        src.frame_tracker.track_objects(
                            video_dict[final_name],
                            species_id,
                            bboxes,
                            frame_number,
                            frame_number + n_tracked_frames,
                        )
                    )
                    for box in tboxes:
                        new_rows.append(
                            (
                                species_id,
                                frame_number + box[0],
                                movie_path,
                                video_dict[final_name][frame_number].shape[1],
                                video_dict[final_name][frame_number].shape[0],
                            )
                            + box[1:]
                        )

        full_rows = pd.DataFrame(
            new_rows,
            columns=[
                "species_id",
                "frame_number",
                "filename",
                "f_w",
                "f_h",
                "x",
                "y",
                "w",
                "h",
            ],
        )

        for name, groups in tqdm(
            full_rows.groupby(["frame_number", "filename"]),
            desc="Saving frames...",
            colour="green",
        ):
            file, ext = os.path.splitext(name[1])
            file_base = os.path.basename(file)
            if out_format == "yolo":
                open(f"{out_path}/labels/{file_base}_frame_{name[0]}.txt", "w").write(
                    "\n".join(
                        [
                            "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                                0 if len(class_list) == 1 else sp_id2mod_id[i[0]],
                                min((i[5] + i[7] / 2) / i[3], 1.0),
                                min((i[6] + i[8] / 2) / i[4], 1.0),
                                min(i[7] / i[3], 1.0),
                                min(i[8] / i[4], 1.0),
                            )
                            for i in groups.values
                        ]
                    )
                )
            Image.fromarray(
                video_dict[fix_text_encoding(name[1])][name[0]][:, :, [2, 1, 0]]
            ).save(f"{out_path}/images/{file_base}_frame_{name[0]}.jpg")
    else:
        train_rows = train_rows[
            ["species_id", "filename", "x_position", "y_position", "width", "height"]
        ]
        new_rows = []

        for name, group in tqdm(train_rows.groupby(["filename", "species_id"])):
            filename, species_id = name[:2]
            filename = project.photo_folder + filename
            named_tuple = tuple([species_id, filename])

            bboxes = []
            bboxes.extend(tuple(i[2:]) for i in group.values)

            for box in bboxes:
                new_rows.append(
                    (
                        species_id,
                        filename,
                        PIL.Image.open(filename).size[0],
                        PIL.Image.open(filename).size[1],
                    )
                    + box
                )

        full_rows = pd.DataFrame(
            new_rows,
            columns=["species_id", "filename", "f_w", "f_h", "x", "y", "w", "h"],
        )

        col_list = list(full_rows.columns)
        fw_pos, fh_pos, x_pos, y_pos, w_pos, h_pos, speciesid_pos = (
            col_list.index("f_w"),
            col_list.index("f_h"),
            col_list.index("x"),
            col_list.index("y"),
            col_list.index("w"),
            col_list.index("h"),
            col_list.index("species_id"),
        )

        for name, groups in full_rows.groupby(["filename"]):
            file, ext = os.path.splitext(name)
            file_base = os.path.basename(file)
            if out_format == "yolo":
                if len(groups.values) == 1 and str(groups.values[0][-1]) == "nan":
                    open(f"{out_path}/labels/{file_base}.txt", "w")
                else:
                    groups = [i for i in groups.values if str(i[-1]) != "nan"]
                    open(f"{out_path}/labels/{file_base}.txt", "w").write(
                        "\n".join(
                            [
                                "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                                    (
                                        0
                                        if len(class_list) == 1
                                        else sp_id2mod_id[i[speciesid_pos]]
                                    ),
                                    min((i[x_pos] + i[w_pos] / 2) / i[fw_pos], 1.0),
                                    min((i[y_pos] + i[h_pos] / 2) / i[fh_pos], 1.0),
                                    min(i[w_pos] / i[fw_pos], 1.0),
                                    min(i[h_pos] / i[fh_pos], 1.0),
                                )
                                for i in groups
                            ]
                        )
                    )
            Image.fromarray(np.asarray(PIL.Image.open(name))).save(
                f"{out_path}/images/{file_base}.jpg"
            )

    logging.info("Frames extracted successfully")

    if len(full_rows) == 0:
        raise Exception(
            "No frames found for the selected species. Please retry with a different configuration."
        )

    split_frames(out_path, perc_test)
