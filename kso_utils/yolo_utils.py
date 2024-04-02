# base imports
import argparse
import time
import cv2 as cv2
import numpy as np
import re
import pims
import sqlite3
import shutil
import yaml
import PIL
import pandas as pd
import logging
import datetime
import requests
import wandb
import imagesize
import base64
import ffmpeg
import mlflow
import ipywidgets as widgets
import matplotlib.pyplot as plt
from jupyter_bbox_widget import BBoxWidget
from IPython.display import display, clear_output, HTML
from PIL import Image as PILImage, ImageDraw

from functools import partial
from tqdm import tqdm
from pathlib import Path
from collections.abc import Callable
from natsort import index_natsorted
from IPython import get_ipython

# util imports
from kso_utils.project_utils import Project

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# globals
frame_device = cv2.cuda_GpuMat()

trackerTypes = [
    "BOOSTING",
    "MIL",
    "KCF",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
    "CSRT",
]


def applyMask(frame: np.ndarray):
    """
    It takes a frame and returns a frame with the top 50 pixels and bottom 100 pixels blacked out

    :param frame: the frame to apply the mask to
    :type frame: np.ndarray
    :return: The frame with the mask applied.
    """
    h, w, c = frame.shape
    cv2.rectangle(frame, (0, h), (0 + w, h - 100), 0, -1)
    cv2.rectangle(frame, (0, 0), (0 + w, 50), 0, -1)
    return frame


def clearImage(frame: np.ndarray):
    """
    We take the maximum value of each channel, and then take the minimum value of the three channels.
    Then we blur the image, and then we take the maximum value of the blurred image and the value 0.5.
    Then we take the maximum value of the difference between the channel and the maximum value of the
    channel, divided by the blurred image, and the maximum value of the channel. Then we divide the
    result by the maximum value of the channel and multiply by 255

    :param frame: the image to be processed
    :return: The clear image
    """
    channels = cv2.split(frame)
    # Get the maximum value of each channel
    # and get the dark channel of each image
    # record the maximum value of each channel
    a_max_dst = [float("-inf")] * len(channels)
    for idx in range(len(channels)):
        a_max_dst[idx] = channels[idx].max()

    dark_image = cv2.min(channels[0], cv2.min(channels[1], channels[2]))

    # Gaussian filtering the dark channel
    dark_image = cv2.GaussianBlur(dark_image, (25, 25), 0)

    image_t = (255.0 - 0.95 * dark_image) / 255.0
    image_t = cv2.max(image_t, 0.5)

    # Calculate t(x) and get the clear image
    for idx in range(len(channels)):
        channels[idx] = (
            cv2.max(
                cv2.add(
                    cv2.subtract(channels[idx].astype(np.float32), int(a_max_dst[idx]))
                    / image_t,
                    int(a_max_dst[idx]),
                ),
                0.0,
            )
            / int(a_max_dst[idx])
            * 255
        )
        channels[idx] = channels[idx].astype(np.uint8)

    return cv2.merge(channels)


def ProcFrames(proc_frame_func: Callable, frames_path: str):
    """
    It takes a function that processes a single frame and a path to a folder containing frames, and
    applies the function to each frame in the folder

    :param proc_frame_func: The function that will be applied to each frame
    :type proc_frame_func: Callable
    :param frames_path: The path to the directory containing the frames
    :type frames_path: str
    :return: The time it took to process all the frames in the folder, and the number of frames processed.
    """
    start = time.time()
    files = Path(frames_path).iterdir()
    for f in files:
        if str(f).endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
            if Path(frames_path, f).exists():
                new_frame = proc_frame_func(cv2.imread(str(Path(frames_path, f))))
                cv2.imwrite(str(Path(frames_path, f)), new_frame)
            else:
                from kso_utils.koster_utils import fix_text_encoding

                new_frame = proc_frame_func(
                    cv2.imread(fix_text_encoding(str(Path(frames_path, f))))
                )
                cv2.imwrite(str(Path(frames_path, f)), new_frame)
    end = time.time()
    return (end - start) * 1000 / len(files), len(files)


def ProcVid(proc_frame_func: Callable, vidPath: str):
    """
    It takes a function that processes a frame and a video path, and returns the average time it takes
    to process a frame and the number of frames in the video

    :param proc_frame_func: This is the function that will be called on each frame
    :type proc_frame_func: Callable
    :param vidPath: The path to the video file
    :type vidPath: str
    :return: The average time to process a frame in milliseconds and the number of frames processed.
    """
    cap = cv2.VideoCapture(vidPath)
    if cap.isOpened() is False:
        logging.error("Error opening video stream or file")
        return
    n_frames = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            n_frames += 1
            proc_frame_func(frame)
        else:
            break
    end = time.time()
    cap.release()
    return (end - start) * 1000 / n_frames, n_frames


def ProcFrameCuda(frame: np.ndarray, size=(416, 416), use_gpu=False):
    """
    It takes a frame, resizes it to a smaller size, converts it to RGB, and then clears it

    :param frame: the frame to be processed
    :type frame: np.ndarray
    :param size: the size of the image to be processed
    :return: the processed frame.
    """
    if use_gpu:
        frame_device.upload(frame)
        frame_device_small = cv2.resize(frame_device, dsize=size)
        fg_device = cv2.cvtColor(frame_device_small, cv2.COLOR_BGR2RGB)
        fg_host = fg_device.download()
        return fg_host
    else:
        frame_device_small = cv2.resize(frame, dsize=size)
        fg_device = cv2.cvtColor(frame_device_small, cv2.COLOR_BGR2RGB)
        return fg_device


def prepare(data_path, percentage_test, out_path):
    """
    It takes a path to a folder containing images, a percentage of the images to be used for testing,
    and a path to the output folder. It then creates two files, train.txt and test.txt, which contain
    the paths to the images to be used for training and testing, respectively

    :param data_path: the path to the dataset
    :param percentage_test: The percentage of the images that we want to be in the test set
    :param out_path: The path to the output directory
    """

    dataset_path = Path(data_path, "images")

    # Create and/or truncate train.txt and test.txt
    file_train = open(Path(data_path, "train.txt"), "w")
    file_test = open(Path(data_path, "test.txt"), "w")

    # Populate train.txt and test.txt
    counter = 1
    index_test = int((1 - percentage_test) / 100 * len(Path(dataset_path).iterdir()))
    latest_movie = ""
    for pathAndFilename in Path(dataset_path).rglob("*.jpg"):
        file_path = Path(pathAndFilename)
        title, _ = file_path.name, file_path.suffix
        movie_name = title.replace("_frame_*", "", regex=True)

        if counter == index_test + 1:
            if movie_name != latest_movie:
                file_test.write(out_path + Path(title).name + ".jpg" + "\n")
            else:
                file_train.write(out_path + Path(title).name + ".jpg" + "\n")
            counter += 1
        else:
            latest_movie = movie_name
            file_train.write(out_path + Path(title).name + ".jpg" + "\n")
            counter += 1


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
    """
    return Path(re.split("_[0-9]+", path)[0]).name.replace("_frame", "")


def clean_species_name(species_name: str):
    """
    Clean species name
    """
    return species_name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def split_frames(data_path: str, perc_test: float):
    """
    Split frames into train and test sets
    """
    dataset_path = Path(data_path)
    images_path = Path(dataset_path, "images")

    # Create and/or truncate train.txt and test.txt
    file_train = open(Path(data_path, "train.txt"), "w")
    # file_test = open(Path(data_path, "test.txt"), "w")
    file_valid = open(Path(data_path, "valid.txt"), "w")

    # Populate train.txt and test.txt
    counter = 1
    index_test = int((1 - perc_test) * len(list(images_path.glob("*.jpg"))))
    latest_movie = ""
    for pathAndFilename in list(images_path.rglob("*.jpg")):
        file_path = Path(pathAndFilename)
        title, _ = file_path.name, file_path.suffix
        movie_name = title.replace("_frame_*", "")

        if counter >= index_test + 1:
            # Avoid leaking frames into test set
            if movie_name != latest_movie or movie_name == title:
                file_valid.write(str(pathAndFilename) + "\n")
            else:
                file_train.write(str(pathAndFilename) + "\n")
            counter += 1
        else:
            latest_movie = movie_name
            # if random.uniform(0, 1) <= 0.5:
            #    file_train.write(pathAndFilename + "\n")
            # else:
            file_train.write(str(pathAndFilename) + "\n")
            counter += 1


def frame_aggregation(
    project: Project,
    server_connection: dict,
    db_connection: sqlite3.Connection,
    out_path: str,
    perc_test: float,
    class_list: list,
    img_size: tuple,
    out_format: str = "yolo",
    remove_nulls: bool = True,
    track_frames: bool = True,
    n_tracked_frames: int = 10,
    agg_df: pd.DataFrame = pd.DataFrame(),
):
    """
    It takes a project, a database, an output path, a percentage of frames to use for testing, a list of
    species to include, an image size, an output format, a boolean to remove null annotations, a boolean
    to track frames, and the number of frames to track, and it returns a dataset of frames with bounding
    boxes for the specified species

    :param project: the project object
    :param server_connection: a dictionary with the connection to the server
    :param db_connection: SQL connection object
    :param out_path: the path to the folder where you want to save the dataset
    :type out_path: str
    :param perc_test: The percentage of frames that will be used for testing
    :type perc_test: float
    :param class_list: list of species to include in the dataset
    :type class_list: list
    :param img_size: tuple, the size of the images to be used for training
    :type img_size: tuple
    :param out_format: str = "yolo", defaults to yolo
    :type out_format: str (optional)
    :param remove_nulls: Remove null annotations from the dataset, defaults to True
    :type remove_nulls: bool (optional)
    :param track_frames: If True, the script will track the bounding boxes for n_tracked_frames frames after the object is detected, defaults to True
    :type track_frames: bool (optional)
    :param n_tracked_frames: number of frames to track after an object is detected, defaults to 10
    :type n_tracked_frames: int (optional)
    """
    # Establish connection to database
    from kso_utils.db_utils import create_connection
    from kso_utils.zooniverse_utils import clean_label

    conn = create_connection(project.db_path)

    # Select the id/s of species of interest
    if class_list[0] == "":
        logging.error(
            "No species were selected. Please select at least one species before continuing."
        )
        return

    # Select the aggregated classifications from the species of interest
    # Commenting this out for now as it seems that we have the correct label in the agg_df already (to be confirmed for different workflows)
    # clean_list = [clean_label(label) for label in class_list]
    train_rows = agg_df[agg_df.label.isin(class_list)]

    # Rename columns if in different format
    train_rows = (
        train_rows.rename(
            columns={"x": "x_position", "y": "y_position", "w": "width", "h": "height"}
        )
        .copy()
        .reset_index()
    )

    # Remove null annotations
    if remove_nulls:
        train_rows = (
            train_rows.dropna(
                subset=["x_position", "y_position", "width", "height"],
            )
            .copy()
            .reset_index()
        )

    # Check if any frames are left after removing null values
    if len(train_rows) == 0:
        logging.error("No frames left. Please adjust aggregation parameters.")
        return

    # Create output folder

    if Path(out_path).is_dir():
        shutil.rmtree(out_path)
    Path(out_path).mkdir()

    # Set up directory structure
    img_dir = Path(out_path, "images")
    label_dir = Path(out_path, "labels")

    # Create image and label directories
    Path(img_dir).mkdir()
    Path(label_dir).mkdir()

    # Create timestamped koster yaml file with model configuration
    species_list = [clean_species_name(sp) for sp in class_list]

    # Write config file
    data = dict(
        path=out_path,
        train="train.txt",
        val="valid.txt",
        nc=len(class_list),
        names=species_list,
    )

    with open(
        Path(
            out_path,
            f"{project.Project_name+'_'+datetime.datetime.now().strftime('%H_%M_%S')}.yaml",
        ),
        "w",
    ) as outfile:
        yaml.dump(data, outfile, default_flow_style=None)

    # Write hyperparameters default file (default hyperparameters from https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch.yaml)
    hyp_data = dict(
        lr0=0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        lrf=0.1,  # final OneCycleLR learning rate (lr0 * lrf)
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # optimizer weight decay 5e-4
        warmup_epochs=3.0,  # warmup epochs (fractions ok)
        warmup_momentum=0.8,  # warmup initial momentum
        warmup_bias_lr=0.1,  # warmup initial bias lr
        box=0.05,  # box loss gain
        cls=0.5,  # cls loss gain
        cls_pw=1.0,  # cls BCELoss positive_weight
        obj=1.0,  # obj loss gain (scale with pixels)
        obj_pw=1.0,  # obj BCELoss positive_weight
        iou_t=0.20,  # IoU training threshold
        anchor_t=4.0,  # anchor-multiple threshold
        # anchors= 3  # anchors per output layer (0 to ignore)
        fl_gamma=0.0,  # focal loss gamma (efficientDet default gamma=1.5)
        hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # image HSV-Value augmentation (fraction)
        degrees=0.0,  # image rotation (+/- deg)
        translate=0.1,  # image translation (+/- fraction)
        scale=0.5,  # image scale (+/- gain)
        shear=0.0,  # image shear (+/- deg)
        perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
        flipud=0.0,  # image flip up-down (probability)
        fliplr=0.5,  # image flip left-right (probability)
        mosaic=1.0,  # image mosaic (probability)
        mixup=0.0,  # image mixup (probability)
        copy_paste=0.0,  # segment copy-paste (probability)
    )

    with open(Path(out_path, "hyp.yaml"), "w") as outfile:
        yaml.dump(hyp_data, outfile, default_flow_style=None)

    # Clean species names
    species_df = pd.read_sql_query(
        "SELECT id, commonName, scientificName FROM species", conn
    )

    # Add species_id to train_rows
    if "species_id" not in train_rows.columns:
        if "ZooClassification" in train_rows.columns:
            train_rows["species_id"] = train_rows.ZooClassification
            # species_df["clean_label"] = species_df.ZooClassification.apply(clean_species_name)
            # species_df["zoo_label"] = species_df.ZooClassification.apply(clean_label)
            sp_id2mod_id = {}
            m_id = 0
            for ix, item in enumerate(species_list):
                # match = species_df[species_df.clean_label == species_list[ix]].id.values
                # if len(match) == 1:
                sp_id2mod_id[item.capitalize().replace("_", " ")] = m_id
                m_id += 1

        else:
            # Allow for both cases where commonName or scientificName was used for annotation
            try:
                train_rows["species_id"] = train_rows["label"].apply(
                    lambda x: (
                        species_df[species_df.commonName == x].id.values[0]
                        if x != "empty"
                        else "empty"
                    ),
                    1,
                )
                species_df["clean_label"] = species_df.commonName.apply(
                    clean_species_name
                )
                species_df["zoo_label"] = species_df.commonName.apply(clean_label)
            except IndexError:

                def get_species_id(row):
                    if row == "empty":
                        return "empty"
                    else:
                        out = species_df[species_df.scientificName == row].id.values
                        if len(out) == 1:
                            return out[0]
                        else:
                            return None

                train_rows["species_id"] = train_rows["label"].apply(
                    lambda x: get_species_id(x), 1
                )
                species_df["clean_label"] = species_df.scientificName.apply(
                    clean_species_name
                )
                species_df["zoo_label"] = species_df.scientificName.apply(clean_label)

                train_rows.drop(columns=["label"], axis=1, inplace=True)

                # Keep only species that can be matched to species_list
                species_df = species_df[species_df.clean_label.isin(species_list)]

            sp_id2mod_id = {}
            m_id = 0
            for ix, item in enumerate(species_list):
                match = species_df[species_df.clean_label == species_list[ix]].id.values
                if len(match) == 1:
                    sp_id2mod_id[match[0]] = m_id
                    m_id += 1

    # Get movie info from server
    from kso_utils.movie_utils import retrieve_movie_info_from_server

    movie_df = retrieve_movie_info_from_server(
        project=project,
        server_connection=server_connection,
        db_connection=db_connection,
    )[0]

    # If at least one movie is linked to the project
    logging.info(f"There are {len(movie_df)} movies")

    if len(movie_df) > 0:
        if (
            "frame_number" in train_rows.columns
            and not pd.isnull(train_rows["frame_number"]).any()
        ):
            movie_bool = True
        else:
            logging.info(
                "There are movies available, but the subject metadata does not contain frame "
                "numbers and will therefore not be used."
            )
            movie_bool = False
    else:
        movie_bool = False

    link_bool = "https_location" in train_rows.columns
    image_bool = project.photo_folder is not None

    if not any([movie_bool, link_bool, image_bool]):
        logging.error(
            "No source of footage for aggregation found. Please check your metadata "
            "and project setup before running this function again."
        )
        return None

    if link_bool and movie_bool:
        # If both movies and subject urls exist, use movie urls since Zooniverse is rate
        # limited
        movie_bool = False

    if movie_bool:
        # Get movie path on the server
        movie_df.rename(columns={"movie_id": "id"}, inplace=True)
        # TODO: Remove weird workaround for datatype conversions (should not be done manually here)
        movie_df["id"] = movie_df["id"].astype(float).astype(int)
        train_rows["movie_id"] = train_rows["movie_id"].astype(float).astype(int)
        train_rows = train_rows.reset_index(drop=True)

        train_rows["movie_path"] = train_rows.merge(
            movie_df, on="movie_id", how="left"
        )["fpath"]

        from kso_utils.movie_utils import get_movie_path

        train_rows["movie_path"] = train_rows["movie_path"].apply(
            lambda x: get_movie_path(
                f_path=x, project=project, server_connection=server_connection
            )
        )

        # Read each movie for efficient frame access
        video_dict = {}
        for i in tqdm(train_rows["movie_path"].unique()):
            try:
                video_dict[i] = pims.MoviePyReader(i)
            except Exception as e:
                try:
                    logging.info(
                        f"Could not use moviepy, switching to regular pims... {e}"
                    )
                    from kso_utils.koster_utils import fix_text_encoding

                    video_dict[fix_text_encoding(str(i))] = pims.Video(
                        fix_text_encoding(str(i))
                    )
                except KeyError:
                    logging.warning("Missing file" + f"{i}")

        # Create full rows
        train_rows = train_rows.sort_values(
            by=["movie_path", "frame_number"], ascending=True
        )

        # Ensure key fields wrt movies are available
        key_fields = [
            "subject_ids",
            "species_id",
            "frame_number",
            "movie_path",
            "x_position",
            "y_position",
            "width",
            "height",
        ]

    else:
        if link_bool:
            key_fields = [
                "subject_ids",
                "species_id",
                "x_position",
                "y_position",
                "width",
                "height",
            ]
        else:
            key_fields = [
                "species_id",
                "filename",
                "x_position",
                "y_position",
                "width",
                "height",
            ]

    # Get relevant fields from dataframe (before groupby)
    train_rows = train_rows[key_fields]

    link_bool = "subject_ids" in key_fields

    group_fields = (
        ["subject_ids", "species_id"]
        if link_bool
        else (
            ["species_id", "frame_number", "movie_path"]
            if movie_bool
            else ["filename", "species_id"]
        )
    )

    print(group_fields)

    new_rows = []
    bboxes = {}
    tboxes = {}

    for name, group in tqdm(train_rows.groupby(group_fields)):
        grouped_fields = list(name[: len(group_fields)])
        if not movie_bool:
            # Get the filenames of the images
            filename = (
                agg_df[agg_df.subject_ids == grouped_fields[0]]["https_location"].iloc[
                    0
                ]
                if link_bool
                else project.photo_folder + grouped_fields[0]
            )
            named_tuple = tuple([grouped_fields[1], filename])
        else:
            # Get movie_path and frame_number
            rev_fields = grouped_fields.reverse()
            named_tuple = tuple([rev_fields])

        if movie_bool:
            from kso_utils.koster_utils import fix_text_encoding

            final_name = (
                name[2] if name[2] in video_dict else fix_text_encoding(name[2])
            )

            if grouped_fields[1] > len(video_dict[final_name]):
                logging.warning(
                    f"Frame out of range for video of length {len(video_dict[final_name])}"
                )

            if final_name in video_dict:
                bboxes[named_tuple], tboxes[named_tuple] = [], []
                bboxes[named_tuple].extend(
                    tuple(i[len(grouped_fields) :]) for i in group.values
                )

                movie_w, movie_h = video_dict[final_name][0].shape[:2]

                for box in bboxes[named_tuple]:
                    new_rows.append(
                        (
                            grouped_fields[-1],
                            grouped_fields[1],
                            grouped_fields[0],
                            movie_h,
                            movie_w,
                        )
                        + box
                    )

                if track_frames:
                    # Track n frames after object is detected
                    tboxes[named_tuple].extend(
                        tracking_frames(
                            video_dict[final_name],
                            grouped_fields[-1],
                            bboxes[named_tuple],
                            grouped_fields[1],
                            grouped_fields[1] + n_tracked_frames,
                        )
                    )
                    for box in tboxes[named_tuple]:
                        new_rows.append(
                            (
                                grouped_fields[-1],
                                grouped_fields[1] + box[0],
                                grouped_fields[0],
                                video_dict[final_name][grouped_fields[1]].shape[1],
                                video_dict[final_name][grouped_fields[1]].shape[0],
                            )
                            + box[1:]
                        )
        else:
            # Track intermediate frames
            bboxes[named_tuple] = []
            bboxes[named_tuple].extend(
                tuple(i[len(grouped_fields) :]) for i in group.values
            )

            if link_bool:
                try:
                    # Attempt to open the image and get its size
                    response = requests.get(filename, stream=True)
                    if response.status_code == 200:
                        image = PIL.Image.open(response.raw)
                        s1, s2 = image.size
                    else:
                        # If the request was unsuccessful, use the fallback size
                        s1, s2 = img_size
                except (IOError, PIL.UnidentifiedImageError) as e:
                    # Handle specific exceptions for image loading failures
                    logging.error(f"Failed to load image: {e}")
                    s1, s2 = img_size
            else:
                s1, s2 = PIL.Image.open(filename).size

            for box in bboxes[named_tuple]:
                new_rows.append(
                    (
                        grouped_fields[-1],  # species_id
                        filename,
                        s1,
                        s2,
                    )
                    + box
                )

    # Final export step
    if movie_bool:
        # Export full rows
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
        f_group_fields = ["frame_number", "filename"]
    else:
        full_rows = pd.DataFrame(
            new_rows,
            columns=[
                "species_id",
                "filename",
                "f_w",
                "f_h",
                "x",
                "y",
                "w",
                "h",
            ],
        )
        f_group_fields = ["filename"]

    # Find indices of important fields
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

    for name, groups in tqdm(
        full_rows.groupby(f_group_fields),
        desc="Saving frames...",
        colour="green",
    ):
        if movie_bool:
            file_path = Path(name[1])
            file, _ = file_path.stem, file_path.suffix
            file_out = Path(out_path, "labels", f"{file}_frame_{name[0]}.txt")
            img_out = Path(out_path, "images", f"{file}_frame_{name[0]}.jpg")
        else:
            file_path = Path(name)
            file, _ = file_path.stem, file_path.suffix
            file_out = Path(out_path, "labels", f"{file}.txt")
            img_out = Path(out_path, "images", f"{file}.jpg")

        # Added condition to avoid bounding boxes outside of maximum size of frame + added 0 class id when working with single class
        if out_format == "yolo":
            if len(groups.values) == 1 and str(groups.values[0][-1]) == "nan":
                # Empty files
                open(file_out, "w")
            else:
                groups = [i for i in groups.values if str(i[-1]) != "nan"]
                open(file_out, "w").write(
                    "\n".join(
                        [
                            "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                                (
                                    0
                                    if len(class_list) == 1
                                    else sp_id2mod_id[i[speciesid_pos]]
                                ),  # single class vs multiple classes
                                min((i[x_pos] + i[w_pos] / 2) / i[fw_pos], 1.0),
                                min((i[y_pos] + i[h_pos] / 2) / i[fh_pos], 1.0),
                                min(i[w_pos] / i[fw_pos], 1.0),
                                min(i[h_pos] / i[fh_pos], 1.0),
                            )
                            for i in groups
                        ]
                    )
                )

        # Save frames to image files
        if movie_bool:
            from kso_utils.koster_utils import fix_text_encoding

            save_name = name[1] if name[1] in video_dict else fix_text_encoding(name[1])
            if save_name in video_dict:
                img_array = video_dict[save_name][name[0]][:, :, [2, 1, 0]]
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                PIL.Image.fromarray(img_array).save(img_out)
        else:
            if link_bool:
                image_output = PIL.Image.open(requests.get(name, stream=True).raw)
            else:
                image_output = np.asarray(PIL.Image.open(name))
            img_array = np.asarray(image_output)
            PIL.Image.fromarray(img_array).save(img_out)

    logging.info("Frames extracted successfully")

    # Check that at least some frames remain after aggregation
    if len(full_rows) == 0:
        raise Exception(
            "No frames found for the selected species. Please retry with a different configuration."
        )

    # Pre-process frames
    # Comment out for the moment as we do not typically need this for all cases
    # process_frames(out_path + "/images", size=tuple(img_size))

    # Create training/test sets
    split_frames(out_path, perc_test)


def createTrackerByName(trackerType: str):
    """
    It creates a tracker based on the tracker name

    :param trackerType: The type of tracker we want to use
    :return: The tracker is being returned.
    """
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        logging.info("Incorrect tracker name")
        logging.info("Available trackers are:")
        for t in trackerTypes:
            logging.info(t)

    return tracker


def tracking_frames(
    video, class_ids: list, bboxes: list, start_frame: int, last_frame: int
):
    """
    It takes a video, a list of bounding boxes, and a start and end frame, and returns a list of tuples
    containing the frame number, and the bounding box coordinates

    :param video: the video to be tracked
    :param class_ids: The class of the object you want to track
    :param bboxes: the bounding boxes of the objects to be tracked
    :param start_frame: the frame number to start tracking from
    :param last_frame: the last frame of the video to be processed
    :return: A list of tuples, where each tuple contains the frame number, x, y, width, and height of the bounding box.
    """

    # Set video to load
    # colors = [(randint(0, 255)) for i in bboxes]

    # Specify the tracker type
    trackerType = "CSRT"

    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    # Extract relevant frame
    frame = video[start_frame]  # [0]

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    t_bbox = []
    t = 0
    # Process video and track objects
    for current_frame in range(int(start_frame) + 1, int(last_frame) + 1):
        frame = video[current_frame]  # [0]

        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)
        if success:
            t += 1
            for i, newbox in enumerate(boxes):
                t_bbox.append(
                    (t, int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3]))
                )

    return t_bbox


def setup_paths(output_folder: str, model_type: str):
    """
    It takes the output folder and returns the path to the data file and the path to the hyperparameters
    file

    :param output_folder: The folder where the output of the experiment is stored
    :type output_folder: str
    :return: The data_path and hyps_path
    """
    if model_type == 1:
        try:
            data_path = [
                str(f)
                for f in Path(output_folder).iterdir()
                if str(f).endswith(".yaml") and "hyp" not in str(f)
            ][-1]
            hyps_path = str(Path(output_folder, "hyp.yaml"))

            # Rewrite main path to images and labels
            with open(data_path, "r") as yamlfile:
                cur_yaml = yaml.safe_load(yamlfile)
                cur_yaml["path"] = str(Path(output_folder).resolve())

            if cur_yaml:
                with open(data_path, "w") as yamlfile:
                    yaml.safe_dump(cur_yaml, yamlfile)

            logging.info("Success! Paths to data.yaml and hyps.yaml found.")
        except Exception as e:
            logging.error(
                f"{e}, Either data.yaml or hyps.yaml was not found in your folder. Ensure they are located in the selected directory."
            )
            return None, None
        return data_path, hyps_path
    elif model_type == 2:
        logging.info("Paths do not need to be changed for this model type.")
        return output_folder, None
    else:
        logging.info(
            "This functionality is currently unavailable for the chosen model type."
        )
        return None, None


def set_config(**kwargs):
    """
    `set_config` takes in a confidence threshold, model name, and evaluation directory and returns a
    configuration object.

    :param conf_thres: This is the confidence threshold for the bounding boxes
    :type conf_thres: float
    :param model: The name of the model you want to use
    :type model: str
    :param eval_dir: The directory where the evaluation images are stored
    :type eval_dir: str
    :return: The config object is being returned.
    """
    config = wandb.config
    for key, value in kwargs.items():
        if key == "model_name" and "model_name" in config:
            pass
        else:
            setattr(config, key, value)
    return config


def add_data(path: str, name: str, registry: str, run):
    """
    > The function `add_data` takes a path to a directory, a name for the directory, and a run
    object, and adds the directory to the run as an artifact

    :param path: the path to the directory you want to upload
    :type path: str
    :param name: The name of the artifact
    :type name: str
    :param run: The run object that you get from calling wandb.init()
    """
    if registry == "wandb":
        my_data = wandb.Artifact(name, type="raw_data")
        if Path(path).is_dir():
            my_data.add_dir(path)
            run.log_artifact(my_data)
        else:
            my_data.add_file(path)
            run.log_artifact(my_data)
    elif registry == "mlflow":
        mlflow.log_artifact(path, artifact_path=name)


def generate_csv_report(
    evaluation_path: str,
    movie_csv_df: pd.DataFrame,
    run,
    log: bool = False,
    registry: str = "wandb",
):
    """
    Generate a CSV report from labels in the evaluation folder.

    :param evaluation_path: The path to the evaluation folder
    :type evaluation_path: str
    :return: A dataframe with columns: filename, class_id, frame_no, x, y, w, h, conf
    """
    labels_path = Path(evaluation_path, "labels")
    data_dict = {}

    for label_file in labels_path.glob("*.txt"):
        try:
            frame_no = int(label_file.stem.split("_")[-1])
        except ValueError:
            logging.error(
                "Custom frames not linked to uploaded movies, no frame numbers available"
            )
            frame_no = None

        with open(label_file, "r") as infile:
            lines = infile.readlines()
            for line in lines:
                parts = line.split()
                class_id, x, y, w, h, conf = parts[:6]
                data_dict.setdefault(str(label_file), []).append(
                    [class_id, frame_no, x, y, w, h, float(conf)]
                )

    dlist = [[key, *i] for key, values in data_dict.items() for i in values]

    # Convert list of lists to output dataframe
    detect_df = pd.DataFrame(
        dlist, columns=["filename", "class_id", "frame_no", "x", "y", "w", "h", "conf"]
    )

    # Filter by survey_start and survey_end if applicable
    if all(col in movie_csv_df for col in ["sampling_start", "sampling_end"]):
        detect_df["movie_filename"] = (
            detect_df["filename"].str.split("/").str[-1].str.rsplit(pat="_", n=1).str[0]
        )
        detect_df["movie_filename"] = detect_df["movie_filename"].apply(
            lambda x: x + ".mp4" if "mp4" not in x else x, 1
        )
        # Rename movie_filename to avoid filename confusion
        movie_csv_df.rename(
            columns={"filename": "movie_filename"},
            inplace=True,
        )

        # Add sampling data
        detect_df = pd.merge(detect_df, movie_csv_df, on="movie_filename")
        if (
            detect_df.sampling_start.dtype == "float"
            and detect_df.sampling_end.dtype == "float"
        ):
            detect_df = detect_df[
                (detect_df.frame_no >= detect_df.sampling_start)
                & (detect_df.frame_no <= detect_df.sampling_end)
            ]
        # Keep only useful columns
        detect_df = detect_df[
            ["filename", "class_id", "frame_no", "x", "y", "w", "h", "conf"]
        ]

    # Sort dataframe by frame_no and filename
    detect_df = detect_df.sort_values(
        by=["frame_no", "filename"],
        key=lambda x: np.argsort(index_natsorted(detect_df["filename"])),
    )

    # Export to CSV
    csv_out = Path(evaluation_path, "annotations.csv")
    print(len(detect_df))
    detect_df.to_csv(csv_out, index=False)

    logging.info(f"Report created at {csv_out}")

    if log and registry == "wandb":
        wandb.log({"predictions": wandb.Table(dataframe=detect_df)})

    return detect_df


def generate_tracking_report(tracker_dir: str, eval_dir: str):
    """
    > It takes the tracking output from the tracker and creates a csv file that can be used for
    evaluation

    :param tracker_dir: The directory where the tracking results are stored
    :type tracker_dir: str
    :param eval_dir: The directory where the evaluation results will be stored
    :type eval_dir: str
    :return: A dataframe with the following columns: filename, class_id, frame_no, tracker_id
    """
    data_dict = {}
    if Path(tracker_dir).exists():
        track_files = [str(f.name) for f in Path(tracker_dir).iterdir()]
    else:
        track_files = []
    if len(track_files) == 0:
        logging.error("No tracks found.")
    else:
        for track_file in track_files:
            if str(track_file).endswith(".txt"):
                data_dict[track_file] = []
                with open(Path(tracker_dir, track_file), "r") as infile:
                    lines = infile.readlines()
                    for line in lines:
                        vals = line.split(" ")
                        print(vals)
                        class_id, frame_no, tracker_id = (
                            vals[-3],
                            vals[0],
                            vals[1],
                        )  # vals[-2], vals[0], vals[1] (track_yolo)
                        data_dict[track_file].append([class_id, frame_no, tracker_id])
        dlist = [
            [str(Path(key).parent / Path(key).stem) + f"_{i[1]}.txt", i[0], i[1], i[2]]
            for key, value in data_dict.items()
            for i in value
        ]
        detect_df = pd.DataFrame.from_records(
            dlist, columns=["filename", "class_id", "frame_no", "tracker_id"]
        )
        csv_out = Path(tracker_dir, "tracking.csv")
        detect_df.sort_values(
            by="frame_no",
            key=lambda x: np.argsort(index_natsorted(detect_df["filename"])),
        ).to_csv(csv_out, index=False)
        logging.info("Report created at {}".format(csv_out))
        return detect_df


def generate_counts(
    eval_dir: str,
    tracker_dir: str,
    artifact_dir: str,
    run,
    log: bool = False,
    registry: str = "wandb",
):
    import torch

    model = torch.load(
        Path(
            [
                str(f)
                for f in Path(artifact_dir).iterdir()
                if f.is_file() and "best.pt" in str(f)
            ][-1]
        )
    )
    names = {i: model["model"].names[i] for i in range(len(model["model"].names))}
    tracker_df = generate_tracking_report(tracker_dir, eval_dir)
    if tracker_df is None:
        logging.error("No tracks to count.")
    else:
        tracker_df["frame_no"] = tracker_df["frame_no"].astype(int)
        tracker_df["species_name"] = tracker_df["class_id"].apply(
            lambda x: names[int(x)]
        )
        logging.info("------- DETECTION REPORT -------")
        logging.info("--------------------------------")
        logging.info(tracker_df.groupby(["species_name"])["tracker_id"].nunique())
        final_df = (
            tracker_df.groupby(["species_name"])["tracker_id"]
            .nunique()
            .to_frame()
            .reset_index()
        )
        if log:
            if registry == "wandb":
                # wandb.init(resume="must", id=run.id)
                wandb.log({"tracking_counts": wandb.Table(dataframe=final_df)})
            elif registry == "mlflow":
                pass
        return final_df


def track_objects(
    name: str,
    source_dir: str,
    artifact_dir: str,
    tracker_folder: str,
    conf_thres: float = 0.5,
    img_size: tuple = (720, 540),
    gpu: bool = False,
    test: bool = False,
):
    """
    This function takes in the source directory of the video, the artifact directory, the tracker
    folder, the confidence threshold, and the image size. It then copies the best model from the
    artifact directory to the tracker folder, and runs the tracking script. It then returns the latest
    tracker folder

    :param source_dir: The directory where the images are stored
    :param artifact_dir: The directory where the model is saved
    :param tracker_folder: The folder where tracker runs will be stored
    :param conf_thres: The confidence threshold for the YOLOv5 model
    :param img_size: The size of the image to be used for tracking. The default is 720, defaults to 720 (optional)
    :return: The latest tracker folder
    """
    import torch
    import src.track_yolo as track
    from types import SimpleNamespace

    # Check that tracker folder specified exists
    if not Path(tracker_folder).exists():
        logging.error("The tracker folder does not exist. Please try again")
        return None

    models = [
        str(f)
        for f in Path(artifact_dir).iterdir()
        if f.is_file()
        and ".pt" in str(f)
        and "osnet" not in str(f)
        and "best" in str(f)
    ]

    if len(models) > 0 and not test:
        best_model = models[0]
    else:
        logging.info("No trained model found, using yolov8 base model...")
        best_model = "yolov8s.pt"

    best_model = Path(best_model)

    track_dict = {
        "name": name,
        "source": source_dir,
        "conf": conf_thres,
        "yolo_model": best_model,
        "reid_model": Path(tracker_folder, "osnet_x0_25_msmt17.pt"),
        "imgsz": img_size,
        "project": Path(f"{tracker_folder}/runs/track/"),
        "device": "0" if torch.cuda.is_available() else "cpu",
        "save": True,
        "save_mot": True,
        "save_txt": True,
        "half": True,
        "iou": 0.7,
        "show": False,
        "show_conf": True,
        "show_labels": True,
        "verbose": True,
        "exist_ok": True,
        "classes": None,
        "vid_stride": 1,
        "line_width": None,
        "tracking_method": "deepocsort",
        "save_id_crops": False,
    }

    args = SimpleNamespace(**track_dict)
    track.run(args)
    tracker_root = Path(tracker_folder, "runs", "track")
    latest_tracker = Path(sorted(Path(tracker_root).iterdir())[-1], "mot")
    logging.info(f"Tracking saved succesfully to {latest_tracker}")
    return latest_tracker


def encode_image(filepath):
    """
    It takes a filepath to an image, opens the image, reads the bytes, encodes the bytes as base64, and
    returns the encoded string

    :param filepath: The path to the image file
    :return: the base64 encoding of the image.
    """
    with open(filepath, "rb") as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), "utf-8")
    return "data:image/jpg;base64," + encoded


# def get_annotator(image_path: str, species_list: list, autolabel_model: str = None):
#     """
#     It takes a path to a folder containing images and annotations, and a list of species names, and
#     returns a widget that allows you to view the images and their annotations, and to edit the
#     annotations

#     :param data_path: the path to the image folder
#     :type data_path: str
#     :param species_list: a list of species names
#     :type species_list: list
#     :return: A VBox widget containing a progress bar and a BBoxWidget.
#     """
#     images = sorted(
#         [
#             f
#             for f in Path(image_path).iterdir()
#             if Path(image_path, f).is_file() and f.suffix == ".jpg"
#         ]
#     )

#     annot_path = Path(Path(image_path).parent, "labels")

#     # a progress bar to show how far we got
#     w_progress = widgets.IntProgress(value=0, max=len(images), description="Progress")
#     w_status = widgets.Label(value="")

#     def get_bboxes(image, bboxes, labels, predict: bool = False):
#         logging.getLogger("yolov5").setLevel(logging.WARNING)
#         if predict:
#             detect.run(
#                 weights=autolabel_model,
#                 source=image,
#                 conf_thres=0.5,
#                 nosave=True,
#                 name="labels",
#             )
#         label_file = [
#             f
#             for f in Path(annot_path).iterdir()
#             if Path(annot_path, f).is_file()
#             and f.suffix == ".txt"
#             and Path(f).stem == Path(image).stem
#         ]
#         if len(label_file) == 1:
#             label_file = label_file[0]
#             with open(Path(annot_path, label_file), "r") as f:
#                 for line in f:
#                     s = line.split(" ")
#                     labels.append(s[0])

#                     left = (float(s[1]) - (float(s[3]) / 2)) * width
#                     top = (float(s[2]) - (float(s[4]) / 2)) * height

#                     bboxes.append(
#                         {
#                             "x": left,
#                             "y": top,
#                             "width": float(s[3]) * width,
#                             "height": float(s[4]) * height,
#                             "label": species_list[int(s[0])],
#                         }
#                     )
#             w_status.value = "Annotations loaded"
#         else:
#             w_status.value = "No annotations found"
#         return bboxes, labels

#     # the bbox widget
#     image = Path(image_path, images[0])
#     width, height = imagesize.get(image)
#     bboxes, labels = [], []
#     if autolabel_model is not None:
#         w_status.value = "Loading annotations..."
#         bboxes, labels = get_bboxes(image, bboxes, labels, predict=True)
#     else:
#         w_status.value = "No predictions, using existing labels if available"
#         bboxes, labels = get_bboxes(image, bboxes, labels)
#     w_bbox = BBoxWidget(image=encode_image(image), classes=species_list)

#     # here we assign an empty list to bboxes but
#     # we could also run a detection model on the file
#     # and use its output for creating initial bboxes
#     w_bbox.bboxes = bboxes

#     # combine widgets into a container
#     w_container = widgets.VBox(
#         [
#             w_status,
#             w_progress,
#             w_bbox,
#         ]
#     )

#     def on_button_clicked(b):
#         w_progress.value = 0
#         image = Path(image_path, images[0])
#         width, height = imagesize.get(image)
#         bboxes, labels = [], []
#         if autolabel_model is not None:
#             w_status.value = "Loading annotations..."
#             bboxes, labels = get_bboxes(image, bboxes, labels, predict=True)
#         else:
#             w_status.value = "No annotations found"
#             bboxes, labels = get_bboxes(image, bboxes, labels)
#         w_bbox.image = encode_image(image)

#         # here we assign an empty list to bboxes but
#         # we could also run a detection model on the file
#         # and use its output for creating inital bboxes
#         w_bbox.bboxes = bboxes
#         w_container.children = tuple(list(w_container.children[1:]))
#         b.close()

#     # when Skip button is pressed we move on to the next file
#     def on_skip():
#         w_progress.value += 1
#         if w_progress.value == len(images):
#             button = widgets.Button(
#                 description="Click to restart.",
#                 disabled=False,
#                 display="flex",
#                 flex_flow="column",
#                 align_items="stretch",
#             )
#             if isinstance(w_container.children[0], widgets.Button):
#                 w_container.children = tuple(list(w_container.children[1:]))
#             w_container.children = tuple([button] + list(w_container.children))
#             button.on_click(on_button_clicked)

#         # open new image in the widget
#         else:
#             image_file = images[w_progress.value]
#             image_p = Path(image_path, image_file)
#             width, height = imagesize.get(image_p)
#             w_bbox.image = encode_image(image_p)
#             bboxes, labels = [], []
#             if autolabel_model is not None:
#                 w_status.value = "Loading annotations..."
#                 bboxes, labels = get_bboxes(image_p, bboxes, labels, predict=True)
#             else:
#                 w_status.value = "No annotations found"
#                 bboxes, labels = get_bboxes(image_p, bboxes, labels)

#             # here we assign an empty list to bboxes but
#             # we could also run a detection model on the file
#             # and use its output for creating initial bboxes
#             w_bbox.bboxes = bboxes

#     w_bbox.on_skip(on_skip)

#     # when Submit button is pressed we save current annotations
#     # and then move on to the next file
#     def on_submit():
#         image_file = images[w_progress.value]
#         width, height = imagesize.get(Path(image_path, image_file))
#         # save annotations for current image
#         label_file = Path(image_file).name.replace(".jpg", ".txt")
#         # if the label_file needs to be created
#         if not Path(annot_path).exists():
#             Path(annot_path).mkdir(parents=True, exist_ok=True)
#         open(Path(annot_path, label_file), "w").write(
#             "\n".join(
#                 [
#                     "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
#                         species_list.index(
#                             i["label"]
#                         ),  # single class vs multiple classes
#                         min((i["x"] + i["width"] / 2) / width, 1.0),
#                         min((i["y"] + i["height"] / 2) / height, 1.0),
#                         min(i["width"] / width, 1.0),
#                         min(i["height"] / height, 1.0),
#                     )
#                     for i in w_bbox.bboxes
#                 ]
#             )
#         )
#         # move on to the next file
#         on_skip()

#     w_bbox.on_submit(on_submit)

#     return w_container


def get_annotations_viewer(data_path: str, species_list: list):
    """
    It takes a path to a folder containing images and annotations, and a list of species names, and
    returns a widget that allows you to view the images and their annotations, and to edit the
    annotations

    :param data_path: the path to the data folder
    :type data_path: str
    :param species_list: a list of species names
    :type species_list: list
    :return: A VBox widget containing a progress bar and a BBoxWidget.
    """
    image_path = Path(data_path, "images")
    annot_path = Path(data_path, "labels")

    images = sorted(
        [f for f in Path(image_path).iterdir() if Path(image_path, f).is_file()]
    )
    annotations = sorted(
        [f for f in Path(annot_path).iterdir() if Path(annot_path, f).is_file()]
    )

    # a progress bar to show how far we got
    w_progress = widgets.IntProgress(value=0, max=len(images), description="Progress")
    # the bbox widget
    image = Path(image_path, images[0])
    width, height = imagesize.get(image)
    label_file = annotations[w_progress.value]
    bboxes = []
    labels = []
    with open(Path(annot_path, label_file), "r") as f:
        for line in f:
            s = line.split(" ")
            labels.append(s[0])

            left = (float(s[1]) - (float(s[3]) / 2)) * width
            top = (float(s[2]) - (float(s[4]) / 2)) * height

            bboxes.append(
                {
                    "x": left,
                    "y": top,
                    "width": float(s[3]) * width,
                    "height": float(s[4]) * height,
                    "label": species_list[int(s[0])],
                }
            )
    w_bbox = BBoxWidget(image=encode_image(image), classes=species_list)

    # here we assign an empty list to bboxes but
    # we could also run a detection model on the file
    # and use its output for creating inital bboxes
    w_bbox.bboxes = bboxes

    # combine widgets into a container
    w_container = widgets.VBox(
        [
            w_progress,
            w_bbox,
        ]
    )

    def on_button_clicked(b):
        w_progress.value = 0
        image = Path(image_path, images[0])
        width, height = imagesize.get(image)
        label_file = annotations[w_progress.value]
        bboxes = []
        labels = []
        with open(Path(annot_path, label_file), "r") as f:
            for line in f:
                s = line.split(" ")
                labels.append(s[0])

                left = (float(s[1]) - (float(s[3]) / 2)) * width
                top = (float(s[2]) - (float(s[4]) / 2)) * height

                bboxes.append(
                    {
                        "x": left,
                        "y": top,
                        "width": float(s[3]) * width,
                        "height": float(s[4]) * height,
                        "label": species_list[int(s[0])],
                    }
                )
        w_bbox.image = encode_image(image)

        # here we assign an empty list to bboxes but
        # we could also run a detection model on the file
        # and use its output for creating inital bboxes
        w_bbox.bboxes = bboxes
        w_container.children = tuple(list(w_container.children[1:]))
        b.close()

    # when Skip button is pressed we move on to the next file
    def on_skip():
        w_progress.value += 1
        if w_progress.value == len(annotations):
            button = widgets.Button(
                description="Click to restart.",
                disabled=False,
                display="flex",
                flex_flow="column",
                align_items="stretch",
            )
            if isinstance(w_container.children[0], widgets.Button):
                w_container.children = tuple(list(w_container.children[1:]))
            w_container.children = tuple([button] + list(w_container.children))
            button.on_click(on_button_clicked)

        # open new image in the widget
        else:
            image_file = images[w_progress.value]
            image_p = Path(image_path, image_file)
            width, height = imagesize.get(image_p)
            w_bbox.image = encode_image(image_p)
            label_file = annotations[w_progress.value]
            bboxes = []
            with open(Path(annot_path, label_file), "r") as f:
                for line in f:
                    s = line.split(" ")
                    left = (float(s[1]) - (float(s[3]) / 2)) * width
                    top = (float(s[2]) - (float(s[4]) / 2)) * height
                    bboxes.append(
                        {
                            "x": left,
                            "y": top,
                            "width": float(s[3]) * width,
                            "height": float(s[4]) * height,
                            "label": species_list[int(s[0])],
                        }
                    )

            # here we assign an empty list to bboxes but
            # we could also run a detection model on the file
            # and use its output for creating initial bboxes
            w_bbox.bboxes = bboxes

    w_bbox.on_skip(on_skip)

    # when Submit button is pressed we save current annotations
    # and then move on to the next file
    def on_submit():
        image_file = images[w_progress.value]
        width, height = imagesize.get(Path(image_path, image_file))
        # save annotations for current image
        open(Path(annot_path, label_file), "w").write(
            "\n".join(
                [
                    "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                        species_list.index(
                            i["label"]
                        ),  # single class vs multiple classes
                        min((i["x"] + i["width"] / 2) / width, 1.0),
                        min((i["y"] + i["height"] / 2) / height, 1.0),
                        min(i["width"] / width, 1.0),
                        min(i["height"] / height, 1.0),
                    )
                    for i in w_bbox.bboxes
                ]
            )
        )
        # move on to the next file
        on_skip()

    w_bbox.on_submit(on_submit)

    return w_container


def get_data_viewer(data_path: str):
    """
    It takes a path to a directory of images, and returns a function that displays the images in that
    directory

    :param data_path: the path to the data folder
    :type data_path: str
    :return: A function that takes in a parameter k and a scale parameter and returns a widget that displays the image at index k in the list of images with the specified scale.
    """
    if "empty_string" in data_path:
        logging.info("No files.")
        return None
    imgs = [
        file
        for file in Path(data_path).iterdir()
        if file.is_file() and file.name.lower().endswith(".jpg")
    ]

    def loadimg(k, scale=0.4):
        display(draw_box(Path(data_path, imgs[k]), scale))

    return widgets.interact(loadimg, k=(0, len(imgs) - 1), scale=(0.1, 1.0))


def draw_box(path: str, scale: float):
    """
    It takes a path to an image and a scale parameter, opens the image and resizes it to the specified scale,
    opens the corresponding label file, and draws a box around each object in the image

    :param path: the path to the image
    :type path: str
    :param scale: scale of the image to show
    :type scale: float
    :return: The image resized to the specified scale with the bounding boxes drawn on it.
    """

    im = PILImage.open(path)
    dw, dh = im._size
    im = im.resize((int(dw * scale), int(dh * scale)))
    d = {
        line.split()[0]: line.split()[1:]
        for line in open(path.replace("images", "labels").replace(".jpg", ".txt"))
    }
    dw, dh = im._size
    img1 = ImageDraw.Draw(im)
    for i, vals in d.items():
        vals = tuple(float(val) for val in vals)
        vals_adjusted = tuple(
            [
                int((vals[0] - vals[2] / 2) * dw),
                int((vals[1] - vals[3] / 2) * dh),
                int((vals[0] + vals[2] / 2) * dw),
                int((vals[1] + vals[3] / 2) * dh),
            ]
        )
        img1.rectangle(vals_adjusted, outline="red", width=2)
    return im


# Function to compare original to modified frames
def choose_files(path: str):
    """
    It creates a dropdown menu of all the files in the specified directory, and displays the selected
    file

    :param path: the path to the folder containing the clips
    :type path: str
    """

    # Add "no movie" option to prevent conflicts
    if path is None:
        logging.error("No path selected.")
        return
    files = np.append([str(i) for i in Path(path).iterdir()], "No file")

    clip_path_widget = widgets.Dropdown(
        options=tuple(np.sort(files)),
        description="Select file:",
        ensure_option=True,
        disabled=False,
        layout=widgets.Layout(width="50%"),
        style={"description_width": "initial"},
    )

    main_out = widgets.Output()
    display(clip_path_widget, main_out)

    # Display the original and modified clips
    def on_change(change):
        with main_out:
            clear_output()
            if change["new"] == "No file":
                logging.info("Choose another file")
            else:
                a = view_file(change["new"])
                display(a)

    clip_path_widget.observe(on_change, names="value")
    return clip_path_widget


# Display the frames using html
def view_file(path: str):
    """
    It takes a path to a file, opens it, and returns a widget that can be displayed in the notebook

    :param path: The path to the file you want to view
    :return: A widget that displays the image or video.
    """
    # Get path of the modified clip selected
    extension = Path(path).suffix
    file = open(path, "rb").read()
    if extension.lower() in [".jpeg", ".png", ".jpg"]:
        widget = widgets.Image(value=file, format=extension)
    elif extension.lower() in [".mp4", ".mov", ".avi"]:
        if Path("linked_content").exists():
            shutil.rmtree("linked_content")
        try:
            Path("linked_content").mkdir()
            logging.info("Opening viewer...")
            stream = ffmpeg.input(path)
            stream = ffmpeg.output(stream, f"linked_content/{Path(path).name}")
            ffmpeg.run(stream)
            widget = HTML(
                f"""
                        <video width=800 height=400 alt="test" controls>
                            <source src="linked_content/{Path(path).name}" type="video/{extension.lower().replace(".", "")}">
                        </video>
                    """
            )
        except Exception as e:
            logging.error(
                f"Cannot write to local files, viewing not currently possible. {e}"
            )
            widget = widgets.Image()

    else:
        logging.error(
            "File format not supported. Supported formats: jpeg, png, jpg, mp4, mov, avi."
        )
        widget.Image()

    return widget


def adjust_tracking(
    tracking_folder: str,
    avg_diff_frames: int,
    min_frames_length: int,
    plot_result: bool = False,
):
    """Clean tracking output by removing noisy class changes and short-duration detections."""
    tracking_df = pd.read_csv(str(Path(tracking_folder, "tracking.csv")))

    import torch

    try:
        # Find the latest model file named "best.pt" in the directory
        model_files = [
            f
            for f in Path(tracking_folder).parent.iterdir()
            if f.is_file() and "best.pt" in str(f)
        ]
        if model_files:
            latest_model_file = sorted(model_files)[-1]
            model = torch.load(latest_model_file)
            names = {
                i: model["model"].names[i] for i in range(len(model["model"].names))
            }
        else:
            raise FileNotFoundError("No model file found")

    except (FileNotFoundError, IndexError) as e:
        # Handle specific exceptions for file not found or index errors
        logging.error(f"Error loading model: {e}")
        logging.error("Using class_id.")
        names = {}

    if plot_result:
        fig, ax = plt.subplots(figsize=(15, 5))

        scatter = ax.scatter(
            x=tracking_df["frame_no"],
            y=tracking_df["tracker_id"],
            c=tracking_df["class_id"],
        )

        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(
            *scatter.legend_elements(), loc="upper left", title="Classes"
        )
        ax.grid()
        ax.add_artist(legend1)
        plt.show()

    def custom_tracking_diff(x):
        """Compute the maximum difference between frame numbers"""
        diff_series = np.diff(x)
        if len(diff_series) > 0:
            return diff_series.max()
        else:
            return 1

    def custom_class(x):
        """Choose class by rounding average of classifications"""
        if len(names) > 0:
            return names[int(np.round(x.median()))]
        else:
            return int(np.round(x.median()))

    diff_df = (
        tracking_df.groupby(["tracker_id"])
        .agg({"frame_no": custom_tracking_diff})
        .sort_values(by="frame_no")
    )
    length_df = (
        tracking_df.groupby("tracker_id")
        .agg({"frame_no": "count", "class_id": custom_class})
        .sort_values(by="frame_no", ascending=False)
    )
    total_df = pd.merge(diff_df, length_df, left_index=True, right_index=True)
    total_df.rename(
        columns={"frame_no_x": "max_frame_diff", "frame_no_y": "frame_length"},
        inplace=True,
    )
    if len(names) > 0:
        total_df.rename(columns={"class_id": "species_name"}, inplace=True)
    total_df = pd.merge(
        total_df,
        tracking_df[["tracker_id", "frame_no"]].groupby("tracker_id").first(),
        on="tracker_id",
    )
    logging.info(
        f"Saving tracking file to {str(Path(tracking_folder, 'tracking_clean.csv'))}"
    )
    filtered_df = total_df[
        (total_df.max_frame_diff <= avg_diff_frames)
        & (total_df.frame_length >= min_frames_length)
    ].sort_index()
    if len(names) > 0:
        logging.info(filtered_df["species_name"].value_counts())
    else:
        logging.info(filtered_df["class_id"].value_counts())
    return filtered_df.to_csv(str(Path(tracking_folder, "tracking_clean.csv")))


# Auxiliary function to obtain a dictionary with the mapping between the class ids used by the detection model and the species names
def get_species_mapping(model, project_name, team_name="koster", registry="wandb"):
    import yaml

    def read_yaml_file(file_path):
        with open(file_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data

    if registry == "wandb":
        import wandb

        api = wandb.Api()

        full_path = f"{team_name}/{project_name}"
        runs = api.runs(full_path)  # Get all runs in the project
        for r in runs:
            # Choose the run corresponding to the model given as parameter
            if r.id == model.split("_")[1]:
                run = api.run(project_name + "/" + r.id)

        # Read species mapping into data dictionary
        try:
            # Attempt to directly read species mapping from the configuration
            data_dict = run.rawconfig["data_dict"]
            species_mapping = data_dict["names"]
        except KeyError:
            try:
                # Attempt to read species mapping from a YAML file specified in the configuration
                data_dict = read_yaml_file(run.rawconfig["data"])
                species_mapping = data_dict["names"]
                species_mapping = {str(i): sp for i, sp in enumerate(species_mapping)}
            except (FileNotFoundError, KeyError):
                # Handle the case where species mapping cannot be found in either location
                logging.error("Error reading species mapping from config or file.")
                species_mapping = {}
    elif registry == "mlflow":
        from mlflow import MlflowClient

        experiment = mlflow.get_experiment_by_name(project_name)
        client = MlflowClient()
        pattern = r"runs:/([^/]+)/weights/best\.pt"
        # Use re.search() to find the match
        try:
            run_id = re.search(pattern, model).group(1)
        except:
            logging.error("No valid run found.")
        if experiment is not None:
            # Get the path of the artifact with the labels and class_id
            artifacts = client.list_artifacts(run_id, path="input_datasets")
            run = mlflow.get_run(run_id)
            artifact_uri = run.info.artifact_uri
            yaml_fpath = [
                Path(artifact_uri, af.path)
                for af in artifacts
                if ".yaml" in af.path and "hyp.yaml" not in af.path
            ][0]

            # Temporarily download the artifact with mapping labels
            local_artifact = mlflow.artifacts.download_artifacts(str(yaml_fpath))

            # Attempt to read species mapping from a YAML file specified in the configuration
            data_dict = read_yaml_file(local_artifact)
            species_mapping = data_dict["names"]
            species_mapping = {str(i): sp for i, sp in enumerate(species_mapping)}
    else:
        logging.error("Registry invalid.")

    return species_mapping


def process_detections(
    project: Project,
    db_connection,
    csv_paths: dict,
    annotations_csv_path: str,
    model_registry: str,
    selected_movies_id: dict = None,
    model: str = None,
    project_name: str = None,
    team_name: str = None,
    source_movies: str = None,
):
    """
    > This function computes the given statistics over the detections obtained by a model on different footages for the species of interest,
    and saves the results in different csv files.

    :param project: the project object
    :param db_connection: SQL connection object
    :param csv_paths: a dictionary with the paths of the csvs used to initiate the db
    :param annotations_csv_path: the path to the folder containing the annotations.csv file or the annotations.csv
    :param selected_movies_id: the ids of the movies selected in earlier steps (note if the selection changes b, mlflow)
    :param model_registry: the name of the model register (e.g wandb, mlflow)
    :param model: the name of the model in wandb used to obtain the detections
    :param project_name: name of the project in wandb
    :param team_name: name of the team in wandb.
    :param source_movies: A string with the path to the movies where the model ran inferences from
    """

    # Read the annotations.csv file
    df = pd.read_csv(Path(annotations_csv_path, "annotations.csv"))

    # Check if the DataFrame is not empty
    if df.empty:
        raise ValueError(
            "There are no labels to aggregate, run the model again with a lower threshold or try a different model."
        )

    # Extract the actual filenames using pathlib
    df["filename"] = df["filename"].apply(lambda path: Path(path).name)

    # Remove frame number and txt extension from filename to represent individual movies
    if project_name == "template_project":
        # Extract unique movie names using regular expression
        df["movie_filename"] = df["filename"].str.rsplit("_", n=1).str[0]
    else:
        df["movie_filename"] = (
            df["filename"].str.split("/").str[-1].str.rsplit(pat="_", n=1).str[0]
        )

    # Drop the filename column to avoid confusion
    df = df.drop("filename", axis=1)

    # Add movie ids info from the movies selected in choose_footage
    if selected_movies_id:
        # Create a new column with the mapped values
        df["movie_id"] = df["movie_filename"].apply(
            lambda x: dict(selected_movies_id).get(x, None)
        )

        # Define the movie col of interest
        movie_group_col = "movie_id"

    else:
        # Define the movie col of interest
        movie_group_col = "movie_filename"

    # Map the class id to species labels
    if model_registry == "wandb":
        # Set the name of the template project
        if project_name == "template_project":
            project_name = "spyfish_aotearoa"

    # Obtain a dictionary with the mapping between the class ids and the species names
    species_mapping = get_species_mapping(
        model, project_name, team_name, model_registry
    )

    # Add a column with the species name corresponding to each class id
    df["commonName"] = df["class_id"].astype(str).map(species_mapping)

    # Get max_n per class detected in each movie per frame
    df["max_n"] = df.groupby([movie_group_col, "frame_no"])["commonName"].transform(
        "count"
    )

    # Specify the columns for which we want unique confidence values
    columns_conf = [movie_group_col, "frame_no", "commonName"]

    # Get the confidence range of each detection per frame and add three columns
    df["min_conf"] = df.groupby(columns_conf)["conf"].transform("min")
    df["mean_conf"] = df.groupby(columns_conf)["conf"].transform("mean")
    df["max_conf"] = df.groupby(columns_conf)["conf"].transform("max")

    # Create a boolean mask for duplicated rows based on the specified columns
    mask_duplicates = df.duplicated(subset=columns_conf, keep=False)

    # Keep only unique rows based on grouped columns
    df = df[~mask_duplicates]

    # Retrieve the max counts and conf.levels of uploaded footage
    if all(column_name in df.columns for column_name in ["movie_id", "commonName"]):
        from kso_utils.db_utils import add_db_info_to_df

        # Combine the movie info with the labels
        df = add_db_info_to_df(
            project=project,
            conn=db_connection,
            csv_paths=csv_paths,
            df=df,
            table_name="movies",
        )

        # Combine the site info with the labels
        df = add_db_info_to_df(
            project=project,
            conn=db_connection,
            csv_paths=csv_paths,
            df=df,
            table_name="sites",
        )

        # Combine the species info with the labels
        df = add_db_info_to_df(
            project=project,
            conn=db_connection,
            csv_paths=csv_paths,
            df=df,
            table_name="species",
        )

    # Set the fps information for movies without info in the sql db
    if "fps" not in df.columns:
        from kso_utils.movie_utils import get_fps_duration

        # Get the fps of the movie
        df["fps"], _ = get_fps_duration(movie_path=source_movies)

        # Set the movie id to 0
        df["movie_id"] = 0

    # Calculate the corresponding second of the frame in the movie
    df["second_in_movie"] = (df["frame_no"] / df["fps"]).astype(int)

    # Report the function ran without issues.
    logging.info(
        f"Detections processed. The dataframe has a total of {df.shape[0]} rows and {df.shape[1]} columns",
    )

    return df


def plot_processed_detections(
    df,
    thres: int = 5,  # number of seconds for thresholding in interval
    int_length: int = 10,
):
    """
    > This function computes the given statistics over the detections obtained by a model on different footages for the species of interest,
    and saves the results in different csv files.
    :param df: df of the aggregated detections
    :param thres: The `thres` parameter is used to filter out columns in the `result_df`
    DataFrame where the corresponding `frame_count` column has a value less than `thres`. This
    means that only columns with a minimum number of frames per interval greater than or equal to
    `thres, defaults to 5
    :param int_length: An integer value specifying the length in seconds of interval for filtering

    """

    if "second_in_movie" not in df.columns:
        logging.error("Aggregation plot not currently supported on this film.")

    # Convert 'second_in_movie' to seconds since 0 (e.g. the start of the movie)
    reference_time = pd.to_datetime("1970-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S")

    df["seconds_since_reference"] = (
        pd.to_datetime(df["second_in_movie"], unit="s") - reference_time
    ).dt.total_seconds()

    # Ensure 'seconds_since_reference' is in datetime format
    df["seconds_since_reference"] = pd.to_datetime(
        df["seconds_since_reference"], unit="s"
    )

    # Group by n-second intervals
    interval = pd.Grouper(key="seconds_since_reference", freq=str(int_length) + "S")

    # Group by species and minute, calculate the count
    max_count_per_species = (
        df.groupby(["movie_id", "commonName", interval])["max_n"].max().reset_index()
    )

    # Enable plotting of matplotlib
    try:
        # Enable inline plotting for Matplotlib
        get_ipython().magic("matplotlib inline")
    except ImportError:
        # Handle the case where IPython is not available
        pass
    except Exception as e:
        # Handle other specific exceptions that may occur
        print(f"Error occurred while enabling inline plotting: {e}")

    import matplotlib.pyplot as plt

    # Plot each movie separately
    movies = max_count_per_species["movie_id"].unique()

    for movie_id in movies:
        movie_data = max_count_per_species[
            max_count_per_species["movie_id"] == movie_id
        ]

        # Create a separate line plot for each species
        species_list = movie_data["commonName"].unique()
        plt.figure(figsize=(10, 6))

        for species in species_list:
            species_data = movie_data[movie_data["commonName"] == species]
            plt.plot(
                species_data["seconds_since_reference"],
                species_data["max_n"],
                label=species,
            )

            plt.xlabel("Timestamp (seconds)")
        plt.ylabel("Max Individuals Recorded in a Minute")
        plt.title(f"Max Individuals Recorded Every Minute for Movie {movie_id}")
        plt.legend()
        plt.show()


def main():
    "Handles argument parsing and launches the correct function."
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to data folder", type=str)
    parser.add_argument(
        "perc_test", help="percentage of data to use as part of test set", type=float
    )
    parser.add_argument("out_path", help="path to save into text files", type=str)
    args = parser.parse_args()

    prepare(args.data_path, args.perc_test, args.out_path)


if __name__ == "__main__":
    main()
