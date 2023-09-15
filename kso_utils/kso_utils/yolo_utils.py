# base imports
import glob
import os
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
import ipywidgets as widgets
from jupyter_bbox_widget import BBoxWidget
from IPython.display import display, clear_output, HTML
from PIL import Image as PILImage, ImageDraw

from functools import partial
from tqdm import tqdm
from pathlib import Path
from collections.abc import Callable
from natsort import index_natsorted

# util imports
from kso_utils.project_utils import Project

try:
    from yolov5.utils import torch_utils
    import yolov5.detect as detect
except ModuleNotFoundError:
    logging.error(
        "Modules yolov5 and yolov5_tracker are required for ML functionality."
    )

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
    files = os.listdir(frames_path)
    for f in files:
        if f.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
            if os.path.exists(str(Path(frames_path, f))):
                new_frame = proc_frame_func(cv2.imread(str(Path(frames_path, f))))
                cv2.imwrite(str(Path(frames_path, f)), new_frame)
            else:
                from kso_utils.movie_utils import unswedify

                new_frame = proc_frame_func(
                    cv2.imread(unswedify(str(Path(frames_path, f))))
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
    index_test = int((1 - percentage_test) / 100 * len(os.listdir(dataset_path)))
    latest_movie = ""
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        movie_name = title.replace("_frame_*", "", regex=True)

        if counter == index_test + 1:
            if movie_name != latest_movie:
                file_test.write(out_path + os.path.basename(title) + ".jpg" + "\n")
            else:
                file_train.write(out_path + os.path.basename(title) + ".jpg" + "\n")
            counter += 1
        else:
            latest_movie = movie_name
            file_train.write(out_path + os.path.basename(title) + ".jpg" + "\n")
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
    return os.path.basename(re.split("_[0-9]+", path)[0]).replace("_frame", "")


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
    index_test = int(
        (1 - perc_test)
        * len([s for s in os.listdir(images_path) if s.endswith(".jpg")])
    )
    latest_movie = ""
    for pathAndFilename in glob.iglob(os.path.join(images_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        movie_name = title.replace("_frame_*", "")

        if counter >= index_test + 1:
            # Avoid leaking frames into test set
            if movie_name != latest_movie or movie_name == title:
                file_valid.write(pathAndFilename + "\n")
            else:
                file_train.write(pathAndFilename + "\n")
            counter += 1
        else:
            latest_movie = movie_name
            # if random.uniform(0, 1) <= 0.5:
            #    file_train.write(pathAndFilename + "\n")
            # else:
            file_train.write(pathAndFilename + "\n")
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
    clean_list = [clean_label(label) for label in class_list]
    train_rows = agg_df[agg_df.label.isin(clean_list)]

    # Rename columns if in different format
    train_rows = train_rows.rename(
        columns={"x": "x_position", "y": "y_position", "w": "width", "h": "height"}
    ).copy()

    # Remove null annotations
    if remove_nulls:
        train_rows = train_rows.dropna(
            subset=["x_position", "y_position", "width", "height"],
        ).copy()

    # Check if any frames are left after removing null values
    if len(train_rows) == 0:
        logging.error("No frames left. Please adjust aggregation parameters.")
        return

    # Create output folder
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    # Set up directory structure
    img_dir = Path(out_path, "images")
    label_dir = Path(out_path, "labels")

    # Create image and label directories
    os.mkdir(img_dir)
    os.mkdir(label_dir)

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
            f"{project.Project_name+'_'+datetime.datetime.now().strftime('%H:%M:%S')}.yaml",
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
    species_df = pd.read_sql_query("SELECT id, label FROM species", conn)
    species_df["clean_label"] = species_df.label.apply(clean_species_name)
    species_df["zoo_label"] = species_df.label.apply(clean_label)

    # Add species_id to train_rows
    if "species_id" not in train_rows.columns:
        train_rows["species_id"] = train_rows["label"].apply(
            lambda x: species_df[species_df.zoo_label == x].id.values[0], 1
        )
        train_rows.drop(columns=["label"], axis=1, inplace=True)

    sp_id2mod_id = {
        species_df[species_df.clean_label == species_list[i]].id.values[0]: i
        for i in range(len(species_list))
    }

    # Get movie info from server
    from kso_utils.movie_utils import retrieve_movie_info_from_server

    movie_df = retrieve_movie_info_from_server(
        project=project,
        server_connection=server_connection,
        db_connection=db_connection,
    )

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

    if movie_bool:
        # Get movie path on the server
        train_rows["movie_path"] = train_rows.merge(
            movie_df, left_on="movie_id", right_on="id", how="left"
        )["fpath"]

        from kso_utils.movie_utils import get_movie_path

        train_rows["movie_path"] = train_rows["movie_path"].apply(
            lambda x: get_movie_path(project, x)
        )

        # Read each movie for efficient frame access
        video_dict = {}
        for i in tqdm(train_rows["movie_path"].unique()):
            try:
                video_dict[i] = pims.MoviePyReader(i)
            except FileNotFoundError:
                try:
                    from kso_utils.movie_utils import unswedify

                    video_dict[unswedify(str(i))] = pims.Video(unswedify(str(i)))
                except KeyError:
                    logging.warning("Missing file" + f"{i}")

        # Create full rows
        train_rows = train_rows.sort_values(
            by=["movie_path", "frame_number"], ascending=True
        )

        # Ensure key fields wrt movies are available
        key_fields = [
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

    group_fields = (
        ["subject_ids", "species_id"]
        if link_bool
        else (
            ["movie_path", "frame_number", "species_id"]
            if movie_bool
            else ["filename", "species_id"]
        )
    )

    new_rows = []
    bboxes = {}
    tboxes = {}

    for name, group in tqdm(train_rows.groupby(group_fields)):
        grouped_fields = name[: len(group_fields)]
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
            from kso_utils.movie_utils import unswedify

            final_name = name[0] if name[0] in video_dict else unswedify(name[0])

            if grouped_fields[1] > len(video_dict[final_name]):
                logging.warning(
                    f"Frame out of range for video of length {len(video_dict[final_name])}"
                )

            if final_name in video_dict:
                bboxes[named_tuple], tboxes[named_tuple] = [], []
                bboxes[named_tuple].extend(
                    tuple(i[len(grouped_fields) :]) for i in group.values
                )
                movie_w, movie_h = video_dict[final_name][0].shape

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
                        track_frames(
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
                                grouped_fields[-1],
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

            for box in bboxes[named_tuple]:
                new_rows.append(
                    (
                        grouped_fields[-1],  # species_id
                        filename,
                        PIL.Image.open(requests.get(filename, stream=True).raw).size[0]
                        if link_bool
                        else PIL.Image.open(filename).size[0],
                        PIL.Image.open(requests.get(filename, stream=True).raw).size[1]
                        if link_bool
                        else PIL.Image.open(filename).size[1],
                    )
                    + box
                )

    ### Final export step
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
            file, ext = os.path.splitext(name[1])
            file_base = os.path.basename(file)
            file_out = f"{out_path}/labels/{file_base}_frame_{name[0]}.txt"
            img_out = f"{out_path}/images/{file_base}_frame_{name[0]}.jpg"
        else:
            file, ext = os.path.splitext(name)
            file_base = os.path.basename(file)
            file_out = f"{out_path}/labels/{file_base}.txt"
            img_out = f"{out_path}/images/{file_base}.jpg"

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
                                0
                                if len(class_list) == 1
                                else sp_id2mod_id[
                                    i[speciesid_pos]
                                ],  # single class vs multiple classes
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
            from kso_utils.movie_utils import unswedify

            save_name = name[1] if name[1] in video_dict else unswedify(name[1])
            if save_name in video_dict:
                PIL.Image.fromarray(
                    video_dict[save_name][name[0]][:, :, [2, 1, 0]]
                ).save(img_out)
        else:
            if link_bool:
                image_output = PIL.Image.open(requests.get(name, stream=True).raw)
            else:
                image_output = np.asarray(PIL.Image.open(name))
            PIL.Image.fromarray(np.asarray(image_output)).save(img_out)

    logging.info("Frames extracted successfully")

    # Check that at least some frames remain after aggregation
    if len(full_rows) == 0:
        raise Exception(
            "No frames found for the selected species. Please retry with a different configuration."
        )

    # Pre-process frames
    process_frames(out_path + "/images", size=tuple(img_size))

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


def track_frames(
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
    for current_frame in range(start_frame + 1, last_frame + 1):
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


def choose_baseline_model(download_path: str, test: bool = False):
    """
    It downloads the latest version of the baseline model from WANDB
    :return: The path to the baseline model.
    """
    api = wandb.Api()
    # weird error fix (initialize api another time)
    api.runs(path="koster/model-registry")
    api = wandb.Api()
    collections = [
        coll
        for coll in api.artifact_type(
            type_name="model", project="koster/model-registry"
        ).collections()
    ]

    model_dict = {}
    for artifact in collections:
        model_dict[artifact.name] = artifact

    model_widget = widgets.Dropdown(
        options=[(name, model) for name, model in model_dict.items()],
        value=None,
        description="Select model:",
        ensure_option=False,
        disabled=False,
        layout=widgets.Layout(width="50%"),
        style={"description_width": "initial"},
    )

    main_out = widgets.Output()
    display(model_widget, main_out)

    def on_change(change):
        with main_out:
            clear_output()
            try:
                for af in model_dict[change["new"].name].versions():
                    artifact_dir = af.download(download_path)
                    artifact_file = [
                        str(Path(artifact_dir, i))
                        for i in os.listdir(artifact_dir)
                        if i.endswith(".pt")
                    ][-1]
                    logging.info(
                        f"Baseline {af.name} successfully downloaded from WANDB"
                    )
                    model_widget.artifact_path = artifact_file
            except Exception as e:
                logging.error(
                    f"Failed to download the baseline model. Please ensure you are logged in to WANDB. {e}"
                )

    model_widget.observe(on_change, names="value")
    if test:
        model_widget.value = model_dict["baseline-yolov5"]
    return model_widget


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
                str(Path(output_folder, _))
                for _ in os.listdir(output_folder)
                if _.endswith(".yaml") and "hyp" not in _
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


def set_config(conf_thres: float, model: str, eval_dir: str):
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
    config.confidence_threshold = conf_thres
    config.model_name = model
    config.evaluation_directory = eval_dir
    return config


def add_data_wandb(path: str, name: str, run):
    """
    > The function `add_data_wandb` takes a path to a directory, a name for the directory, and a run
    object, and adds the directory to the run as an artifact

    :param path: the path to the directory you want to upload
    :type path: str
    :param name: The name of the artifact
    :type name: str
    :param run: The run object that you get from calling wandb.init()
    """
    my_data = wandb.Artifact(name, type="raw_data")
    my_data.add_dir(path)
    run.log_artifact(my_data)


def generate_csv_report(evaluation_path: str, run, wandb_log: bool = False):
    """
    > We read the labels from the `labels` folder, and create a dictionary with the filename as the key,
    and the list of labels as the value. We then convert this dictionary to a dataframe, and write it to
    a csv file

    :param evaluation_path: The path to the evaluation folder
    :type evaluation_path: str
    :return: A dataframe with the following columns:
        filename, class_id, frame_no, x, y, w, h, conf
    """
    labels = os.listdir(Path(evaluation_path, "labels"))
    data_dict = {}
    for f in labels:
        frame_no = int(f.split("_")[-1].replace(".txt", ""))
        data_dict[f] = []
        with open(Path(evaluation_path, "labels", f), "r") as infile:
            lines = infile.readlines()
            for line in lines:
                class_id, x, y, w, h, conf = line.split(" ")
                data_dict[f].append([class_id, frame_no, x, y, w, h, float(conf)])
    dlist = [[key, *i] for key, value in data_dict.items() for i in value]
    detect_df = pd.DataFrame.from_records(
        dlist, columns=["filename", "class_id", "frame_no", "x", "y", "w", "h", "conf"]
    )
    csv_out = Path(evaluation_path, "annotations.csv")
    detect_df.sort_values(
        by="frame_no", key=lambda x: np.argsort(index_natsorted(detect_df["filename"]))
    ).to_csv(csv_out, index=False)
    logging.info("Report created at {}".format(csv_out))
    if wandb_log:
        wandb.init(resume="must", id=run.id)
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
    if os.path.exists(tracker_dir):
        track_files = os.listdir(tracker_dir)
    else:
        track_files = []
    if len(track_files) == 0:
        logging.error("No tracks found.")
    else:
        for track_file in track_files:
            if track_file.endswith(".txt"):
                data_dict[track_file] = []
                with open(Path(tracker_dir, track_file), "r") as infile:
                    lines = infile.readlines()
                    for line in lines:
                        vals = line.split(" ")
                        class_id, frame_no, tracker_id = vals[0], vals[1], vals[2]
                        data_dict[track_file].append([class_id, frame_no, tracker_id])
        dlist = [
            [os.path.splitext(key)[0] + f"_{i[1]}.txt", i[0], i[1], i[2]]
            for key, value in data_dict.items()
            for i in value
        ]
        detect_df = pd.DataFrame.from_records(
            dlist, columns=["filename", "class_id", "frame_no", "tracker_id"]
        )
        csv_out = Path(eval_dir, "tracking.csv")
        detect_df.sort_values(
            by="frame_no",
            key=lambda x: np.argsort(index_natsorted(detect_df["filename"])),
        ).to_csv(csv_out, index=False)
        logging.info("Report created at {}".format(csv_out))
        return detect_df


def generate_counts(
    eval_dir: str, tracker_dir: str, artifact_dir: str, run, wandb_log: bool = False
):
    import torch

    model = torch.load(
        Path(
            [
                f
                for f in Path(artifact_dir).iterdir()
                if f.is_file() and ".pt" in str(f)
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
        if wandb_log:
            wandb.init(resume="must", id=run.id)
            wandb.log({"tracking_counts": wandb.Table(dataframe=final_df)})
        return final_df


def track_objects(
    name: str,
    source_dir: str,
    artifact_dir: str,
    tracker_folder: str,
    conf_thres: float = 0.5,
    img_size: tuple = (720, 540),
    gpu: bool = False,
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
    import yolov5_tracker.track as track

    # Check that tracker folder specified exists
    if not os.path.exists(tracker_folder):
        logging.error("The tracker folder does not exist. Please try again")
        return None

    model_path = [
        f
        for f in Path(artifact_dir).iterdir()
        if f.is_file()
        and ".pt" in str(f)
        and "osnet" not in str(f)
        and "best" in str(f)
    ][0]

    best_model = Path(model_path)

    if not gpu:
        track.run(
            name=name,
            source=source_dir,
            conf_thres=conf_thres,
            yolo_weights=best_model,
            reid_weights=Path(tracker_folder, "osnet_x0_25_msmt17.pt"),
            imgsz=img_size,
            project=Path(f"{tracker_folder}/runs/track/"),
            save_vid=True,
            save_conf=True,
            save_txt=True,
        )
    else:
        track.run(
            name=name,
            source=source_dir,
            conf_thres=conf_thres,
            yolo_weights=best_model,
            reid_weights=Path(tracker_folder, "osnet_x0_25_msmt17.pt"),
            imgsz=img_size,
            project=Path(f"{tracker_folder}/runs/track/"),
            device=torch_utils.select_device(""),
            save_vid=True,
            save_conf=True,
            save_txt=True,
            half=True,
        )

    tracker_root = os.path.join(tracker_folder, "runs", "track")
    latest_tracker = os.path.join(
        tracker_root, sorted(os.listdir(tracker_root))[-1], "tracks"
    )
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


def get_annotator(image_path: str, species_list: list, autolabel_model: str = None):
    """
    It takes a path to a folder containing images and annotations, and a list of species names, and
    returns a widget that allows you to view the images and their annotations, and to edit the
    annotations

    :param data_path: the path to the image folder
    :type data_path: str
    :param species_list: a list of species names
    :type species_list: list
    :return: A VBox widget containing a progress bar and a BBoxWidget.
    """
    images = sorted(
        [
            f
            for f in os.listdir(image_path)
            if os.path.isfile(os.path.join(image_path, f)) and f.endswith(".jpg")
        ]
    )

    annot_path = os.path.join(Path(image_path).parent, "labels")

    # a progress bar to show how far we got
    w_progress = widgets.IntProgress(value=0, max=len(images), description="Progress")
    w_status = widgets.Label(value="")

    def get_bboxes(image, bboxes, labels, predict: bool = False):
        logging.getLogger("yolov5").setLevel(logging.WARNING)
        if predict:
            detect.run(
                weights=autolabel_model,
                source=image,
                conf_thres=0.5,
                nosave=True,
                name="labels",
            )
        label_file = [
            f
            for f in os.listdir(annot_path)
            if os.path.isfile(os.path.join(annot_path, f))
            and f.endswith(".txt")
            and Path(f).stem == Path(image).stem
        ]
        if len(label_file) == 1:
            label_file = label_file[0]
            with open(os.path.join(annot_path, label_file), "r") as f:
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
            w_status.value = "Annotations loaded"
        else:
            w_status.value = "No annotations found"
        return bboxes, labels

    # the bbox widget
    image = os.path.join(image_path, images[0])
    width, height = imagesize.get(image)
    bboxes, labels = [], []
    if autolabel_model is not None:
        w_status.value = "Loading annotations..."
        bboxes, labels = get_bboxes(image, bboxes, labels, predict=True)
    else:
        w_status.value = "No predictions, using existing labels if available"
        bboxes, labels = get_bboxes(image, bboxes, labels)
    w_bbox = BBoxWidget(image=encode_image(image), classes=species_list)

    # here we assign an empty list to bboxes but
    # we could also run a detection model on the file
    # and use its output for creating initial bboxes
    w_bbox.bboxes = bboxes

    # combine widgets into a container
    w_container = widgets.VBox(
        [
            w_status,
            w_progress,
            w_bbox,
        ]
    )

    def on_button_clicked(b):
        w_progress.value = 0
        image = os.path.join(image_path, images[0])
        width, height = imagesize.get(image)
        bboxes, labels = [], []
        if autolabel_model is not None:
            w_status.value = "Loading annotations..."
            bboxes, labels = get_bboxes(image, bboxes, labels, predict=True)
        else:
            w_status.value = "No annotations found"
            bboxes, labels = get_bboxes(image, bboxes, labels)
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
        if w_progress.value == len(images):
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
            image_p = os.path.join(image_path, image_file)
            width, height = imagesize.get(image_p)
            w_bbox.image = encode_image(image_p)
            bboxes, labels = [], []
            if autolabel_model is not None:
                w_status.value = "Loading annotations..."
                bboxes, labels = get_bboxes(image_p, bboxes, labels, predict=True)
            else:
                w_status.value = "No annotations found"
                bboxes, labels = get_bboxes(image_p, bboxes, labels)

            # here we assign an empty list to bboxes but
            # we could also run a detection model on the file
            # and use its output for creating initial bboxes
            w_bbox.bboxes = bboxes

    w_bbox.on_skip(on_skip)

    # when Submit button is pressed we save current annotations
    # and then move on to the next file
    def on_submit():
        image_file = images[w_progress.value]
        width, height = imagesize.get(os.path.join(image_path, image_file))
        # save annotations for current image
        label_file = Path(image_file).name.replace(".jpg", ".txt")
        # if the label_file needs to be created
        if not os.path.exists(annot_path):
            Path(annot_path).mkdir(parents=True, exist_ok=True)
        open(os.path.join(annot_path, label_file), "w").write(
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
    image_path = os.path.join(data_path, "images")
    annot_path = os.path.join(data_path, "labels")

    images = sorted(
        [
            f
            for f in os.listdir(image_path)
            if os.path.isfile(os.path.join(image_path, f))
        ]
    )
    annotations = sorted(
        [
            f
            for f in os.listdir(annot_path)
            if os.path.isfile(os.path.join(annot_path, f))
        ]
    )

    # a progress bar to show how far we got
    w_progress = widgets.IntProgress(value=0, max=len(images), description="Progress")
    # the bbox widget
    image = os.path.join(image_path, images[0])
    width, height = imagesize.get(image)
    label_file = annotations[w_progress.value]
    bboxes = []
    labels = []
    with open(os.path.join(annot_path, label_file), "r") as f:
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
        image = os.path.join(image_path, images[0])
        width, height = imagesize.get(image)
        label_file = annotations[w_progress.value]
        bboxes = []
        labels = []
        with open(os.path.join(annot_path, label_file), "r") as f:
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
            image_p = os.path.join(image_path, image_file)
            width, height = imagesize.get(image_p)
            w_bbox.image = encode_image(image_p)
            label_file = annotations[w_progress.value]
            bboxes = []
            with open(os.path.join(annot_path, label_file), "r") as f:
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
        width, height = imagesize.get(os.path.join(image_path, image_file))
        # save annotations for current image
        open(os.path.join(annot_path, label_file), "w").write(
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
    imgs = list(filter(lambda fn: fn.lower().endswith(".jpg"), os.listdir(data_path)))

    def loadimg(k, scale=0.4):
        display(draw_box(os.path.join(data_path, imgs[k]), scale))

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
    files = np.append([path + i for i in os.listdir(path)], "No file")

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
    extension = os.path.splitext(path)[1]
    file = open(path, "rb").read()
    if extension.lower() in [".jpeg", ".png", ".jpg"]:
        widget = widgets.Image(value=file, format=extension)
    elif extension.lower() in [".mp4", ".mov", ".avi"]:
        if os.path.exists("linked_content"):
            shutil.rmtree("linked_content")
        try:
            os.mkdir("linked_content")
            logging.info("Opening viewer...")
            stream = ffmpeg.input(path)
            stream = ffmpeg.output(stream, f"linked_content/{os.path.basename(path)}")
            ffmpeg.run(stream)
            widget = HTML(
                f"""
                        <video width=800 height=400 alt="test" controls>
                            <source src="linked_content/{os.path.basename(path)}" type="video/{extension.lower().replace(".", "")}">
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
