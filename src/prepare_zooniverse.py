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
from kso_utils.koster_utils import unswedify
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
    """
    It takes a project, a database, an output path, a percentage of frames to use for testing, a list of
    species to include, an image size, an output format, a boolean to remove null annotations, a boolean
    to track frames, and the number of frames to track, and it returns a dataset of frames with bounding
    boxes for the specified species
    
    :param project: the project object
    :param db_info_dict: a dictionary containing the path to the database and the database name
    :type db_info_dict: dict
    :param out_path: the path to the folder where you want to save the dataset
    :type out_path: str
    :param perc_test: The percentage of frames that will be used for testing
    :type perc_test: float
    :param class_list: list of species to include in the dataset
    :type class_list: list
    :param img_size: tuple, the size of the images to be used for training
    :type img_size: tuple
    :param out_format: str = "yolo",, defaults to yolo
    :type out_format: str (optional)
    :param remove_nulls: Remove null annotations from the dataset, defaults to True
    :type remove_nulls: bool (optional)
    :param track_frames: If True, the script will track the bounding boxes for n_tracked_frames frames
    after the object is detected, defaults to True
    :type track_frames: bool (optional)
    :param n_tracked_frames: number of frames to track after an object is detected, defaults to 10
    :type n_tracked_frames: int (optional)
    """
    # Establish connection to database
    conn = create_connection(db_info_dict["db_path"])

    # Select the id/s of species of interest
    if class_list[0] == "":
        logging.error(
            "No species were selected. Please select at least one species before continuing."
        )
    else:
        species_ref = get_species_ids(project, class_list)

    # Select the aggregated classifications from the species of interest
    if len(class_list) == 1:
        train_rows = pd.read_sql_query(
            f"SELECT a.subject_id, b.id, b.movie_id, b.filename, b.frame_number, a.species_id, a.x_position, a.y_position, a.width, a.height FROM \
            agg_annotations_frame AS a INNER JOIN subjects AS b ON a.subject_id=b.id WHERE \
            species_id=='{tuple(species_ref)[0]}' AND subject_type='frame'",
            conn,
        )
    else:
        train_rows = pd.read_sql_query(
            f"SELECT b.frame_number, b.movie_id, b.filename, a.species_id, a.x_position, a.y_position, a.width, a.height FROM \
            agg_annotations_frame AS a INNER JOIN subjects AS b ON a.subject_id=b.id WHERE species_id IN {tuple(species_ref)} AND subject_type='frame'",
            conn,
        )

    # Remove null annotations
    if remove_nulls:
        train_rows = train_rows.dropna(
            subset=["x_position", "y_position", "width", "height"]
        )

    if len(train_rows) == 0:
        logging.error("No frames left. Please adjust aggregation parameters.")

    # Get movie info from server
    movie_df = retrieve_movie_info_from_server(
        project=project, db_info_dict=db_info_dict
    )

    # Create output folder
    if not os.path.isdir(out_path):
        os.mkdir(Path(out_path))

    # Set up directory structure
    img_dir = Path(out_path, "images")
    label_dir = Path(out_path, "labels")

    # Create image and label directories
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)

    if os.path.isdir(label_dir):
        shutil.rmtree(label_dir)

    os.mkdir(img_dir)
    os.mkdir(label_dir)

    # Create timestamped koster yaml file with model configuration
    species_list = [clean_species_name(sp) for sp in class_list]

    # Write config file
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

    sp_id2mod_id = {
        species_df[species_df.clean_label == species_list[i]].id.values[0]: i
        for i in range(len(species_list))
    }

    # If at least one movie is linked to the project
    logging.info(f"There are {len(movie_df)} movies")

    if len(movie_df) > 0:

        train_rows["movie_path"] = train_rows.merge(
            movie_df, left_on="movie_id", right_on="id", how="left"
        )["spath"]

        train_rows["movie_path"] = train_rows["movie_path"].apply(
            lambda x: get_movie_url(project, db_info_dict, x)
        )

        video_dict = {}
        for i in tqdm(train_rows["movie_path"].unique()):
            try:
                video_dict[i] = pims.MoviePyReader(i)
            except FileNotFoundError:
                try:
                    video_dict[unswedify(str(i))] = pims.Video(unswedify(str(i)))
                except KeyError:
                    logging.warning("Missing file" + f"{i}")

        # Ensure column order
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
        bboxes = {}
        tboxes = {}

        # Create full rows
        train_rows = train_rows.sort_values(
            by=["movie_path", "frame_number"], ascending=True
        )
        for name, group in tqdm(
            train_rows.groupby(["movie_path", "frame_number", "species_id"])
        ):
            movie_path, frame_number, species_id = name[:3]
            named_tuple = tuple([species_id, frame_number, movie_path])

            final_name = name[0] if name[0] in video_dict else unswedify(name[0])
            if frame_number > len(video_dict[final_name]):
                logging.warning(
                    f"Frame out of range for video of length {len(video_dict[final_name])}"
                )
                frame_number = frame_number // 2
            if final_name in video_dict:
                bboxes[named_tuple], tboxes[named_tuple] = [], []
                bboxes[named_tuple].extend(tuple(i[3:]) for i in group.values)
                movie_h = video_dict[final_name][0].shape[1]
                movie_w = video_dict[final_name][0].shape[0]

                for box in bboxes[named_tuple]:
                    new_rows.append(
                        (
                            species_id,
                            frame_number,
                            movie_path,
                            movie_h,
                            movie_w,
                        )
                        + box
                    )

                if track_frames:
                    # Track n frames after object is detected
                    tboxes[named_tuple].extend(
                        src.frame_tracker.track_objects(
                            video_dict[final_name],
                            species_id,
                            bboxes[named_tuple],
                            frame_number,
                            frame_number + n_tracked_frames,
                        )
                    )
                    for box in tboxes[named_tuple]:
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

        for name, groups in tqdm(
            full_rows.groupby(["frame_number", "filename"]),
            desc="Saving frames...",
            colour="green",
        ):
            file, ext = os.path.splitext(name[1])
            file_base = os.path.basename(file)
            # Added condition to avoid bounding boxes outside of maximum size of frame + added 0 class id when working with single class
            if out_format == "yolo":
                open(f"{out_path}/labels/{file_base}_frame_{name[0]}.txt", "w").write(
                    "\n".join(
                        [
                            "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                                0
                                if len(class_list) == 1
                                else sp_id2mod_id[
                                    i[0]
                                ],  # single class vs multiple classes
                                min((i[5] + i[7] / 2) / i[3], 1.0),
                                min((i[6] + i[8] / 2) / i[4], 1.0),
                                min(i[7] / i[3], 1.0),
                                min(i[8] / i[4], 1.0),
                            )
                            for i in groups.values
                        ]
                    )
                )

            # Save frames to image files
            save_name = name[1] if name[1] in video_dict else unswedify(name[1])
            if save_name in video_dict:
                Image.fromarray(video_dict[save_name][name[0]][:, :, [2, 1, 0]]).save(
                    f"{out_path}/images/{file_base}_frame_{name[0]}.jpg"
                )
    else:
        train_rows = train_rows[
            [
                "species_id",
                "filename",
                "x_position",
                "y_position",
                "width",
                "height",
            ]
        ]

        new_rows = []
        bboxes = {}
        tboxes = {}

        for name, group in tqdm(train_rows.groupby(["filename", "species_id"])):
            filename, species_id = name[:2]
            filename = project.photo_folder + filename
            named_tuple = tuple([species_id, filename])

            # Track intermediate frames
            bboxes[named_tuple] = []
            bboxes[named_tuple].extend(tuple(i[2:]) for i in group.values)

            for box in bboxes[named_tuple]:
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

        # Export full rows
        for name, groups in full_rows.groupby(["filename"]):
            file, ext = os.path.splitext(name)
            file_base = os.path.basename(file)
            # Added condition to avoid bounding boxes outside of maximum size of frame + added 0 class id when working with single class
            if out_format == "yolo":
                if len(groups.values) == 1 and str(groups.values[0][-1]) == "nan":
                    open(f"{out_path}/labels/{file_base}.txt", "w")
                else:
                    groups = [i for i in groups.values if str(i[-1]) != "nan"]
                    open(f"{out_path}/labels/{file_base}.txt", "w").write(
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
            save_name = name
            Image.fromarray(np.asarray(PIL.Image.open(save_name))).save(
                f"{out_path}/images/{file_base}.jpg"
            )

    logging.info("Frames extracted successfully")

    if len(full_rows) == 0:
        raise Exception(
            "No frames found for the selected species. Please retry with a different configuration."
        )

    # Pre-process frames (Turned off since we now implement transformations separately)
    # process_frames(out_path + "/images", size=tuple(img_size))

    # Create training/test sets
    split_frames(out_path, perc_test)
