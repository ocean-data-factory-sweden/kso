# module imports
import os, re, json, argparse, glob, pims, random, shutil, yaml
import pandas as pd
import numpy as np
import src.frame_tracker
import logging
import datetime
import PIL

from collections import OrderedDict, Counter
from pathlib import Path
from functools import partial
from ast import literal_eval
from tqdm import tqdm
from PIL import Image
from kso_utils.db_utils import create_connection
from kso_utils.koster_utils import unswedify
from kso_utils.t3_utils import retrieve_movie_info_from_server
import kso_utils.tutorials_utils as t_utils
from src.prepare_input import ProcFrameCuda, ProcFrames

# Logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# utility functions
def process_frames(frames_path, size=(416, 416)):
    # Run tests
    gpu_time_0, n_frames = ProcFrames(partial(ProcFrameCuda, size=size), frames_path)
    logging.info(f"Processing performance: {n_frames} frames, {gpu_time_0:.2f} ms/frame")

def process_path(path):
    """
    Process a single path
    """
    return os.path.basename(re.split("_[0-9]+", path)[0]).replace("_frame", "")

def clean_species_name(species_name):
    """
    Clean species name
    """
    return species_name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")

def split_frames(data_path, perc_test):
    """
    Split frames into train and test sets
    """
    dataset_path = Path(data_path)
    images_path = Path(dataset_path, "images")

    # Create and/or truncate train.txt and test.txt
    file_train = open(Path(data_path, "train.txt"), "w")
    file_test = open(Path(data_path, "test.txt"), "w")
    file_valid = open(Path(data_path, "valid.txt"), "w")

    # Populate train.txt and test.txt
    counter = 1
    index_test = int((1-perc_test) * len([s for s in os.listdir(images_path) if s.endswith('.jpg')]))
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
            #if random.uniform(0, 1) <= 0.5:
            #    file_train.write(pathAndFilename + "\n")
            #else:
            file_train.write(pathAndFilename + "\n")
            counter += 1

def frame_aggregation(project_name, db_info_dict, out_path, perc_test, class_list, img_size: tuple, out_format: str = "yolo"):
    """
    Track frames and save to out_path
    """
    # Establish connection to database
    conn = create_connection(db_info_dict["db_path"])
    # Process class list for SQL query    
    class_list = [i.replace(",", "").replace("[", "").replace("]", "") for i in class_list]

    if len(class_list) > 0:
        if len(class_list) == 1:
            species_ref = pd.read_sql_query(
                f"SELECT id FROM species WHERE label=='{class_list[0]}'", conn
            )["id"].tolist()
        else:
            species_ref = pd.read_sql_query(
                f"SELECT id FROM species WHERE label IN {tuple(class_list)}", conn
            )["id"].tolist()
    else:
        species_ref = pd.read_sql_query(f"SELECT id FROM species", conn)["id"].tolist()


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
    train_rows = train_rows.dropna(subset=["x_position", "y_position", "width", "height"])

    # Add dataset metadata to dataset table in koster db
    bboxes = {}
    tboxes = {}
    new_rows = []
    
    movie_df = retrieve_movie_info_from_server(project_name=project_name, db_info_dict=db_info_dict)
    movie_folder = t_utils.get_project_info(project_name, "movie_folder")
    
    if not movie_folder == "None":
        
        train_rows["movie_path"] = movie_folder + train_rows.merge(movie_df, 
                                              left_on="movie_id", right_on="id", how='left')["fpath"]
        video_dict = {}
        for i in train_rows["movie_path"].unique():
            try:
                video_dict[i] = pims.Video(i)
            except:
                try:
                    video_dict[unswedify(i)] = pims.Video(unswedify(i))
                except:
                    logging.warning("Missing file"+f"{i}")
                
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

        for name, group in tqdm(
            train_rows.groupby(["movie_path", "frame_number", "species_id"])
        ):
            movie_path, frame_number, species_id = name[:3]
            named_tuple = tuple([species_id, frame_number, movie_path])

            # Track intermediate frames
            final_name = name[0] if name[0] in video_dict else unswedify(name[0])
            if final_name in video_dict:
                bboxes[named_tuple], tboxes[named_tuple] = [], []
                bboxes[named_tuple].extend(tuple(i[3:]) for i in group.values)
                tboxes[named_tuple].extend(
                    src.frame_tracker.track_objects(
                        video_dict[final_name],
                        species_id,
                        bboxes[named_tuple],
                        frame_number,
                        frame_number + 10,
                    )
                )

                for box in bboxes[named_tuple]:
                    new_rows.append(
                        (
                            species_id,
                            frame_number,
                            movie_path,
                            video_dict[final_name][frame_number].shape[1],
                            video_dict[final_name][frame_number].shape[0],
                        )
                        + box
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

        # Export txt files
        full_rows = pd.DataFrame(
            new_rows,
            columns=[
                "species_id",
                "frame",
                "movie_path",
                "f_w",
                "f_h",
                "x",
                "y",
                "w",
                "h",
            ],
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

        for name, group in tqdm(
            train_rows.groupby(["filename", "species_id"])
        ):
            filename, species_id = name[:2]
            filename = t_utils.get_project_info(project_name, "frame_folder") + filename
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

        # Export txt files
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
        print(len(full_rows))


    # Create output folder
    if not os.path.isdir(out_path):
        os.mkdir(Path(out_path))

    # Set up directory structure
    img_dir = Path(out_path, "images")
    label_dir = Path(out_path, "labels")

    # Create directories
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)

    if os.path.isdir(label_dir):
        shutil.rmtree(label_dir)

    os.mkdir(img_dir)
    os.mkdir(label_dir)

    # Create koster yaml file with model configuration
    species_list = [clean_species_name(sp) for sp in class_list]

    # Write config file
    data = dict(
        train=str(Path(out_path, "train.txt")),
        val=str(Path(out_path, "valid.txt")),
        nc=len(class_list),
        names=species_list,
    )

    with open(Path(out_path, f"{project_name+'_'+datetime.datetime.now().strftime('%H:%M:%S')}.yaml"), "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=None)

    # Write hyperparameters default file (default hyperparameters from https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch.yaml)
    hyp_data = dict(lr0= 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    lrf= 0.1,  # final OneCycleLR learning rate (lr0 * lrf)
                    momentum= 0.937,  # SGD momentum/Adam beta1
                    weight_decay= 0.0005,  # optimizer weight decay 5e-4
                    warmup_epochs= 3.0,  # warmup epochs (fractions ok)
                    warmup_momentum= 0.8,  # warmup initial momentum
                    warmup_bias_lr= 0.1,  # warmup initial bias lr
                    box= 0.05,  # box loss gain
                    cls= 0.5,  # cls loss gain
                    cls_pw= 1.0,  # cls BCELoss positive_weight
                    obj= 1.0,  # obj loss gain (scale with pixels)
                    obj_pw= 1.0,  # obj BCELoss positive_weight
                    iou_t= 0.20,  # IoU training threshold
                    anchor_t= 4.0,  # anchor-multiple threshold
                    # anchors= 3  # anchors per output layer (0 to ignore)
                    fl_gamma= 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
                    hsv_h= 0.015, # image HSV-Hue augmentation (fraction)
                    hsv_s= 0.7,  # image HSV-Saturation augmentation (fraction)
                    hsv_v= 0.4,  # image HSV-Value augmentation (fraction)
                    degrees= 0.0,  # image rotation (+/- deg)
                    translate= 0.1,  # image translation (+/- fraction)
                    scale= 0.5,  # image scale (+/- gain)
                    shear= 0.0,  # image shear (+/- deg)
                    perspective= 0.0,  # image perspective (+/- fraction), range 0-0.001
                    flipud= 0.0,  # image flip up-down (probability)
                    fliplr= 0.5,  # image flip left-right (probability)
                    mosaic= 1.0,  # image mosaic (probability)
                    mixup= 0.0,  # image mixup (probability)
                    copy_paste= 0.0  # segment copy-paste (probability)
                )

    with open(Path(out_path, "hyp.yaml"), "w") as outfile:
        yaml.dump(hyp_data, outfile, default_flow_style=None)

    # Clean species names
    species_df = pd.read_sql_query(
        f"SELECT id, label FROM species", conn
    )
    species_df["clean_label"] = species_df.label.apply(clean_species_name)

    sp_id2mod_id = {
        species_df[species_df.clean_label == species_list[i]].id.values[0]: i
        for i in range(len(species_list))
    }

    if not movie_folder == "None":
        for name, groups in full_rows.groupby(["frame", "movie_path"]):
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
                                else sp_id2mod_id[i[0]],  # single class vs multiple classes
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
                                    else sp_id2mod_id[i[speciesid_pos]],  # single class vs multiple classes
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

    # Clear images
    if len(full_rows) == 0:
        raise Exception("No frames found for the selected species. Please retry with a different configuration.")
    process_frames(out_path + "/images", size=tuple(img_size))

    # Create training/test sets
    split_frames(out_path, perc_test)

def main():
    "Handles argument parsing and launches the correct function."
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path",
        "-o",
        help="output to txt files in YOLO format",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out_format",
        "-f",
        help="format of annotations for export",
        type=str,
        default=r"yolo",
    )
    parser.add_argument(
        "--class_list",
        "-c",
        help="list of classes to use for dataset",
        type=str,
        nargs="*",
        default="",
    )
    parser.add_argument(
        "-db",
        "--db_path",
        type=str,
        help="the absolute path to the database file",
        default=r"koster_lab.db",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--movie_dir",
        type=str,
        help="the directory of movie files",
        default=r"/uploads/",
        required=True,
    )
    parser.add_argument(
        "-pt",
        "--perc_test",
        type=float,
        help="proportion of data to use for testing",
        default=0.2,
        required=True,
    )
    parser.add_argument(
        "-img",
        "--img_size",
        type=int,
        help="image size for model training",
        nargs="+",
        required=True,
    )

    args = parser.parse_args()
    conn = create_connection(args.db_path)
    
    args.class_list = [i.replace(",", "").replace("[", "").replace("]", "") for i in args.class_list]

    if len(args.class_list) > 0:
        if len(args.class_list) == 1:
            species_ref = pd.read_sql_query(
                f"SELECT id FROM species WHERE label=='{args.class_list[0]}'", conn
            )["id"].tolist()
        else:
            species_ref = pd.read_sql_query(
                f"SELECT id FROM species WHERE label IN {tuple(args.class_list)}", conn
            )["id"].tolist()
    else:
        species_ref = pd.read_sql_query(f"SELECT id FROM species", conn)["id"].tolist()


    if len(args.class_list) == 1:
        train_rows = pd.read_sql_query(
            f"SELECT a.subject_id, b.id, b.movie_id, b.frame_number, a.species_id, a.x_position, a.y_position, a.width, a.height FROM \
            agg_annotations_frame AS a INNER JOIN subjects AS b ON a.subject_id=b.id WHERE \
            species_id=='{tuple(species_ref)[0]}' AND subject_type='frame'",
            conn,
        )
    else:
        train_rows = pd.read_sql_query(
            f"SELECT b.frame_number, b.movie_id, a.species_id, a.x_position, a.y_position, a.width, a.height FROM \
            agg_annotations_frame AS a INNER JOIN subjects AS b ON a.subject_id=b.id WHERE species_id IN {tuple(species_ref)} AND subject_type='frame'",
            conn,
        )

    # Add dataset metadata to dataset table in koster db
    bboxes = {}
    tboxes = {}
    new_rows = []
    
    movie_df = pd.read_sql_query("SELECT id, fpath FROM movies", conn)
    
    train_rows["movie_path"] = train_rows.merge(movie_df, 
                                              left_on="movie_id", right_on="id", how='left')["fpath"]
    
    #train_rows["movie_path"] = args.movie_dir + '/' + train_rows["filename"].apply(process_path)

    video_dict = {}
    for i in train_rows["movie_path"].unique():
        try:
            video_dict[i] = pims.Video(i)
        except:
            try:
                video_dict[unswedify(i)] = pims.Video(unswedify(i))
            except:
                logging.warning("Missing file"+f"{i}")
                
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
    
    for name, group in tqdm(
        train_rows.groupby(["movie_path", "frame_number", "species_id"])
    ):
        movie_path, frame_number, species_id = name[:3]
        named_tuple = tuple([species_id, frame_number, movie_path])

        # Track intermediate frames
        final_name = name[0] if name[0] in video_dict else unswedify(name[0])
        if final_name in video_dict:
            bboxes[named_tuple], tboxes[named_tuple] = [], []
            bboxes[named_tuple].extend(tuple(i[3:]) for i in group.values)
            tboxes[named_tuple].extend(
                src.frame_tracker.track_objects(
                    video_dict[final_name],
                    species_id,
                    bboxes[named_tuple],
                    frame_number,
                    frame_number + 10,
                )
            )

            for box in bboxes[named_tuple]:
                new_rows.append(
                    (
                        species_id,
                        frame_number,
                        movie_path,
                        video_dict[final_name][frame_number].shape[1],
                        video_dict[final_name][frame_number].shape[0],
                    )
                    + box
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

    # Export txt files
    full_rows = pd.DataFrame(
        new_rows,
        columns=[
            "species_id",
            "frame",
            "movie_path",
            "f_w",
            "f_h",
            "x",
            "y",
            "w",
            "h",
        ],
    )

    # Create output folder
    if not os.path.isdir(args.out_path):
        os.mkdir(Path(args.out_path))

    # Set up directory structure
    img_dir = Path(args.out_path, "images")
    label_dir = Path(args.out_path, "labels")

    # Create directories
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)

    if os.path.isdir(label_dir):
        shutil.rmtree(label_dir)

    os.mkdir(img_dir)
    os.mkdir(label_dir)

    # Create koster yaml file with model configuration
    species_list = [clean_species_name(sp) for sp in args.class_list]

    data = dict(
        train=str(Path(args.out_path, "train.txt")),
        val=str(Path(args.out_path, "valid.txt")),
        nc=len(args.class_list),
        names=species_list,
    )

    with open(Path(args.out_path, "koster.yaml"), "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=None)

    species_df = pd.read_sql_query(
        f"SELECT id, label FROM species", conn
    )

    species_df["clean_label"] = species_df.label.apply(clean_species_name)

    sp_id2mod_id = {
        species_df[species_df.clean_label == species_list[i]].id.values[0]: i
        for i in range(len(species_list))
    }

    for name, groups in full_rows.groupby(["frame", "movie_path"]):
        file, ext = os.path.splitext(name[1])
        file_base = os.path.basename(file)
        # Added condition to avoid bounding boxes outside of maximum size of frame + added 0 class id when working with single class
        if args.out_format == "yolo":
            open(f"{args.out_path}/labels/{file_base}_frame_{name[0]}.txt", "w").write(
                "\n".join(
                    [
                        "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                            0
                            if len(args.class_list) == 1
                            else sp_id2mod_id[i[0]],  # single class vs multiple classes
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
                f"{args.out_path}/images/{file_base}_frame_{name[0]}.jpg"
            )

    logging.info("Frames extracted successfully")

    # Clear images
    if len(full_rows) == 0:
        raise Exception("No frames found for the selected species. Please retry with a different configuration.")
    process_frames(args.out_path + "/images", size=tuple(args.img_size))

    # Create training/test sets
    split_frames(args.out_path, args.perc_test)


if __name__ == "__main__":
    main()
