# module imports
import os, re, json, argparse, glob, pims, random, shutil, yaml
import pandas as pd
import numpy as np
import frame_tracker

from collections import OrderedDict, Counter
from pathlib import Path
from functools import partial
from ast import literal_eval
from tqdm import tqdm
from PIL import Image
from tutorial_utils.db_utils import create_connection, unswedify, clean_species_name
from prepare_input import ProcFrameCuda, ProcFrames

# utility functions
def process_frames(frames_path, size=(416, 416)):
    # Run tests
    gpu_time_0, n_frames = ProcFrames(partial(ProcFrameCuda, size=size), frames_path)
    print(f"Processing performance: {n_frames} frames, {gpu_time_0:.2f} ms/frame")


def process_path(path):
    print(path)
    return os.path.basename(re.split("_[0-9]+", path)[0]).replace("_frame", "")

def split_frames(data_path, perc_test):
    dataset_path = Path(data_path)
    images_path = Path(dataset_path, "images")

    # Create and/or truncate train.txt and test.txt
    file_train = open(Path(data_path, "train.txt"), "w")
    file_valid = open(Path(data_path, "valid.txt"), "w")
    file_test = open(Path(data_path, "test.txt"), "w")

    files = list(glob.iglob(os.path.join(images_path, "*.jpg")))
    random.seed(777)
    random.shuffle(files)
    test_array = random.sample(range(len(files)), k=int(perc_test * len(files)))

    # Populate train.txt and test.txt
    counter = 0
    for pathAndFilename in files:
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if counter in test_array:
            file_test.write(pathAndFilename + "\n")
        else:
            if random.uniform(0, 1) >= perc_test:
                file_train.write(pathAndFilename + "\n")
            else:
                file_valid.write(pathAndFilename + "\n")
        counter = counter + 1
    print("Training and test set completed")


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
                print("Missing file", i)
                
    
                

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
                frame_tracker.track_objects(
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

    # See species frame distribution
    #print(full_rows.groupby("species_id").count())

    # Create output folder
    if not os.path.isdir(args.out_path):
        os.mkdir(Path(args.out_path))

    # Set up directory structure
    img_dir = Path(args.out_path, "images")
    label_dir = Path(args.out_path, "labels")

    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)

    os.mkdir(img_dir)

    if os.path.isdir(label_dir):
        shutil.rmtree(label_dir)

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

    print("Frames extracted successfully")

    # Clear images
    if len(full_rows) == 0:
        raise Exception("No frames found for the selected species. Please retry with a different configuration.")
    process_frames(args.out_path + "/images", size=tuple(args.img_size))

    # Create training/test sets
    split_frames(args.out_path, args.perc_test)


if __name__ == "__main__":
    main()
