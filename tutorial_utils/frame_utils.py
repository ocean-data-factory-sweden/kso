import os, sys, re
import argparse
import pims
import db_utils
import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm


def drawBoxes(df, movie_dir, out_path):
    df["movie_path"] = (
        movie_dir
        + "/"
        + df["filename"].apply(
            lambda x: os.path.basename(x.rsplit("_frame_")[0]) + ".mov"
        )
    )
    movie_dict = {i: pims.Video(i) for i in df["movie_path"].unique()}
    df["annotation"] = df[["x_position", "y_position", "width", "height"]].apply(
        lambda x: tuple([x[0], x[1], x[2], x[3]]), 1
    )
    df = df.drop(columns=["x_position", "y_position", "width", "height"])
    for name, group in tqdm(
        df.groupby(["movie_path", "frame_number", "species_id", "filename"])
    ):
        frame = movie_dict[name[0]][name[1]]
        boxes = [tuple(i[4:])[0] for i in group.values]
        for box in boxes:
            # Calculating end-point of bounding box based on starting point and w, h
            end_box = tuple([int(box[0] + box[2]), int(box[1] + box[3])])
            # changed color and width to make it visible
            cv.rectangle(frame, (int(box[0]), int(box[1])), end_box, (255, 0, 0), 1)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        cv.imwrite(out_path + "/" + os.path.basename(name[3]), frame)


def main():
    "Handles argument parsing and launches the correct function."
    parser = argparse.ArgumentParser()
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
        "-o",
        "--output_dir",
        type=str,
        help="the directory to save the frames",
        default=r"/database/frames/",
        required=True,
    )
    args = parser.parse_args()
    conn = db_utils.create_connection(args.db_path)
    df = pd.read_sql_query(
        "SELECT b.filename, b.frame_number, a.species_id, a.x_position, a.y_position, a.width, a.height FROM agg_annotations_frame AS a LEFT JOIN subjects AS b ON a.subject_id=b.id",
        conn,
    )
    drawBoxes(df, args.movie_dir, args.output_dir)
    print("Frames exported successfully")


if __name__ == "__main__":
    main()
