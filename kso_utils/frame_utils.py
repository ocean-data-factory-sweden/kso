# base imports
import logging
import pims
import cv2
from sklearn.cluster import DBSCAN
from collections import Counter
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image as PILImage, ImageDraw

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def drawBoxes(df: pd.DataFrame, movie_dir: str, out_path: str):
    """
    For each unique movie, create a dictionary of movie paths and their corresponding pims.Video
    objects. Then, for each unique movie, frame number, species id, and filename, get the corresponding
    frame, get the bounding boxes for that frame, and draw the bounding boxes on the frame. Then, write
    the frame to the output directory

    :param df: the dataframe containing the bounding box coordinates
    :param movie_dir: The directory where the movies are stored
    :param out_path: The path to the directory where you want to save the images with the bounding boxes
           drawn on them
    :return:
    """
    df["movie_path"] = df["filename"].apply(
        lambda x: str(
            (Path(movie_dir) / Path(x).name.rsplit("_frame_")[0]).with_suffix(".mp4")
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
            cv2.rectangle(frame, (int(box[0]), int(box[1])), end_box, (255, 0, 0), 1)
        out_dir = Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Recursively add permissions to folders created
        for root, dirs, files in out_dir.iterdir():
            Path(root).chmod(0o777)
        cv2.imwrite(out_dir / Path(name[3]).name, frame)


def bb_iou(boxA, boxB):
    """
    The function takes two bounding boxes, computes the area of intersection, and divides it by the area
    of the union of the two boxes

    :param boxA: The first bounding box
    :param boxB: The ground truth box
    :return: The IOU value
    """

    # Compute edges
    temp_boxA = boxA.copy()
    temp_boxB = boxB.copy()
    temp_boxA[2], temp_boxA[3] = (
        temp_boxA[0] + temp_boxA[2],
        temp_boxA[1] + temp_boxA[3],
    )
    temp_boxB[2], temp_boxB[3] = (
        temp_boxB[0] + temp_boxB[2],
        temp_boxB[1] + temp_boxB[3],
    )

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(temp_boxA[0], temp_boxB[0])
    yA = max(temp_boxA[1], temp_boxB[1])
    xB = min(temp_boxA[2], temp_boxB[2])
    yB = min(temp_boxA[3], temp_boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 1
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((temp_boxA[2] - temp_boxA[0]) * (temp_boxA[3] - temp_boxA[1]))
    boxBArea = abs((temp_boxB[2] - temp_boxB[0]) * (temp_boxB[3] - temp_boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return 1 - iou


def filter_bboxes(
    total_users: int, users: list, bboxes: list, obj: float, eps: float, iua: float
):
    """
    If at least half of the users who saw this frame decided that there was an object, then we cluster
    the bounding boxes based on the IoU criterion. If at least 80% of users agree on the annotation,
    then we accept the cluster assignment

    :param total_users: total number of users who saw this frame
    :param users: list of user ids
    :param bboxes: list of bounding boxes
    :param obj: the minimum fraction of users who must have seen an object in order for it to be considered
    :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood
    :param iua: the minimum percentage of users who must agree on a bounding box for it to be accepted
    """

    # If at least half of those who saw this frame decided that there was an object
    user_count = pd.Series(users).nunique()
    if user_count / total_users >= obj:
        # Get clusters of annotation boxes based on iou criterion
        cluster_ids = DBSCAN(min_samples=1, metric=bb_iou, eps=eps).fit_predict(bboxes)
        # Count the number of users within each cluster
        counter_dict = Counter(cluster_ids)
        # Accept a cluster assignment if at least 80% of users agree on annotation
        passing_ids = [k for k, v in counter_dict.items() if v / user_count >= iua]

        indices = np.isin(cluster_ids, passing_ids)

        final_boxes = []
        for i in passing_ids:
            # Compute median over all accepted bounding boxes
            boxes = np.median(np.array(bboxes)[np.where(cluster_ids == i)], axis=0)
            final_boxes.append(boxes)

        return indices, final_boxes

    else:
        return [], bboxes


def draw_annotations_in_frame(im: PILImage.Image, class_df_subject: pd.DataFrame):
    """
    > The function takes an image and a dataframe of annotations and returns the image with the
    annotations drawn on it

    :param im: the image object of type PILImage
    :param class_df_subject: a dataframe containing the annotations for a single subject
    :return: The image with the annotations
    """
    # Calculate image size
    dw, dh = im._size

    # Draw rectangles of each annotation
    img1 = ImageDraw.Draw(im)

    # Merge annotation info into a tuple
    class_df_subject["vals"] = class_df_subject[["x", "y", "w", "h"]].values.tolist()

    for index, row in class_df_subject.iterrows():
        # Specify the vals object
        vals = row.vals

        # Adjust annotantions to image size
        vals_adjusted = tuple(
            [
                int(vals[0]),
                int(vals[1]),
                int((vals[0] + vals[2])),
                int((vals[1] + vals[3])),
            ]
        )

        # Draw annotation
        img1.rectangle(vals_adjusted, width=2)

    return im
