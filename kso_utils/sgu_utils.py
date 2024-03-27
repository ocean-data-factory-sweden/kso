# base imports
import os
import cv2
import pandas as pd
import logging
import imghdr
from tqdm import tqdm
from pathlib import Path
import splitfolders
import glob

tqdm.pandas()

# util imports
import kso_utils.project_utils as project_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def create_classification_dataset(
    data_path: str, out_path: str, test_size: float, seed: int = 1337
):
    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
        # Recursively add permissions to folders created
        for root, dirs, files in out_path.iterdir():
            root.chmod(0o777)

    splitfolders.ratio(
        data_path, output=out_path, seed=seed, ratio=(1 - test_size, 0, test_size)
    )
    logging.info(f"Training and test datasets saved at {out_path}")


def get_patch(row, out_path: str, pixels: int = 224, label_col: str = "sub_type"):
    try:
        img = cv2.imread(row.fpath)
    except Exception as e:
        logging.info(e)
        logging.info(f"No such image, {row.fpath}")
        return

    for ix, (pos_X, pos_Y) in enumerate(zip(row.pos_X, row.pos_Y)):
        # Use conversion between current XY position and actual pixel values
        coord = (pos_X / 15, pos_Y / 15)

        # Discard images where pos_X is negative
        if coord[0] < 0:
            logging.error(f"Negative X value in {Path(row.fpath).name}. Skipping...")
            return

        # Discard images where label is NA
        if not isinstance(row[label_col][ix], str):
            logging.error(f"Invalid label in {Path(row.fpath).name}. Skipping...")
            return

        cropped_ys, cropped_ye = int(coord[1] - pixels / 2), int(coord[1] + pixels / 2)
        cropped_xs, cropped_xe = int(coord[0] - pixels / 2), int(coord[0] + pixels / 2)

        if cropped_ys < 0:
            y_diff = -1 * cropped_ys
            cropped_ys += y_diff
            cropped_ye += y_diff

        if cropped_xs < 0:
            x_diff = -1 * cropped_xs
            cropped_xs += x_diff
            cropped_xe += x_diff

        # Specify cropped patch size
        cropped_image = img[
            cropped_ys:cropped_ye,
            cropped_xs:cropped_xe,
        ]

        # Get label
        label = row[label_col][ix]
        label_path = Path(out_path, label)
        label_path.mkdir(parents=True, exist_ok=True)

        # Recursively add permissions to folders created
        for root, dirs, files in label_path.iterdir():
            root.chmod(0o777)

        # Write patches to a folder
        patch_filename = f"{Path(row.fpath).stem}_patch_{row.point[ix]}.jpg"
        cv2.imwrite(str(label_path / patch_filename), cropped_image)


def get_patches(
    root_path: str,
    meta_filename: str,
    pixels: int,
    out_path: str,
    label_col: str = "sub_type",
):
    """
    The function takes as input a folder with images, a metadata-sheet, a height/width in pixels, and an
    output path, and gives as output square patches from all points specified in the sheet, with size
    equal to pixels:pixels

    :param root_path: the path to the folder containing the images and the metadata-sheet
    :type root_path: str
    :param meta_filename: the name of the metadata file, which is an excel file
    :type meta_filename: str
    :param pixels: the size of the square patch you want to extract (usually 224)
    :type pixels: int
    :param out_path: the path to the folder where you want to save the patches
    :type out_path: str
    :return: nothing, but it creates a folder with patches from the images in the root_path folder.
    """

    path_to_folder = Path(root_path)

    # get list of image files
    orig_names = [
        f
        for f in os.listdir(path_to_folder)
        if imghdr.what(Path(path_to_folder, f)) is not None
    ]

    image_list = [f.lower() for f in orig_names]
    img_2_orig = {k: v for k, v in zip(image_list, orig_names)}

    def find_image(path):
        try:
            path = img_2_orig[path]
        except KeyError:
            path = None
        return path

    # df: dataframe based on SGU metadata-sheet
    df = pd.read_excel(Path(path_to_folder, meta_filename), engine="openpyxl")
    df["fpath"] = df["image_name"].apply(
        lambda x: str(Path(path_to_folder.as_posix(), find_image(x))), 1
    )

    if label_col == "sub_type":
        # Assumption: Si not found in metadata, and UkSu refers to unknown substrate
        df = df[~df["sub_type"].isin(["UkSu"])]
        # Assumption: St, LaSt combined, Bo and LaBo combined
        df["sub_type"] = df["sub_type"].replace(["LaSt"], "St")
        df["sub_type"] = df["sub_type"].replace(["LaBo"], "Bo")

    if label_col == "bio_type":
        # Assumption: UdOrg removed
        df = df[~df["sub_type"].isin(["UdOrg"])]

    df = df.groupby(["fpath"], as_index=False)[
        ["pos_X", "pos_Y", label_col, "point"]
    ].agg(lambda x: list(x))

    # create patch folder
    if not os.path.exists(f"{out_path}"):
        Path(out_path).mkdir(parents=True, exist_ok=True)
        # Recursively add permissions to folders created
        [os.chmod(root, 0o777) for root, dirs, files in os.walk(out_path)]
    df.progress_apply(lambda x: get_patch(x, out_path, pixels, label_col), axis=1)
    logging.info(
        f"Patch creation completed successfully. Total patches: {len(glob.glob(out_path + '/**/*.jpg', recursive=True))}"
    )


def process_sgu_photos_csv(project: project_utils.Project):
    """
    It takes the local csv files with photos and surveys information and returns a dataframe with the
    photos information

    :param project: The project object
    :return: A dataframe with the photos information
    """
    # Add survey info to the photos information
    photos_df = project.local_photos_csv.merge(
        project.local_surveys_csv.rename(columns={"ID": "SurveyID"}),
        on="SurveyID",
        how="left",
    )

    # TO DO Include server's path to the photo files
    photos_df["fpath"] = photos_df["filename"]

    # Rename to match schema format
    photos_df = photos_df.rename(
        columns={
            "SiteID": "site_id",  # site id for the db
            "SurveyDate": "created_on",
        }
    )

    return photos_df
