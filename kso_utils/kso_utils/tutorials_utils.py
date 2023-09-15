# base imports
import os
import time
import json
import math
import shutil
import pandas as pd
import numpy as np
import logging
import subprocess
import datetime
import random
import imagesize
import requests
import multiprocessing
import ffmpeg as ffmpeg_python
from base64 import b64encode
from io import BytesIO
from urllib.parse import urlparse
from csv_diff import compare
from pathlib import Path
from PIL import Image as PILImage, ImageDraw

# widget imports
from tqdm import tqdm
from jupyter_bbox_widget import BBoxWidget
from IPython.display import HTML, display, clear_output
import ipywidgets as widgets

# Util imports
from kso_utils.project_utils import Project
import kso_utils.movie_utils as movie_utils

# server imports
from kso_utils.server_utils import ServerType

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

############ CSV/iPysheet FUNCTIONS ################


def log_meta_changes(
    project: Project,
    meta_key: str,
    new_sheet_df: pd.DataFrame,
    csv_paths: dict,
):
    """Records changes to csv files in log file (json format)"""

    diff = {
        "timestamp": int(time.time()),
        "change_info": compare(
            {
                int(k): v
                for k, v in pd.read_csv(csv_paths[meta_key]).to_dict("index").items()
            },
            {int(k): v for k, v in new_sheet_df.to_dict("index").items()},
        ),
    }

    if len(diff) == 0:
        logging.info("No changes were logged")
        return

    else:
        try:
            with open(Path(project.csv_folder, "change_log.json"), "r+") as f:
                try:
                    existing_data = json.load(f)
                except json.decoder.JSONDecodeError:
                    existing_data = []
                existing_data.append(diff)
                f.seek(0)
                json.dump(existing_data, f)
        except FileNotFoundError:
            with open(Path(project.csv_folder, "change_log.json"), "w") as f:
                json.dump([diff], f)
        logging.info(
            f"Changelog updated at: {Path(project.csv_folder, 'change_log.json')}"
        )
        return


def process_source(source):
    """
    If the source is a string, write the string to a file and return the file name. If the source is a
    list, return the list. If the source is neither, return None

    :param source: The source of the data. This can be a URL, a file, or a list of URLs or files
    :return: the value of the source variable.
    """
    try:
        source.value
        if source.value is None:
            raise AttributeError("Value is None")
        return write_urls_to_file(source.value)
    except AttributeError:
        try:
            source.selected
            return source.selected
        except AttributeError:
            return None


def write_urls_to_file(movie_list: list, filepath: str = "/tmp/temp.txt"):
    """
    > This function takes a list of movie urls and writes them to a file
    so that they can be passed to the detect method of the ML models

    :param movie_list: list
    :type movie_list: list
    :param filepath: The path to the file to write the urls to, defaults to /tmp/temp.txt
    :type filepath: str (optional)
    :return: The filepath of the file that was written to.
    """
    try:
        iter(movie_list)
    except TypeError:
        logging.error(
            "No source movies found in selected path or path is empty. Please fix the previous selection"
        )
        return
    with open(filepath, "w") as fp:
        fp.write("\n".join(movie_list))
    return filepath


def get_project_info(projects_csv: str, project_name: str, info_interest: str):
    """
    > This function takes in a csv file of project information, a project name, and a column of interest
    from the csv file, and returns the value of the column of interest for the project name

    :param projects_csv: the path to the csv file containing the list of projects
    :param project_name: The name of the project you want to get the info for
    :param info_interest: the column name of the information you want to get from the project info
    :return: The project info
    """

    # Read the latest list of projects
    projects_df = pd.read_csv(projects_csv)

    # Get the info_interest from the project info
    project_info = projects_df[projects_df["Project_name"] == project_name][
        info_interest
    ].unique()[0]

    return project_info


# Function to check if an url is valid or not
def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# Function to extract the videos
def extract_example_clips(
    output_clip_path: str, start_time_i: int, clip_length: int, movie_path: str
):
    """
    > Extracts a clip from a movie file, and saves it to a new file

    :param output_clip_path: The path to the output clip
    :param start_time_i: The start time of the clip in seconds
    :param clip_length: The length of the clip in seconds
    :param movie_path: the path to the movie file
    """

    # Extract the clip
    if not os.path.exists(output_clip_path):
        subprocess.call(
            [
                "ffmpeg",
                "-ss",
                str(start_time_i),
                "-t",
                str(clip_length),
                "-i",
                str(movie_path),
                "-c",
                "copy",
                "-an",  # removes the audio
                "-force_key_frames",
                "1",
                str(output_clip_path),
            ]
        )

        os.chmod(output_clip_path, 0o777)


def check_clip_size(clips_list: list):
    """
    > This function takes a list of file paths and returns a dataframe with the file path and size of
    each file. If the size is too large, we suggest compressing them as a first step.

    :param clips_list: list of file paths to the clips you want to check
    :type clips_list: list
    :return: A dataframe with the file path and size of each clip
    """

    # Get list of files with size
    if clips_list is None:
        logging.error("No clips found.")
        return None
    files_with_size = [
        (file_path, os.path.getsize(file_path) / float(1 << 20))
        for file_path in clips_list
    ]

    df = pd.DataFrame(files_with_size, columns=["File_path", "Size"])

    if df["Size"].ge(8).any():
        logging.info(
            "Clips are too large (over 8 MB) to be uploaded to Zooniverse. Compress them!"
        )
        return df
    else:
        logging.info(
            "Clips are a good size (below 8 MB). Ready to be uploaded to Zooniverse"
        )
        return df


def modify_clips(
    clip_i: str, modification_details: dict, output_clip_path: str, gpu_available: bool
):
    """
    > This function takes in a clip, a dictionary of modification details, and an output path, and then
    modifies the clip using the details provided

    :param clip_i: the path to the clip to be modified
    :param modification_details: a dictionary of the modifications to be made to the clip
    :param output_clip_path: The path to the output clip
    :param gpu_available: If you have a GPU, set this to True. If you don't, set it to False
    """
    if gpu_available:
        # Unnest the modification detail dict
        df = pd.json_normalize(modification_details, sep="_")
        # Commenting out b_v as it causes gpu runs to fail
        # b_v = df.filter(regex="bv$", axis=1).values[0][0] + "M"

        subprocess.call(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                clip_i,
                "-c:a",
                "copy",
                "-c:v",
                "h264_nvenc",
                # "-b:v",
                # b_v,
                output_clip_path,
            ]
        )

    else:
        # Set up input prompt
        init_prompt = f"ffmpeg_python.input('{clip_i}')"
        default_output_prompt = f".output('{output_clip_path}', crf=20, pix_fmt='yuv420p', vcodec='libx264')"
        full_prompt = init_prompt
        mod_prompt = ""

        # Set up modification
        for transform in modification_details.values():
            if "filter" in transform:
                mod_prompt += transform["filter"]
            else:
                # Unnest the modification detail dict
                df = pd.json_normalize(modification_details, sep="_")
                crf = df.filter(regex="crf$", axis=1).values[0][0]
                out_prompt = f".output('{output_clip_path}', crf={crf}, preset='veryfast', pix_fmt='yuv420p', vcodec='libx264')"

        if len(mod_prompt) > 0:
            full_prompt += mod_prompt
        if out_prompt:
            full_prompt += out_prompt
        else:
            full_prompt += default_output_prompt

        # Run the modification
        try:
            eval(full_prompt).run(capture_stdout=True, capture_stderr=True)
            os.chmod(output_clip_path, 0o777)
        except ffmpeg_python.Error as e:
            logging.info("stdout: {}", e.stdout.decode("utf8"))
            logging.info("stderr: {}", e.stderr.decode("utf8"))
            raise e

    logging.info(f"Clip {clip_i} modified successfully")


def review_clip_selection(clip_selection, movie_i: str, clip_modification):
    """
    > This function reviews the clips that will be created from the movie selected

    :param clip_selection: the object that contains the results of the clip selection
    :param movie_i: the movie you want to create clips from
    :param clip_modification: The modification that will be applied to the clips
    """
    start_trim = clip_selection.kwargs["clips_range"][0]
    end_trim = clip_selection.kwargs["clips_range"][1]

    # Review the clips that will be created
    logging.info(
        f"You are about to create {round(clip_selection.result)} clips from {movie_i}"
    )
    logging.info(
        f"starting at {datetime.timedelta(seconds=start_trim)} and ending at {datetime.timedelta(seconds=end_trim)}"
    )
    logging.info(f"The modification selected is {clip_modification}")


# Func to expand seconds
def expand_list(df: pd.DataFrame, list_column: str, new_column: str):
    """
    We take a dataframe with a column that contains lists, and we expand that column into a new
    dataframe with a new column that contains the items in the list

    :param df: the dataframe you want to expand
    :param list_column: the column that contains the list
    :param new_column: the name of the new column that will be created
    :return: A dataframe with the list column expanded into a new column.
    """
    lens_of_lists = df[list_column].apply(len)
    origin_rows = range(df.shape[0])
    destination_rows = np.repeat(origin_rows, lens_of_lists)
    non_list_cols = [idx for idx, col in enumerate(df.columns) if col != list_column]
    expanded_df = df.iloc[destination_rows, non_list_cols].copy()
    expanded_df[new_column] = [item for items in df[list_column] for item in items]
    expanded_df.reset_index(inplace=True, drop=True)
    return expanded_df


# Function to extract the videos
def extract_clips(
    movie_path: str,
    clip_length: int,
    upl_second_i: int,
    output_clip_path: str,
    modification_details: dict,
    gpu_available: bool,
):
    """
    This function takes in a movie path, a clip length, a starting second index, an output clip path, a
    dictionary of modification details, and a boolean indicating whether a GPU is available. It then
    extracts a clip from the movie, and applies the modifications specified in the dictionary.

    The function is written in such a way that it can be used to extract clips from a movie, and apply
    modifications to the clips.

    :param movie_path: The path to the movie file
    :param clip_length: The length of the clip in seconds
    :param upl_second_i: The second in the video to start the clip
    :param output_clip_path: The path to the output clip
    :param modification_details: a dictionary of dictionaries, where each dictionary contains the
           details of a modification to be made to the video. The keys of the dictionary are the names of the
           modifications, and the values are dictionaries containing the details of the modification.
    :param gpu_available: If you have a GPU, set this to True. If you don't, set it to False
    """
    if not modification_details and gpu_available:
        # Create clips without any modification
        subprocess.call(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-ss",
                str(upl_second_i),
                "-t",
                str(clip_length),
                "-i",
                movie_path,
                "-threads",
                "4",
                "-an",  # removes the audio
                "-c:a",
                "copy",
                "-c:v",
                "h264_nvenc",
                str(output_clip_path),
            ]
        )
        os.chmod(str(output_clip_path), 0o777)

    elif modification_details and gpu_available:
        # Unnest the modification detail dict
        df = pd.json_normalize(modification_details, sep="_")
        # Commenting out b_v as it causes gpu runs to fail
        # b_v = df.filter(regex="bv$", axis=1).values[0][0] + "M"

        subprocess.call(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-ss",
                str(upl_second_i),
                "-t",
                str(clip_length),
                "-i",
                movie_path,
                "-threads",
                "4",
                "-an",  # removes the audio
                "-c:a",
                "copy",
                "-c:v",
                "h264_nvenc",
                # "-b:v",
                # b_v,
                str(output_clip_path),
            ]
        )
        os.chmod(str(output_clip_path), 0o777)
    else:
        # Set up input prompt
        init_prompt = f"ffmpeg_python.input('{movie_path}')"
        full_prompt = init_prompt
        mod_prompt = ""
        output_prompt = ""
        def_output_prompt = f".output('{str(output_clip_path)}', ss={str(upl_second_i)}, t={str(clip_length)}, movflags='+faststart', crf=20, pix_fmt='yuv420p', vcodec='libx264')"

        # Set up modification
        for transform in modification_details.values():
            if "filter" in transform:
                mod_prompt += transform["filter"]

            else:
                # Unnest the modification detail dict
                df = pd.json_normalize(modification_details, sep="_")
                crf = df.filter(regex="crf$", axis=1).values[0][0]
                output_prompt = f".output('{str(output_clip_path)}', crf={crf}, ss={str(upl_second_i)}, t={str(clip_length)}, movflags='+faststart', preset='veryfast', pix_fmt='yuv420p', vcodec='libx264')"

        # Run the modification
        try:
            if len(mod_prompt) > 0:
                full_prompt += mod_prompt
            if len(output_prompt) > 0:
                full_prompt += output_prompt
            else:
                full_prompt += def_output_prompt
            eval(full_prompt).run(capture_stdout=True, capture_stderr=True)
            os.chmod(str(output_clip_path), 0o777)
        except ffmpeg_python.Error as e:
            logging.info("stdout: {}", e.stdout.decode("utf8"))
            logging.info("stderr: {}", e.stderr.decode("utf8"))
            raise e

        logging.info("Clips extracted successfully")


def check_frame_size(frame_paths: list):
    """
    It takes a list of file paths, gets the size of each file, and returns a dataframe with the file
    path and size of each file

    :param frame_paths: a list of paths to the frames you want to check
    :return: A dataframe with the file path and size of each frame.
    """

    # Get list of files with size
    files_with_size = [
        (file_path, os.stat(file_path).st_size) for file_path in frame_paths
    ]

    df = pd.DataFrame(files_with_size, columns=["File_path", "Size"])

    # Change bytes to MB
    df["Size"] = df["Size"] / 1000000

    if df["Size"].ge(1).any():
        logging.info(
            "Frames are too large (over 1 MB) to be uploaded to Zooniverse. Compress them!"
        )
        return df
    else:
        logging.info(
            "Frames are a good size (below 1 MB). Ready to be uploaded to Zooniverse"
        )
        return df


# Function to compare original to modified frames
def compare_frames(df):
    if not isinstance(df, pd.DataFrame):
        df = df.df

    # Save the paths of the clips
    original_frame_paths = df["frame_path"].unique()

    # Add "no movie" option to prevent conflicts
    original_frame_paths = np.append(original_frame_paths, "No frame")

    clip_path_widget = widgets.Dropdown(
        options=tuple(np.sort(original_frame_paths)),
        description="Select original frame:",
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
            if change["new"] == "No frame":
                logging.info("It is OK to modify the frames again")
            else:
                a = view_frames(df, change["new"])
                display(a)

    clip_path_widget.observe(on_change, names="value")


# Display the frames using html
def view_frames(df: pd.DataFrame, frame_path: str):
    # Get path of the modified clip selected
    modified_frame_path = df[df["frame_path"] == frame_path].modif_frame_path.values[0]
    extension = os.path.splitext(frame_path)[1]

    img1 = open(frame_path, "rb").read()
    wi1 = widgets.Image(value=img1, format=extension, width=400, height=500)
    img2 = open(modified_frame_path, "rb").read()
    wi2 = widgets.Image(value=img2, format=extension, width=400, height=500)
    a = [wi1, wi2]
    wid = widgets.HBox(a)

    return wid


def launch_table(agg_class_df: pd.DataFrame, subject_type: str):
    """
    It takes in a dataframe of aggregated classifications and a subject type, and returns a dataframe
    with the columns "subject_ids", "label", "how_many", and "first_seen"

    :param agg_class_df: the dataframe that you want to launch
    :param subject_type: "clip" or "subject"
    """
    if subject_type == "clip":
        a = agg_class_df[["subject_ids", "label", "how_many", "first_seen"]]
    else:
        a = agg_class_df

    return a


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


def view_subject(subject_id: int, class_df: pd.DataFrame, subject_type: str):
    """
    It takes a subject id, a dataframe containing the annotations for that subject, and the type of
    subject (clip or frame) and returns an HTML object that can be displayed in a notebook

    :param subject_id: The subject ID of the subject you want to view
    :type subject_id: int
    :param class_df: The dataframe containing the annotations for the class of interest
    :type class_df: pd.DataFrame
    :param subject_type: The type of subject you want to view. This can be either "clip" or "frame"
    :type subject_type: str
    """
    if subject_id in class_df.subject_ids.tolist():
        # Select the subject of interest
        class_df_subject = class_df[class_df.subject_ids == subject_id].reset_index(
            drop=True
        )

        # Get the location of the subject
        subject_location = class_df_subject["https_location"].unique()[0]

    else:
        raise Exception("The reference data does not contain media for this subject.")

    if len(subject_location) == 0:
        raise Exception("Subject not found in provided annotations")

    # Get the HTML code to show the selected subject
    if subject_type == "clip":
        html_code = f"""
        <html>
        <div style="display: flex; justify-content: space-around">
        <div>
          <video width=500 controls>
          <source src={subject_location} type="video/mp4">
        </video>
        </div>
        <div>{class_df_subject[['label','first_seen','how_many']].value_counts().sort_values(ascending=False).to_frame().to_html()}</div>
        </div>
        </html>"""

    elif subject_type == "frame":
        # Read image
        response = requests.get(subject_location)
        im = PILImage.open(BytesIO(response.content))

        # if label is not empty draw rectangles
        if class_df_subject.label.unique()[0] != "empty":
            # Create a temporary image with the annotations drawn on it
            im = draw_annotations_in_frame(im, class_df_subject)

        # Remove previous temp image if exist
        try:
            with open("test.txt", "w") as temp_file:
                temp_file.write("Testing write access.")
            temp_image_path = "temp.jpg"
        except:
            # Specify volume allocated by SNIC
            snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"
            temp_image_path = f"{snic_path}/tmp_dir/temp.jpg"

        finally:
            # Remove temporary file
            if os.path.exists("test.txt"):
                os.remove("test.txt")

        # Remove temp image if it exists
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Save the new image
        im.save(temp_image_path)

        # Load image data (used to enable viewing in Colab)
        img = open(temp_image_path, "rb").read()
        data_url = "data:image/jpeg;base64," + b64encode(img).decode()

        html_code = f"""
        <html>
        <div style="display: flex; justify-content: space-around">
        <div>
          <img src={data_url} type="image/jpeg" width=500>
        </img>
        </div>
        <div>{class_df_subject[['label','colour']].value_counts().sort_values(ascending=False).to_frame().to_html()}</div>
        </div>
        </html>"""
    else:
        Exception("Subject type not supported.")
    return HTML(html_code)


def launch_viewer(class_df: pd.DataFrame, subject_type: str):
    """
    > This function takes a dataframe of classifications and a subject type (frame or video) and
    displays a dropdown menu of subjects of that type. When a subject is selected, it displays the
    subject and the classifications for that subject

    :param class_df: The dataframe containing the classifications
    :type class_df: pd.DataFrame
    :param subject_type: The type of subject you want to view. This can be either "frame" or "video"
    :type subject_type: str
    """

    # If subject is frame assign a color to each label
    if subject_type == "frame":
        # Create a list of unique labels
        list_labels = class_df.label.unique().tolist()

        # Generate a list of random colors for each label
        random_color_list = []
        for index, item in enumerate(list_labels):
            random_color_list = random_color_list + [
                "#" + "".join([random.choice("ABCDEF0123456789") for i in range(6)])
            ]

        # Add a column with the color for each label
        class_df["colour"] = class_df.apply(
            lambda row: random_color_list[list_labels.index(row.label)], axis=1
        )

    # Select the subject
    options = tuple(
        class_df[class_df["subject_type"] == subject_type]["subject_ids"]
        .apply(int)
        .apply(str)
        .unique()
    )
    subject_widget = widgets.Dropdown(
        options=options,
        description="Subject id:",
        ensure_option=True,
        disabled=False,
    )

    main_out = widgets.Output()
    display(subject_widget, main_out)

    # Display the subject and classifications on change
    def on_change(change):
        with main_out:
            a = view_subject(int(change["new"]), class_df, subject_type)
            clear_output()
            display(a)

    subject_widget.observe(on_change, names="value")


def explore_classifications_per_subject(class_df: pd.DataFrame, subject_type: str):
    """
    > This function takes a dataframe of processed classifications and a subject type (clip or frame) and displays
    the classifications for a given subject

    :param class_df: the dataframe of classifications
    :type class_df: pd.DataFrame
    :param subject_type: "clip" or "frame"
    """

    # Select the subject
    subject_widget = widgets.Dropdown(
        options=tuple(class_df.subject_ids.apply(int).apply(str).unique()),
        description="Subject id:",
        ensure_option=True,
        disabled=False,
    )

    main_out = widgets.Output()
    display(subject_widget, main_out)

    # Display the subject and classifications on change
    def on_change(change):
        with main_out:
            a = class_df[class_df.subject_ids == int(change["new"])]
            if subject_type == "clip":
                a = a[
                    [
                        "classification_id",
                        "user_name",
                        "label",
                        "how_many",
                        "first_seen",
                    ]
                ]
            else:
                a = a[
                    [
                        "x",
                        "y",
                        "w",
                        "h",
                        "label",
                        "https_location",
                        "subject_ids",
                        "frame_number",
                        "movie_id",
                    ]
                ]
            clear_output()
            display(a)

    subject_widget.observe(on_change, names="value")


def encode_image(filepath):
    """
    It takes a filepath to an image, opens the image, reads the bytes, encodes the bytes as base64, and
    returns the encoded string

    :param filepath: The path to the image file
    :return: the base64 encoding of the image.
    """
    with open(filepath, "rb") as f:
        image_bytes = f.read()
    encoded = str(b64encode(image_bytes), "utf-8")
    return "data:image/jpg;base64," + encoded


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

    if any([len(images), len(annotations)]) == 0:
        logging.error("No annotations to display")
        return None

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
            w_progress.value = 0

        # open new image in the widget

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

    # when Submit button is pressed we save current annotations
    # and then move on to the next file
    def on_submit():
        image_file = images[w_progress.value]
        width, height = imagesize.get(os.path.join(image_path, image_file))
        label_file = annotations[w_progress.value]
        # save annotations for current image
        with open(os.path.join(annot_path, label_file), "w") as f:
            f.write(
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

    w_bbox.on_skip(on_skip)
    w_bbox.on_submit(on_submit)

    return w_container


def choose_test_prop():
    """
    > The function `choose_test_prop()` creates a slider widget that allows the user to choose the
    proportion of the data to be used for testing
    :return: A widget object
    """

    w = widgets.FloatSlider(
        value=0.2,
        min=0.0,
        max=1.0,
        step=0.1,
        description="Test proportion:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )

    display(w)
    return w


def choose_eval_params():
    """
    It creates one slider for confidence threshold
    :return: the value of the slider.
    """

    z1 = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.1,
        description="Confidence threshold:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )

    display(z1)
    return z1


def choose_train_params(model_type: str):
    """
    It creates two sliders, one for batch size, one for epochs
    :return: the values of the sliders.
    """
    v = widgets.FloatLogSlider(
        value=1,
        base=2,
        min=0,  # max exponent of base
        max=10,  # min exponent of base
        step=1,  # exponent step
        description="Batch size:",
        readout=True,
        readout_format="d",
    )

    z = widgets.IntSlider(
        value=1,
        min=0,
        max=1000,
        step=10,
        description="Epochs:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    h = widgets.IntText(description="Height:", value=128)
    w = widgets.IntText(description="Width:", value=128)
    s = widgets.IntText(description="Image size:", value=128)

    def on_value_change(change):
        height = h.value
        width = w.value
        return [height, width]

    h.observe(on_value_change, names="value")
    w.observe(on_value_change, names="value")
    s.observe(on_value_change, names="value")

    if model_type == 1:
        box = widgets.HBox([v, z, h, w])
        display(box)
        return v, z, h, w
    elif model_type == 2:
        box = widgets.HBox([v, z, s])
        display(box)
        return v, z, s, None
    else:
        logging.warning("Model in experimental stage.")
        box = widgets.HBox([v, z])
        display(box)
        return v, z, None, None


def choose_experiment_name():
    """
    It creates a text box that allows you to enter a name for your experiment
    :return: The text box widget.
    """
    exp_name = widgets.Text(
        value="exp_name",
        placeholder="Choose an experiment name",
        description="Experiment name:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(exp_name)
    return exp_name


def choose_entity():
    """
    It creates a text box that allows you to enter your username or teamname of WandB
    :return: The text box widget.
    """
    entity = widgets.Text(
        value="koster",
        placeholder="Give your user or team name",
        description="User or Team name:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(entity)
    return entity


def choose_model_type():
    """
    It creates a dropdown box that allows you to choose a model type
    :return: The dropdown box widget.
    """
    model_type = widgets.Dropdown(
        value=None,
        description="Required model type:",
        options=[
            (
                "Object Detection (e.g. identifying individuals in an image using rectangles)",
                1,
            ),
            (
                "Image Classification (e.g. assign a class or label to an entire image)",
                2,
            ),
            (
                "Instance Segmentation (e.g. fit a suitable mask on top of identified objects)",
                3,
            ),
            ("Custom model (currently only Faster RCNN)", 4),
        ],
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        layout={"width": "max-content"},
        style={"description_width": "initial"},
    )
    display(model_type)
    return model_type


def choose_conf():
    w = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1.0,
        step=0.1,
        description="Confidence threshold:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(w)
    return w


def choose_text(name: str):
    text_widget = widgets.Text(
        description=f"Please enter a suitable {name} ",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(text_widget)
    return text_widget


class WidgetMaker(widgets.VBox):
    def __init__(self):
        """
        The function creates a widget that allows the user to select which workflows to run

        :param workflows_df: the dataframe of workflows
        """
        self.widget_count = widgets.IntText(
            description="Number of authors:",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )

        self.bool_widget_holder = widgets.HBox(
            layout=widgets.Layout(
                width="100%", display="inline-flex", flex_flow="row wrap"
            )
        )
        children = [
            self.widget_count,
            self.bool_widget_holder,
        ]
        self.widget_count.observe(self._add_bool_widgets, names=["value"])
        super().__init__(children=children)

    def _add_bool_widgets(self, widg):
        num_bools = widg["new"]
        new_widgets = []
        for _ in range(num_bools):
            new_widget = widgets.Text(
                display="flex",
                flex_flow="column",
                align_items="stretch",
                style={"description_width": "initial"},
            ), widgets.Text(
                display="flex",
                flex_flow="column",
                align_items="stretch",
                style={"description_width": "initial"},
            )
            new_widget[0].description = "Author Name: " + f" #{_}"
            new_widget[1].description = "Organisation: " + f" #{_}"
            new_widgets.extend(new_widget)
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}

    @property
    def author_dict(self):
        init_dict = {w.description: w.value for w in self.bool_widget_holder.children}
        names, organisations = [], []
        for i in range(0, len(init_dict), 2):
            names.append(list(init_dict.values())[i])
            organisations.append(list(init_dict.values())[i + 1])
        return {n: org for n, org in zip(names, organisations)}


def create_clips(
    available_movies_df: pd.DataFrame,
    movie_i: str,
    movie_path: str,
    clip_selection,
    project: Project,
    modification_details: dict,
    gpu_available: bool,
    pool_size: int = 4,
):
    """
    This function takes a movie and extracts clips from it

    :param available_movies_df: the dataframe with the movies that are available for the project
    :param movie_i: the name of the movie you want to extract clips from
    :param movie_path: the path to the movie you want to extract clips from
    :param clip_selection: a ClipSelection object
    :param project: the project object
    :param modification_details: a dictionary with the following keys:
    :param gpu_available: True or False, depending on whether you have a GPU available to use
    :param pool_size: the number of threads to use to extract the clips, defaults to 4 (optional)
    :return: A dataframe with the clip_path, clip_filename, clip_length, upl_seconds, and clip_modification_details
    """

    # Store the desired length of the clips
    clip_length = clip_selection.kwargs["clip_length"]

    # Store the starting seconds of the clips
    if isinstance(clip_selection.result, int):
        # Clipping video from a range of seconds (e.g. 10-180)
        # Store the starting and ending of the range
        start_trim = clip_selection.kwargs["clips_range"][0]
        end_trim = clip_selection.kwargs["clips_range"][1]

        # Create a list with the starting seconds of the clips
        list_clip_start = [
            list(
                range(
                    start_trim,
                    start_trim
                    + math.floor((end_trim - start_trim) / clip_length) * clip_length,
                    clip_length,
                )
            )
        ]

        if not clip_selection.result == len(list_clip_start[0]):
            if clip_selection.result < len(list_clip_start[0]):
                # Choose random starting points based on the number of samples (example clips, by default this is 3)
                list_clip_start = [
                    np.random.choice(
                        list_clip_start[0], size=clip_selection.result, replace=False
                    )
                ]
            else:
                logging.info(
                    "There was an issue estimating the starting seconds for the clips."
                )

    else:
        # Clipping specific sections of a video at random (e.g. starting at 10, 20, 180)
        # Store the starting seconds of the clips
        list_clip_start = [clip_selection.result["clip_start_time"]]

    # Filter the df for the movie of interest
    movie_i_df = available_movies_df[
        available_movies_df["filename"] == movie_i
    ].reset_index(drop=True)

    # Add the list of starting seconds to the df
    movie_i_df["list_clip_start"] = list_clip_start

    # Reshape the dataframe with the starting seconds for the new clips
    potential_start_df = expand_list(movie_i_df, "list_clip_start", "upl_seconds")

    # Add the length of the clips to df (to keep track of the length of each uploaded clip)
    potential_start_df["clip_length"] = clip_length

    # Specify the temp folder to host the clips
    movie_path_folder = Path(movie_path).parent
    clips_folder = str(Path(movie_path_folder, "tmp_dir", movie_i + "_zooniverseclips"))

    # Set the filename of the clips
    potential_start_df["clip_filename"] = (
        movie_i
        + "_clip_"
        + potential_start_df["upl_seconds"].astype(str)
        + "_"
        + str(clip_length)
        + ".mp4"
    )

    # Set the path of the clips
    potential_start_df["clip_path"] = potential_start_df["clip_filename"].apply(
        lambda x: str(Path(clips_folder, x)), 1
    )

    # Create the folder to store the videos if not exist
    if os.path.exists(clips_folder):
        shutil.rmtree(clips_folder)
    Path(clips_folder).mkdir(parents=True, exist_ok=True)
    # Recursively add permissions to folders created
    [os.chmod(root, 0o777) for root, dirs, files in os.walk(clips_folder)]

    logging.info("Extracting clips")

    # Read each movie and extract the clips
    for index, row in tqdm(
        potential_start_df.iterrows(), total=potential_start_df.shape[0]
    ):
        # Extract the videos and store them in the folder
        extract_clips(
            movie_path,
            clip_length,
            row["upl_seconds"],
            row["clip_path"],
            modification_details,
            gpu_available,
        )

    # Add information on the modification of the clips
    potential_start_df["clip_modification_details"] = str(modification_details)

    return potential_start_df


def create_modified_clips(
    project: Project,
    clips_list: list,
    movie_i: str,
    modification_details: dict,
    gpu_available: bool,
    pool_size: int = 4,
):
    """
    This function takes a list of clips, a movie name, a dictionary of modifications, a project, and a
    GPU availability flag, and returns a list of modified clips

    :param clips_list: a list of the paths to the clips you want to modify
    :param movie_i: the path to the movie you want to extract clips from
    :param modification_details: a dictionary with the modifications to be applied to the clips. The keys are the names of the modifications and the values are the parameters of the modifications
    :param project: the project object
    :param gpu_available: True if you have a GPU available, False if you don't
    :param pool_size: the number of parallel processes to run, defaults to 4 (optional)
    :return: The modified clips
    """

    # Specify the folder to host the modified clips
    mod_clip_folder = "modified_" + movie_i + "_clips"

    # Specify the temp folder to host the clips
    movie_path_folder = Path(movie_i).parent
    mod_clips_folder = str(Path(movie_path_folder, "tmp_dir", mod_clip_folder))

    # Remove existing modified clips
    if os.path.exists(mod_clips_folder):
        shutil.rmtree(mod_clips_folder)

    if len(modification_details.values()) > 0:
        # Create the folder to store the videos if not exist
        if not os.path.exists(mod_clips_folder):
            Path(mod_clips_folder).mkdir(parents=True, exist_ok=True)
            # Recursively add permissions to folders created
            [os.chmod(root, 0o777) for root, dirs, files in os.walk(mod_clips_folder)]

        # Specify the number of parallel items
        pool = multiprocessing.Pool(pool_size)

        # Create empty list to keep track of new clips
        modified_clips = []
        results = []
        # Create the information for each clip and extract it (printing a progress bar)
        for clip_i in clips_list:
            # Create the filename and path of the modified clip
            output_clip_name = "modified_" + os.path.basename(clip_i)
            output_clip_path = Path(mod_clips_folder, output_clip_name)

            # Add the path of the clip to the list
            modified_clips = modified_clips + [output_clip_path]

            # Modify the clips and store them in the folder
            results.append(
                pool.apply_async(
                    modify_clips,
                    (
                        clip_i,
                        modification_details,
                        output_clip_path,
                        gpu_available,
                    ),
                )
            )

        pool.close()
        pool.join()
        return modified_clips
    else:
        logging.info("No modification selected")


def format_to_gbif(
    project: Project,
    db_connection,
    df: pd.DataFrame,
    csv_paths: dict,
    classified_by: str,
    zoo_info_dict: dict = {},
):
    """
    > This function takes a df of biological observations classified by citizen scientists, biologists or ML algorithms and returns a df of species occurrences to publish in GBIF/OBIS.
    :param project: the project object
    :param db_connection: SQL connection object
    :param csv_paths: a dictionary with the paths of the csvs used to initiate the db
    :param df: the dataframe containing the aggregated classifications
    :param classified_by: the entity who classified the object of interest, either "citizen_scientists", "biologists" or "ml_algorithms"
    :param zoo_info_dict: dictionary with the workflow/subjects/classifications retrieved from Zooniverse project
    :return: a df of species occurrences to publish in GBIF/OBIS.
    """

    # If classifications have been created by citizen scientists
    if classified_by == "citizen_scientists":
        #### Retrieve species/labels information #####
        # Create a df with unique workflow ids and versions of interest
        work_df = (
            df[["workflow_id", "workflow_version"]].drop_duplicates().astype("int")
        )

        # Correct for some weird zooniverse version behaviour
        work_df["workflow_version"] = work_df["workflow_version"] - 1

        # Store df of all the common names and the labels into a list of df
        from kso_utils.zooniverse_utils import get_workflow_labels

        commonName_labels_list = [
            get_workflow_labels(zoo_info_dict["workflows"], x, y)
            for x, y in zip(work_df["workflow_id"], work_df["workflow_version"])
        ]

        # Concatenate the dfs and select only unique common names and the labels
        commonName_labels_df = pd.concat(commonName_labels_list).drop_duplicates()

        # Drop the clips classified as nothing here or other
        df = df[~df["label"].isin(["OTHER", "NOTHINGHERE"])]

        # Combine the labels with the commonNames of the classifications
        comb_df = pd.merge(df, commonName_labels_df, how="left", on="label")

        from kso_utils.db_utils import add_db_info_to_df

        # Query info about the species of interest
        comb_df = add_db_info_to_df(
            project, db_connection, csv_paths, comb_df, "species"
        )

        # Identify the second of the original movie when the species first appears
        comb_df["second_in_movie"] = comb_df["clip_start_time"] + comb_df["first_seen"]

        # Select the max count of each species on each movie
        comb_df = comb_df.sort_values("how_many").drop_duplicates(
            ["movie_id", "commonName"], keep="last"
        )

        # Rename columns to match Darwin Data Core Standards
        comb_df = comb_df.rename(
            columns={
                "created_on": "eventDate",
                "how_many": "individualCount",
                "commonName": "vernacularName",
            }
        )

        # Create relevant columns for GBIF
        comb_df["occurrenceID"] = (
            project.Project_name
            + "_"
            + comb_df["siteName"]
            + "_"
            + comb_df["eventDate"].astype(str)
            + "_"
            + comb_df["second_in_movie"].astype(str)
            + "_"
            + comb_df["vernacularName"].astype(str)
        )

        # Set the basis of record as machine observation
        comb_df["basisOfRecord"] = "MachineObservation"

        # If coord uncertainity doesn't exist set to 30 metres
        comb_df["coordinateUncertaintyInMeters"] = comb_df.get(
            "coordinateUncertaintyInMeters", 30
        )

        # Select columns relevant for GBIF occurrences
        comb_df = comb_df[
            [
                "occurrenceID",
                "basisOfRecord",
                "vernacularName",
                "scientificName",
                "eventDate",
                "countryCode",
                "taxonRank",
                "kingdom",
                "decimalLatitude",
                "decimalLongitude",
                "geodeticDatum",
                "coordinateUncertaintyInMeters",
                "individualCount",
            ]
        ]

        return comb_df

    # If classifications have been created by biologists
    if classified_by == "biologists":
        logging.info("This sections is currently under development")

    # If classifications have been created by ml algorithms
    if classified_by == "ml_algorithms":
        logging.info("This sections is currently under development")
    else:
        raise ValueError(
            "Specify who classified the species of interest (citizen_scientists, biologists or ml_algorithms)"
        )
