# -*- coding: utf-8 -*-
# base imports
import logging
import sqlite3
import ftfy
import pandas as pd
from pathlib import Path

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def fix_text_encoding(string: str):
    """This function corrects for text encoding errors, which occur when there is
    for example an ä,å,ö present."""
    return ftfy.fix_text(string)


def fix_text_encoding_folder(folder_name):
    """
    This function corrects for text encoding errors, which occur when there is
    for example an ä,å,ö present. It runs through all the file and folder names
    of the directory you give it. It uses the package ftfy, which recognizes
    which encoding the text has based on the text itself, and it encodes/decodes
    it to utf8.
    This function was tested on a Linux and Windows device with package version
    6.1.1. With package version 5.8 it did not work.

    This function can replace the unswedify and reswedify functions from
    koster_utils, but this is not implemented yet.
    """
    for item in Path(folder_name).iterdir():
        if item.is_dir():
            for sub_item in item.iterdir():
                old_path = sub_item
                new_path = sub_item.parent / ftfy.fix_text(sub_item.name)
                old_path.rename(new_path)


def get_koster_col_names(table_name: str):
    """Return a dictionary with the project-specific column names of a csv of interest
    This function helps matching the schema format without modifying the column names of the original csv.

    :param table_name: a string of the name of the schema table of interest
    :return: a dictionary with the names of the columns
    """

    if table_name == "movies":
        # Save the column names of interest in a dict
        col_names_dic = {
            "SamplingStart": "sampling_start",
            "SamplingEnd": "sampling_end",
        }
    else:
        # Create empty data frame as there are no project-specific
        # columns for this table
        col_names_dic = {}

    return col_names_dic


# Function to process subjects uploaded automatically
def auto_subjects(subjects_df: pd.DataFrame, auto_date: str):
    """
    The function `auto_subjects` selects and extracts metadata from subjects that were automatically
    uploaded after a specified date.

    :param subjects_df: `subjects_df` is a pandas DataFrame containing information about subjects, such
    as their IDs, project IDs, and creation dates
    :type subjects_df: pd.DataFrame
    :param auto_date: auto_date is a string parameter that represents the date from which the function
    should select automatically uploaded frames. The function will only select frames that were created
    after this date
    :type auto_date: str
    :return: The function `auto_subjects` returns a pandas DataFrame containing metadata information
    extracted from subjects that were automatically uploaded after a specified date.
    """
    # Select automatically uploaded frames
    auto_subjects_df = subjects_df[(subjects_df["created_at"] > auto_date)]

    from kso_utils.zooniverse_utils import extract_metadata

    # Extract metadata from automatically uploaded subjects
    auto_subjects_df, auto_subjects_meta = extract_metadata(auto_subjects_df)

    # Combine metadata info with the subjects df
    auto_subjects_df = pd.concat([auto_subjects_df, auto_subjects_meta], axis=1)

    return auto_subjects_df


# Function to process subjects uploaded manually
def manual_subjects(subjects_df: pd.DataFrame, manual_date: str, auto_date: str):
    """
    The function extracts metadata from manually uploaded clips and processes it to combine with the
    subjects dataframe.

    :param subjects_df: A pandas DataFrame containing information about subjects, including metadata and
    creation dates
    :type subjects_df: pd.DataFrame
    :param manual_date: The date from which to start selecting clips uploaded manually
    :type manual_date: str
    :param auto_date: It seems like the parameter auto_date is missing from the code snippet. Can you
    provide more information on what this parameter represents?
    :type auto_date: str
    :return: a pandas DataFrame containing information about clips that were uploaded manually, along
    with their metadata and processed information.
    """
    from kso_utils.zooniverse_utils import extract_metadata

    # Select clips uploaded manually
    man_clips_df = (
        subjects_df[
            (subjects_df["metadata"].str.contains(".mp4"))
            & (subjects_df["created_at"].between(manual_date, auto_date))
        ]
        .reset_index(drop=True)
        .reset_index()
    )

    # Specify the type of subject
    man_clips_df["subject_type"] = "clip"

    # Extract metadata from manually uploaded clips
    man_clips_df, man_clips_meta = extract_metadata(man_clips_df)

    if len(man_clips_meta) > 0:
        # Process the metadata of manually uploaded clips
        man_clips_meta = process_manual_clips(man_clips_meta)

        # Combine metadata info with the subjects df
        man_clips_df = pd.concat([man_clips_df, man_clips_meta], axis=1)

    return man_clips_df


# Function to process the metadata of clips that were uploaded manually
def process_manual_clips(meta_df: pd.DataFrame):
    """
    The function processes metadata of manual clips by extracting relevant information such as clip
    start and end times and the filename of the original movie.

    :param meta_df: The input parameter `meta_df` is a Pandas DataFrame containing metadata information
    about video clips. It is assumed that the DataFrame has a column named "filename" which contains the
    name of the video clip file in the format "original_movie_name_start_time.mp4". The function
    processes this information to extract
    :type meta_df: pd.DataFrame
    :return: a pandas DataFrame containing the filename of the clips, the filename of the original
    movie, the starting time of the clips in relation to the original movie, and the end time of the
    clips in relation to the original movie.
    """
    # Select the filename of the clips and remove extension type
    clip_filenames = meta_df["filename"].str.replace(".mp4", "", regex=True)

    # Get the starting time of clips in relation to the original movie
    # split the filename and select the last section
    meta_df["clip_start_time"] = clip_filenames.str.rsplit("_", 1).str[-1]

    # Extract the filename of the original movie
    meta_df["movie_filename"] = meta_df.apply(
        lambda x: x["filename"]
        .replace("_" + x["clip_start_time"], "")
        .replace(".mp4", ".mov"),
        axis=1,
    )

    # Get the end time of clips in relation to the original movie
    meta_df["clip_start_time"] = pd.to_numeric(
        meta_df["clip_start_time"], downcast="signed"
    )
    meta_df["clip_end_time"] = meta_df["clip_start_time"] + 10

    # Select only relevant columns
    meta_df = meta_df[
        ["filename", "movie_filename", "clip_start_time", "clip_end_time"]
    ]

    return meta_df


def get_movies_id(df: pd.DataFrame, conn: sqlite3.Connection):
    """
    This function retrieves movie IDs based on movie filenames from a database and merges them with a
    given DataFrame.

    :param df: A pandas DataFrame containing information about movie filenames and clip subjects
    :type df: pd.DataFrame
    :param conn: SQL connection object
    :return: a pandas DataFrame with the movie_ids added to the input DataFrame based on matching movie
    filenames with the movies table in a SQLite database. The function drops the movie_filename column
    before returning the DataFrame.
    """
    from kso_utils.db_utils import get_df_from_db_table

    # Query id and filenames from the movies table
    movies_df = get_df_from_db_table(conn, "movies")[["id", "filename"]]
    movies_df = movies_df.rename(
        columns={"id": "movie_id", "filename": "movie_filename"}
    )

    # Check all the movies have a unique ID
    df_unique = df.movie_filename.unique()
    movies_df_unique = movies_df.movie_filename.unique()
    diff_filenames = set(df_unique).difference(movies_df_unique)

    if diff_filenames:
        raise ValueError(
            f"There are clip subjects that don't have movie_id. The movie filenames are {diff_filenames}"
        )

    # Reference the manually uploaded subjects with the movies table
    df = pd.merge(df, movies_df, how="left", on="movie_filename")

    # Drop the movie_filename column
    df = df.drop(columns=["movie_filename"])

    return df


def process_koster_subjects(subjects: pd.DataFrame, conn: sqlite3.Connection):
    """
    This function takes in a dataframe of subjects and a path to the database and returns a dataframe of
    subjects with updated metadata

    :param subjects: the dataframe of subjects from the database
    :type subjects: pd.DataFrame
    :param conn: SQL connection object
    :return: A dataframe with all the subjects that have been uploaded to the database.
    """

    # Set the date when the metadata of subjects uploaded matches/doesn't match schema.py requirements
    # Specify the date when the metadata of subjects uploaded matches schema.py
    auto_date = "2020-05-29 00:00:00 UTC"

    # Specify the starting date when clips were manually uploaded
    manual_date = "2019-11-17 00:00:00 UTC"

    # Select automatically uploaded subjects
    auto_subjects_df = auto_subjects(subjects, auto_date=auto_date)

    # Select manually uploaded subjects
    manual_subjects_df = manual_subjects(
        subjects, manual_date=manual_date, auto_date=auto_date
    )

    if len(manual_subjects_df) > 0:
        # Include movie_ids to the metadata
        manual_subjects_df = get_movies_id(manual_subjects_df, conn=conn)

        # Combine all uploaded subjects
        subjects = pd.merge(manual_subjects_df, auto_subjects_df, how="outer")

    else:
        subjects = auto_subjects_df

    return subjects


def process_clips_koster(annotations, row_class_id: str, rows_list: list):
    """
    For each annotation, if the task is T4, then for each species annotated, flatten the relevant
    answers and save the species of choice, class and subject id

    :param annotations: the list of annotations for a given classification
    :param row_class_id: the classification id of the row
    :param rows_list: list
    :type rows_list: list
    :return: A list of dictionaries, each dictionary containing the classification id, the label, the first time seen and how many individuals were seen.
    """

    nothing_values = [
        "NOANIMALSPRESENT",
        "ICANTRECOGNISEANYTHING",
        "ISEENOTHING",
        "NOTHINGHERE",
    ]

    for ann_i in annotations:
        if ann_i["task"] == "T4":
            # Select each species annotated and flatten the relevant answers
            for value_i in ann_i["value"]:
                choice_i = {}
                # If choice = 'nothing here', set follow-up answers to blank
                if value_i["choice"] in nothing_values:
                    f_time = ""
                    inds = ""
                # If choice = species, flatten follow-up answers
                else:
                    answers = value_i["answers"]
                    f_time, inds = None, None
                    for k in answers.keys():
                        if "FIRSTTIME" in k:
                            f_time = answers[k].replace("S", "")
                        if "INDIVIDUAL" in k:
                            inds = answers[k]
                        elif "FIRSTTIME" not in k and "INDIVIDUAL" not in k:
                            f_time, inds = None, None

                # Save the species of choice, class and subject id
                choice_i.update(
                    {
                        "classification_id": row_class_id,
                        "label": value_i["choice"],
                        "first_seen": f_time,
                        "how_many": inds,
                    }
                )

                rows_list.append(choice_i)

    return rows_list


def process_koster_movies_csv(movies_df: pd.DataFrame):
    """
    It takes a dataframe of movies and returns a dataframe of movies with the following changes:

    - The filename is standardized
    - The filename is unswedified
    - The filename is renamed to fpath
    - The SamplingStart and SamplingEnd columns are renamed to sampling_start and sampling_end

    :param movies_df: the dataframe containing the movies
    :return: A dataframe with the columns:
        - filename
        - fpath
        - sampling_start
        - sampling_end
    """
    # Standarise the filename
    movies_df["filename"] = movies_df["filename"].str.normalize("NFD")

    # Ensure the filename has standard characters
    movies_df["filename"] = movies_df["filename"].apply(lambda x: fix_text_encoding(x))

    # TO DO Include server's path to the movie files
    movies_df["fpath"] = (
        movies_df["filename"].replace(".MP4", ".mp4").replace(".mov", ".mp4")
    )

    # Rename relevant fields
    movies_df = movies_df.rename(
        columns={
            "SamplingStart": "sampling_start",
            "SamplingEnd": "sampling_end",
        }
    )

    return movies_df
