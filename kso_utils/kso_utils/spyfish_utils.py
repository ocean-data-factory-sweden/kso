# base imports
import os
import sqlite3
import logging
import pandas as pd
import numpy as np


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def get_spyfish_col_names(table_name: str):
    """Return a dictionary with the project-specific column names of a csv of interest
    This function helps matching the schema format without modifying the column names of the original csv.

    :param table_name: a string of the name of the schema table of interest
    :return: a dictionary with the names of the columns
    """

    if table_name == "sites":
        # Save the column names of interest in a dict
        col_names_dic = {
            "SiteName": "SiteNameOld",  # Confusing but gets rid off duplication issues
            "SiteID": "siteName",
            "Latitude": "decimalLatitude",
            "Longitude": "decimalLongitude",
            "schema_site_id": "site_id",
        }

    elif table_name == "movies":
        # Save the column names of interest in a dict
        col_names_dic = {
            "EventDate": "created_on",
            "SamplingStart": "sampling_start",
            "SamplingEnd": "sampling_end",
            "RecordedBy": "author",
            "SiteID": "siteName",
            "LinkToVideoFile": "fpath",
        }

    else:
        # Create empty data frame as there are no project-specific
        # columns for this table
        col_names_dic = {}

    return col_names_dic


def process_spyfish_subjects(
    project,
    server_connection: dict,
    subjects: pd.DataFrame,
    db_connection: sqlite3.Connection,
):
    """
    It takes a dataframe of subjects and a path to the database, and returns a dataframe of subjects
    with the following columns:

    - filename, clip_start_time,clip_end_time,frame_number,subject_type,ScientificName,frame_exp_sp_id,movie_id

    The function does this by:

    - Merging "#Subject_type" and "Subject_type" columns to "subject_type"
    - Renaming columns to match the db format
    - Calculating the clip_end_time
    - Matching 'ScientificName' to species id and save as column "frame_exp_sp_id"
    - Matching site code to name from movies sql and get movie_id to save it as "movie_id"

    :param project: the project object
    :param server_connection: A dictionary with the client and sftp_client
    :param subjects: the dataframe of subjects to be processed
    :param db_connection: SQL connection object
    :return: A dataframe with the columns:
        - filename, clip_start_time,clip_end_time,frame_number,subject_type,ScientificName,frame_exp_sp_id,movie_id
    """
    # Fix weird bug where Subject_type and subject_type are used for some clips
    if "subject_type" in subjects.columns and "Subject_type" in subjects.columns:
        subjects["subject_type"] = subjects["subject_type"].fillna(
            subjects["Subject_type"]
        )
        subjects = subjects.drop(columns=["Subject_type"])

    # Fix weird bug where #Subject_type and subject_type are used for some clips
    if "subject_type" in subjects.columns and "#Subject_type" in subjects.columns:
        subjects["subject_type"] = subjects["subject_type"].fillna(
            subjects["#Subject_type"]
        )
        subjects = subjects.drop(columns=["#Subject_type"])

    # Rename common non-standard column names
    rename_cols = {
        "#VideoFilename": "filename",
        "#Subject_type": "subject_type",
        "Subject_type": "subject_type",
    }

    if "movie_id" in subjects.columns:
        # Drop movie_id because it's not a reliable way to trace back movies,
        # we will use VideoFilename/filename instead
        subjects = subjects.drop(columns="movie_id")

    # Rename columns to match the db format
    subjects = subjects.rename(columns=rename_cols)

    ##### Match site code to name from movies sql and get movie_id to save it as "movie_id"
    # Manually update the name of the original movies that have
    # been renamed after the clips were uploaded
    # Download the filename lookup csv file
    server_csv = project.key + "/update_of_movie_filenames_24_02_2023.csv"
    local_i_csv = "update_of_movie_filenames_24_02_2023.csv"

    if not os.path.exists(local_i_csv):
        from kso_utils.server_utils import download_object_from_s3

        download_object_from_s3(
            client=server_connection["client"],
            bucket=project.bucket,
            key=server_csv,
            filename=local_i_csv,
        )

    # Store the values to be replace 1first as df and then in a dictionary
    renames_df = pd.read_csv(local_i_csv)
    filenames_dict = dict(zip(renames_df["OLD"], renames_df["NEW"]))

    # Replace the old filenames
    subjects["filename"] = (
        subjects["filename"].map(filenames_dict).fillna(subjects["filename"])
    )

    # Fix issue with some filenames missing the .mp4
    subjects["filename"] = subjects["filename"].apply(
        lambda x: x + ".mp4" if not x.endswith(".mp4") else x
    )

    return subjects


def process_clips_spyfish(annotations, row_class_id, rows_list: list):
    """
    For each annotation, if the task is T0, then for each species annotated, flatten the relevant
    answers and save the species of choice, class and subject id.

    :param annotations: the list of annotations for a given subject
    :param row_class_id: the classification id
    :param rows_list: a list of dictionaries, each dictionary is a row in the output dataframe
    :return: A list of dictionaries, each dictionary containing the classification id, the label, the first seen time and the number of individuals.
    """

    for ann_i in annotations:
        if ann_i["task"] == "T0":
            # Select each species annotated and flatten the relevant answers
            for value_i in ann_i["value"]:
                choice_i = {}
                # If choice = 'nothing here', set follow-up answers to blank
                if value_i["choice"] == "NOTHINGHERE":
                    f_time = ""
                    inds = ""
                # If choice = species, flatten follow-up answers
                else:
                    answers = value_i["answers"]
                    for k in answers.keys():
                        if "EARLIESTPOINT" in k:
                            f_time = answers[k].replace("S", "")
                        if "HOWMANY" in k:
                            inds = answers[k]
                            # Deal with +20 fish options
                            if inds == "2030":
                                inds = "25"
                            if inds == "3040":
                                inds = "35"
                        elif "EARLIESTPOINT" not in k and "HOWMANY" not in k:
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


def spyfish_subject_metadata(df: pd.DataFrame, csv_paths: dict):
    """
    It takes a dataframe of subject metadata and returns a dataframe of subject metadata that is ready
    to be uploaded to Zooniverse

    :param df: the dataframe of all the detections
    :param csv_paths: paths to the csv from the project object
    :return: A dataframe with the columns of interest for uploading to Zooniverse.
    """

    # Get extra movie information
    movies_df = pd.read_csv(csv_paths["local_movies_csv"])

    df = df.merge(movies_df.drop(columns=["filename"]), how="left", on="movie_id")

    # Get extra survey information
    surveys_df = pd.read_csv(csv_paths["local_surveys_csv"])

    df = df.merge(surveys_df, how="left", on="SurveyID")

    # Get extra site information
    sites_df = pd.read_csv(csv_paths["local_sites_csv"])

    df = df.merge(
        sites_df.drop(columns=["LinkToMarineReserve"]), how="left", on="SiteID"
    )

    # Convert datetime to string to avoid JSON seriazible issues
    df["EventDate"] = df["EventDate"].astype(str)

    df = df.rename(
        columns={
            "LinkToMarineReserve": "!LinkToMarineReserve",
            "UID": "#UID",
            "scientificName": "ScientificName",
            "EventDate": "#EventDate",
            "first_seen_movie": "#TimeOfMaxSeconds",
            "frame_number": "#frame_number",
            "filename": "#VideoFilename",
            "SiteID": "#SiteID",
            "SiteCode": "#SiteCode",
            "clip_start_time": "upl_seconds",
        }
    )

    # Select only columns of interest
    upload_to_zoo = df[
        [
            "frame_path",
            "Year",
            "ScientificName",
            "Depth",
            "!LinkToMarineReserve",
            "#EventDate",
            "#TimeOfMaxSeconds",
            "#frame_number",
            "#VideoFilename",
            "#SiteID",
            "#SiteCode",
        ]
    ].reset_index(drop=True)

    return upload_to_zoo


def add_spyfish_survey_info(movies_df: pd.DataFrame, csv_paths: dict):
    """
    It takes a dataframe of movies and returns it with the survey_specific info

    :param df: the dataframe of all the detections
    :param csv_paths: paths to the csv from the project object
    :return: A dataframe with the columns of interest for uploading to Zooniverse.
    """
    # Read info about the movies
    movies_csv = pd.read_csv(csv_paths["local_movies_csv"])

    # Select only movie ids and survey ids
    movies_csv = movies_csv[["movie_id", "SurveyID"]]

    # Combine the movie_id and survey information
    movies_df = pd.merge(
        movies_df, movies_csv, how="left", left_on="id", right_on="movie_id"
    ).drop(columns=["movie_id"])

    # Read info about the surveys
    surveys_df = pd.read_csv(
        csv_paths["local_surveys_csv"],
        parse_dates=["SurveyStartDate"],
        infer_datetime_format=True,
    )

    # Combine the movie_id and survey information
    movies_df = pd.merge(
        movies_df,
        surveys_df,
        how="left",
        left_on="SurveyID",
        right_on="SurveyID",
    )

    return movies_df
