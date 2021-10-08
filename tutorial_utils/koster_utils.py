# -*- coding: utf-8 -*-
import io, os, json, csv
import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
from collections import OrderedDict, Counter
from datetime import datetime
import tutorial_utils.db_utils as db_utils


# Function to prevent issues with Swedish characters
def unswedify(string):
    """Convert ä and ö to utf-8"""
    return (
        string.encode("utf-8")
        .replace(b"\xc3\xa4", b"a\xcc\x88")
        .replace(b"\xc3\xb6", b"o\xcc\x88")
        .decode("utf-8")
    )

# Function to extract metadata from subjects
def extract_metadata(subj_df):

    # Reset index of df
    subj_df = subj_df.reset_index(drop=True).reset_index()

    # Flatten the metadata information
    meta_df = pd.json_normalize(subj_df.metadata.apply(json.loads))

    # Drop metadata and index columns from original df
    subj_df = subj_df.drop(columns=["metadata", "index"])

    return subj_df, meta_df

# Function to process subjects uploaded automatically
def auto_subjects(subjects_df, auto_date):
    
    # Select automatically uploaded frames
    auto_subjects_df = subjects_df[(subjects_df["created_at"] > auto_date)]

    # Extract metadata from automatically uploaded subjects
    auto_subjects_df, auto_subjects_meta = extract_metadata(auto_subjects_df)
    
    # Combine metadata info with the subjects df
    auto_subjects_df = pd.concat([auto_subjects_df, auto_subjects_meta], axis=1)
    
    return auto_subjects_df

# Function to process subjects uploaded manually
def manual_subjects(subjects_df, manual_date, auto_date):
    
    # Select clips uploaded manually
    man_clips_df = (
        subjects_df[
            (subjects_df["metadata"].str.contains(".mp4"))
            & (
                subjects_df["created_at"].between(
                    manual_date, auto_date
                )
            )
        ]
        .reset_index(drop=True)
        .reset_index()
    )

    # Specify the type of subject
    man_clips_df["subject_type"] = "clip"

    # Extract metadata from manually uploaded clips
    man_clips_df, man_clips_meta = extract_metadata(man_clips_df)

    # Process the metadata of manually uploaded clips
    man_clips_meta = process_manual_clips(man_clips_meta)

    # Combine metadata info with the subjects df
    man_clips_df = pd.concat([man_clips_df, man_clips_meta], axis=1)
    
    return man_clips_df
    
    # Function to get the movie_ids based on movie filenames
def get_movies_id(df, db_path):

    # Create connection to db
    conn = db_utils.create_connection(db_path)

    # Query id and filenames from the movies table
    movies_df = pd.read_sql_query("SELECT id, filename FROM movies", conn)
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


# Function to process the metadata of clips that were uploaded manually
def process_manual_clips(meta_df):

    # Select the filename of the clips and remove extension type
    clip_filenames = meta_df["filename"].str.replace(".mp4", "", regex=True)

    # Get the starting time of clips in relation to the original movie
    # split the filename and select the last section
    meta_df["clip_start_time"] = (
        clip_filenames.str.rsplit("_", 1).str[-1]
    )

    # Extract the filename of the original movie
    meta_df["movie_filename"] = meta_df.apply(
        lambda x: x["filename"].replace("_" + x["clip_start_time"], "").replace(".mp4", ".mov"), axis=1
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


# Function to get the list of duplicated subjects
def get_duplicatesdf():
    
    # Define the path to the csv files with initial info to build the db
    db_csv_info = "../db_starter/db_csv_info/" 

    # Define the path to the csv file with ids of the duplicated subjects
    for file in Path(db_csv_info).rglob("*.csv"):
        if 'duplicat' in file.name:
            duplicates_csv = file
            
    # Load the csv with information about duplicated subjects
    duplicatesdf = pd.read_csv(duplicates_csv)
    
    return duplicatesdf


# Function to select the first subject of those that are duplicated
def clean_duplicated_subjects(subjects):
    
    # Get the duplicates df
    duplicatesdf = get_duplicatesdf()
    
    # Include a column with unique ids for duplicated subjects 
    subjects = pd.merge(subjects, duplicatesdf, how="left", left_on="subject_id", right_on="dupl_subject_id")
    
    # Replace the id of duplicated subjects for the id of the first subject
    subjects.subject_id = np.where(subjects.single_subject_id.isnull(), subjects.subject_id, subjects.single_subject_id)
    
    #Select only unique subjects
    subjects = subjects.drop_duplicates(subset='subject_id', keep='first')
    
    return subjects


def process_koster_subjects(subjects, db_path):
    
    ## Set the date when the metadata of subjects uploaded matches/doesn't match schema.py requirements

    # Specify the date when the metadata of subjects uploaded matches schema.py
    auto_date = "2020-05-29 00:00:00 UTC"

    # Specify the starting date when clips were manually uploaded
    manual_date = "2019-11-17 00:00:00 UTC"

    ## Update subjects automatically uploaded 

    # Select automatically uploaded subjects
    auto_subjects_df = auto_subjects(subjects, auto_date = auto_date)

    ## Update subjects manually uploaded
    # Select manually uploaded subjects
    manual_subjects_df = manual_subjects(subjects, manual_date = manual_date, auto_date = auto_date)

    # Include movie_ids to the metadata
    manual_subjects_df = get_movies_id(manual_subjects_df, db_path)
   
    # Combine all uploaded subjects
    subjects = pd.merge(manual_subjects_df, auto_subjects_df, how="outer")
        
    return subjects

# Function to combine classifications received on duplicated subjects
def combine_annot_from_duplicates(annot_df):

    # Get the duplicates df
    duplicatesdf = get_duplicatesdf()
    
    # Include a column with unique ids for duplicated subjects
    annot_df = pd.merge(
        annot_df, duplicatesdf, how="left", left_on="subject_ids", right_on="dupl_subject_id"
    )

    # Replace the id of duplicated subjects for the id of the first subject
    annot_df["subject_ids"] = np.where(
        annot_df.single_subject_id.isnull(),
        annot_df.subject_ids,
        annot_df.single_subject_id,
    )

    return annot_df


def process_clips_koster(annotations, row_class_id, rows_list):
    
    nothing_values = ["NOANIMALSPRESENT","ICANTRECOGNISEANYTHING","ISEENOTHING","NOTHINGHERE"]
    
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

def bb_iou(boxA, boxB):

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


def filter_bboxes(total_users, users, bboxes, obj, eps, iua):

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
