##ZOOniverse utils
import io
import getpass
import pandas as pd
import json
import numpy as np
from panoptes_client import (
    SubjectSet,
    Subject,
    Project,
    Panoptes,
)

from ast import literal_eval
from tutorial_utils.koster_utils import process_koster_subjects, clean_duplicated_subjects, combine_annot_from_duplicates
from tutorial_utils.spyfish_utils import process_spyfish_subjects
import tutorial_utils.db_utils as db_utils

def zoo_credentials():
    zoo_user = getpass.getpass('Enter your Zooniverse user')
    zoo_pass = getpass.getpass('Enter your Zooniverse password')
    
    return zoo_user, zoo_pass


class AuthenticationError(Exception):
    pass


# Function to authenticate to Zooniverse
def auth_session(username, password, project_n):

    # Connect to Zooniverse with your username and password
    auth = Panoptes.connect(username=username, password=password)

    if not auth.logged_in:
        raise AuthenticationError("Your credentials are invalid. Please try again.")

    # Specify the project number of the koster lab
    project = Project(project_n)

    return project


# Function to retrieve information from Zooniverse
def retrieve_zoo_info(username: str, password: str, project_name: str, zoo_info: str):

    # Specify location of the latest list of projects
    projects_csv = "../db_starter/projects_list.csv" 
    
    # Read the latest list of projects
    projects_df = pd.read_csv(projects_csv)
    
    project_n = projects_df[projects_df["Project_name"]==project_name]["Zooniverse_number"].unique()[0]
    
    print("Connecting to the Zooniverse project")

    # Connect to the Zooniverse project
    project = auth_session(username, password, project_n)

    # Create an empty dictionary to host the dfs of interest
    info_df = {}

    for info_n in zoo_info:
        print("Retrieving", info_n, "from Zooniverse")

        # Get the information of interest from Zooniverse
        export = project.get_export(info_n)

        try:
            # Save the info as pandas data frame
            export_df = pd.read_csv(io.StringIO(export.content.decode("utf-8")))
            
            # If KSO deal with duplicated subjects
            if project_name == "Koster Seafloor Obs":

                # Clear duplicated subjects
                if info_n == "subjects":
                    export_df = clean_duplicated_subjects(export_df)

                # Combine classifications from duplicated subjects to unique subject id
                if info_n == "classifications":
                    export_df = combine_annot_from_duplicates(export_df)

        except:
            raise ValueError("Request time out, please try again in 1 minute.")

        # Ensure subject_ids match db format
        if info_n == "classifications":
            export_df["subject_ids"] = export_df["subject_ids"].astype(np.int64)
                    
        # Add df to dictionary
        info_df[info_n] = export_df
        
        print(info_n, "were retrieved successfully")

    return project_n, info_df


# Function to extract metadata from subjects
def extract_metadata(subj_df):

    # Reset index of df
    subj_df = subj_df.reset_index(drop=True).reset_index()

    # Flatten the metadata information
    meta_df = pd.json_normalize(subj_df.metadata.apply(json.loads))

    # Drop metadata and index columns from original df
    subj_df = subj_df.drop(
        columns=[
            "metadata",
            "index",
        ]
    )

    return subj_df, meta_df


def populate_subjects(subjects, project_name, db_path):

    # Check if the Zooniverse project is the KSO
    if project_name == "Koster Seafloor Obs":

        subjects = process_koster_subjects(subjects, db_path)

    else:

        # Extract metadata from uploaded subjects
        subjects_df, subjects_meta = extract_metadata(subjects)

        # Combine metadata info with the subjects df
        subjects = pd.concat([subjects_df, subjects_meta], axis=1)

        # Check if the Zooniverse project is the Spyfish
        if project_name == "Spyfish Aotearoa":

            subjects = process_spyfish_subjects(subjects, db_path)

    # Set subject_id information as id
    subjects = subjects.rename(columns={"subject_id": "id"})

    # Extract the html location of the subjects
    subjects["https_location"] = subjects["locations"].apply(lambda x: literal_eval(x)["0"])
    
    # Set the columns in the right order
    subjects = subjects[
        [
            "id",
            "subject_type",
            "filename",
            "clip_start_time",
            "clip_end_time",
            "frame_exp_sp_id",
            "frame_number",
            "workflow_id",
            "subject_set_id",
            "classifications_count",
            "retired_at",
            "retirement_reason",
            "created_at",
            "https_location",
            "movie_id",
        ]
    ]

    # Ensure that subject_ids are not duplicated by workflow
    subjects = subjects.drop_duplicates(subset="id")

    # Test table validity
    db_utils.test_table(subjects, "subjects", keys=["movie_id"])

    # Add values to subjects
    db_utils.add_to_table(db_path, "subjects", [tuple(i) for i in subjects.values], 15)
    
    ##### Print how many subjects are in the db
    # Create connection to db
    conn = db_utils.create_connection(db_path)
    
    # Query id and subject type from the subjects table
    subjects_df = pd.read_sql_query("SELECT id, subject_type FROM subjects", conn)
    frame_subjs = subjects_df[subjects_df["subject_type"]=="frame"].shape[0]
    clip_subjs = subjects_df[subjects_df["subject_type"]=="clip"].shape[0]
    
    
    print("The database has a total of", frame_subjs, "frame subjects and", clip_subjs, "clip subjects have been updated")

# Relevant for ML and upload frames tutorials
def populate_agg_annotations(annotations, subj_type, db_path):

    conn = db_utils.create_connection(db_path)
    
    # Query id and subject type from the subjects table
    subjects_df = pd.read_sql_query("SELECT id, frame_exp_sp_id FROM subjects", conn)

    # Combine annotation and subject information
    annotations = annotations.rename(columns={"subject_ids": "subject_id"})
    annotations_df = pd.merge(
        annotations,
        subjects_df,
        how="left",
        left_on="subject_id",
        right_on="id",
        validate="many_to_one",
    )

    # Update agg_annotations_clip table
    if subj_type == "clip":
        print("WIP")
        
    # Update agg_annotations_frame table
    if subj_type == "frame":
        
        # Select relevant columns
        annotations_df = annotations_df[["frame_exp_sp_id", "x", "y", "w", "h", "subject_id"]].dropna(
            subset=["x", "y", "w", "h"])
        
        # Test table validity
        db_utils.test_table(annotations_df, "agg_annotations_frame", keys=["frame_exp_sp_id"])

        # Add values to agg_annotations_frame
        db_utils.add_to_table(
            db_path,
            "agg_annotations_frame",
            [(None,) + tuple(i) for i in annotations_df.values],
            7,
        )

    



