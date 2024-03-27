# base imports
import io
import re
import os
import getpass
import pandas as pd
import json
import logging
import math
import numpy as np
import gdown
import datetime
import ffmpeg
import shutil
import sqlite3
from tqdm import tqdm
from panoptes_client import Panoptes, panoptes, Subject, SubjectSet
from panoptes_client import Project as zooProject
from ast import literal_eval
from pathlib import Path

# util imports
from kso_utils.project_utils import Project
from kso_utils.db_utils import add_db_info_to_df
import kso_utils.db_utils as db_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.server_utils as server_utils

# Widget imports
import ipywidgets as widgets


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

##########################
# General Zoo purpose functions
##########################


def zoo_credentials():
    zoo_user = getpass.getpass("Enter your Zooniverse user")
    zoo_pass = getpass.getpass("Enter your Zooniverse password")
    return zoo_user, zoo_pass


class AuthenticationError(Exception):
    pass


def connect_zoo_project(project: Project, zoo_cred=False):
    """
    It takes a project name as input, and returns a Zooniverse project object
    zoo_cred is an argument that can pass [username, password] to log in into zooniverse.
    This is used in the automatic tests in gitlab called autotests.py.
    when it is set to False, then the credentials are retrieved from the interacitve widget.
    :param project: the KSO project you are working
    :return: A Zooniverse project object.
    """
    if zoo_cred is False:
        # Save your Zooniverse user name and password.
        zoo_user, zoo_pass = zoo_credentials()
    else:
        zoo_user = zoo_cred[0]
        zoo_pass = zoo_cred[1]

    # Get the project-specific zooniverse number
    project_n = project.Zooniverse_number

    # Connect to the Zooniverse project
    zoo_project = auth_session(zoo_user, zoo_pass, project_n)

    logging.info("Connected to Zooniverse")

    return zoo_project


# Function to authenticate to Zooniverse
def auth_session(username: str, password: str, project_n: int):
    """
    It connects to the Zooniverse with your username and password, and then returns the project number
    you specify

    :param username: Your Zooniverse username
    :param password: your Zooniverse password
    :param project_n: The project number of the project you want to download data from
    :return: The project number of the koster lab
    """

    # Connect to Zooniverse with your username and password
    auth = Panoptes.connect(username=username, password=password)

    if not auth.logged_in:
        raise AuthenticationError("Your credentials are invalid. Please try again.")

    # Specify the project number of the koster lab
    try:
        project = zooProject(int(float(project_n)))
        return project
    except Exception as e:
        logging.error(e)


# Function to retrieve information from Zooniverse
def retrieve_zoo_info(
    project: Project,
    zoo_project,
    zoo_info: str,
    generate_export: bool = False,
):
    """
    This function retrieves the information of interest from Zooniverse and saves it as a pandas data
    frame

    :param project: the kso project object
    :param zoo_project: the Zooniverse project object
    :param zoo_info: a list of the info you want to retrieve from Zooniverse
    :type zoo_info: str
    :param generate_export: boolean determining whether to generate a new export and wait for it to be ready or to just download the latest export
    :return: A dictionary of dataframes.
    """

    # Create an empty dictionary to host the dfs of interest
    info_df = {}

    for info_n in zoo_info:
        logging.info(f"Retrieving {info_n} from Zooniverse")

        try:
            # Get the information of interest from Zooniverse
            if generate_export:
                try:
                    export = zoo_project.get_export(
                        info_n, generate=generate_export, wait=True, wait_timeout=1800
                    )
                except panoptes.PanoptesAPIException:
                    logging.error(
                        "Export generation time out, retrieving the last available information..."
                    )
                    export = zoo_project.get_export(info_n, generate=False)
            else:
                export = zoo_project.get_export(info_n, generate=generate_export)

            # Save the info as pandas data frame
            try:
                export_df = pd.read_csv(
                    io.StringIO(export.content.decode("utf-8")), low_memory=False
                )
            except pd.errors.ParserError as e:
                # Handle the case where a ParserError occurs
                logging.error(f"ParserError occurred while reading CSV data: %s {e}")
                # Optionally, provide a more specific error message or handle the error accordingly
                export_df = None

        except AttributeError:
            logging.info(
                "No connection with Zooniverse, retrieving template info from google drive."
            )
            if info_n == "classifications":
                url = "https://drive.google.com/file/d/1DvJ2nOrG32MR2D7faAJZXMNbEm_ra3rb/view?usp=sharing"
            if info_n == "subjects":
                url = "https://drive.google.com/file/d/18AWRPx3erL25IHekncgKfI_kXHFYAl8e/view?usp=sharing"
            if info_n == "workflows":
                url = "https://drive.google.com/file/d/1bZ6CSxJLxeoX8xVgMU7ZqL76RZDv-09A/view?usp=sharing"
            export = gdown.download(url, info_n + ".csv", quiet=False, fuzzy=True)
            export_df = pd.read_csv(export)

        if len(export_df) > 0:
            # Clear duplicated subjects
            if info_n == "subjects":
                # Select info from each unique subject to avoid loading subject info
                # multiple times (because the subject has been classified in multiple workflows)
                export_df = export_df.drop_duplicates(subset="subject_id", keep="first")

        else:
            raise ValueError(
                "The export is empty. This may be due to a "
                "request time out, please try again in 1 minute."
            )

        # Ensure subject_ids and workflow_ids match db format
        if info_n == "classifications":
            export_df["subject_ids"] = export_df["subject_ids"].astype(np.int64)

        # Add df to dictionary
        info_df[info_n] = export_df
        project.info_df = info_df
        logging.info(f"{info_n} retrieved successfully")

    return info_df


# Function to extract metadata from subjects
def extract_metadata(subj_df: pd.DataFrame):
    """
    The function extracts metadata from a pandas DataFrame and returns two separate DataFrames, one with
    the metadata flattened and one without the metadata.

    :param subj_df: A pandas DataFrame containing subject data, including metadata information in JSON
    format
    :type subj_df: pd.DataFrame
    :return: The function `extract_metadata` returns two dataframes: `subj_df` and `meta_df`. `subj_df`
    is the original input dataframe with the "metadata" and "index" columns dropped, and with the index
    reset. `meta_df` is a new dataframe that contains the flattened metadata information extracted from
    the "metadata" column of the input dataframe.
    """

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


# Function to clean label (no non-alpha characters)
def clean_label(label_string: str):
    label_string = label_string.replace("_", "")
    label_string = label_string.upper()
    label_string = label_string.replace(" ", "")
    pattern = r"[^A-Za-z0-9]+"
    cleaned_string = re.sub(pattern, "", label_string)
    return cleaned_string


##########################
# Workflow-specific functions
##########################


class WidgetWorkflowSelection(widgets.VBox):
    def __init__(self, workflows_df: pd.DataFrame):
        """
        The function creates a widget that allows the user to select which workflows to run

        :param workflows_df: the dataframe of workflows
        """

        # Estimate the maximum number of workflows available
        max_workflows = workflows_df.display_name.nunique()

        self.workflows_df = workflows_df
        self.widget_count = widgets.BoundedIntText(
            value=0,
            min=0,
            max=max_workflows,
            description="Number of workflows:",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )
        self.bool_widget_holder = widgets.HBox(
            layout=widgets.Layout(
                width="70%", display="inline-flex", flex_flow="row wrap"
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
            from kso_utils.widgets import choose_workflows

            new_widget = choose_workflows(self.workflows_df)
            for wdgt in new_widget:
                wdgt.description = wdgt.description + f" #{_}"
            new_widgets.extend(new_widget)
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}


def get_workflow_labels(
    workflow_df: pd.DataFrame, workflow_id: int, workflow_version: int
):
    """
    > This function takes a df of workflows of interest and retrieves the labels and common names of the choices cit scientists have in a survey task in Zooniverse.
    the function is a modified version of the 'get_workflow_info' function by @lcjohnso
    https://github.com/zooniverse/Data-digging/blob/6e9dc5db6f6125316616c4b04ae5fc4223826a25/scripts_GeneralPython/get_workflow_info.pybiological observations classified by citizen scientists, biologists or ML algorithms and returns a df of species occurrences to publish in GBIF/OBIS.
    :param workflow_df: df of the workflows of the Zooniverse project of interest,
    :param workflow_id: integer of the workflow id of interest,
    :param workflow_version: integer of the workflow version of interest.
    :return: a df with the common name and label of the annotations for the workflow.
    """
    # initialize the output
    workflow_info = {}

    # parse the tasks column as a json so we can work with it (it just loads as a string)
    workflow_df["tasks_json"] = [json.loads(q) for q in workflow_df["tasks"]]
    workflow_df["strings_json"] = [json.loads(q) for q in workflow_df["strings"]]

    # identify the row of the workflow dataframe we want to extract
    is_theworkflow = (workflow_df["workflow_id"] == workflow_id) & (
        workflow_df["version"] == workflow_version
    )

    # extract it
    theworkflow = workflow_df[is_theworkflow]

    # pandas is a little weird about accessing stuff sometimes
    # we should only have 1 row in theworkflow but the row index will be retained
    # from the full workflow_df, so we need to figure out what it is
    i_wf = theworkflow.index[0]

    # extract the tasks as a json
    tasks = theworkflow["tasks_json"][i_wf]
    strings = theworkflow["strings_json"][i_wf]

    workflow_info = tasks.copy()

    tasknames = workflow_info.keys()
    workflow_info["tasknames"] = tasknames

    # now that we've extracted the actual task names, add the first task
    workflow_info["first_task"] = theworkflow["first_task"].values[0]

    # now join workflow structure to workflow label content for each task
    for task in tasknames:
        # Check if the task has multiple choices
        if isinstance(workflow_info[task], dict):
            # Create an empty dictionary to host the dfs of interest
            label_common_name_dict = {"commonName": [], "label": []}
            for choice in workflow_info[task]["choices"]:
                label_common_name_dict["label"].append(choice)
                choice_name = strings[workflow_info[task]["choices"][choice]["label"]]
                label_common_name_dict["commonName"].append(choice_name)

        if task == "T0":
            break

    return pd.DataFrame.from_dict(label_common_name_dict)


##########################
# Classification-specific functions
##########################


# Flatten the classifications provided the cit. scientists
def process_zoo_classifications(
    project: Project,
    server_connection: dict,
    db_connection,
    workflow_widget_checks,
    workflows_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    csv_paths: dict,
    classifications_data: pd.DataFrame,
    subject_type: str,
):
    """
    This function select subjects from the workflows of interest, populates the sql subjects table, process the classifications received and returns a dataframe of annotations.

    :param project: the project object
    :param server_connection: A dictionary with the client and sftp_client
    :param db_connection: SQL connection object
    :param workflow_widget_checks: the widget with information of the selected workflows
    :param workflows_df: dataframe with the project workflows information (retrieved from Zooniverse)
    :param subjects_df: dataframe with the project subjects information (retrieved from Zooniverse)
    :param csv_paths: a dictionary with the paths of the csvs used to initiate the db
    :param classifications_data: the dataframe of classifications from the Zooniverse API
    :param subject_type: This is the type of subject you want to retrieve classifications for. This
           can be either "clip" or "frame"
    """

    # Store the names of the workflows and their versions
    workflow_names, workflow_versions = [], []
    for i in range(0, len(workflow_widget_checks), 3):
        workflow_names.append(list(workflow_widget_checks.values())[i])
        workflow_versions.append(list(workflow_widget_checks.values())[i + 2])

    # Get the ids of the workflows of interest
    selected_zoo_workflows = [
        workflows_df[workflows_df.display_name == wf_name].workflow_id.unique()[0]
        for wf_name in workflow_names
    ]

    if not isinstance(selected_zoo_workflows, list):
        selected_zoo_workflows = literal_eval(selected_zoo_workflows)

    # Classifications selected##############
    # Select only classification from the workflows of interest
    classes_df = classifications_data[
        classifications_data["workflow_id"].isin(selected_zoo_workflows)
    ].copy()

    # Get the unique subject IDs from the classes DataFrame
    unique_subject_ids = classes_df["subject_ids"].unique()

    # Count the number of unique subject IDs
    number_of_unique_subjects = len(unique_subject_ids)

    # Report the number of classifications retrieved
    logging.info(
        f"{classes_df.shape[0]:,} Zooniverse classifications have been retrieved"
        f" from {number_of_unique_subjects:,} subjects"
    )

    # Subjects selected########################
    # Select only subjects from the workflows of interest
    subjects_series = subjects_df[
        subjects_df["subject_id"].isin(unique_subject_ids)
    ].copy()

    from kso_utils.db_utils import drop_table

    # Safely remove subjects table
    drop_table(conn=db_connection, table_name="subjects")

    if len(subjects_series) > 0:
        # Fill or re-fill subjects table
        populate_subjects(project, server_connection, db_connection, subjects_series)
    else:
        logging.error("No subjects to populate database from the workflows selected.")

    # Combine classifications and subject info#####
    # Combine the classifications and subjects dataframes
    classes_df = add_db_info_to_df(
        project,
        db_connection,
        csv_paths,
        classes_df,
        "subjects",
        "id, subject_type, filename, clip_start_time, clip_end_time, frame_exp_sp_id, frame_number, subject_set_id, https_location, movie_id",
    )

    # Flatten the classifications provided the cit. scientists

    # Create an empty list to store the annotations
    rows_list = []

    # Loop through each classification submitted by the users
    for index, row in classes_df.iterrows():
        # Load annotations as json format
        annotations = json.loads(row["annotations"])

        # Process clip annotations
        if subject_type == "clip":
            # Select the information from the species identification task
            if project.Zooniverse_number == 9747:
                from kso_utils.koster_utils import process_clips_koster

                rows_list = process_clips_koster(
                    annotations, row["classification_id"], rows_list
                )

            # Check if the Zooniverse project is the Spyfish
            if project.Project_name == "Spyfish_Aotearoa":
                from kso_utils.spyfish_utils import process_clips_spyfish

                rows_list = process_clips_spyfish(
                    annotations, row["classification_id"], rows_list
                )

            # Process clips as the default method
            else:
                rows_list = process_clips_default(
                    annotations, row["classification_id"], rows_list
                )

        # Process frame classifications
        elif subject_type == "frame":
            # Select the information from all the labelled animals (e.g. task = T0)
            for ann_i in annotations:
                # Temporary fix (find out what T2 is and if it is always relevant)
                if ann_i["task"] in ["T0", "T2"]:
                    if ann_i["value"] == []:
                        # Specify the frame was classified as empty
                        choice_i = {
                            "classification_id": row["classification_id"],
                            "x": None,
                            "y": None,
                            "w": None,
                            "h": None,
                            "label": "empty",
                        }
                        rows_list.append(choice_i)

                    else:
                        # Select each species annotated and flatten the relevant answers
                        for i in ann_i["value"]:
                            choice_i = {
                                "classification_id": row["classification_id"],
                                "x": int(i["x"]) if "x" in i else None,
                                "y": int(i["y"]) if "y" in i else None,
                                "w": int(i["width"]) if "width" in i else None,
                                "h": int(i["height"]) if "height" in i else None,
                                "label": (
                                    str(i["tool_label"]) if "tool_label" in i else None
                                ),
                            }
                            rows_list.append(choice_i)

        else:
            logging.error("The subject_type is not valid")

    # Specify the cols specific to each subject type
    if subject_type == "clip":
        subject_cols = ["first_seen", "how_many"]
    else:
        subject_cols = ["x", "y", "w", "h"]

    # Combine common columns for the flattened annotations with subject type specific
    annot_cols = ["classification_id", "label"]
    annot_cols.extend(subject_cols)

    # Create a data frame of the flatten annotations
    flat_annot_df = pd.DataFrame(rows_list, columns=annot_cols)

    # Ensure the empty cells are replace with NAN
    flat_annot_df[subject_cols] = (
        flat_annot_df[subject_cols]
        .astype(str)
        .apply(lambda x: x.str.strip())
        .replace("", np.nan)
        .replace("None", np.nan)
    )

    # Ensure the subject type specific columns are numeric
    flat_annot_df[subject_cols] = flat_annot_df[subject_cols].astype("float64")

    # Combine the flatten the classifications with the subject information

    # Add subject information to each annotation
    annot_df = pd.merge(
        flat_annot_df,
        classes_df.drop(columns=["annotations"]),
        how="left",
        on="classification_id",
    )

    # Specify relevant columns
    annot_cols.extend(
        [
            "https_location",
            "subject_type",
            "subject_ids",
            "workflow_id",
            "workflow_name",
            "workflow_version",
            "user_name",
            "filename",
            "clip_start_time",
            "clip_end_time",
            "frame_exp_sp_id",
            "frame_number",
            "movie_id",
        ]
    )

    # Select only relevant columns
    annot_df = annot_df[annot_cols]

    # Select only classification from the workflows of interest
    annot_df = annot_df[annot_df.workflow_id.isin(selected_zoo_workflows)].copy()

    # Report the number of annotations flattened
    logging.info(
        f"{annot_df.shape[0]:,} Zooniverse annotations have been flattened"
        f" from {annot_df.subject_ids.nunique():,} subjects"
    )

    return pd.DataFrame(annot_df)


def process_clips_default(annotations: pd.DataFrame, row_class_id, rows_list: list):
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


def aggregate_classifications(
    project: Project,
    processed_classifications: pd.DataFrame,
    subject_type: str,
    agg_params: list,
):
    """
    We take the processed citizen scientits classifications and
    aggregated them based on aggregration paramaeters

    :param df: the processed classifications dataframe
    :param subject_type: the type of subject, either "frame" or "clip"
    :param agg_params: list of parameters for the aggregation
    :return: the aggregated classifications classifications.
    """

    # Check if we have the right number of parameters for the type of subject
    if subject_type == "clip":
        subject_parameters = 2
    elif subject_type == "frame":
        subject_parameters = 5

    if not len(agg_params) == subject_parameters:
        logging.error("Incorrect agg_params length for subject type")
        return

    logging.info("Aggregating the classifications")

    # We take the processed_classifications and aggregate them.
    if subject_type == "frame":
        # Get the aggregation parameters
        agg_users, min_users, agg_obj, agg_iou, agg_iua = [i.value for i in agg_params]
        # Report selected parameters
        logging.info(
            f"Aggregation parameters are: Agg. threshold "
            f"{agg_users} "
            f"Min. users "
            f"{min_users} "
            f"Obj threshold "
            f"{agg_obj} "
            f"IOU "
            f"{agg_iou} "
            f"Int. agg. "
            f"{agg_iua} "
        )

        # Aggregate frames based on volunteer consensus of the labels
        agg_labels_df = aggregate_labels(
            processed_classifications, agg_users, min_users
        )

        # Get rid of the "empty" labels if other species are among the volunteer consensus
        agg_labels_df = agg_labels_df[
            ~((agg_labels_df["class_n_agg"] > 1) & (agg_labels_df["label"] == "empty"))
        ]

        # Select frames aggregated only as empty (to add them later)
        empty_class_df = agg_labels_df[agg_labels_df["label"] == "empty"]

        # Temporary exclude frames already aggregated as empty
        agg_labels_df = agg_labels_df[
            ~agg_labels_df["classification_id"].isin(
                empty_class_df.classification_id.unique()
            )
        ]

        # Map the position of the annotation parameters
        col_list = list(agg_labels_df.columns)
        x_pos, y_pos, w_pos, h_pos, class_id_pos, subject_id_pos = (
            col_list.index("x"),
            col_list.index("y"),
            col_list.index("w"),
            col_list.index("h"),
            col_list.index("classification_id"),
            col_list.index("subject_ids"),
        )

        # Get prepared annotations
        new_rows = []

        # Specify the cols to group the annotations by
        group_cols = ["subject_ids", "label"]

        # loop through each group and estimate the overlapped annotations
        for name, group in agg_labels_df.groupby(group_cols):
            subj_id, label = name
            total_users = agg_labels_df[
                (agg_labels_df.subject_ids == subj_id) & (agg_labels_df.label == label)
            ]["user_name"].nunique()

            # Filter bboxes using IOU metric (essentially a consensus metric)
            # Keep only bboxes where mean overlap exceeds this threshold
            from kso_utils.frame_utils import filter_bboxes

            indices, new_group = filter_bboxes(
                total_users=total_users,
                users=[i[class_id_pos] for i in group.values],
                bboxes=[
                    np.array([i[x_pos], i[y_pos], i[w_pos], i[h_pos]])
                    for i in group.values
                ],
                obj=agg_obj,
                eps=agg_iou,
                iua=agg_iua,
            )

            subject_ids = [i[subject_id_pos] for i in group.values[indices]]

            for ix, box in zip(subject_ids, new_group):
                new_rows.append(
                    (
                        label,
                        ix,
                    )
                    + tuple(box)
                )

        # Specify the names of the relevant columns for the agggregated df
        cols_agg_class_df = [
            "label",
            "subject_ids",
            "x",
            "y",
            "w",
            "h",
        ]

        # Combine the new rows into a df
        agg_class_df = pd.DataFrame(
            new_rows,
            columns=cols_agg_class_df,
        )

        # Clear some weird labels
        agg_class_df["label"] = agg_class_df["label"].apply(
            lambda x: x.split("(")[0].strip()
        )

        # Select only relevant columns from the empty df
        empty_class_df = empty_class_df[cols_agg_class_df]

        # Add the frames aggregated as "empty"
        agg_class_df = pd.concat([agg_class_df, empty_class_df])

        # Specify the columns from the processed_classifications to drop
        # to avoid duplicates
        cols_processed_classifications = [
            x for x in cols_agg_class_df if x != "subject_ids"
        ]

        # temporarily drop the unnecessary cols from the processed classifications
        temp_processed_classifications = processed_classifications.drop(
            columns=cols_processed_classifications
        )

        # Add the subject,site,movie.. info related to the subject id
        # from the processed classifications
        agg_class_df = agg_class_df.merge(
            temp_processed_classifications, on="subject_ids"
        )

    else:
        # Get the aggregation parameters
        if not isinstance(agg_params, list):
            agg_users, min_users = [i.value for i in agg_params]
        else:
            agg_users, min_users = agg_params

        # Aggregate clips based on their labels
        agg_class_df = aggregate_labels(processed_classifications, agg_users, min_users)

    # Drop unnecessary columns
    agg_class_df = agg_class_df.drop(columns=["classification_id", "user_name"])

    # Keep only one raw per subject and label
    agg_class_df = agg_class_df.drop_duplicates()

    logging.info(
        f"{agg_class_df.shape[0]}"
        " classifications aggregated out of "
        f"{processed_classifications.subject_ids.nunique()}"
        " unique subjects available"
    )

    return agg_class_df


def aggregate_labels(
    processed_classifications: pd.DataFrame, agg_users: float, min_users: int
):
    """
    > This function takes a dataframe of classifications and returns a dataframe of classifications that
    have been filtered by the number of users that classified each subject and the proportion of users
    that agreed on their annotations

    :param processed_classifications: the dataframe of all the classifications
    :param agg_users: the proportion of users that must agree on a classification for it to be included
           in the final dataset
    :param min_users: The minimum number of users that must have classified a subject for it to be
           included in the final dataset
    :return: a dataframe with the aggregated labels.
    """
    # Calculate the number of users that classified each subject
    processed_classifications["n_users"] = processed_classifications.groupby(
        "subject_ids"
    )["classification_id"].transform("nunique")

    # Select classifications with at least n different user classifications
    processed_classifications = processed_classifications[
        processed_classifications.n_users >= min_users
    ].reset_index(drop=True)

    # Calculate the proportion of unique classifications (it can have multiple annotations) per subject
    processed_classifications["class_n"] = processed_classifications.groupby(
        ["subject_ids", "label"]
    )["classification_id"].transform("nunique")

    # Calculate the proportion of users that agreed on their annotations
    processed_classifications["aggreg"] = (
        processed_classifications.class_n / processed_classifications.n_users
    )

    # Select annotations based on agreement threshold
    agg_class_df = processed_classifications[
        processed_classifications.aggreg >= agg_users
    ].reset_index(drop=True)

    # Specify the columns to group by and calculate the median to
    # of the second where the animal/object is and number of animals
    list_cols_group = ["subject_ids", "label"]
    cols_median = ["how_many", "first_seen"]

    # If the df has columns to calculate the median (usually clips)
    if all(item in agg_class_df.columns for item in cols_median):
        # Prevent issues when trying to calculate the median of empty cells
        agg_class_df[cols_median] = agg_class_df[cols_median].fillna(0)

        # Extract the median of the second where the animal/object is and number of animals
        agg_class_df[cols_median] = (
            agg_class_df.groupby(list_cols_group, observed=True)[cols_median]
            .transform("median")
            .astype(int)
        )

    # Calculate the number of unique classifications aggregated per subject
    agg_class_df["class_n_agg"] = agg_class_df.groupby(["subject_ids"])[
        "label"
    ].transform("nunique")

    return agg_class_df


def add_subject_site_movie_info_to_class(
    project: Project,
    db_connection,
    csv_paths: dict,
    class_df: pd.DataFrame,
):
    """
    It takes a dataframe of clips or frames, and adds metadata about the site and project to it
    :param df: the dataframe with the media to upload
    :param project: the project object
    :param species_list: a list of the species that should be on the frames
    :param csv_paths: a dictionary with the paths of the csvs used to initiate the db
    :param class_df: a data frame with the classifications
    :param aggregated: a boolean value specifyin if the classifications are aggregated or not
    """

    # Combine the aggregated clips and movies dataframes
    class_df = add_db_info_to_df(
        project,
        db_connection,
        csv_paths,
        class_df,
        "movies",
        "id, created_on, fps, duration, sampling_start, sampling_end, author, site_id, fpath",
    )

    # Combine the aggregated clips and sites dataframes
    class_df = add_db_info_to_df(project, db_connection, csv_paths, class_df, "sites")

    return class_df


def launch_classifications_table(agg_class_df: pd.DataFrame, subject_type: str):
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


##########################
# Subject-specific functions
##########################


def get_workflow_ids(workflows_df: pd.DataFrame, workflow_names: list):
    # The function that takes a list of workflow names and returns a list of workflow
    # ids.
    return [
        workflows_df[workflows_df.display_name == wf_name].workflow_id.unique()[0]
        for wf_name in workflow_names
    ]


def get_classifications(
    project: Project,
    conn: sqlite3.Connection,
    workflow_dict: dict,
    workflows_df: pd.DataFrame,
    subj_type: str,
    class_df: pd.DataFrame,
):
    """
    It takes in a dictionary of workflows, a dataframe of workflows, the type of subject (frame or
    clip), a dataframe of classifications, the path to the database, and the project name. It returns a
    dataframe of classifications

    :param project: the project object
    :param conn: SQL connection object
    :param workflow_dict: a dictionary of the workflows you want to retrieve classifications for. The
        keys are the workflow names, and the values are the workflow IDs, workflow versions, and the minimum
        number of classifications per subject
    :type workflow_dict: dict
    :param workflows_df: the dataframe of workflows from the Zooniverse project
    :type workflows_df: pd.DataFrame
    :param subj_type: "frame" or "clip"
    :param class_df: the dataframe of classifications from the database
    :return: A dataframe with the classifications for the specified project and workflow.
    """

    names, workflow_versions = [], []
    for i in range(0, len(workflow_dict), 3):
        names.append(list(workflow_dict.values())[i])
        workflow_versions.append(list(workflow_dict.values())[i + 2])

    workflow_ids = get_workflow_ids(workflows_df, names)

    # Filter classifications of interest
    classes = []
    for id, version in zip(workflow_ids, workflow_versions):
        class_df_id = class_df[
            (class_df.workflow_id == id) & (class_df.workflow_version >= version)
        ].reset_index(drop=True)
        classes.append(class_df_id)
    classes_df = pd.concat(classes)

    # Add information about the subject
    # Query id and subject type from the subjects table
    subjects_df = db_utils.get_df_from_db_table(conn, "subjects")

    if subj_type == "frame":
        # Select only frame subjects
        subjects_df = subjects_df[subjects_df["subject_type"] == "frame"]

        # Select columns relevant for frame subjects
        subjects_df = subjects_df[
            [
                "id",
                "subject_type",
                "https_location",
                "filename",
                "frame_number",
                "movie_id",
            ]
        ]

    else:
        # Select only clip subjects
        subjects_df = subjects_df[subjects_df["subject_type"] == "clip"]

        # Select columns relevant for clip subjects
        subjects_df = subjects_df[
            [
                "id",
                "subject_type",
                "https_location",
                "filename",
                "clip_start_time",
                "movie_id",
            ]
        ]

    # Ensure id format matches classification's subject_id
    classes_df["subject_ids"] = classes_df["subject_ids"].astype("Int64")
    subjects_df["id"] = subjects_df["id"].astype("Int64")

    # Add subject information based on subject_ids
    classes_df = pd.merge(
        classes_df,
        subjects_df,
        how="left",
        left_on="subject_ids",
        right_on="id",
    )

    if classes_df[["subject_type", "https_location"]].isna().any().any():
        # Exclude classifications from missing subjects
        filtered_class_df = classes_df.dropna(
            subset=["subject_type", "https_location"], how="any"
        ).reset_index(drop=True)

        # Report on the issue
        logging.info(
            f"There are {(classes_df.shape[0]-filtered_class_df.shape[0])}"
            f" classifications out of {classes_df.shape[0]}"
            f" missing subject info. Maybe the subjects have been removed from Zooniverse?"
        )

        classes_df = filtered_class_df

    logging.info(
        f"{classes_df.shape[0]} Zooniverse classifications have been retrieved"
    )

    return classes_df


def populate_subjects(
    project: Project,
    server_connection: dict,
    db_connection,
    subjects: pd.DataFrame,
):
    """
    Populate the subjects table with the subject metadata
    :param project: the project object
    :param server_connection: A dictionary with the client and sftp_client
    :param subjects: a dataframe of the subset of subjects from the workflows of interest
    :param db_connection: SQL connection object

    """

    # Check if the Zooniverse project is the KSO
    if project.Project_name == "Koster_Seafloor_Obs":
        from kso_utils.koster_utils import process_koster_subjects

        subjects = process_koster_subjects(subjects, db_connection)

    else:
        # Extract metadata from uploaded subjects
        subjects_df, subjects_meta = extract_metadata(subjects)

        # Combine metadata info with the subjects df
        subjects = pd.concat([subjects_df, subjects_meta], axis=1)

        # Check if the Zooniverse project is the Spyfish
        if project.Project_name == "Spyfish_Aotearoa":
            from kso_utils.spyfish_utils import process_spyfish_subjects

            # Process the spyfish movies
            subjects = process_spyfish_subjects(
                project, server_connection, subjects, db_connection
            )

    # Ensure that subject_ids are not duplicated by workflow
    subjects = subjects.drop_duplicates(subset="subject_id")

    # Rename columns that start with "#"
    hash_columns = {col: col.replace("#", "") for col in subjects.columns}

    # Rename columns to match the db format
    subjects = subjects.rename(columns=hash_columns)

    # Fix subject_type missing info and non-standard format
    # Fix weird bug where Subject_type is used instead of subject_type for the column name for some clips
    if "Subject_type" in subjects.columns and "subject_type" in subjects.columns:
        subjects["subject_type"] = subjects[["subject_type", "Subject_type"]].apply(
            lambda x: x[1] if isinstance(x[1], str) else x[0], 1
        )
        subjects.drop(columns=["Subject_type"], inplace=True)

    # Fix similar weird bug to above (Jannes)
    if "upl_seconds" in subjects.columns and "clip_start_time" in subjects.columns:
        subjects["clip_start_time"] = subjects[
            ["clip_start_time", "upl_seconds"]
        ].apply(lambda x: x[1] if not np.isnan(x[1]) else x[0], 1)
        subjects.drop(columns=["upl_seconds"], inplace=True)

    # Fix similar weird bug to above (Jannes)
    if "VideoFilename" in subjects.columns and "filename" in subjects.columns:
        subjects["filename"] = subjects[["filename", "VideoFilename"]].apply(
            lambda x: x[1] if isinstance(x[1], str) else x[0], 1
        )
        subjects.drop(columns=["VideoFilename"], inplace=True)

    # Rename common non-standard column names
    rename_cols = {
        "subject_id": "id",
        "VideoFilename": "filename",
        "upl_seconds": "clip_start_time",
        "Subject_type": "subject_type",
    }

    # Rename columns to match the db format
    subjects = subjects.rename(columns=rename_cols)

    if hasattr(subjects["subject_type"], "columns"):
        # Avoid having two subject_type columns (one from Zoo one from the db)
        subjects["subject_type0"] = subjects["subject_type"].iloc[:, 0]
        subjects["subject_type1"] = subjects["subject_type"].iloc[:, 1]

        # Update with non-empty values
        subjects["combined_subject_type"] = subjects["subject_type0"].combine_first(
            subjects["subject_type1"]
        )

        # Drop unnecesary cols
        subjects.drop(
            columns=["subject_type", "subject_type0", "subject_type1"], inplace=True
        )

        # Rename the combined subject type
        subjects = subjects.rename(columns={"combined_subject_type": "subject_type"})

    # Extract the html location of the subjects
    subjects["https_location"] = subjects["locations"].apply(
        lambda x: literal_eval(x)["0"]
    )

    # Check for empty values in subject_type
    if subjects["subject_type"].isna().any():
        # Manually write the subject type based on the extension of the subject location (link in Zooniverse)
        from kso_utils.movie_utils import get_movie_extensions

        # Calculate the number of subjects with empty subject type info
        num_updated_empty = subjects[subjects["subject_type"].isna()].shape[0]

        # Update empty values with "clip" based on https_location extension
        subjects.loc[
            subjects["subject_type"].isna(),
            "subject_type",
        ] = subjects[
            subjects["subject_type"].isna()
        ]["https_location"].apply(
            lambda url: (
                "clip"
                if any(url.endswith(ext) for ext in get_movie_extensions())
                else "frame"
            )
        )
        logging.info(
            f"Updated {num_updated_empty} empty rows with subject_type based on extension."
        )

    # Ensure only subjects with the right format get populated
    right_types = ["frame", "clip"]
    if subjects[subjects.subject_type.isin(right_types)].all().all():
        # Count the number of rows to be excluded
        logging.info(
            f"{subjects[~subjects.subject_type.isin(right_types)].shape[0]}"
            f" subjects missing the right subject_type info"
        )

        # Select only rows with the right subject_type info
        subjects = subjects[subjects["subject_type"].isin(right_types)]

    if "movie_id" not in subjects.columns:
        from kso_utils.db_utils import get_df_from_db_table

        # Match site code to name from movies sql and get movie_id to save it as "movie_id"
        # Query id and filenames from the movies table
        movies_df = get_df_from_db_table(db_connection, "movies")[["id", "filename"]]

        # Rename columns to match subject df
        movies_df = movies_df.rename(columns={"id": "movie_id"})

        # Reference the movienames with the id movies table
        subjects = pd.merge(subjects, movies_df, how="left", on="filename")

    # Warn users if original movies can't be traced
    # back using the subject information
    if subjects.movie_id.isnull().values.any():
        # Store the classifications that don't have movie info
        movie_missing = subjects[subjects.movie_id.isnull()]

        # Report the number of classifications retrieved
        logging.info(
            f"The original movies of {movie_missing.id.nunique():,}"
            f" subjects couldn't be traced back, ensure the subject information"
            f" related to the orignal movie matches the information in the movies.csv,"
            #             f" The trouble subjects are: {movie_missing.id.unique()}"
            f" The trouble filename of the movies are: {movie_missing.filename.unique()}"
        )

    if subjects["subject_type"].value_counts().idxmax() == "clip":
        # Calculate the clip_end_time
        subjects["clip_end_time"] = (
            subjects["clip_start_time"] + subjects["clip_length"]
        )

    if subjects["subject_type"].value_counts().idxmax() == "frame":
        # Ensure only one value per expected species id
        # this value is not crucial as we use the labels
        # to process the actual classifications
        # Modify the DataFrame to retain only the first value from each list
        subjects["frame_exp_sp_id"] = subjects["frame_exp_sp_id"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
        )

    # Get the names required for the subject table
    from kso_utils.db_utils import get_column_names_db

    required_cols = list(get_column_names_db(db_connection, "subjects").values())

    # Convert lists to sets and find the difference
    difference = list(set(required_cols) - set(subjects.columns))

    if difference:
        logging.info(f"Creating empty values for the {difference} columns")
        # Create columns to match schema if they don't exist
        [subjects.__setitem__(i, subjects.get(i, np.nan)) for i in difference]

    # Set the subject columns in the right order
    subjects = subjects[required_cols]

    from kso_utils.db_utils import test_table, add_to_table, get_df_from_db_table

    # Test table validity
    test_table(subjects, "subjects", keys=["id"])

    # Add values to subjects
    add_to_table(
        conn=db_connection,
        table_name="subjects",
        values=[tuple(i) for i in subjects.values],
        num_fields=11,
    )

    # log how many subjects are in the db
    # Query id and subject type from the subjects table
    subjects_df = get_df_from_db_table(db_connection, "subjects")

    logging.debug(
        f"The subjects database has now a total of " f"{subjects_df.shape[0]} subjects"
    )


##########################
# Upload clips functions
##########################


def check_movies_uploaded_zoo(project: Project, db_connection, selected_movies: list):
    """
    This function takes in a movie name and a dictionary containing the path to the database and returns
    a boolean value indicating whether the movie has already been uploaded to Zooniverse

    :param project: the project object
    :param selected_movies: the name of the movie(s) you want to check
    :type selected_movies: list
    :param db_connection: SQL connection object
    """
    from kso_utils.db_utils import get_df_from_db_table

    # Query info about the clip subjects uploaded to Zooniverse from the db
    subjects_df = get_df_from_db_table(conn=db_connection, table_name="subjects")

    # Select only columns of interest
    subjects_df = subjects_df[
        [
            "id",
            "subject_type",
            "filename",
            "clip_start_time",
            "clip_end_time",
            "movie_id",
        ]
    ]

    # Select only clip subjects
    subjects_df = subjects_df[subjects_df["subject_type"] == "clip"]

    # Save the video filenames of the clips uploaded to Zooniverse
    clips_uploaded = subjects_df[
        subjects_df["filename"].apply(
            lambda x: any(movie in x for movie in selected_movies)
        )
    ]

    if clips_uploaded.empty:
        logging.info(f"{selected_movies} has not been uploaded to Zooniverse yet")
    else:
        logging.info(f"{clips_uploaded} clips have already been uploaded.")


# Function to extract the videos
def extract_clips(
    movie_path: str,
    clip_length: int,
    upl_second_i: int,
    output_clip_path: str,
    modification_details: dict,
    gpu_available: bool,
    remove_audio: bool = True,
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
    :param remove_audio: Boolen parameter specifying if remove the audio or not
    """
    # Ensure the clip doesn't already exist
    output_clip_path_check = Path(output_clip_path)

    if output_clip_path_check.exists():
        logging.info(
            f"The clip {output_clip_path} already exists, please remove it and run the code again."
        )
        pass

    from kso_utils.movie_utils import check_ffmpeg_availability

    # Ensure ffmpeg is installed
    if not check_ffmpeg_availability():
        raise FileNotFoundError(
            "ffmpeg is not found in your system's PATH. Please install or add its path to the PATH environment variable."
        )

    # Set up input and output default prompts
    input_path = movie_path
    output_path = str(output_clip_path)

    input_options = {}
    output_options = {}

    # Add GPU-related options if available
    if gpu_available:
        input_options["hwaccel"] = "cuda"
        output_options["c"] = "copy"

    # Add audio-related options if needed
    if remove_audio:
        output_options["an"] = None

    # Access modifications from clip_modification_widget
    if modification_details:
        # Set up modification
        for (
            modif_number,
            modification_details_dict,
        ) in modification_details.checks.items():
            if "filter" in modification_details_dict:
                output_options["vf"] = modification_details_dict["filter"]
            elif "b:v" in modification_details_dict:
                # Specify the select size (MB) of the clip
                output_options["b:v"] = modification_details_dict["b:v"]

                # add the cuda-compatible vcodec
                if gpu_available:
                    # Remove the copy format argument
                    del output_options["c"]
                    output_options["c:v"] = "h264_nvenc"
                    logging.info("Compressing using h264_nvenc")

    input_options["ss"] = upl_second_i
    input_options["t"] = clip_length

    # Run the ffmpeg clip extraction code
    try:
        ffmpeg.input(input_path, **input_options).output(
            output_path, **output_options
        ).run(overwrite_output=True)

    except ffmpeg.Error as e:
        logging.error("ffmpeg error occurred.")
        logging.error(f"stderr: {e}")
        if e.stdout is not None:
            logging.error(f"stdout: {e.stdout.decode('utf8')}")
        if e.stderr is not None:
            logging.error(f"stderr: {e.stderr.decode('utf8')}")
        raise e

    # Ensure the clip was extracted
    if not os.path.exists(output_clip_path):
        raise FileNotFoundError("The clip wasn't extracted")

    else:
        os.chmod(output_clip_path, 0o777)
        logging.info("Clip extracted successfully")


def create_clips(
    available_movies_df: pd.DataFrame,
    selected_movies: str,
    movies_paths: str,
    clip_selection,
    project: Project,
    modification_details: dict,
    gpu_available: bool,
    is_example: bool,
    pool_size: int = 4,
):
    """
    This function takes a movie and extracts clips from it

    :param available_movies_df: the dataframe with the movies that are available for the project
    :param selected_movies: the name(s) of the movie(s) you want to extract clips from
    :param movies_paths: the path(s) to the movie(s) you want to extract clips from
    :param clip_selection: a ClipSelection object
    :param project: the project object
    :param modification_details: a dictionary with the following keys:
    :param gpu_available: True or False, depending on whether you have a GPU available to use
    :param is_example: If True and there are clip modifications, two clips will be generated (original and modified).
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
    selected_movies_df = available_movies_df[
        available_movies_df["filename"] == selected_movies
    ].reset_index(drop=True)

    # Add the list of starting seconds to the df
    selected_movies_df["list_clip_start"] = list_clip_start

    # Use explode() to expand the list column
    clips_start_df = selected_movies_df.explode("list_clip_start")

    # Rename the exploded column
    clips_start_df.rename(columns={"list_clip_start": "upl_seconds"}, inplace=True)

    # Add the length of the clips to df (to keep track of the length of each uploaded clip)
    clips_start_df["clip_length"] = clip_length

    # Specify output path for zooniverse clip extraction
    if project.server == "SNIC":
        if Path("/mimer").exists():
            temp_path = "/mimer/NOBACKUP/groups/snic2021-6-9/tmp_dir"
        elif Path("/tmp").exists() and not Path("/mimer").exists():
            temp_path = "/tmp"
        else:
            logging.error("No suitable writable path found.")
            return
    else:
        temp_path = "."
    clips_folder = str(Path(temp_path, selected_movies + "_zooniverseclips"))

    # Set the filename of the clips
    clips_start_df["clip_filename"] = clips_start_df.apply(
        lambda row: f"{selected_movies}_clip_{row['upl_seconds']}_{clip_length}.mp4",
        axis=1,
    )

    # Set the path of the clips
    clips_start_df["clip_path"] = clips_start_df["clip_filename"].apply(
        lambda x: str(Path(clips_folder, x))
    )

    # Create the folder to store the videos if not exist
    if os.path.exists(clips_folder):
        shutil.rmtree(clips_folder)
    Path(clips_folder).mkdir(parents=True, exist_ok=True)
    # Recursively add permissions to folders created
    [os.chmod(root, 0o777) for root, dirs, files in os.walk(clips_folder)]

    # Add information on the modification of the clips
    clips_start_df["clip_modification_details"] = str(modification_details)

    if is_example and modification_details.children[0].value > 0:
        # Set the filename of the original clips to be compared against the modified
        clips_start_df["clip_example_original_filename"] = clips_start_df.apply(
            lambda row: f"{selected_movies}_clip_example_original_{row['upl_seconds']}_{clip_length}.mp4",
            axis=1,
        )

        # Set the path of the clips
        clips_start_df["clip_example_original_path"] = clips_start_df[
            "clip_example_original_filename"
        ].apply(lambda x: str(Path(clips_folder, x)))

        logging.info("Extracting original clips")
        # Read each movie and extract the original clips
        for index, row in tqdm(
            clips_start_df.iterrows(), total=clips_start_df.shape[0]
        ):
            # Extract the videos and store them in the folder
            extract_clips(
                movie_path=movies_paths,
                clip_length=clip_length,
                upl_second_i=row["upl_seconds"],
                output_clip_path=row["clip_example_original_path"],
                modification_details={},
                gpu_available=gpu_available,
            )
        logging.info("Original clips extracted")

    logging.info("Extracting clips")
    # Read each movie and extract the clips from the original videos
    for index, row in tqdm(clips_start_df.iterrows(), total=clips_start_df.shape[0]):
        extract_clips(
            movie_path=movies_paths,
            clip_length=clip_length,
            upl_second_i=row["upl_seconds"],
            output_clip_path=row["clip_path"],
            modification_details=modification_details,
            gpu_available=gpu_available,
        )
    logging.info("Clips successfully extracted")

    return clips_start_df


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
        (file_path, Path(file_path).stat().st_size / float(1 << 20))
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


def set_zoo_clip_metadata(
    project: Project,
    generated_clipsdf: pd.DataFrame,
    sitesdf: pd.DataFrame,
    moviesdf: pd.DataFrame,
):
    """
    This function updates the dataframe of clips to be uploaded with
    metadata about the site and project

    :param project: the project object
    :param generated_clipsdf: a df with the information of the clips to be uploaded
    :param sitesdf: a df with the information of the sites of the project
    :param moviesdf: a df with the information of the movies of the project
    :return: upload_to_zoo, sitename, created_on

    """

    # Add spyfish-specific info
    if project.Project_name in ["Spyfish_Aotearoa", "Spyfish_BOPRC"]:
        # Rename the site columns to match standard cols names
        sitesdf = sitesdf.rename(columns={"schema_site_id": "id", "SiteID": "siteName"})

    # Rename the id column to match generated_clipsdf
    sitesdf = sitesdf.rename(columns={"id": "site_id", "siteName": "#siteName"})

    # Combine site info to the generated_clips df
    if "site_id" in generated_clipsdf.columns:
        upload_to_zoo = generated_clipsdf.merge(sitesdf, on="site_id")
        sitename = upload_to_zoo["#siteName"].unique()[0]
    else:
        raise ValueError("Sites table empty. Perhaps try to rebuild the initial db.")

    # Rename columns to match schema names
    # (fields that begin with “#” or “//” will never be shown to volunteers)
    # (fields that begin with "!" will only be available for volunteers on the Talk section, after classification)
    upload_to_zoo = upload_to_zoo.rename(
        columns={
            "id": "movie_id",
            "created_on": "#created_on",
            "clip_length": "#clip_length",
            "filename": "#VideoFilename",
            "clip_modification_details": "#clip_modification_details",
        }
    )

    # Convert datetime to string to avoid JSON seriazible issues
    upload_to_zoo["#created_on"] = upload_to_zoo["#created_on"].astype(str)
    created_on = upload_to_zoo["#created_on"].unique()[0]

    # Select only relevant columns
    upload_to_zoo = upload_to_zoo[
        [
            "movie_id",
            "clip_path",
            "upl_seconds",
            "#clip_length",
            "#created_on",
            "#VideoFilename",
            "#siteName",
            "#clip_modification_details",
        ]
    ]

    # Add information about the type of subject
    upload_to_zoo["Subject_type"] = "clip"

    # Add spyfish-specific info
    if project.Project_name == "Spyfish_Aotearoa":
        # Rename columns to match schema names
        sitesdf = sitesdf.rename(
            columns={
                "LinkToMarineReserve": "!LinkToMarineReserve",
            }
        )

        # Select only relevant columns
        sitesdf = sitesdf[["!LinkToMarineReserve", "#siteName", "ProtectionStatus"]]

        # Include site info to the df
        upload_to_zoo = upload_to_zoo.merge(sitesdf, on="#siteName")

    if project.Project_name == "Koster_Seafloor_Obs":
        # Rename columns to match schema names
        sitesdf = sitesdf.rename(
            columns={
                "decimalLatitude": "#decimalLatitude",
                "decimalLongitude": "#decimalLongitude",
                "geodeticDatum": "#geodeticDatum",
                "countryCode": "#countryCode",
            }
        )

        # Select only relevant columns
        sitesdf = sitesdf[
            [
                "#siteName",
                "#decimalLatitude",
                "#decimalLongitude",
                "#geodeticDatum",
                "#countryCode",
            ]
        ]

        # Include site info to the df
        upload_to_zoo = upload_to_zoo.merge(
            sitesdf, left_on="#siteName", right_on="#siteName"
        )

    # Prevent NANs on any column
    if upload_to_zoo.isnull().values.any():
        logging.info(
            f"The following columns have NAN values {upload_to_zoo.columns[upload_to_zoo.isna().any()].tolist()}"
        )

    logging.info(f"The metadata for the {upload_to_zoo.shape[0]} subjects is ready.")

    return upload_to_zoo, sitename, created_on


def upload_clips_to_zooniverse(
    project: Project,
    upload_to_zoo: pd.DataFrame,
    sitename: str,
    created_on: str,
):
    """
    It takes a dataframe of clips and metadata, creates a new subject set, and uploads the clips to
    Zooniverse

    :param upload_to_zoo: the dataframe of clips to upload
    :param sitename: the name of the site you're uploading clips from
    :param created_on: the date the clips were created
    :param project: the project object
    """

    # Estimate the number of clips
    n_clips = upload_to_zoo.shape[0]

    # Create a new subject set to host the clips
    subject_set = SubjectSet()
    subject_set_name = "clips_" + sitename + "_" + str(int(n_clips)) + "_" + created_on
    subject_set.links.project = project.Zooniverse_number
    subject_set.display_name = subject_set_name
    subject_set.save()

    logging.info(f"{subject_set_name} subject set created")

    # Save the df as the subject metadata
    subject_metadata = upload_to_zoo.set_index("clip_path").to_dict("index")

    # Upload the clips to Zooniverse (with metadata)
    new_subjects = []

    logging.info("Uploading subjects to Zooniverse")
    for clip_path, metadata in tqdm(
        subject_metadata.items(), total=len(subject_metadata)
    ):
        # Create a subject
        subject = Subject()

        # Add project info
        subject.links.project = project.Zooniverse_number

        # Add location of clip
        subject.add_location(clip_path)

        # Add metadata
        subject.metadata.update(metadata)

        # Save subject info
        subject.save()
        new_subjects.append(subject)

    # Upload all subjects
    subject_set.add(new_subjects)

    logging.info("Subjects uploaded to Zooniverse")


##########################
# Upload frames functions
##########################
def extract_frames_for_zoo(
    project: Project,
    species_list: list,
    zoo_info: dict,
    agg_df: pd.DataFrame,
    db_connection,
    server_connection,
    n_frames_subject,
    subsample_up_to,
):
    """
    > This function allows you to choose a species of interest, and then it will fetch a random
    sample of frames from the database for that species

    :param n_frames_subject: number of frames to fetch per subject, defaults to 3
    :type n_frames_subject: int (optional)
    :param subsample_up_to: If you have a lot of frames for a given species, you can subsample them.
           This parameter controls how many subjects you want to subsample to, defaults to 100
    :type subsample_up_to: int (optional)
    """

    # Roadblock to check if species list is empty
    if len(species_list) == 0:
        raise ValueError(
            "No species were selected. Please select at least one species before continuing."
        )

    # Select only aggregated classifications of species of interest
    sp_agg_df = agg_df[agg_df["label"].isin(species_list)].reset_index(drop=True)

    # Subsample up to n subjects per label
    if sp_agg_df["label"].value_counts().max() > subsample_up_to:
        logging.info(
            f"Subsampling up to {subsample_up_to} subjects of the species selected"
        )
        # randomise the order and select up to n per group
        sp_agg_df = sp_agg_df.sample(frac=1).groupby("label").head(subsample_up_to)

    # Identify the second of the original movie when the species first appears
    sp_agg_df["first_seen_movie"] = (
        sp_agg_df["clip_start_time"] + sp_agg_df["first_seen"]
    )

    # Add information of the original movies associated with the subjects
    # (e.g. the movie that was clipped from)
    (
        movies_df,
        no_available_movies_df,
        no_info_movies_df,
    ) = movie_utils.retrieve_movie_info_from_server(
        project=project,
        server_connection=server_connection,
        db_connection=db_connection,
    )

    # Include movies' filepath and fps to the df
    comb_df = sp_agg_df.merge(movies_df, on="movie_id", suffixes=("", "_df2"))

    # Prevent trying to extract frames from movies that are not accessible
    if len(comb_df[~comb_df.exists]) > 0:
        logging.error(
            f"There are {len(comb_df) - comb_df.exists.sum()} out of"
            "{len(frames_df)} subjects with original movies that are not accessible"
        )

    # Get csv paths
    csv_paths = server_utils.download_init_csv(
        project, ["movies", "species", "photos", "surveys", "sites"], server_connection
    )

    # Create a commonName col based on the zoo label to match schema format
    comb_df["commonName"] = comb_df["label"]

    # Combine the aggregated clips and species dataframes
    comb_df = db_utils.add_db_info_to_df(
        project=project,
        conn=db_connection,
        df=comb_df,
        csv_paths=csv_paths,
        table_name="species",
        cols_interest="id, commonName, scientificName",
    )

    # Create a list with the frames to be extracted and save into frame_number column
    comb_df["frame_number"] = comb_df[["first_seen_movie", "fps"]].apply(
        lambda x: [
            int((x["first_seen_movie"] + j) * x["fps"]) for j in range(n_frames_subject)
        ],
        1,
    )

    # Reshape df to have each frame as rows
    lst_col = "frame_number"

    comb_df = pd.DataFrame(
        {
            col: np.repeat(comb_df[col].values, comb_df[lst_col].str.len())
            for col in comb_df.columns.difference([lst_col])
        }
    ).assign(**{lst_col: np.concatenate(comb_df[lst_col].values)})[
        comb_df.columns.tolist()
    ]

    # Drop unnecessary columns
    comb_df.drop(["subject_ids"], inplace=True, axis=1)

    # Check the frames haven't been uploaded to Zooniverse
    comb_df = check_frames_uploaded(db_connection, comb_df)

    # Specify the temp location to store the frames
    temp_frames_folder = "_".join(species_list) + "_frames/"
    # Make sure that folder names do not exceed a certain length
    if len(temp_frames_folder) > 260:
        curr = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        temp_frames_folder = f"{curr}_various_frames/"

    if project.server == "SNIC":
        # Specify volume allocated by SNIC
        if Path("/mimer").exists():
            folder_name = "/mimer/NOBACKUP/groups/snic2021-6-9/tmp_dir/frames"
        elif Path("/tmp").exists() and not Path("/mimer").exists():
            folder_name = "/tmp/frames"
        else:
            logging.error("No suitable writable path found.")
            return
        frames_folder = str(Path(folder_name, temp_frames_folder))
    else:
        frames_folder = temp_frames_folder
    #         if hasattr(project, "output_path"):
    #             mod_frames_folder = project.output_path + frames_folder

    # Extract the frames from the videos, store them in the temp location
    # and save the df with information about the frames in the projectprocessor
    generated_frames = movie_utils.extract_frames(
        project=project,
        server_connection=server_connection,
        df=comb_df,
        frames_folder=frames_folder,
    )
    return generated_frames


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


# Function to gather information of frames already uploaded to Zooniverse
def check_frames_uploaded(
    db_connection,
    frames_df: pd.DataFrame,
):
    from kso_utils.db_utils import get_df_from_db_table

    # Get info of the subjects uploaded to zooniverse from the db
    subjects_df = get_df_from_db_table(db_connection, "subjects")

    # Select only frame subjects
    subjects_df = subjects_df[subjects_df["subject_type"] == "frame"]

    if subjects_df.empty:
        return frames_df

    # Create a list of the species about to upload (scientificName)
    list_species = frames_df["scientificName"].unique()

    # Query id and sci. names from the species table
    species_df = get_df_from_db_table(db_connection, "species")[
        ["id", "scientificName"]
    ]

    # Rename columns to match subject df
    species_df = species_df.rename(columns={"id": "frame_exp_sp_id"})

    # Reference the expected species on the uploaded subjects
    subjects_df = pd.merge(
        subjects_df,
        species_df,
        how="left",
        on="frame_exp_sp_id",
    )

    # Select only those subjects that should have the species of interest
    uploaded_frames_df = subjects_df[subjects_df["scientificName"].isin(list_species)]

    uploaded_frames_df = uploaded_frames_df[
        ["movie_id", "frame_number", "frame_exp_sp_id"]
    ]

    # Filter out frames that have already been uploaded
    if (
        len(uploaded_frames_df) > 0
        and not uploaded_frames_df["frame_number"].isnull().any()
    ):
        logging.info(
            "There are some frames already uploaded in Zooniverse for the species selected. \
            Checking if those are the frames you are trying to upload"
        )
        # Ensure that frame_number is an integer
        uploaded_frames_df["frame_number"] = uploaded_frames_df["frame_number"].astype(
            int
        )
        frames_df["frame_number"] = frames_df["frame_number"].astype(int)
        merge_df = (
            pd.merge(
                frames_df,
                uploaded_frames_df,
                left_on=["movie_id", "frame_number"],
                right_on=["movie_id", "frame_number"],
                how="left",
                indicator=True,
            )["_merge"]
            == "both"
        )

        # Exclude frames that have already been uploaded
        # trunk-ignore(flake8/E712)
        frames_df = frames_df[~merge_df]
        if len(frames_df) == 0:
            logging.error("All of the frames you have selected are already uploaded.")
        else:
            logging.info(
                f"There are {len(frames_df)} frames with the species of interest not uploaded to Zooniverse yet.",
            )

    else:
        logging.info(
            "There are no frames uploaded in Zooniverse for the species selected."
        )

    return frames_df


# Function modify the frames
def modify_frames(
    project: Project,
    frames_to_upload_df: pd.DataFrame,
    species_i: list,
    modification_details: dict,
):
    if len(species_i) == 0:
        species_i = ["custom_species"]

    # Specify the folder to host the modified frames
    if project.server == "SNIC":
        # Specify volume allocated by SNIC
        if Path("/mimer").exists():
            folder_name = "/mimer/NOBACKUP/groups/snic2021-6-9/tmp_dir/frames"
        elif Path("/tmp").exists() and not Path("/mimer").exists():
            folder_name = "/tmp/frames"
        else:
            logging.error("No suitable writable path found.")
            return
        mod_frames_folder = Path(
            folder_name, "modified_" + "_".join(species_i) + "_frames"
        )
    else:
        mod_frames_folder = Path("modified_" + "_".join(species_i) + "_frames")
        if hasattr(project, "output_path"):
            mod_frames_folder = project.output_path / mod_frames_folder

    # Specify the path of the modified frames
    frames_to_upload_df["modif_frame_path"] = frames_to_upload_df["frame_path"].apply(
        lambda x: str(mod_frames_folder / Path(x).name)
    )

    # Remove existing modified clips
    if mod_frames_folder.exists():
        shutil.rmtree(mod_frames_folder)

    if len(modification_details.values()) > 0:
        # Save the modification details to include as subject metadata
        frames_to_upload_df["frame_modification_details"] = str(modification_details)

        # Create the folder to store the videos if not exist
        if not mod_frames_folder.exists():
            mod_frames_folder.mkdir(parents=True, exist_ok=True)
            # Recursively add permissions to folders created
            [os.chmod(root, 0o777) for root, _, _ in os.walk(mod_frames_folder)]

        # Modify the clips###
        # Read each clip and modify them (showing a progress bar)
        for index, row in tqdm(
            frames_to_upload_df.iterrows(), total=frames_to_upload_df.shape[0]
        ):
            if not row["modif_frame_path"].exists():
                # Set up input prompt
                init_prompt = f'ffmpeg.input("{row["frame_path"]}")'
                full_prompt = init_prompt
                # Set up modification
                for transform in modification_details.values():
                    if "filter" in transform:
                        mod_prompt = transform["filter"]
                        full_prompt += mod_prompt

                # Run the modification
                try:
                    logging.info(full_prompt)
                    eval(full_prompt).run(capture_stdout=True, capture_stderr=True)
                    row["modif_frame_path"].chmod(0o777)
                except ffmpeg.Error as e:
                    logging.info("stdout: {}", e.stdout.decode("utf8"))
                    logging.info("stderr: {}", e.stderr.decode("utf8"))
                    raise e

        logging.info("Frames modified successfully")

    else:
        # Save the modification details to include as subject metadata
        frames_to_upload_df["modif_frame_path"] = frames_to_upload_df["frame_path"]

    return frames_to_upload_df


# Function to set the metadata of the frames to be uploaded to Zooniverse
def set_zoo_frame_metadata(
    project: Project,
    db_connection,
    df: pd.DataFrame,
    species_list: list,
    csv_paths: dict,
):
    """
    It takes a dataframe of clips or frames, and adds metadata about the site and project to it

    :param df: the dataframe with the media to upload
    :param project: the project object
    :param species_list: a list of the species that should be on the frames
    :param csv_paths: a dictionary with the paths of the csvs used to initiate the db
    :return: upload_to_zoo, sitename, created_on
    """
    project_name = project.Project_name

    if not isinstance(df, pd.DataFrame):
        df = df.df

    if (
        "modif_frame_path" in df.columns
        and "no_modification" not in df["modif_frame_path"].values
    ):
        df["frame_path"] = df["modif_frame_path"].apply(str)

    # Roadblock to prevent uploading frames to template project:
    if project.Zooniverse_number == 9754:
        raise ValueError(
            "You are not allowed to upload frames to the template Zooniverse project."
        )

    # Set project-specific metadata
    if project.Zooniverse_number == 9747:
        df = add_db_info_to_df(
            project, db_connection, csv_paths, df, "sites", "id, siteName"
        )
        upload_to_zoo = df[
            [
                "frame_path",
                "frame_number",
                "species_id",
                "movie_id",
                "created_on",
                "siteName",
            ]
        ]

    elif project_name == "SGU":
        upload_to_zoo = df[["frame_path", "species_id", "filename"]]

    elif project_name == "Spyfish_Aotearoa":
        from kso_utils.spyfish_utils import spyfish_subject_metadata

        upload_to_zoo = spyfish_subject_metadata(df, csv_paths=csv_paths)
    else:
        logging.error("This project is not a supported Zooniverse project.")

    # Add information about the type of subject
    upload_to_zoo = upload_to_zoo.copy()
    upload_to_zoo.loc[:, "subject_type"] = "frame"
    upload_to_zoo = upload_to_zoo.rename(columns={"species_id": "frame_exp_sp_id"})

    # Check there are no empty values (prevent issues uploading subjects)
    if upload_to_zoo.isnull().values.any():
        logging.error(
            "There are some values missing from the data you are trying to upload."
        )

    return upload_to_zoo


# Function to upload frames to Zooniverse
def upload_frames_to_zooniverse(
    project: Project,
    upload_to_zoo: pd.DataFrame,
    species_list: list,
):
    """
    It takes a dataframe of frames, and upload it to Zooniverse

    :param df: the dataframe with the media to upload
    :param project: the project object
    :param species_list: a list of the species that should be on the frames
    :return: upload_to_zoo, sitename, created_on
    """

    # Retireve zooniverse project name and number
    project_name = project.Project_name
    project_number = project.Zooniverse_number

    # Estimate the number of frames
    n_frames = upload_to_zoo.shape[0]

    if project_name == "Koster_Seafloor_Obs":
        created_on = upload_to_zoo["created_on"].unique()[0]
        sitename = upload_to_zoo["siteName"].unique()[0]

        # Name the subject set
        subject_set_name = (
            "frames_"
            + str(int(n_frames))
            + "_"
            + "_".join(species_list)
            + "_"
            + sitename
            + "_"
            + created_on
        )

    elif project_name == "SGU":
        surveys_df = pd.read_csv(project.csv_paths["local_surveys_csv"])
        created_on = surveys_df["SurveyDate"].unique()[0]
        folder_name = Path(upload_to_zoo["frame_path"].iloc[0]).parent.name
        sitename = folder_name

        # Name the subject set
        subject_set_name = (
            "frames_"
            + str(int(n_frames))
            + "_"
            + "_".join(species_list)
            + "_"
            + sitename
            + "_"
            + created_on
        )

    elif project_name == "Spyfish_Aotearoa":
        reserve = upload_to_zoo["!LinkToMarineReserve"].unique()[0]

        # Name the subject for frames from multiple sites/movies
        subject_set_name = (
            "frames_"
            + str(int(n_frames))
            + "_"
            + reserve
            + "_"
            + "_".join(species_list)
            + datetime.date.today().strftime("_%d_%m_%Y")
        )

    else:
        sitename = upload_to_zoo["siteName"].unique()[0]

        # Name the subject for frames from multiple sites/movies
        subject_set_name = (
            "frames_"
            + str(int(n_frames))
            + "_"
            + sitename
            + "_"
            + "_".join(species_list)
            + datetime.date.today().strftime("_%d_%m_%Y")
        )

    # Create a new subject set to host the frames
    subject_set = SubjectSet()
    subject_set.links.project = project_number
    subject_set.display_name = subject_set_name
    subject_set.save()

    logging.info(f"{subject_set_name} subject set created")

    # Save the df as the subject metadata
    subject_metadata = upload_to_zoo.set_index("frame_path").to_dict("index")

    # Upload the clips to Zooniverse (with metadata)
    new_subjects = []

    logging.info("Uploading subjects to Zooniverse...")
    for frame_path, metadata in tqdm(
        subject_metadata.items(), total=len(subject_metadata)
    ):
        subject = Subject()

        subject.links.project = project_number
        subject.add_location(frame_path)

        logging.info(frame_path)
        subject.metadata.update(metadata)

        logging.info(metadata)
        subject.save()
        logging.info("Subject saved")
        new_subjects.append(subject)

    # Upload videos
    subject_set.add(new_subjects)
    logging.info("Subjects uploaded to Zooniverse")
