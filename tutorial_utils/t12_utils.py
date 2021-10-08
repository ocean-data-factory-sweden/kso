import pandas as pd
import numpy as np
import json, io
from ast import literal_eval
from tutorial_utils.zooniverse_utils import auth_session
import tutorial_utils.db_utils as db_utils
from tutorial_utils.koster_utils import filter_bboxes, process_clips_koster
from tutorial_utils.spyfish_utils import process_clips_spyfish
from tutorial_utils import db_utils
from collections import OrderedDict, Counter
from IPython.display import HTML, display, update_display, clear_output
import ipywidgets as widgets
from ipywidgets import interact
import asyncio
from itables import show


def choose_project():
    
    # Specify location of the latest list of projects
    projects_csv = "../db_starter/projects_list.csv" 
    
    # Read the latest list of projects
    projects_df = pd.read_csv(projects_csv)
    
    # Display the project options
    choose_project = widgets.Dropdown(
        options=projects_df.Project_name.unique().tolist(),
        value=projects_df.Project_name.unique().tolist()[0],
        description="Project:",
        disabled=False,
    )
    
    display(choose_project)
    return choose_project


def choose_agg_parameters(subject_type: str):
    agg_users = widgets.FloatSlider(
        value=0.8,
        min=0,
        max=1.0,
        step=0.1,
        description="Aggregation threshold:",
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
    display(agg_users)
    min_users = widgets.IntSlider(
        value=3,
        min=1,
        max=15,
        step=1,
        description="Min numbers of users:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(min_users)
    if subject_type == "frame":
        agg_obj = widgets.FloatSlider(
            value=0.8,
            min=0,
            max=1.0,
            step=0.1,
            description="Object threshold:",
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
        display(agg_obj)
        agg_iou = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.1,
            description="IOU Epsilon:",
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
        display(agg_iou)
        agg_iua = widgets.FloatSlider(
            value=0.8,
            min=0,
            max=1.0,
            step=0.1,
            description="Inter user agreement:",
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
        display(agg_iua)
        return agg_users, min_users, agg_obj, agg_iou, agg_iua
    else:
        return agg_users, min_users


def choose_workflows(workflows_df):

    layout = widgets.Layout(width="auto", height="40px")  # set width and height

    # Display the names of the workflows
    workflow_name = widgets.Dropdown(
        options=workflows_df.display_name.unique().tolist(),
        value=workflows_df.display_name.unique().tolist()[0],
        description="Workflow name:",
        disabled=False,
    )

    # Display the type of subjects
    subj_type = widgets.Dropdown(
        options=["frame", "clip"],
        value="clip",
        description="Subject type:",
        disabled=False,
    )

    display(workflow_name)
    display(subj_type)

    return workflow_name, subj_type


def choose_w_version(workflows_df, workflow_id):

    layout = widgets.Layout(width="auto", height="40px")  # set width and height

    # Estimate the versions of the workflow available
    versions_available = workflows_df[workflows_df.display_name==workflow_id].version.unique().tolist()

    # Display the versions of the workflow available
    w_version = widgets.Dropdown(
        options=list(map(float, versions_available)),
        value=float(versions_available[0]),
        description="Minimum workflow version:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )

    display(w_version)
    return w_version


def get_classifications(
    workflow_id: int, workflow_version: float, subj_type, class_df, db_path
):
    
    # Filter classifications of interest
    class_df = class_df[
        (class_df.workflow_id == workflow_id)
        & (class_df.workflow_version >= workflow_version)
    ].reset_index(drop=True)
    
    
    # Add information about the subject
    # Create connection to db
    conn = db_utils.create_connection(db_path)
    
    if subj_type == "frame":
        # Query id and subject type from the subjects table
        subjects_df = pd.read_sql_query("SELECT id, subject_type, https_location, frame_number, movie_id FROM subjects", conn)
        
    else:
        # Query id and subject type from the subjects table
        subjects_df = pd.read_sql_query("SELECT id, subject_type, https_location, clip_start_time, movie_id FROM subjects", conn)
    
    # Ensure id format matches classification's subject_id
    class_df["subject_ids"] = class_df["subject_ids"].astype('Int64')
    subjects_df["id"] = subjects_df["id"].astype('Int64')
    
    
    # Add subject information based on subject_ids
    class_df = pd.merge(
        class_df,
        subjects_df,
        how="left",
        left_on="subject_ids",
        right_on="id",
    )
    
    if class_df[["subject_type", "https_location","movie_id"]].isna().any().any():
        # Exclude classifications from missing subjects
        filtered_class_df = class_df.dropna(subset=["subject_type",
                                                    "https_location",
                                                    "movie_id"], 
                                            how='any').reset_index(drop=True)
        
        # Report on the issue
        print("There are", 
              (class_df.shape[0]-filtered_class_df.shape[0]), 
              "classifications out of",
              class_df.shape[0],
              "missing subject info. Maybe the subjects have been removed from Zooniverse?")
        
        class_df = filtered_class_df
        
    
    print("Zooniverse classifications have been retrieved")

    return class_df


def aggregrate_labels(raw_class_df, agg_users, min_users):
    # Calculate the number of users that classified each subject
    raw_class_df["n_users"] = raw_class_df.groupby("subject_ids")[
        "classification_id"
    ].transform("nunique")

    # Select classifications with at least n different user classifications
    raw_class_df = raw_class_df[raw_class_df.n_users >= min_users].reset_index(drop=True)

    # Calculate the proportion of unique classification (it can have multiple annotations) per subject
    raw_class_df["class_n"] = raw_class_df.groupby(["subject_ids", "label"])[
        "classification_id"
    ].transform("count")
    
    # Calculate the proportion of users that agreed on their annotations
    raw_class_df["class_prop"] = raw_class_df.class_n / raw_class_df.n_users
    
    # Select annotations based on agreement threshold
    agg_class_df = raw_class_df[raw_class_df.class_prop >= agg_users].reset_index(drop=True)
    
    return agg_class_df


def aggregrate_classifications(df, subj_type, project_name: str, agg_params):

    print("Aggregrating the classifications")
    if subj_type == "frame":
        # Get the aggregration parameters
        agg_users, min_users, agg_obj, agg_iou, agg_iua = [i.value for i in agg_params]
        
        # Process the raw classifications
        raw_class_df = process_frames(df, project_name)
        
        # Aggregrate frames based on their labels
        agg_labels_df = aggregrate_labels(raw_class_df, agg_users, min_users)
        
        # Select frames aggregrated as empty
        agg_labels_df_empty = agg_labels_df[agg_labels_df["label"]=="empty"]
        agg_labels_df_empty = agg_labels_df_empty.rename(columns={'frame_number': 'start_frame'})
        agg_labels_df_empty = agg_labels_df_empty[[
                "label",
                "subject_ids",
                "x",
                "y",
                "w",
                "h",
            ]]
        
        # Exclude empty frames
        agg_labels_df = agg_labels_df[agg_labels_df["label"]!="empty"]
        
        # Map the position of the annotation parameters
        col_list = list(agg_labels_df.columns)
        x_pos, y_pos, w_pos, h_pos, user_pos, subject_id_pos = (
            col_list.index("x"),
            col_list.index("y"),
            col_list.index("w"),
            col_list.index("h"),
            col_list.index("user_name"),
            col_list.index("subject_ids"),
        )

        # Get prepared annotations
        new_rows = []
        
        for name, group in agg_labels_df.groupby(["movie_id", "label", "frame_number"]):
            movie_id, label, start_frame = name

            total_users = agg_labels_df[
                (agg_labels_df.movie_id == movie_id)
                & (agg_labels_df.label == label)
                & (agg_labels_df.frame_number == start_frame)
            ]["user_name"].nunique()
            
            # Filter bboxes using IOU metric (essentially a consensus metric)
            # Keep only bboxes where mean overlap exceeds this threshold
            indices, new_group = filter_bboxes(
                total_users=total_users,
                users=[i[user_pos] for i in group.values],
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
                        movie_id,
                        label,
                        start_frame,
                        ix,
                    )
                    + tuple(box)
                )

        agg_class_df = pd.DataFrame(
            new_rows,
            columns=[
                "movie_id",
                "label",
                "start_frame",
                "subject_ids",
                "x",
                "y",
                "w",
                "h",
            ],
        )

        agg_class_df["subject_type"] = "frame"
        
        # Add the empty frames
        agg_class_df = pd.concat([agg_class_df,agg_labels_df_empty])
        
        # Select the aggregated labels
        agg_class_df = agg_class_df[["subject_ids", "label", "x", "y", "w", "h"]].drop_duplicates()
        
        # Add the http info
        agg_class_df =  pd.merge(
        agg_class_df,
        raw_class_df[["subject_ids","https_location","subject_type", "movie_id"]].drop_duplicates(),
        how="left",
        on="subject_ids"
    )
  
    else:
        # Get the aggregration parameters
        agg_users, min_users = [i.value for i in agg_params]
        
        # Process the raw classifications
        raw_class_df = process_clips(df, project_name)
        
        # Aggregrate clips based on their labels
        agg_class_df = aggregrate_labels(raw_class_df, agg_users, min_users)
        
        # Extract the median of the second where the animal/object is and number of animals
        agg_class_df = agg_class_df.groupby(["subject_ids", "https_location", "subject_type", "label"], as_index=False)
        agg_class_df = pd.DataFrame(agg_class_df[["how_many", "first_seen"]].median())

    # Add username info to raw class
    raw_class_df = pd.merge(
        raw_class_df,
        df[["classification_id","user_name"]],
        how="left",
        on="classification_id"
    )
    
    print(agg_class_df.shape[0], "classifications aggregated out of", df.subject_ids.nunique(), "unique subjects available")
    
    return agg_class_df, raw_class_df


def process_clips(df: pd.DataFrame, project_name):

    # Create an empty list
    rows_list = []

    # Loop through each classification submitted by the users
    for index, row in df.iterrows():
        # Load annotations as json format
        annotations = json.loads(row["annotations"])

        # Select the information from the species identification task
        if project_name == "Koster Seafloor Obs":
            rows_list = process_clips_koster(annotations, row["classification_id"], rows_list)
            
        # Check if the Zooniverse project is the Spyfish
        if project_name == "Spyfish Aotearoa":
            rows_list = process_clips_spyfish(annotations, row["classification_id"], rows_list)

    # Create a data frame with annotations as rows
    annot_df = pd.DataFrame(
        rows_list, columns=["classification_id", "label", "first_seen", "how_many"]
    )
    
    # Specify the type of columns of the df
    annot_df["how_many"] = pd.to_numeric(annot_df["how_many"])
    annot_df["first_seen"] = pd.to_numeric(annot_df["first_seen"])

    # Add subject id to each annotation
    annot_df = pd.merge(
        annot_df,
        df.drop(columns=["annotations"]),
        how="left",
        on="classification_id",
    )
    
    #Select only relevant columns
    annot_df = annot_df[
        [
            "classification_id",
            "label",
            "how_many", 
            "first_seen",
            "https_location",
            "subject_type",
            "subject_ids",
        ]
    ]
    
    return pd.DataFrame(annot_df)


def process_frames(df: pd.DataFrame, project_name):

    # Create an empty list
    rows_list = []
    
    # Loop through each classification submitted by the users and flatten them
    for index, row in df.iterrows():
        # Load annotations as json format
        annotations = json.loads(row["annotations"])

        # Select the information from all the labelled animals (e.g. task = T0)
        for ann_i in annotations:
            if ann_i["task"] == "T0":

                if not ann_i["value"]:
                    # Specify the frame was classified as empty
                    choice_i = {
                            "classification_id": row["classification_id"],
                            # If value_i is not empty flatten labels
                            "x": None,
                            "y": None,
                            "w": None,
                            "h": None,
                            "label": "empty",
                        }
                
                else:
                    # Select each species annotated and flatten the relevant answers
                    for i in ann_i["value"]:
                        choice_i = {
                            "classification_id": row["classification_id"],
                            # If value_i is not empty flatten labels
                            "x": int(i["x"]) if "x" in i else None,
                            "y": int(i["y"]) if "y" in i else None,
                            "w": int(i["width"]) if "width" in i else None,
                            "h": int(i["height"]) if "height" in i else None,
                            "label": str(i["tool_label"]) if "tool_label" in i else None,
                        }
                        

                rows_list.append(choice_i)

    # Create a data frame with annotations as rows
    flat_annot_df = pd.DataFrame(
        rows_list, columns=["classification_id", "x", "y", "w", "h", "label"]
    )

    
    # Add other classification information to the flatten classifications
    annot_df = pd.merge(
        flat_annot_df,
        df,
        how="left",
        on="classification_id",
    )

    
    #Select only relevant columns
    annot_df = annot_df[
        [
            "classification_id",
            "x", "y", "w", "h", 
            "label",
            "https_location",
            "subject_type",
            "subject_ids",
            "frame_number",
            "user_name",
            "movie_id"
        ]
    ]
    
    return pd.DataFrame(annot_df)


def view_subject(subject_id: int,  class_df: pd.DataFrame, subject_type: str):
    try:

        subject_location = class_df[class_df.subject_ids == subject_id]["https_location"].unique()[0]
    except:
        raise Exception("The reference data does not contain media for this subject.")
    if not subject_location:
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
        <div>{class_df[class_df.subject_ids == subject_id][['label','first_seen','how_many']].value_counts().sort_values(ascending=False).to_frame().to_html()}</div>
        </div>
        </html>"""
    elif subject_type == "frame":
        html_code = f"""
        <html>
        <div style="display: flex; justify-content: space-around">
        <div>
          <img src={subject_location} type="image/jpeg" width=500>
        </img>
        </div>
        <div>{class_df[class_df.subject_ids == subject_id]['label'].value_counts().sort_values(ascending=False).to_frame().to_html()}</div>
        </div>
        </html>"""
    else:
        Exception("Subject type not supported.")
    return HTML(html_code)


def launch_viewer(class_df: pd.DataFrame, subject_type: str):

    # Select the subject
    options = tuple(class_df[class_df["subject_type"] == subject_type]["subject_ids"].apply(int).apply(str).unique())
    subject_widget = widgets.Combobox(
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
                
                
    subject_widget.observe(on_change, names='value')


def explore_classifications_per_subject(class_df: pd.DataFrame, subject_type):

    # Select the subject
    subject_widget = widgets.Combobox(
                    options= tuple(class_df.subject_ids.apply(int).apply(str).unique()),
                    description="Subject id:",
                    ensure_option=True,
                    disabled=False,
                )
    
    main_out = widgets.Output()
    display(subject_widget, main_out)
    
    # Display the subject and classifications on change
    def on_change(change):
        with main_out:
            a = class_df[class_df.subject_ids==int(change["new"])]
            if subject_type == "clip":
                a = a[['classification_id', 'user_name', 'label', 'how_many', 'first_seen']]
            else:
                a = a[["x", "y", "w", "h", 
            "label",
            "https_location",
            "subject_ids",
            "frame_number",
            "movie_id"]]
            clear_output()
            show(a)
                
                
    subject_widget.observe(on_change, names='value')
