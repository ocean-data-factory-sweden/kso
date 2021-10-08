import pandas as pd
import numpy as np
import json, io
from ast import literal_eval
from utils.zooniverse_utils import auth_session
from db_setup.process_frames import filter_bboxes
from utils import db_utils
from collections import OrderedDict
from IPython.display import HTML, display, update_display, clear_output
import ipywidgets as widgets

def get_exports(user, password):
    
    # Connect to the Zooniverse project
    project = auth_session(user, password)

    # Get the classifications from the project
    c_export = project.get_export("classifications")
    s_export = project.get_export("subjects")

    # Save the response as pandas data frame
    class_df = pd.read_csv(
        io.StringIO(c_export.content.decode("utf-8")),
        usecols=[
            "user_name",
            "subject_ids",
            "subject_data",
            "classification_id",
            "workflow_id",
            "workflow_version",
            "created_at",
            "annotations",
        ],
    )
                

    subjects_df = pd.read_csv(
        io.StringIO(s_export.content.decode("utf-8")),
    )
                
    total_df = pd.merge(class_df, subjects_df[["subject_id", "workflow_id", "locations"]], 
               left_on=['subject_ids', "workflow_id"], right_on=["subject_id", "workflow_id"])
                
    total_df["locations"] = total_df["locations"].apply(lambda x: literal_eval(x)["0"])
    
    return total_df, class_df


def process_clips(df: pd.DataFrame, class_df: pd.DataFrame, workflow_id: int, workflow_version: float):
    df = df[(df.workflow_id == workflow_id) & (df.workflow_version >= workflow_version)].reset_index()
    # Create an empty list
    rows_list = []

    # Loop through each classification submitted by the users
    for index, row in df.iterrows():
        # Load annotations as json format
        annotations = json.loads(row["annotations"])

        # Select the information from the species identification task
        for ann_i in annotations:
            if ann_i["task"] == "T4":

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
                            if "FIRSTTIME" in k:
                                f_time = answers[k].replace("S", "")
                            if "INDIVIDUAL" in k:
                                inds = answers[k]
                            elif "FIRSTTIME" not in k and "INDIVIDUAL" not in k:
                                f_time, inds = None, None

                    # Save the species of choice, class and subject id
                    choice_i.update(
                        {
                            "classification_id": row["classification_id"],
                            "label": value_i["choice"],
                            "first_seen": f_time,
                            "how_many": inds,
                        }
                    )

                    rows_list.append(choice_i)

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
        class_df.drop(columns=["annotations"]),
        how="left",
        on="classification_id",
    )

    # Clear duplicated subjects
    #if args.duplicates_file_id:
    #    annot_df = db_utils.combine_duplicates(annot_df, args.duplicates_file_id)

    # Calculate the number of users that classified each subject
    annot_df["n_users"] = annot_df.groupby("subject_ids")[
        "classification_id"
    ].transform("nunique")
    
    annot_df['retired'] = annot_df["subject_data"].apply(lambda x: [v["retired"] for k,v in json.loads(x).items()][0])
    
    return annot_df


def process_frames(db_path: str, df: pd.DataFrame, workflow_id: int, workflow_version: float, duplicates_file_id: str,
                   n_users: int = 5,
                   object_thresh: float = 0.8, iou_epsilon: float = 0.5, inter_user_agreement: float = 0.5):
    
    # Filter w2 classifications
    w2_data = df[
        (df.workflow_id == workflow_id)
        & (df.workflow_version >= workflow_version)
    ].reset_index()

    # Clear duplicated subjects
    if duplicates_file_id:
        w2_data = db_utils.combine_duplicates(w2_data, duplicates_file_id)

    #Drop NaN columns
    w2_data = w2_data.drop(['dupl_subject_id', 'single_subject_id'], 1)

    ## Check if subjects have been uploaded
    # Get species id for each species
    conn = db_utils.create_connection(db_path)
    
    # Calculate the number of users that classified each subject
    w2_data["n_users"] = w2_data.groupby("subject_ids")["classification_id"].transform(
        "nunique"
    )

    # Select frames with at least n different user classifications
    w2_data = w2_data[w2_data.n_users >= n_users]

    # Drop workflow and n_users columns
    w2_data = w2_data.drop(
        columns=[
            "workflow_id",
            "workflow_version",
            "n_users",
            "created_at",
            "subject_id",
            "locations",
        ]
    )
    
    # Extract the video filename and annotation details
    subject_data_df = pd.DataFrame(
        w2_data["subject_data"]
        .apply(
            lambda x: [
                {
                    "movie_id": v["movie_id"],
                    "frame_number": v["frame_number"],
                    "label": v["label"],
                }
                for k, v in json.loads(x).items()  # if v['retired']
            ][0],
            1,
        )
        .tolist()
    )

    w2_data = pd.concat(
        [w2_data.reset_index().drop("index", 1), subject_data_df],
        axis=1,
        ignore_index=True,
    )
    
    w2_data = w2_data[w2_data.columns[1:]]

    w2_data.columns = [
        "classification_id",
        "user_name",
        "annotations",
        "subject_data",
        "subject_ids",
        "movie_id",
        "frame_number",
        "label",
    ]

    movies_df = pd.read_sql_query("SELECT id, filename FROM movies", conn)
    movies_df = movies_df.rename(columns={"id": "movie_id"})
    
    w2_data = pd.merge(w2_data, movies_df, how="left", on="movie_id")

    # Convert to dictionary entries
    w2_data["movie_id"] = w2_data["movie_id"].apply(lambda x: {"movie_id": x})
    w2_data["frame_number"] = w2_data["frame_number"].apply(
        lambda x: {"frame_number": x}
    )
    w2_data["label"] = w2_data["label"].apply(lambda x: {"label": x})
    w2_data["user_name"] = w2_data["user_name"].apply(lambda x: {"user_name": x})
    w2_data["subject_id"] = w2_data["subject_ids"].apply(lambda x: {"subject_id": x})
    
    
    w2_data["annotation"] = w2_data["annotations"].apply(
        lambda x: literal_eval(x)[0]["value"], 1
    )

    # Extract annotation metadata
    w2_data["annotation"] = w2_data[
        ["movie_id", "frame_number", "label", "annotation", "user_name", "subject_id"]
    ].apply(
        lambda x: [
            OrderedDict(
                list(x["movie_id"].items())
                + list(x["frame_number"].items())
                + list(x["label"].items())
                + list(x["annotation"][i].items())
                + list(x["user_name"].items())
                + list(x["subject_id"].items())
            )
            for i in range(len(x["annotation"]))
        ]
        if len(x["annotation"]) > 0
        else [
            OrderedDict(
                list(x["movie_id"].items())
                + list(x["frame_number"].items())
                + list(x["label"].items())
                + list(x["user_name"].items())
                + list(x["subject_id"].items())
            )
        ],
        1,
    )

    # Convert annotation to format which the tracker expects
    ds = [
        OrderedDict(
            {
                "user": i["user_name"],
                "movie_id": i["movie_id"],
                "label": i["label"],
                "start_frame": i["frame_number"],
                "x": int(i["x"]) if "x" in i else None,
                "y": int(i["y"]) if "y" in i else None,
                "w": int(i["width"]) if "width" in i else None,
                "h": int(i["height"]) if "height" in i else None,
                "subject_id": int(i["subject_id"]) if "subject_id" in i else None,
            }
        )
        for i in w2_data.annotation.explode()
        if i is not None and i is not np.nan
    ]

    # Get prepared annotations
    w2_full = pd.DataFrame(ds)
    w2_annotations = w2_full[w2_full["x"].notnull()]

    new_rows = []
    final_indices = []
    for name, group in w2_annotations.groupby(["movie_id", "label", "start_frame"]):
        movie_id, label, start_frame = name

        total_users = w2_full[
            (w2_full.movie_id == movie_id)
            & (w2_full.label == label)
            & (w2_full.start_frame == start_frame)
        ]["user"].nunique()

        # Filter bboxes using IOU metric (essentially a consensus metric)
        # Keep only bboxes where mean overlap exceeds this threshold
        indices, new_group = filter_bboxes(
            total_users=total_users,
            users=[i[0] for i in group.values],
            bboxes=[np.array((i[4], i[5], i[6], i[7])) for i in group.values],
            obj=object_thresh,
            eps=iou_epsilon,
            iua=inter_user_agreement,
        )

        subject_ids = [i[8] for i in group.values[indices]]

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

    w2_annotations = pd.DataFrame(
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

    # Filter out invalid movies
    w2_annotations = w2_annotations[w2_annotations["movie_id"].notnull()][
        ["label", "x", "y", "w", "h", "subject_ids"]
    ]

    return w2_annotations

def view_subject(subject_id: int, df: pd.DataFrame, annot_df: pd.DataFrame):
    try:
        subject_location = df[df.subject_id == subject_id]["locations"].iloc[0]
    except:
        raise Exception("The reference data does not contain media for this subject.")
    if len(annot_df[annot_df.subject_ids == subject_id]) == 0: 
        raise Exception("Subject not found in provided annotations")
       
    
    # Get the HTML code to show the selected subject
    if ".mp4" in subject_location:
        html_code =f"""
        <html>
        <div style="display: flex; justify-content: space-around">
        <div>
          <video width=500 controls>
          <source src={subject_location} type="video/mp4">
        </video>
        </div>
        <div>{annot_df[annot_df.subject_ids == subject_id]['label'].value_counts().sort_values(ascending=False).to_frame().to_html()}</div>
        </div>
        </html>"""
    else:
        html_code =f"""
        <html>
        <div style="display: flex; justify-content: space-around">
        <div>
          <img src={subject_location} type="image/jpeg" width=500>
        </img>
        </div>
        <div>{annot_df[annot_df.subject_ids == subject_id]['label'].value_counts().sort_values(ascending=False).to_frame().to_html()}</div>
        </div>
        </html>"""
    return HTML(html_code)


def launch_viewer(total_df: pd.DataFrame, clips_df: pd.DataFrame, frames_df: pd.DataFrame):
    
    v = widgets.ToggleButtons(
        options=['Frames', 'Clips'],
        description='Subject type:',
        disabled=False,
        button_style='success',
    )

    subject_df = clips_df

    def on_tchange(change):
        global subject_df
        with main_out:
            if change['type'] == 'change' and change['name'] == 'value':
                if change['new'] == "Frames":
                    subject_df = frames_df
                else:
                    subject_df = clips_df
                clear_output()
                w = widgets.Dropdown(
                    options=subject_df.subject_ids.unique().tolist(),
                    value=subject_df.subject_ids.unique().tolist()[0],
                    description='Subject id:',
                    disabled=False,
                )
                w.observe(on_change)
                display(w)
                global out
                out = widgets.Output()
                display(out)

    def on_change(change):
        global subject_df
        with out:
            if change['type'] == 'change' and change['name'] == 'value':
                a = view_subject(change['new'], total_df, subject_df)
                clear_output()
                display(a)

    v.observe(on_tchange)
    display(v)
    main_out = widgets.Output()
    display(main_out)