# base imports
import logging
import os
import random
import subprocess
import pandas as pd
import numpy as np
import cv2
import math

# widget imports
import ipysheet
import folium
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import HTML, display, clear_output
from ipywidgets import interactive, Layout
from folium.plugins import MiniMap
from pathlib import Path
import asyncio

# util imports
from kso_utils.video_reader import VideoReader
from kso_utils.project_utils import Project
import kso_utils.movie_utils as movie_utils
from kso_utils.db_utils import create_connection
import kso_utils.tutorials_utils as t_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

######################################################################
############Functions for interactive widgets#########################
######################################################################


# Function to update widget based on user interaction (eg. click)
def wait_for_change(widget1: widgets.Widget, widget2: widgets.Widget):
    future = asyncio.Future()

    def getvalue(change):
        future.set_result(change.description)
        widget1.on_click(getvalue, remove=True)
        widget2.on_click(getvalue, remove=True)

    widget1.on_click(getvalue)
    widget2.on_click(getvalue)
    return future


def single_wait_for_change(widget, value):
    future = asyncio.Future()

    def getvalue(change):
        future.set_result(change.new)
        widget.unobserve(getvalue, value)

    widget.observe(getvalue, value)
    return future


######################################################################
#####################Common widgets###################################
######################################################################


def choose_species(db_connection, species_list=None):
    """
    This function generates a widget to select the species of interest
    :param project: the project object

    """
    from kso_utils.db_utils import get_df_from_db_table

    if species_list is None:
        # Get a list of the species available from the db
        species_df = get_df_from_db_table(db_connection, "species")

        species_list = species_df["commonName"].unique().tolist()

    # Roadblock to check if species list is empty
    if len(species_list) == 0:
        raise ValueError(
            "Your database contains no species, please add at least one species before continuing."
        )

    # Generate the widget
    w = widgets.SelectMultiple(
        options=species_list,
        value=[species_list[0]],
        description="Species",
        disabled=False,
    )

    display(w)
    return w


def choose_project(
    projects_csv: str = "../kso_utils/kso_utils/db_starter/projects_list.csv",
):
    """
    > This function takes a csv file with a list of projects and returns a dropdown menu with the
    projects listed

    :param projects_csv: str = "../kso_utils/kso_utils/db_starter/projects_list.csv", defaults to ../kso_utils/db_starter/projects_list.csv
    :type projects_csv: str (optional)
    :return: A dropdown widget with the project names as options.
    """

    # Check path to the list of projects is a csv
    if os.path.exists(projects_csv) and not projects_csv.endswith(".csv"):
        logging.error("A csv file was not selected. Please try again.")

    # If list of projects doesn't exist retrieve it from github
    if not os.path.exists(projects_csv):
        projects_csv = "https://github.com/ocean-data-factory-sweden/kso_utils/blob/main/kso_utils/db_starter/projects_list.csv?raw=true"

    projects_df = pd.read_csv(projects_csv)

    if "Project_name" not in projects_df.columns:
        logging.error(
            "We were unable to find any projects in that file, \
                      please choose a projects csv file that matches our template."
        )

    # Display the project options
    choose_project = widgets.Dropdown(
        options=projects_df.Project_name.unique().tolist(),
        value=projects_df.Project_name.unique().tolist()[0],
        description="Project:",
        disabled=False,
    )

    display(choose_project)
    return choose_project


def gpu_select():
    """
    If the user selects "No GPU", then the function will return a boolean value of False. If the user
    selects "Colab GPU", then the function will install the GPU requirements and return a boolean value
    of True. If the user selects "Other GPU", then the function will return a boolean value of True
    :return: The gpu_available variable is being returned.
    """

    def gpu_output(gpu_option):
        if gpu_option == "No GPU":
            logging.info("You are set to start the modifications")
            # Set GPU argument
            gpu_available = False
            return gpu_available

        if gpu_option == "Colab GPU":
            # Install the GPU requirements
            if not os.path.exists("./colab-ffmpeg-cuda/bin/."):
                try:
                    logging.info(
                        "Installing the GPU requirements. PLEASE WAIT 10-20 SECONDS"
                    )  # Install ffmpeg with GPU version
                    subprocess.check_call(
                        "git clone https://github.com/fritolays/colab-ffmpeg-cuda.git",
                        shell=True,
                    )
                    subprocess.check_call(
                        "cp -r ./colab-ffmpeg-cuda/bin/. /usr/bin/", shell=True
                    )
                    logging.info("GPU Requirements installed!")

                except subprocess.CalledProcessError as e:
                    logging.error(
                        f"There was an issues trying to install the GPU requirements, {e}"
                    )

            # Set GPU argument
            gpu_available = True
            return gpu_available

        if gpu_option == "Other GPU":
            # Set GPU argument
            gpu_available = True
            return gpu_available

    # Select the gpu availability
    gpu_output_interact = interactive(
        gpu_output,
        gpu_option=widgets.RadioButtons(
            options=["No GPU", "Colab GPU", "Other GPU"],
            value="No GPU",
            description="Select GPU availability:",
            disabled=False,
        ),
    )

    display(gpu_output_interact)
    return gpu_output_interact


# Select the movie you want
def select_movie(available_movies_df: pd.DataFrame):
    """
    > This function takes in a dataframe of available movies and returns a widget that allows the user
    to select a movie of interest

    :param available_movies_df: a dataframe containing the list of available movies
    :return: The widget object
    """

    # Get the list of available movies
    available_movies_tuple = tuple(sorted(available_movies_df.filename.unique()))

    # Widget to select the movie
    select_movie_widget = widgets.Dropdown(
        options=available_movies_tuple,
        description="Movie of interest:",
        ensure_option=False,
        value=None,
        disabled=False,
        layout=widgets.Layout(width="50%"),
        style={"description_width": "initial"},
    )

    display(select_movie_widget)
    return select_movie_widget


def choose_folder(start_path: str = ".", folder_type: str = ""):
    """
    > This function enables users to select the folder of interest to retrieve or save files to/from.

    :param start_path: a string with the path of the origin for the folder
    :param folder_type: a string with the names of the type of folder required
    :return: A path of the folder of interest
    """
    # Specify the output folder
    fc = FileChooser(start_path)
    fc.title = f"Choose location of {folder_type}"
    display(fc)
    return fc


def choose_footage(
    project: Project,
    server_connection: dict,
    db_connection,
    start_path: str = ".",
    folder_type: str = "",
):
    """
    > This function enables users to select movies for ML purposes.

    :param project: the project object
    :param server_connection: a dictionary with the connection to the server
    :param db_connection: SQL connection object
    :param start_path: a string with the path of the origin for the folder
    :param folder_type: a string with the names of the type of folder required
    :return: A path of the folder of interest
    """
    if project.server == "AWS":
        available_movies_df, _, _ = movie_utils.retrieve_movie_info_from_server(
            project=project,
            server_connection=server_connection,
            db_connection=db_connection,
        )
        movie_dict = {
            name: movie_utils.get_movie_path(f_path, project, server_connection)
            for name, f_path in available_movies_df[0][["filename", "fpath"]].values
        }
        movie_widget = widgets.SelectMultiple(
            options=[(name, movie) for name, movie in movie_dict.items()],
            description="Select movie(s):",
            ensure_option=False,
            disabled=False,
            layout=widgets.Layout(width="50%"),
            style={"description_width": "initial"},
        )

        display(movie_widget)
        return movie_widget

    else:
        # Specify the output folder
        fc = FileChooser(start_path)
        fc.title = f"Choose location of {folder_type}"
        display(fc)
        return fc


######################################################################
###################Common ZOO widgets#################################
######################################################################


def request_latest_zoo_info():
    """
    Display a widget that allows to select whether to retrieve the last available information,
    or to request the latest information.

    :return: an interactive widget object with the value of the boolean

    """

    def generate_export(retrieve_option):
        if retrieve_option == "No, just download the last available information":
            generate = False

        elif retrieve_option == "Yes":
            generate = True

        return generate

    latest_info = interactive(
        generate_export,
        retrieve_option=widgets.RadioButtons(
            options=["Yes", "No, just download the last available information"],
            value="No, just download the last available information",
            layout={"width": "max-content"},
            description="Do you want to request the most up-to-date Zooniverse information?",
            disabled=False,
            style={"description_width": "initial"},
        ),
    )

    display(latest_info)
    display(
        HTML(
            """<font size="2px">If yes, a new data export will be requested and generated with the latest information of Zooniverse (this may take some time)<br>
    Otherwise, the latest available export will be downloaded (some recent information may be missing!!).<br><br>
    If the waiting time for the generation of a new data export ends, the last available information will be retrieved. However, that information <br>
    will probably correspond to the newly generated export.
    </font>"""
        )
    )

    return latest_info


def choose_agg_parameters(subject_type: str = "clip"):
    """
    > This function creates a set of sliders that allow you to set the parameters for the aggregation
    algorithm

    :param subject_type: The type of subject you are aggregating. This can be either "frame" or "video"
    :type subject_type: str
    :return: the values of the sliders.
        Aggregation threshold: (0-1) Minimum proportion of citizen scientists that agree in their classification of the clip/frame.
        Min numbers of users: Minimum number of citizen scientists that need to classify the clip/frame.
        Object threshold (0-1): Minimum proportion of citizen scientists that agree that there is at least one object in the frame.
        IOU Epsilon (0-1): Minimum area of overlap among the classifications provided by the citizen scientists so that they will be considered to be in the same cluster.
        Inter user agreement (0-1): The minimum proportion of users inside a given cluster that must agree on the frame annotation for it to be accepted.
    """
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
    # Create HTML widget for description
    description_widget = HTML(
        f"<p>Minimum proportion of citizen scientists that agree in their classification of the {subject_type}.</p>"
    )
    # Display both widgets in a VBox
    display(agg_users)
    display(description_widget)
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
    # Create HTML widget for description
    description_widget = HTML(
        f"<p>Minimum number of citizen scientists that need to classify the {subject_type}.</p>"
    )
    # Display both widgets in a VBox
    display(min_users)
    display(description_widget)
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
        # Create HTML widget for description
        description_widget = HTML(
            "<p>Minimum proportion of citizen scientists that agree that there is at least one object in the frame.</p>"
        )
        # Display both widgets in a VBox
        display(agg_obj)
        display(description_widget)
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
        # Create HTML widget for description
        description_widget = HTML(
            "<p>Minimum area of overlap among the citizen science classifications to be considered as being in the same cluster.</p>"
        )
        # Display both widgets in a VBox
        display(agg_iou)
        display(description_widget)
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
        # Create HTML widget for description
        description_widget = HTML(
            "<p>The minimum proportion of users inside a given cluster that must agree on the frame annotation for it to be accepted.</p>"
        )
        # Display both widgets in a VBox
        display(agg_iua)
        display(description_widget)

        return agg_users, min_users, agg_obj, agg_iou, agg_iua
    else:
        return agg_users, min_users


def choose_w_version(workflows_df: pd.DataFrame, workflow_id: str):
    """
    It takes a workflow ID and returns a dropdown widget with the available versions of the workflow

    :param workflows_df: a dataframe containing the workflows available in the Galaxy instance
    :param workflow_id: The name of the workflow you want to run
    :return: A tuple containing the widget and the list of versions available.
    """

    # Estimate the versions of the workflow available
    versions_available = (
        workflows_df[workflows_df.display_name == workflow_id].version.unique().tolist()
    )

    if len(versions_available) > 1:
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

    else:
        raise ValueError("There are no versions available for this workflow.")

    # display(w_version)
    return w_version, list(map(float, versions_available))


def choose_workflows(workflows_df: pd.DataFrame):
    """
    It creates a dropdown menu for the user to choose a workflow name, a dropdown menu for the user to
    choose a subject type, and a dropdown menu for the user to choose a workflow version

    :param workflows_df: a dataframe containing the workflows you want to choose from
    :type workflows_df: pd.DataFrame
    """

    layout = widgets.Layout(width="auto", height="40px")  # set width and height

    # Display the names of the workflows
    workflow_name = widgets.Dropdown(
        options=workflows_df.display_name.unique().tolist(),
        value=workflows_df.display_name.unique().tolist()[0],
        description="Workflow name:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
        layout=layout,
    )

    # Display the type of subjects
    subj_type = widgets.Dropdown(
        options=["frame", "clip"],
        value="clip",
        description="Subject type:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
        layout=layout,
    )

    workflow_version, versions = choose_w_version(workflows_df, workflow_name.value)

    def on_change(change):
        with out:
            if change["name"] == "value":
                clear_output()
                workflow_version.options = choose_w_version(
                    workflows_df, change["new"]
                )[1]
                workflow_name.observe(on_change)

    out = widgets.Output()
    display(out)

    workflow_name.observe(on_change)
    return workflow_name, subj_type, workflow_version


######################################################################
#####################Tutorial 1 widgets###############################
######################################################################


def map_sites(project: Project, csv_paths: dict):
    """
    > This function takes a dictionary of database information and a project object as input, and
    returns a map of the sites in the database

    :param project: The project object
    :param csv_paths: a dictionary with the paths of the csv files used to initiate te db
    :return: A map with all the sites plotted on it.
    """
    # Read the csv file with site information
    sites_df = pd.read_csv(csv_paths["local_sites_csv"])

    # Identify columns containing "Latitude" or "Longitude"
    latitude_columns = [col for col in sites_df.columns if "Latitude" in col]
    longitude_columns = [col for col in sites_df.columns if "Longitude" in col]

    # Ensure we have at least one column of each
    if not latitude_columns or not longitude_columns:
        raise ValueError("No 'Latitude' or 'Longitude' columns found.")

    # Rearrange columns to put Latitude and Longitude first
    new_columns = (
        latitude_columns
        + longitude_columns
        + [
            col
            for col in sites_df.columns
            if col not in latitude_columns + longitude_columns
        ]
    )

    # Create a new DataFrame with rearranged columns
    sites_df = sites_df[new_columns]

    # Set initial location to first site
    init_location = [
        sites_df.iloc[0][latitude_columns],
        sites_df.iloc[0][longitude_columns],
    ]

    # Create the initial kso map
    kso_map = folium.Map(location=init_location, width=900, height=600)

    # Iterate through rows to add markers for each site
    for index, row in sites_df.iterrows():
        site_info = row.to_list()
        latitude = row[latitude_columns]
        longitude = row[longitude_columns]

        # Create a CircleMarker for the site
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=14,
            popup=site_info,
        ).add_to(kso_map)

    # Add a minimap to the corner for reference
    kso_map = kso_map.add_child(MiniMap())

    # Return the map
    return kso_map


def select_sheet_range(project: Project, csv_paths: dict, orig_csv: str):
    """
    > This function loads the csv file of interest into a pandas dataframe and enables users
    to pick a range of rows and columns to display

    :param project: the project object
    :param csv_paths: a dictionary with the paths of the csv files used to initiate the db
    :param orig_csv: the original csv file name
    :type orig_csv: str
    :return: A dataframe with the sites information
    """

    # Load the csv with the information of interest
    df = pd.read_csv(csv_paths[orig_csv])

    df_range_rows = widgets.SelectionRangeSlider(
        options=range(0, len(df.index) + 1),
        index=(0, len(df.index)),
        description="Rows to display",
        orientation="horizontal",
        layout=Layout(width="90%", padding="35px"),
        style={"description_width": "initial"},
    )

    display(df_range_rows)

    df_range_columns = widgets.SelectMultiple(
        options=df.columns,
        description="Columns",
        disabled=False,
        layout=Layout(width="50%", padding="35px"),
    )

    display(df_range_columns)

    return df, df_range_rows, df_range_columns


def display_ipysheet_changes(isheet: ipysheet.Sheet, df_filtered: pd.DataFrame):
    """
    It takes the dataframe from the ipysheet and compares it to the dataframe from the local csv file.
    If there are any differences, it highlights them and returns the dataframe with the changes
    highlighted

    :param isheet: The ipysheet object that contains the data
    :param sites_df_filtered: a pandas dataframe with information of a range of sites
    :return: A tuple with the highlighted changes and the sheet_df
    """
    # Convert ipysheet to pandas
    sheet_df = ipysheet.to_dataframe(isheet)

    # Check the differences between the modified and original spreadsheets
    sheet_diff_df = pd.concat([df_filtered, sheet_df]).drop_duplicates(keep=False)

    # If changes in dataframes display them and ask the user to confirm them
    if sheet_diff_df.empty:
        logging.info("No changes were made.")
        return sheet_df, sheet_df
    else:
        # Retrieve the column name of the id of interest (Sites, movies,..)
        id_col = [col for col in df_filtered.columns if "_id" in col][0]

        # Concatenate DataFrames and distinguish each frame with the keys parameter
        df_all = pd.concat(
            [df_filtered.set_index(id_col), sheet_df.set_index(id_col)],
            axis="columns",
            keys=["Origin", "Update"],
        )

        # Rearrange columns to have them next to each other
        df_final = df_all.swaplevel(axis="columns")[
            [x for x in df_filtered.columns if x != id_col]
        ]

        # Create a function to highlight the changes
        def highlight_diff(data, color="yellow"):
            attr = "background-color: {}".format(color)
            other = data.xs("Origin", axis="columns", level=-1)
            return pd.DataFrame(
                np.where(data.ne(other, level=0), attr, ""),
                index=data.index,
                columns=data.columns,
            )

        # Return the df with the changes highlighted
        highlight_changes = df_final.style.apply(highlight_diff, axis=None)

        return highlight_changes, sheet_df


def choose_movie_review():
    """
    This function creates a widget that allows the user to choose between two methods to review the
    movies.csv file.
    :return: The widget is being returned.
    """
    choose_movie_review_widget = widgets.RadioButtons(
        options=[
            "Basic: Checks for available movies and empty cells in movies.csv",
            "Advanced: Basic + Check movies format and movies with missing information",
        ],
        description="Select the movies review method:",
        disabled=False,
        layout=Layout(width="95%"),
        style={"description_width": "initial"},
    )
    display(
        HTML(
            """<font size="2px">In the Basic review, we check:<br>
            all movies in the "movies.csv" are in the "movie_folder",<br>
            all movies in the "movie_folder" are in the "movies.csv" and,<br>
            for empty cells in the fps, duration, sampling_start and sampling_end columns of the "movies.csv". If there are empty cells, retrieves the information and saves it into the movies.csv.<br><br>
    In the Advanced review, in addition to the basic checks, we also check:<br>
    the format, frame rate and codec of the movies are correct. If not, automatically standarises the movies.<br>
    Note the advanced review can take a while to standarise all the movies<br>
    </font>"""
        )
    )

    display(choose_movie_review_widget)

    return choose_movie_review_widget


def update_meta(
    project: Project,
    conn,
    server_connection: dict,
    sheet_df: pd.DataFrame,
    df: pd.DataFrame,
    meta_name: str,
    csv_paths: dict,
    test=False,
):
    """
    `update_meta` takes a new table, a meta name, and updates the local and server meta files

    :param sheet_df: The dataframe of the sheet you want to update
    :param meta_name: the name of the metadata file (e.g. "movies")
    """

    from kso_utils.db_utils import process_test_csv
    from kso_utils.server_utils import update_csv_server

    # Create button to confirm changes
    confirm_button = widgets.Button(
        description="Yes, details are correct",
        layout=widgets.Layout(width="25%"),
        style={"description_width": "initial"},
        button_style="danger",
    )

    # Create button to deny changes
    deny_button = widgets.Button(
        description="No, I will go back and fix them",
        layout=widgets.Layout(width="45%"),
        style={"description_width": "initial"},
        button_style="danger",
    )

    # Save changes in survey csv locally and in the server
    async def f(sheet_df, df, meta_name):
        if not test:
            x = await wait_for_change(
                confirm_button, deny_button
            )  # <---- Pass both buttons into the function
        else:
            x = "Yes, details are correct"
        if (
            x == "Yes, details are correct"
        ):  # <--- use if statement to trigger different events for the two buttons
            logging.info("Checking if changes can be incorporated to the database")

            # Retrieve the column name of the id of interest (Sites, movies,..)
            id_col = [col for col in df.columns if "_id" in col][0]

            # Replace the different values based on id
            df_orig = df.copy()
            df_new = sheet_df.copy()
            df_orig.set_index(id_col, inplace=True)
            df_new.set_index(id_col, inplace=True)
            df_orig.update(df_new)
            df_orig.reset_index(drop=False, inplace=True)

            # Process the csv of interest and tests for compatibility with sql table
            df_to_db = process_test_csv(
                conn=conn, project=project, local_df=df_orig, init_key=meta_name
            )

            # Log changes locally
            t_utils.log_meta_changes(
                project=project,
                meta_key="local_" + meta_name + "_csv",
                new_sheet_df=sheet_df,
                csv_paths=csv_paths,
            )

            # Save the updated df locally
            df_orig.to_csv(csv_paths["local_" + meta_name + "_csv"], index=False)
            logging.info("The local csv file has been updated")

            if project.server == "AWS":
                # Save the updated df in the server
                update_csv_server(
                    project=project,
                    csv_paths=csv_paths,
                    server_connection=server_connection,
                    orig_csv="server_" + meta_name + "_csv",
                    updated_csv="local_" + meta_name + "_csv",
                )

        else:
            logging.info("Run this cell again when the changes are correct!")

    logging.info("")
    logging.info("Are the changes above correct?")
    display(
        widgets.HBox([confirm_button, deny_button])
    )  # <----Display both buttons in an HBox
    if not test:
        asyncio.create_task(f(sheet_df, df, meta_name))
    else:
        f(sheet_df=sheet_df, df=df, meta_name=meta_name)


def open_csv(
    df: pd.DataFrame, df_range_rows: widgets.Widget, df_range_columns: widgets.Widget
):
    """
    > This function loads the dataframe with the information of interest, filters the range of rows and columns selected and then loads the dataframe into
    an ipysheet

    :param df: a pandas dataframe of the information of interest:
    :param df_range_rows: the rows range widget selection:
    :param df_range_columns: the columns range widget selection:
    :return: A (subset) dataframe with the information of interest and the same data in an interactive sheet
    """
    # Extract the first and last row to display
    range_start = int(df_range_rows[0])
    range_end = int(df_range_rows[1])

    # Display the range of sites selected
    logging.info(f"Displaying # {range_start} to # {range_end}")

    # Filter the dataframe based on the selection: rows and columns
    df_filtered_row = df.filter(items=range(range_start, range_end), axis=0)
    if not len(df_range_columns) == 0:
        df_filtered = df_filtered_row.filter(items=df_range_columns, axis=1)
        # Display columns
        logging.info(f"Displaying {df_range_columns}")
    else:
        df_filtered = df_filtered_row.filter(items=df.columns, axis=1)
        # Display columns
        logging.info(f"Displaying {df.columns.tolist()}")

    # Load the df as ipysheet
    sheet = ipysheet.from_dataframe(df_filtered)

    return df_filtered, sheet


######################################################################
#####################Tutorial 2 widgets###############################
######################################################################


def choose_new_videos_to_upload():
    """
    Simple widget for uploading videos from a file browser.
    returns the list of movies to be added.
    Supports multi-select file uploads
    """

    movie_list = []

    fc = FileChooser()
    fc.title = "First choose your directory of interest"
    " and then the movies you would like to upload"
    logging.info("Choose the file that you want to upload: ")

    def change_dir(chooser):
        sel.options = os.listdir(chooser.selected)
        fc.children[1].children[2].layout.display = "none"
        sel.layout.visibility = "visible"

    fc.register_callback(change_dir)

    sel = widgets.SelectMultiple(options=os.listdir(fc.selected))

    display(fc)
    display(sel)

    sel.layout.visibility = "hidden"

    button_add = widgets.Button(description="Add selected file")
    output_add = widgets.Output()

    logging.info(
        "Showing paths to the selected movies:\nRerun cell to reset\n--------------"
    )

    display(button_add, output_add)

    def on_button_add_clicked(b):
        with output_add:
            if sel.value is not None:
                for movie in sel.value:
                    if Path(movie).suffix in [".mp4", ".mov"]:
                        movie_list.append([Path(fc.selected, movie), movie])
                        logging.info(Path(fc.selected, movie))
                    else:
                        logging.error("Invalid file extension")
                    fc.reset()

    button_add.on_click(on_button_add_clicked)
    return movie_list


######################################################################
#####################Tutorial 3 widgets###############################
######################################################################


# Display the number of clips to generate based on the user's selection
def to_clips(clip_length, clips_range, is_example: bool):
    # Calculate the number of clips
    if is_example:
        clips = 3
    else:
        clips = int((clips_range[1] - clips_range[0]) / clip_length)

    logging.info(f"Number of clips to generate: {clips}")

    return clips


def select_n_clips(
    project: Project,
    db_connection,
    movie_i: str,
    is_example: bool,
):
    """
    > The function `select_random_clips` takes in a movie name and
    returns a dictionary containing the starting points of the clips and the
    length of the clips.

    :param project: the project object
    :param db_connection: SQL connection object
    :param movie_i: the name of the movie of interest
    :return: A dictionary with the starting points of the clips and the length of the clips.
    """

    # Query info about the movie of interest
    movie_df = pd.read_sql_query(
        f"SELECT filename, duration, sampling_start, sampling_end FROM movies WHERE filename='{movie_i}'",
        db_connection,
    )

    # Select the number of clips to upload
    # Create a boolean widget and hide it so that it can be added to the interactive layout
    example_widget = widgets.Checkbox(value=is_example, description="Random examples")
    example_widget.layout.visibility = "hidden"
    clip_length_number = widgets.interactive(
        to_clips,
        is_example=example_widget,
        clip_length=select_clip_length(),
        clips_range=widgets.IntRangeSlider(
            value=[movie_df.sampling_start.values, movie_df.sampling_end.values],
            min=0,
            max=int(movie_df.duration.values),
            step=1,
            description="Movie range to generate clips from (seconds):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%"),
        ),
    )
    display(clip_length_number)
    return clip_length_number


def select_clip_length():
    """
    > This function creates a dropdown widget that allows the user to select the length of the clips
    :return: The widget is being returned.
    """
    # Widget to record the length of the clips
    ClipLength_widget = widgets.Dropdown(
        options=[10, 5],
        value=10,
        description="Length of clips (seconds):",
        style={"description_width": "initial"},
        ensure_option=True,
        disabled=False,
    )

    return ClipLength_widget


class clip_modification_widget(widgets.VBox):
    def __init__(self):
        """
        The function creates a widget that allows the user to select which modifications to run
        """
        self.widget_count = widgets.IntText(
            description="Number of modifications:",
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
            new_widget = select_modification()
            for wdgt in [new_widget]:
                wdgt.description = wdgt.description + f" #{_}"
            new_widgets.extend([new_widget])
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}


# Function to specify the frame modification
def select_modification():
    # Widget to select the frame modification

    frame_modifications = {
        "Color_correction": {
            "filter": ".filter('curves', '0/0 0.396/0.67 1/1', \
                                        '0/0 0.525/0.451 1/1', \
                                        '0/0 0.459/0.517 1/1')"
        }
        # borrowed from https://www.element84.com/blog/color-correction-in-space-and-at-sea
        ,
        "Zoo_low_compression": {
            "crf": "25",
        },
        "Zoo_medium_compression": {
            "crf": "27",
        },
        "Zoo_high_compression": {
            "crf": "30",
        },
        "Blur_sensitive_info": {
            "filter": ".drawbox(0, 0, 'iw', 'ih*(15/100)', color='black' \
                            ,thickness='fill').drawbox(0, 'ih*(95/100)', \
                            'iw', 'ih*(15/100)', color='black', thickness='fill')",
            "None": {},
        },
    }

    select_modification_widget = widgets.Dropdown(
        options=[(a, b) for a, b in frame_modifications.items()],
        description="Select modification:",
        ensure_option=True,
        disabled=False,
        style={"description_width": "initial"},
    )

    return select_modification_widget


# Display the clips side-by-side
def view_clips(example_clips: list, modified_clip_path: str):
    """
    > This function takes in a list of example clips and a path to a modified clip, and returns a widget
    that displays the original and modified clips side-by-side

    :param example_clips: a list of paths to the original clips
    :param modified_clip_path: The path to the modified clip you want to view
    :return: A widget that displays the original and modified videos side-by-side.
    """

    # Get the path of the modified clip selected
    example_clip_name = os.path.basename(modified_clip_path).replace("modified_", "")
    example_clip_path = next(
        filter(lambda x: os.path.basename(x) == example_clip_name, example_clips), None
    )

    # Get the extension of the video
    extension = Path(example_clip_path).suffix

    # Open original video
    vid1 = open(example_clip_path, "rb").read()
    wi1 = widgets.Video(value=vid1, format=extension, width=400, height=500)

    # Open modified video
    vid2 = open(modified_clip_path, "rb").read()
    wi2 = widgets.Video(value=vid2, format=extension, width=400, height=500)

    # Display videos side-by-side
    a = [wi1, wi2]
    wid = widgets.HBox(a)

    return wid


def compare_clips(example_clips: list, modified_clips: list):
    """
    > This function allows you to select a clip from the modified clips and displays the original and
    modified clips side by side

    :param example_clips: The original clips
    :param modified_clips: The list of clips that you want to compare to the original clips
    """

    # Add "no movie" option to prevent conflicts
    modified_clips = np.append(modified_clips, "0 No movie")

    clip_path_widget = widgets.Dropdown(
        options=tuple(modified_clips),
        description="Select original clip:",
        ensure_option=True,
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )

    main_out = widgets.Output()
    display(clip_path_widget, main_out)

    # Display the original and modified clips
    def on_change(change):
        with main_out:
            clear_output()
            if change["new"] == "0 No movie":
                logging.info("It is OK to modify the clips again")
            else:
                a = view_clips(example_clips, change["new"])
                display(a)

    clip_path_widget.observe(on_change, names="value")


######################################################################
#####################Tutorial 4 widgets###############################
######################################################################


def extract_custom_frames(
    input_path,
    output_dir,
    skip_start=None,
    skip_end=None,
    num_frames=None,
    frame_skip=None,
    backend: str = "cv",
):
    """
    This function extracts frames from a video file and saves them as JPEG images.

    :param input_path: The file path of the input movie file that needs to be processed
    :param output_dir: The directory where the extracted frames will be saved as JPEG files
    :param num_frames: The number of frames to extract from the input video. If this parameter is
    provided, the function will randomly select num_frames frames to extract from the video
    :param frame_skip: frame_skip is an optional parameter that determines how many frames to skip
    between extracted frames. For example, if frame_skip is set to 10, then every 10th frame will be
    extracted. If frame_skip is not provided, then all frames will be extracted
    """

    if backend == "cv":
        # Open the input movie file
        cap = cv2.VideoCapture(input_path)

        # Get base filename
        input_stem = Path(input_path).stem

        # Get the total number of frames in the movie
        num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        skip_start = int(skip_start * fps)
        skip_end = int(skip_end * fps)

    elif backend == "av":
        # Open the input movie file
        reader = VideoReader(path=input_path)

        # Get base filename
        input_stem = Path(input_path).stem

        # Get the total number of frames in the movie
        num_frames_total = reader._n_frames
        fps = reader._fps
        skip_start = int(skip_start * fps)
        skip_end = int(skip_end * fps)
    else:
        raise ValueError("Unsupported backend.")

    frame_start = 0 if skip_start is None else skip_start
    frame_end = num_frames_total if skip_end is None else num_frames_total - skip_end

    # Determine which frames to extract based on the input parameters
    if num_frames is not None:
        # Note: current workaround uses every 500 frames to avoid frame seeking error
        frames_to_extract = random.sample(
            range(frame_start, frame_end, 500), num_frames
        )
    elif frame_skip is not None:
        frames_to_extract = range(frame_start, frame_end, frame_skip)
    else:
        frames_to_extract = range(frame_end)

    output_files, input_movies = [], []

    if backend == "cv":
        # Loop through the frames and extract the selected ones
        for frame_idx in frames_to_extract:
            # Set the frame index for the next frame to read
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # Read the next frame
            ret, frame = cap.read()

            if ret:
                # Construct the output filename for this frame
                output_filename = os.path.join(
                    output_dir, f"{input_stem}_frame_{frame_idx}.jpg"
                )

                # Write the frame to a JPEG file
                cv2.imwrite(output_filename, frame)

                # Add output filename to list of files
                output_files.append(output_filename)

                # Add movie filename to list
                input_movies.append(Path(input_path).name)

        # Release the video capture object
        cap.release()

    elif backend == "av":
        # Loop through the frames and extract the selected ones
        for frame_idx in frames_to_extract:
            # Set the frame index for the next frame to read
            frame = reader.request_frame(frame_idx)

            # Get the image
            frame = frame.image

            # Construct the output filename for this frame
            output_filename = os.path.join(
                output_dir, f"{input_stem}_frame_{frame_idx}.jpg"
            )

            # Write the frame to a JPEG file
            cv2.imwrite(output_filename, frame)

            # Add output filename to list of files
            output_files.append(output_filename)

            # Add movie filename to list
            input_movies.append(Path(input_path).name)

    return pd.DataFrame(
        np.column_stack([input_movies, output_files, frames_to_extract]),
        columns=["movie_filename", "frame_path", "frame_number"],
    )
