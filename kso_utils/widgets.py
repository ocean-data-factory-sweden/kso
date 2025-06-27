# base imports
import cv2
import logging
import numpy as np
import os
import pandas as pd
import random
import subprocess
import requests
from io import BytesIO
from base64 import b64encode

# widget imports
import ipysheet
import folium
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import HTML, display, clear_output, Image as IPImage
from ipywidgets import interactive, Layout, Video
from folium.plugins import MiniMap
from pathlib import Path
import asyncio
from PIL import Image as PILImage

# util imports
from kso_utils.video_reader import VideoReader
from kso_utils.project_utils import Project, get_projects_csv_file
import kso_utils.movie_utils as movie_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

######################################
# ###### Common widgets ###########
# #####################################




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


def choose_project():
    """
    This function generates a dropdown menu with project names listed based on a CSV file.

    :param projects_csv: Path to the CSV file with a list of projects, defaults to "../kso_utils/db_starter/projects_list.csv"
    :type projects_csv: str, optional
    :return: A dropdown widget with the project names as options
    :rtype: ipywidgets.Dropdown
    """
    projects_csv_path = get_projects_csv_file()
    projects_df = pd.read_csv(projects_csv_path)

    if "Project_name" not in projects_df.columns:
        raise ValueError("The CSV file does not contain a 'Project_name' column.")

    # Create the dropdown widget
    choose_project_widget = widgets.Dropdown(
        options=projects_df["Project_name"].unique(),
        value=projects_df["Project_name"].iloc[0],
        description="Project:",
        disabled=False,
    )

    display(choose_project_widget)
    return choose_project_widget


def gpu_select():
    """
    This function allows the user to select GPU availability and installs GPU requirements if needed.

    If the user selects "No GPU", the function returns False.
    If the user selects "Colab GPU", the function installs GPU requirements and returns True.
    If the user selects "Other GPU", the function returns True.

    :return: The gpu_available variable
    :rtype: bool
    """

    def gpu_output(gpu_option):
        if gpu_option == "No GPU":
            print("You are set to start the modifications")
            # Set GPU argument
            gpu_available = False
            return gpu_available

        if gpu_option == "Colab GPU":
            # Install the GPU requirements
            if not Path("./colab-ffmpeg-cuda/bin").exists():
                try:
                    print(
                        "Installing the GPU requirements. PLEASE WAIT 10-20 SECONDS"
                    )  # Install ffmpeg with GPU version

                    # Use subprocess for shell commands
                    subprocess.run(
                        [
                            "git",
                            "clone",
                            "https://github.com/XniceCraft/ffmpeg-colab.git",
                        ]
                    )
                    subprocess.run(["chmod", "755", "./ffmpeg-colab/install"])
                    subprocess.run(["./ffmpeg-colab/install"])
                    print("Installation finished!")
                    subprocess.run(["rm", "-fr", "/content/ffmpeg-colab"])

                    print("GPU Requirements installed!")

                except subprocess.CalledProcessError as e:
                    print(
                        f"There was an issue trying to install the GPU requirements: {e}"
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


# Select the movie(s) you want
def _select_movie(available_movies_df: pd.DataFrame):
    """
    > This function takes in a dataframe of available movies and returns a widget that allows the user
    to select movie(s) of interest

    :param available_movies_df: a dataframe containing the list of available movies
    :return: The widget object
    """

    # Get the list of available movies
    available_movies_tuple = tuple(sorted(available_movies_df.filename.unique()))

    # Widget to select the movie
    select_movie_widget = widgets.SelectMultiple(
        options=available_movies_tuple,
        description="Select movie(s):",
        ensure_option=False,
        disabled=False,
        layout=widgets.Layout(width="50%"),
        style={"description_width": "initial"},
    )

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


def choose_footage_source():
    # Create radio buttons
    source_widget = widgets.RadioButtons(
        options=["Existing Footage", "New Footage"],
        value="Existing Footage",
        description="Choose footage source:",
    )

    # Display the radio buttons
    display(source_widget)
    return source_widget


def choose_footage(
    df: pd.DataFrame,
    project: Project,
    server_connection: dict,
    footage_source: str,
    preview_media: bool,
):
    """
    > The function `choose_footage` is a function that takes in a `self` argument and returns a
    function `f` that takes in three arguments: `project`, `csv_paths`, and `available_movies_df`. The
    function `f` is an asynchronous function that takes in the value of the `selected_movies` widget
    and previews the movies if specified
    :param df: the dataframe of available movies
    :param project: the project object
    :param server_connection: a dictionary with the connection to the server
    :param footage_source: a string specifying whether the footage is already in the system or is new
    :param preview_media: a boolean parameter to display or not the movie selected

    """

    if footage_source == "Existing Footage":
        # Initiate and display the movie_widget output
        movie_output = widgets.Output()

        # Display the available movies
        select_movie_widg = _select_movie(df)
        # print(select_movie_widg)

        def update_movie(change):

            selected_movies = change["new"]

            # Get the df and paths of the selected movies
            (
                selected_movies_paths,
                selected_movies,
                selected_movies_df,
                selected_movies_ids,
            ) = movie_utils.get_info_selected_movies(
                selected_movies=selected_movies,
                footage_source=footage_source,
                df=df,
                project=project,
                server_connection=server_connection,
            )
            print(selected_movies_paths, selected_movies, selected_movies_df,selected_movies_ids)
            project.selected_movies = selected_movies

            # Display the movie
            if preview_media:
                # with movie_output:
                #   clear_output()
                previews = []

                # Display/preview each selected movie
                for (
                    index,
                    movie_row,
                ) in selected_movies_df.iterrows():
                    movie_path = selected_movies_paths[index]
                    movie_metadata = pd.DataFrame(
                        [movie_row.values], columns=movie_row.index
                    )

                    html = _preview_movie(
                        movie_path=movie_path,
                        movie_metadata=movie_metadata,
                    )

                    previews.append(html)

                project.previews = previews

        # Observe changes in the widget
        # todo kalindi
        
        # select_movie_widg.value = ('HOR_20240408_BUV_HOR_002_01.mp4',)
        # update_movie({"new" : ('HOR_20240408_BUV_HOR_002_01.mp4',))
        # print(select_movie_widg.value)
        # print(select_movie_widg.options)
        
        select_movie_widg.observe(update_movie, "value")
        # select_movie_widg.value = (select_movie_widg.options[0],)
        select_movie_widg.value = ('HOR_20240408_BUV_HOR_010_01.mp4',)
        display(select_movie_widg)  
        
        

    elif footage_source == "New Footage":

        def on_folder_selected(change):
            selected_folder = select_movie_widg.selected

            if selected_folder is not None:
                print(f"Selected folder: {selected_folder}")
            else:
                print("No folder selected")

        select_movie_widg = choose_folder(
            start_path=(
                project.movie_folder
                if project.movie_folder not in [None, "None"]
                else "."
            ),
            folder_type="new footage",
        )

        select_movie_widg.register_callback(on_folder_selected)

    else:
        logging.info("Select a valid option from the choose_footage_source function")

    return select_movie_widg


######################################
# ###### Common ZOO widgets ###########
# #####################################


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


def choose_aggregation_users(info_dict: dict):
    """
    > This function shows a widget for selecting the users to keep the annotations from. The options include:
    - Keep the annotations made by all users. This is the default option.
    - Select specific users to keep their annotations and discard the rest.

    :param info_dict: a dictionary with the citizen scientits classifications
    """

    def get_users_list(info_dict: dict, option: str):
        if "specific" in option:
            users = _select_users(info_dict)
            return users
        else:
            clear_output()
            return []

    users = interactive(
        get_users_list,
        info_dict=widgets.fixed(info_dict),
        option=widgets.RadioButtons(
            options=[
                "Keep annotations of all users",
                "Select specific users",
            ],
            layout={"width": "max-content"},
            default="Keep annotations of all users",
            description="Which annotations would you like to keep?",
            disabled=False,
            style={"description_width": "initial"},
        ),
    )
    display(users)
    return users


def _select_users(info_dict: dict):
    """
    Returns a widget showing all the citizen scientits that made classifications in the project,
    sorted in descending order according to the number of classifications they have made.
    This facilitates the selection of top contributors to keep their annotations and discard the rest.

    :param info_dict: a dictionary with the citizen scientits classifications
    """

    users = info_dict["user_name"]
    unique_counts = users.value_counts().sort_values(ascending=False)
    unique_counts = unique_counts.index.to_list()

    w = widgets.SelectMultiple(
        options=unique_counts,
        description="Select users:",
        ensure_option=False,
        disabled=False,
        layout=widgets.Layout(width="50%"),
        rows=len(unique_counts),
        style={"description_width": "initial"},
    )

    description_widget = HTML(
        "<p>The users are sorted in descending order based on the number of annotations they have done.</p>"
    )

    # Display both widgets in a VBox
    display(w)
    display(description_widget)
    return w


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
        description="Min number of users:",
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
            value=0.5,
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
            value=0.7,
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
            value=0.4,
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


def _choose_w_version(workflows_df: pd.DataFrame, workflow_id: str):
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

    if len(versions_available) >= 1:
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

    workflow_version, versions = _choose_w_version(workflows_df, workflow_name.value)

    def on_change(change):
        with out:
            if change["name"] == "value":
                clear_output()
                workflow_version.options = _choose_w_version(
                    workflows_df, change["new"]
                )[1]
                workflow_name.observe(on_change)

    out = widgets.Output()
    display(out)

    workflow_name.observe(on_change)
    return workflow_name, subj_type, workflow_version


######################################
# ###### TUT 1 widgets ###########
# #####################################


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


# Function to preview underwater movies
def _preview_movie(
    movie_path: str,
    movie_metadata: pd.DataFrame,
):
    """
    It takes a movie filename and its associated metadata and returns a widget object that can be displayed in the notebook

    :param movie_path: the filename of the movie you want to preview
    :param movie_metadata: the metadata of the movie you want to preview
    :return: HTML object
    """

    # Adjust the width of the video and metadata sections based on your preference
    video_width = "80%"  # Adjust as needed
    metadata_width = "20%"  # Adjust as needed

    metadata_html = movie_metadata.T.to_html()

    if "http" in movie_path:
        video_html = f"""
          <video width="{video_width}" controls>
              <source src="{movie_path}" type="video/mp4">
              Your browser does not support the video tag.
          </video>
        """
        combined_html = f"""
        <div style="display: flex;">
            <div style="flex: 1;">{video_html}</div>
            <div style="flex: 1; width: {metadata_width}; overflow: auto;">{metadata_html}</div>
        </div>
        """
        display_widget = HTML(combined_html)

    else:
        video_widget = widgets.Video.from_file(movie_path, width=video_width)

        metadata_widget = widgets.HTML(
            value=metadata_html,
            layout=widgets.Layout(width=metadata_width, overflow="auto"),
        )

        # Create a horizontal box layout to display video and metadata side by side
        display_widget = widgets.HBox([video_widget, metadata_widget])

    display(display_widget)

    return display_widget


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


######################################
# ###### Tut 3 widgets ###########
# #####################################


# Display the number of clips to generate based on the user's selection
def _to_clips(clip_length, clips_range, is_example: bool):
    # Calculate the number of clips to generate
    clips = int((clips_range[1] - clips_range[0]) / clip_length)

    if is_example and clips > 5:
        logging.info(
            f"Number of clips to generate: {clips}. We recommend to create less than 5 examples"
        )
    else:
        logging.info(f"Number of clips to generate: {clips}")

    return clips


def select_n_clips(
    project: Project,
    db_connection,
    selected_movies: list,
    is_example: bool,
):
    """
    > The function `select_random_clips` takes in a movie name and
    returns a dictionary containing the starting points of the clips and the
    length of the clips.

    :param project: the project object
    :param db_connection: SQL connection object
    :param movie_i: a list with the name of the movie(s) of interest
    :return: A dictionary with the starting points of the clips and the length of the clips.
    """

    # Query info about the movie of interest
    movie_df = pd.read_sql_query(
        f"SELECT filename, duration, sampling_start, sampling_end FROM movies WHERE filename='{selected_movies}'",
        db_connection,
    )
    print(movie_df)
    # Select the number of clips to upload
    # Create a boolean widget and hide it so that it can be added to the interactive layout
    example_widget = widgets.Checkbox(value=is_example, description="Random examples")
    example_widget.layout.visibility = "hidden"

    # todo kalindi
    movie_df.duration = 2000
    movie_df.sampling_start = 329 
    movie_df.sampling_end = 339
    
    clip_length_number = widgets.interactive(
        _to_clips,
        is_example=example_widget,
        clip_length=_select_clip_length(),
        clips_range=widgets.IntRangeSlider(
            value=[movie_df.sampling_start.values, movie_df.sampling_end.values],
            # value=[0, 2000],

            min=0,
            # TODO: kalindi
            max=int(movie_df.duration.values),
            # max=2000,
            step=1,
            description="Movie range to generate clips from (seconds):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="90%"),
        ),
    )
    display(clip_length_number)
    return clip_length_number


def _select_clip_length():
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
            new_widget = _select_modification()
            for wdgt in [new_widget]:
                wdgt.description = wdgt.description + f" #{_}"
            new_widgets.extend([new_widget])
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}


# Function to specify the frame modification
def _select_modification():

    # Widget to select the frame modification

    frame_modifications = {
        "Zoo_low_compression": {
            "b:v": "10M",
        },
        "Zoo_medium_compression": {
            "b:v": "5M",
        },
        "Zoo_high_compression": {
            "b:v": "3M",
        },
        "Blur_sensitive_info": {
            "filter": '.drawbox(0, 0, "iw", "ih*(15/100)", color="black", thickness="fill").drawbox(0, "ih*(95/100)", "iw", "ih*(15/100)", color="black", thickness="fill")',
        },
        # Increase brightness (useful for low-light)
        "Increase_brightness": {
            "filter": '.filter("eq", brightness=0.1)',  # Slight increase in brightness
        },
        # Increase contrast (useful for low-light)
        "Increase_contrast": {
            "filter": '.filter("eq", contrast=1.5)',  # Higher contrast for better distinction
        },
        # Increase both brightness and contrast together
        "Increase_brightness_and_contrast": {
            "filter": '.filter("eq", brightness=0.1, contrast=1.5)',  # Adjust both brightness and contrast at once
        },
        # Denoising (useful in low-light noisy environments)
        "Denoise": {
            "filter": '.filter("hqdn3d", 3, 3, 6, 6)',  # Reduces noise
        },
        # Sharpen (useful when image appears blurry)
        "Sharpen": {
            "filter": '.filter("unsharp", 5, 5, 1.0, 5, 5, 0.0)',  # Enhances image sharpness
        },
        # Gamma correction (useful in low-light)
        "Adjust_gamma": {
            "filter": '.filter("eq", gamma=1.2)',  # Adjust gamma to bring out details in darker areas
        },
        # Saturation boost (to make colors more vivid in low-light)
        "Increase_saturation": {
            "filter": '.filter("eq", saturation=1.5)',  # Boosts color saturation
        },
        # Grayscale (for contrast-focused low-light viewing)
        "Grayscale": {
            "filter": '.filter("format", "gray")',  # Converts image to grayscale
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
def _view_clips(example_clips: list, modified_clips: list, modified_clip_selected: str):
    """
    > This function takes in a list of example clips and a path to a modified clip, and returns a widget
    that displays the original and modified clips side-by-side

    :param example_clips: a list of paths to the original clips
    :param modified_clip_path: The path to the modified clip you want to view
    :return: A widget that displays the original and modified videos side-by-side.
    """

    # Get the path of the original clip based on the selected modified clip
    example_clip_selected = example_clips[
        modified_clips.tolist().index(modified_clip_selected)
    ]

    # Get the extension of the video
    extension = Path(example_clip_selected).suffix

    # Open original video
    vid1 = open(example_clip_selected, "rb").read()
    wi1 = widgets.Video(value=vid1, format=extension, width=400, height=500)

    # Open modified video
    vid2 = open(modified_clip_selected, "rb").read()
    wi2 = widgets.Video(value=vid2, format=extension, width=400, height=500)

    # Display videos side-by-side
    wid = widgets.HBox([wi1, wi2])

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
                a = _view_clips(example_clips, modified_clips, change["new"])
                display(a)

    clip_path_widget.observe(on_change, names="value")


######################################
# ###### Tut 4 widgets ###########
# #####################################


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
        # Note: if frame-seeking fails, use every 500th frame instead
        frames_to_extract = random.sample(range(frame_start, frame_end, 1), num_frames)
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
                output_filename = str(
                    Path(output_dir) / f"{input_stem}_frame_{frame_idx}.jpg"
                )

                # Write the frame to a JPEG file
                cv2.imwrite(str(output_filename), frame)

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
            output_filename = str(
                Path(output_dir) / f"{input_stem}_frame_{frame_idx}.jpg"
            )

            # Write the frame to a JPEG file
            cv2.imwrite(str(output_filename), frame)

            # Add output filename to list of files
            output_files.append(output_filename)

            # Add movie filename to list
            input_movies.append(Path(input_path).name)

    return pd.DataFrame(
        np.column_stack([input_movies, output_files, frames_to_extract]),
        columns=["movie_filename", "frame_path", "frame_number"],
    )


# Display the frames using html
def _view_frames(df: pd.DataFrame, frame_path: str):
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
                a = _view_frames(df, change["new"])
                display(a)

    clip_path_widget.observe(on_change, names="value")


######################################
# ###### Tut 5 widgets ###########
# #####################################


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
        # Retrieve species/labels information #####
        # Create a df with unique workflow ids and versions of interest
        work_df = (
            df[["workflow_id", "workflow_version"]].drop_duplicates().astype("int")
        )

        # Correct for some weird zooniverse version behaviour
        #         work_df["workflow_version"] = work_df["workflow_version"] - 1

        # Store df of all the common names and the labels into a list of df
        from kso_upload_copy.notebooks.classify.hold.zooniverse_utils import get_workflow_labels

        commonName_labels_list = [
            get_workflow_labels(zoo_info_dict["workflows"], x, y)
            for x, y in zip(work_df["workflow_id"], work_df["workflow_version"])
        ]

        # Concatenate the dfs and select only unique common names and the labels
        commonName_labels_df = pd.concat(commonName_labels_list).drop_duplicates()

        # Drop the clips classified as nothing here or other
        df = df[~df["label"].isin(["OTHER", "NOTHINGHERE", "HUMANOBJECTS"])]

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
        # Rename columns to match Darwin Data Core Standards
        df = df.rename(
            columns={
                "created_on": "eventDate",
                "max_n": "individualCount",
                "commonName": "vernacularName",
            }
        )

        # Create relevant columns for GBIF
        df["occurrenceID"] = (
            project.Project_name
            + "_"
            + df["siteName"]
            + "_"
            + df["eventDate"].astype(str)
            + "_"
            + df["second_in_movie"].astype(str)
            + "_"
            + df["vernacularName"].astype(str)
        )

        # Set the basis of record as machine observation
        df["basisOfRecord"] = "MachineObservation"

        # If coord uncertainity doesn't exist set to 30 metres
        df["coordinateUncertaintyInMeters"] = df.get(
            "coordinateUncertaintyInMeters", 30
        )

        # Select columns relevant for GBIF occurrences
        df = df[
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

        return df
    else:
        raise ValueError(
            "Specify who classified the species of interest (citizen_scientists, biologists or ml_algorithms)"
        )


######################################
# ###### Tut 8 widgets ###########
# #####################################


def _view_subject(subject_id: int, class_df: pd.DataFrame, subject_type: str):
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
            from kso_utils.frame_utils import draw_annotations_in_frame

            # Create a temporary image with the annotations drawn on it
            im = draw_annotations_in_frame(im, class_df_subject)

        # Remove previous temp image if exist
        try:
            with open("test.txt", "w") as temp_file:
                temp_file.write("Testing write access.")
            temp_image_path = "temp.jpg"

        except Exception as e:
            # Handle the case when file creation fails
            logging.error(f"Failed to create temporary file: {e}")

            # Specify volume allocated by SNIC if available
            if Path("/mimer").exists():
                snic_tmp_path = "/mimer/NOBACKUP/groups/snic2021-6-9/tmp_dir"
            elif Path("/tmp").exists():
                snic_tmp_path = "/tmp"
            else:
                logging.error("No suitable writable path found.")
                return

            temp_image_path = str(Path(snic_tmp_path, "temp.jpg"))

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


def launch_classifications_viewer(class_df: pd.DataFrame, subject_type: str):
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
        sorted(
            class_df[class_df["subject_type"] == subject_type]["subject_ids"]
            .apply(int)
            .apply(str)
            .unique()
        )
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
            a = _view_subject(int(change["new"]), class_df, subject_type)
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
        options=tuple(sorted(class_df.subject_ids.apply(int).apply(str).unique())),
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


def select_viewer():
    viewer_dropdown = widgets.Dropdown(
        options=["image", "video"],
        value="image",
        description="Select viewer:",
        disabled=False,
    )

    output = widgets.Output()

    def on_change(change):
        with output:
            output.clear_output()
            if change.new == "image":
                _image_viewer()
            elif change.new == "video":
                _movie_viewer()
            else:
                logging.error("Invalid option selected.")

    viewer_dropdown.observe(on_change, names="value")

    display(viewer_dropdown)
    display(output)


def _image_viewer():
    def check_image(change):
        filepath = file_chooser.selected
        if filepath.lower().endswith((".jpg", ".jpeg")):
            logging.info(f"Showing image: {filepath}")
            show_image(filepath)
        else:
            logging.error("Please select a JPG or JPEG file.")

    def show_image(image_path):
        with image_output:
            clear_output(wait=True)
            img = IPImage(filename=image_path)
            display(img)

    image_output = widgets.Output()

    file_chooser = FileChooser(os.getcwd())
    file_chooser.use_dir_icons = True
    file_chooser.filter_pattern = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    file_chooser.register_callback(check_image)

    file_chooser_widget = widgets.VBox(
        [widgets.Label("Select a JPG or JPEG file:"), file_chooser, image_output]
    )
    display(file_chooser_widget)


def _movie_viewer():
    def check_movie(change):
        filepath = file_chooser.selected
        if filepath.endswith(".mp4"):
            # Implement your function to show the movie here
            logging.info(f"Showing movie: {filepath}")
            show_movie(filepath)
        else:
            logging.error("Please select an MP4 file.")

    def show_movie(
        movie_path: str,
    ):
        """
        This function takes a movie file path as input, reads the video file, and returns the video object.

        :param movie_path: The `movie_path` parameter is a string that represents the file path to a movie
        file
        :type movie_path: str
        :return: The function `show_movie` returns the `Video` object created from the file located at the
        `movie_path` provided as input.
        """
        with movie_output:
            clear_output(wait=True)
            video = Video.from_file(movie_path)
            display(video)

    movie_output = widgets.Output()

    file_chooser = FileChooser(os.getcwd())
    file_chooser.use_dir_icons = True
    file_chooser.filter_pattern = "*.mp4"
    file_chooser.register_callback(check_movie)

    file_chooser_widget = widgets.VBox(
        [widgets.Label("Select an MP4 file:"), file_chooser, movie_output]
    )
    display(file_chooser_widget)
