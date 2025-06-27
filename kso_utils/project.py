# base imports
import os
import time
import sys
import logging
import datetime
import numpy as np
import pandas as pd
import ipywidgets as widgets
import ffmpeg
import shutil
from pathlib import Path
import ipysheet
from IPython.display import display, clear_output
from typing import List

# util imports
import kso_utils.project_utils as project_utils
import kso_utils.db_utils as db_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.server_utils as server_utils
import kso_utils.general as g_utils
import kso_utils.widgets as kso_widgets
import kso_utils.yolo_utils as yolo_utils
import kso_utils.zooniverse_utils as zoo_utils



# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

logging.info("hi41i")


class ProjectProcessor:
    # The ProjectProcessor class initializes various attributes and methods for processing a project,
    # including importing modules, setting up a database, and loading metadata.
    def __init__(self, project: project_utils.Project):
        self.project = project
        self.db_connection = None
        self.INIT_KEYS = ["movies", "species", "photos", "surveys", "sites"]
        self.server_connection = {}
        self.csv_paths = {}
        self.zoo_info = {}
        self.annotation_engine = None
        self.annotations = pd.DataFrame()
        self.classifications = pd.DataFrame()
        self.generated_clips = pd.DataFrame()
        self.species_of_interest = []
        self.selected_movies_id = {}

        # Get server details and connect to server
        self._connect_to_server()

        # Map initial csv files
        self._map_init_csv()

        # Create empty db and populate with local csv files data
        self._setup_db()

    def __repr__(self):
        return f"ProjectProcessor(project={self.project})"

    @property
    def keys(self) -> List[str]:
        """Log keys of ProjectProcessor object"""
        logging.debug("Stored variable names.")
        return list(self.__dict__.keys())

    #############
    # Functions to initiate the project
    #############

    def _connect_to_server(self):
        """
        It connects to the server and returns the server info
        :return: The server_connection is added to the ProjectProcessor class.
        """
        try:
            self.server_connection = server_utils.connect_to_server(self.project)
        except Exception as e:
            logging.error(f"Server connection could not be established. Details: {e}")

    def _map_init_csv(self):
        """
        This function maps the csv files, download them from the server (if needed) and
        stores the server/local paths of the csv files
        """

        # Create the folder to store the csv files if not exist
        csv_folder = Path(self.project.csv_folder)
        if not csv_folder.exists():
            csv_folder.mkdir(parents=True, exist_ok=True)

            # Recursively add permissions to folders created
            for root, dirs, files in csv_folder.iterdir():
                root.chmod(0o777)

        # Download csv files from the server if needed and store their server path
        self.csv_paths = server_utils.download_init_csv(
            self.project, self.INIT_KEYS, self.server_connection
        )

        # Store the paths of the local csv files
        self._load_meta()

    def _load_meta(self):
        """
        It loads the metadata from the relevant local csv files into the `csv_paths` dictionary
        """
        # Retrieve a list with all the csv files in the folder with initial csvs
        csv_folder = Path(self.project.csv_folder)
        local_csv_files = [str(f) for f in csv_folder.glob("*.csv")]

        # Store the paths of the local csv files of interest into the "csv_paths" dictionary
        for filename in local_csv_files:
            # Select only csv files that are relevant to start the db
            for init_key in self.INIT_KEYS:
                if init_key in filename:
                    # Specify the key in the dictionary of the CSV file
                    csv_key = f"local_{init_key}_csv"

                    # Store the path of the CSV file
                    self.csv_paths[csv_key] = filename

                    # Read the local CSV file into a pandas DataFrame
                    df = pd.read_csv(filename)
                    setattr(self, csv_key, df)

                    # Temporary workaround for sites_csv (Pylint needs an explicit declaration)
                    if csv_key == "local_sites_csv":
                        self.local_sites_csv = df

    def _setup_db(self):
        """
        The function creates a database and populates it with the data from the local csv files.
        It also return the db connection
        :return: The database connection object.
        """
        # Create a new database for the project and get a connection
        self.db_connection = db_utils.create_db(self.project.db_path)

        # Retrieves the table names of the sql db
        table_names = db_utils.get_schema_table_names(self.db_connection)

        # Select only attributes of the propjectprocessor that are df of local csvs
        local_dfs = [
            key
            for key in self.keys
            if key.startswith("local_") and key.split("_")[1] in table_names
        ]

        # Sort the local dfs in reverse alphabetically to load sites before movies
        local_dfs = sorted(local_dfs, reverse=True)

        # Populate the db with initial info from the local_csvs
        for df_key in local_dfs:
            init_key = df_key.split("_")[1]
            local_df = getattr(self, df_key)
            db_utils.populate_db(
                project=self.project,
                conn=self.db_connection,
                local_df=local_df,
                init_key=init_key,
            )

    #############
    # t1
    #############
    def map_sites(self):
        return kso_widgets.map_sites(project=self.project, csv_paths=self.csv_paths)

    def get_movie_info(self):
        """
        This function checks what movies from the movies csv are available and returns
        three df with those available in folder/server and movies.csv, only available
        in movies.csv and only available in folder/server
        """
        (
            self.available_movies_df,
            self.no_available_movies_df,
            self.no_info_movies_df,
        ) = movie_utils.retrieve_movie_info_from_server(
            project=self.project,
            db_connection=self.db_connection,
            server_connection=self.server_connection,
        )

        logging.info("Information of available movies has been retrieved")

    def choose_footage_source(self):
        """
        Enables users to select exisiting (already uploaded)
        or new (local) footage
        """
        # Choose the source of the footage
        self.source_footage_widget = kso_widgets.choose_footage_source()

    def choose_footage(self, preview_media: bool = False):
        """
        Function that enables users to select footage to upload/process/classify and previews the movies if specified
        """


        # Check if the source_footage is available and has the right format
        if not hasattr(self, "source_footage_widget"):
            # Set the source footage as "existing footage"
            # (choose_footage_source func is unavailable in tut#1 and #3 to ensure the use of db)
            self.source_footage = "Existing Footage"

        elif self.source_footage_widget.value is None:
            logging.info(
                "Select a valid option from the choose_footage_source function"
            )

        else:
            self.source_footage = self.source_footage_widget.value



        # Check if the necessary available_movies_df attribute is available
        if not hasattr(self, "available_movies_df"):
            if self.source_footage == "Existing Footage":
                logging.info("Creating the available_movies_df attribute")
                self.get_movie_info()
            else:
                logging.info("Creating an empty available_movies_df for the new movies")
                self.available_movies_df = pd.DataFrame()

    
        # Call the choose_footage function and save the widget to pp
        self.footage_selected_widget = kso_widgets.choose_footage(
            df=self.available_movies_df,
            project=self.project,
            footage_source=self.source_footage,
            server_connection=self.server_connection,
            preview_media=preview_media,
        )

    def check_selected_movies(self):
        """
        Function that loads the paths and other information of the selected footage to the ProjectProcessors
        """
        (
            self.selected_movies_paths,
            self.selected_movies,
            self.selected_movies_df,
            self.selected_movies_ids,
        ) = movie_utils.get_info_selected_movies(
            selected_movies=(
                self.footage_selected_widget.selected
                if self.footage_selected_widget.value is None
                else self.footage_selected_widget.value
            ),
            footage_source=self.source_footage,
            df=self.available_movies_df,
            project=self.project,
            server_connection=self.server_connection,

            
        )
     


    def check_movies_meta(
        self,
        review_method: str,
        gpu_available: bool = False,
    ):
        """
        > The function `check_movies_csv` loads the csv with movies information and checks if it is empty

        :param review_method: The method used to review the movies
        :param gpu_available: Boolean, whether or not a GPU is available
        """
        # Check if the necessary attribute is available
        if not hasattr(self, "available_movies_df") or self.available_movies_df is None:
            raise AttributeError(
                "Please run 'get_movie_info' before 'choose_footage' to set 'available_movies_df'."
            )

        df = movie_utils.check_movies_meta(
            project=self.project,
            csv_paths=self.csv_paths,
            db_connection=self.db_connection,
            available_movies_df=self.available_movies_df,
            no_info_movies_df=self.no_info_movies_df,
            server_connection=self.server_connection,
            review_method=review_method,
            gpu_available=gpu_available,
        )
        if df is not None:
            self.temp_local_movies = df

    def check_species_meta(self):
        return db_utils.check_species_meta(
            csv_paths=self.csv_paths, conn=self.db_connection
        )

    #############
    # t2
    #############

    #############
    # t3
    #############

    def connect_zoo_project(self, generate_export: bool = False, zoo_cred=False):
        """
        This function connects to Zooniverse, saves the connection
        to the project processor and retrieves
        the subjects, workflows and classifications.
        If the project is template, retrieves the info from the Gdrive.
        :return: The zoo_info is being returned.
        """
        # Connect to Zooniverse if project is not template
        if self.project.Project_name == "Template project":
            self.zoo_project = {}

        else:
            if self.project.Zooniverse_number is not None:
                # connect to Zooniverse
                self.zoo_project = zoo_utils.connect_zoo_project(
                    self.project, zoo_cred=zoo_cred
                )
            else:
                logging.error("This project is not registered with Zooniverse.")
                return

        # Retrieve the Zooniverse information
        self.zoo_info = zoo_utils.retrieve_zoo_info(
            self.project,
            self.zoo_project,
            zoo_info=["subjects", "workflows", "classifications"],
            generate_export=generate_export,
        )

    def check_movies_uploaded_zoo(
        self,
    ):
        """
        This function checks if a movie has been uploaded to Zooniverse

        :param selected_movies: The name of the movie(s) you want to check if it's uploaded
        :type selected_movies: list
        """
        print(self)
        zoo_utils.check_movies_uploaded_zoo(
            project=self.project,
            db_connection=self.db_connection,
            selected_movies=self.selected_movies,
        )

    def generate_zoo_clips(
        self,
        use_gpu: bool = False,
        pool_size: int = 4,
        is_example: bool = False,
    ):
        """
         > This function takes a movie name and path, and returns a list of clips from that movie

        :param use_gpu: If you have a GPU, set this to True, defaults to False
         :type use_gpu: bool (optional)
         :param pool_size: number of threads to use for clip extraction, defaults to 4
         :type pool_size: int (optional)
         :param is_example: If True, the clips will be selected randomly. If False, the clips will be
                selected based on the number of clips and the length of each clip, defaults to False
         :type is_example: bool (optional)
        """
        # Roadblock to ensure only one movie has been selected
        # Option to generate clips from multiple movies is not available at this point
        if len(self.selected_movies) > 1 and isinstance(self.selected_movies, list):
            logging.error(
                "The option to generate clips from multiple movies is not available at this point. Please select only one movie to generate clips from"
            )
            return None

        # Select the clips to be extracted
        clip_selection = kso_widgets.select_n_clips(
            project=self.project,
            db_connection=self.db_connection,
            selected_movies=str(self.selected_movies[0]),
            is_example=is_example,
        )
        clip_modification = kso_widgets.clip_modification_widget()

        button = widgets.Button(
            description="Click to extract clips",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        # print(self.available_movies_df, self.selected_movies[0], self.selected_movies_paths[0], clip_selection, self.project, clip_modification)
        def on_button_clicked(b):
            
            self.generated_clips = zoo_utils.create_clips(
                available_movies_df=self.available_movies_df,
                selected_movies=str(self.selected_movies[0]),
                movie_path=str(self.selected_movies_paths[0]),
                clip_selection=clip_selection,
                project=self.project,
                modification_details=clip_modification,
                gpu_available=use_gpu,
                pool_size=pool_size,
                is_example=is_example,
            )

            if self.project.Project_name != "Spyfish_Aotearoa":
                # Excludes Spyfish as it doesn't use the site_id column but sitename
                # Temporary workaround to ensure site_id is an integer
                self.generated_clips["site_id"] = (
                    self.generated_clips["site_id"].astype(float).astype(np.int64)
                )
        print("Attaching button callback...")
        button.on_click(on_button_clicked)
        display(clip_modification)
        display(button)
        on_button_clicked("1")

    def check_clip_size(self):
        """
        > This function takes a list of file paths and returns a dataframe with the file path and size of
        each file. If the size is too large, we suggest compressing them as a first step.
        """
        return zoo_utils.check_clip_size(clips_list=self.generated_clips.clip_path)

    def upload_zoo_subjects(self, subject_type: str):
        """
        This function uploads clips or frames to Zooniverse, depending on the subject_type argument

        :param
        :param subject_type: str = "clip" or "frame"
        :type subject_type: str
        """
        if subject_type == "clip":
            # Add declaration to avoid pylint error
            upload_df, sitename, created_on = zoo_utils.set_zoo_clip_metadata(
                project=self.project,
                generated_clipsdf=self.generated_clips,
                sitesdf=self.local_sites_csv,
                moviesdf=self.local_movies_csv,
            )
            zoo_utils.upload_clips_to_zooniverse(
                project=self.project,
                upload_to_zoo=upload_df,
                sitename=sitename,
                created_on=created_on,
            )
            # Clean up subjects after upload
            for temp_clip in upload_df["clip_path"].unique().tolist():
                temp_clip_path = Path(temp_clip)
                if temp_clip_path.exists():
                    temp_clip_path.unlink()

            logging.info("Clips temporarily stored locally has been removed")

        elif subject_type == "frame":
            upload_df = zoo_utils.set_zoo_frame_metadata(
                project=self.project,
                db_connection=self.db_connection,
                df=self.generated_frames,
                species_list=self.species_of_interest,
                csv_paths=self.csv_paths,
            )
            zoo_utils.upload_frames_to_zooniverse(
                project=self.project,
                upload_to_zoo=upload_df,
                species_list=self.species_of_interest,
            )

        else:
            logging.error("Select the right type of subject (e.g. frame or clip)")

    #############
    # t4
    #############

    def choose_zoo_workflows(self):
        """
        The function process the available Zooniverse workflows and enables
        users to select those of interest
        :return: A widget displaying the different workflows available.
        """
        self.workflow_widget = zoo_utils.WidgetWorkflowSelection(
            self.zoo_info["workflows"]
        )
        display(self.workflow_widget)

    def process_zoo_classifications(self):
        """
        It samples subjects from the workflows selected, populates the subjects db,
        sample the classifications from the workflows of interest,
        process them and saves them to the Zooniverse attribute of the project processor

        """
        workflow_checks = self.workflow_widget.checks

        # Retrieve a subset of the subjects from the workflows of interest and
        # populate the sql subjects table and flatten the classifications provided the cit. scientists
        self.processed_zoo_classifications = zoo_utils.process_zoo_classifications(
            project=self.project,
            server_connection=self.server_connection,
            db_connection=self.db_connection,
            workflow_widget_checks=workflow_checks,
            workflows_df=self.zoo_info["workflows"],
            subjects_df=self.zoo_info["subjects"],
            csv_paths=self.csv_paths,
            classifications_data=self.zoo_info["classifications"],
            subject_type=workflow_checks["Subject type: #0"],
        )

    def aggregate_zoo_classifications(self, agg_params, users: list):
        workflow_checks = self.workflow_widget.checks

        if isinstance(users, list):
            # If users is already a list, select all user classifications
            classifications_filtered = self.processed_zoo_classifications
        else:
            # Convert users widget to a list
            users_list = list(users.value) if users else None

            if users_list:
                # Obtain classifications only from the selected users
                classifications_filtered = self.processed_zoo_classifications[
                    self.processed_zoo_classifications["user_name"].isin(users_list)
                ].copy()
            else:
                logging.warning(
                    "Processing the classifications of all users as no user was selected."
                )
                classifications_filtered = self.processed_zoo_classifications

        # Check if all items in 'poly_points' are "nan" as a string
        if "poly_points" in classifications_filtered.columns:
            all_nan = (classifications_filtered["poly_points"] == "nan").all()
        else:
            all_nan = False
        if not "poly_points" in classifications_filtered.columns or all_nan:
            self.aggregated_zoo_classifications = zoo_utils.aggregate_classifications(
                self.project,
                classifications_filtered,
                workflow_checks["Subject type: #0"],
                agg_params,
            )
            # Return nan values for testing
            self.aggregated_zoo_classifications["poly_points"] = "nan"
        else:
            # Use all polygons
            self.aggregated_zoo_classifications = classifications_filtered

    def extract_zoo_frames(self, n_frames_subject: int = 3, subsample_up_to: int = 100):
        if not isinstance(self.species_of_interest, list):
            self.species_of_interest = self.species_of_interest.value
        species_list = self.species_of_interest

        self.generated_frames = zoo_utils.extract_frames_for_zoo(
            project=self.project,
            zoo_info=self.zoo_info,
            species_list=species_list,
            db_connection=self.db_connection,
            server_connection=self.server_connection,
            agg_df=self.aggregated_zoo_classifications,
            n_frames_subject=n_frames_subject,
            subsample_up_to=subsample_up_to,
        )

    def modify_zoo_frames(self):
        """
        This function takes a dataframe of frames to upload, a species of interest, a project, and a
        dictionary of modifications to make to the frames, and returns a dataframe of modified frames.
        """

        frame_modification = kso_widgets.clip_modification_widget()

        button = widgets.Button(
            description="Click to modify frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.modified_frames = zoo_utils.modify_frames(
                project=self.project,
                frames_to_upload_df=self.generated_frames.reset_index(drop=True),
                species_i=self.species_of_interest,
                modification_details=frame_modification.checks,
            )

        button.on_click(on_button_clicked)
        display(frame_modification)
        display(button)

    def generate_custom_frames(
        self,
        skip_start: int,
        skip_end: int,
        input_path: str,
        output_path: str,
        num_frames: int = None,
        frames_skip: int = None,
        backend: str = "cv",
    ):
        """
        This function generates custom frames from input movie files and saves them in an output directory.

        :param input_path: The directory path where the input movie files are located
        :type input_path: str
        :param output_path: The directory where the extracted frames will be saved
        :type output_path: str
        :param num_frames: The number of frames to extract from each video file. If not specified, all
        frames will be extracted
        :type num_frames: int
        :param frames_skip: The `frames_skip` parameter is an optional integer that specifies the number of
        frames to skip between each extracted frame. For example, if `frames_skip` is set to 2, every other
        frame will be extracted. If `frames_skip` is not specified, all frames will be extracted
        :type frames_skip: int
        :return: the results of calling the `parallel_map` function with the `extract_custom_frames` function from
        the `t4_utils` module, passing in the `movie_files` list as the input and the `args` tuple
        containing `output_dir`, `num_frames`, and `frames_skip`. The `parallel_map` function is a custom
        function that applies the given function to each element of a list of movie_files.
        """
        if backend not in ["av", "cv"]:
            raise ValueError(
                "Unsupported backend. "
                "Choose either 'av' or 'cv' for pyav and OpenCV."
            )

        frame_modification = kso_widgets.clip_modification_widget()
        species_list = kso_widgets.choose_species(self.db_connection)

        button = widgets.Button(
            description="Click to modify frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            movie_files = sorted(
                [
                    str(f)
                    for f in Path(input_path).iterdir()
                    if f.is_file()
                    and f.suffix.lower() in [".mov", ".mp4", ".avi", ".mkv", ".mpg"]
                ]
            )

            if not Path(output_path).exists():
                Path(output_path).mkdir(parents=True)
                Path(output_path).chmod(0o777)

            results = g_utils.parallel_map(
                kso_widgets.extract_custom_frames,
                movie_files,
                args=(
                    [output_path] * len(movie_files),
                    [skip_start] * len(movie_files),
                    [skip_end] * len(movie_files),
                    [num_frames] * len(movie_files),
                    [frames_skip] * len(movie_files),
                    [backend] * len(movie_files),
                ),
            )
            if len(results) > 0:
                self.frames_to_upload_df = pd.concat(results)
                self.frames_to_upload_df["species_id"] = pd.Series(
                    [db_utils.get_species_ids(self.db_connection, species_list.value)]
                    * len(self.frames_to_upload_df)
                )
                self.frames_to_upload_df = self.frames_to_upload_df.merge(
                    db_utils.get_df_from_db_table(self.db_connection, "movies").rename(
                        columns={"id": "movie_id"}
                    ),
                    how="left",
                    left_on="movie_filename",
                    right_on="filename",
                )
                # Ensure necessary metadata fields are available
                self.frames_to_upload_df = self.frames_to_upload_df[
                    [
                        "frame_path",
                        "siteName",
                        "movie_id",
                        "created_on",
                        "frame_number",
                        "species_id",
                    ]
                ]

            else:
                logging.error("No results.")
                self.frames_to_upload_df = pd.DataFrame()
            self.project.output_path = output_path
            self.generated_frames = zoo_utils.modify_frames(
                project=self.project,
                frames_to_upload_df=self.frames_to_upload_df.reset_index(drop=True),
                species_i=species_list.value,
                modification_details=frame_modification.checks,
            )
            self.modified_frames = self.generated_frames

        button.on_click(on_button_clicked)
        display(frame_modification)
        display(button)

    def check_frame_size(self):
        """
        It takes a list of file paths, gets the size of each file, and returns a dataframe with the file
        path and size of each file

        :param frame_paths: a list of paths to the frames you want to check
        :return: A dataframe with the file path and size of each frame.
        """
        # Check the size of the frames
        return zoo_utils.check_frame_size(
            frame_paths=self.generated_frames["frame_path"].unique()
        )

    # Function to compare original to modified frames
    def compare_frames(self, df):
        # Function to compare original to modified frames
        kso_widgets.compare_frames(df)

    #############
    # t8
    #############
    def explore_processed_classifications_per_subject(self):
        """
        It displays the processed classifications for a given subject

        """
        # Display the displays the processed classifications for a given subject
        kso_widgets.explore_classifications_per_subject(
            class_df=self.processed_zoo_classifications,
            subject_type=self.workflow_widget.checks["Subject type: #0"],
        )

    def launch_classifications_table(self):
        """
        It takes in a dataframe of aggregated classifications and a subject type, and returns a dataframe
        with the columns "subject_ids", "label", "how_many", and "first_seen"
        """
        agg_class_df = zoo_utils.launch_classifications_table(
            agg_class_df=self.aggregated_zoo_classifications,
            subject_type=self.workflow_widget.checks["Subject type: #0"],
        )

        return agg_class_df

    def launch_classifications_viewer(self):
        """
        > This function takes a dataframe of classifications and a subject type (frame or video) and
        displays a dropdown menu of subjects of that type. When a subject is selected, it displays the
        subject and the classifications for that subject
        """
        kso_widgets.launch_classifications_viewer(
            class_df=self.aggregated_zoo_classifications,
            subject_type=self.workflow_widget.checks["Subject type: #0"],
        )

    def download_classications_csv(self, class_df):
        # Add the site and movie information to the classifications based on the subject information
        class_df = zoo_utils.add_subject_site_movie_info_to_class(
            self.project, self.db_connection, self.csv_paths, class_df
        )

        # Download the processed classifications as a csv file
        csv_filename = (
            self.project.Project_name
            + str(datetime.date.today())
            + "classifications.csv"
        )

        class_df.to_csv(csv_filename, index=False)

        logging.info(f"The classications have been downloaded to {csv_filename}")

    def get_annotations_viewer(self, folder_path: str, annotation_classes: list):
        """
        > This function takes in a folder path and a list of annotation classes and returns a widget that
        allows you to view the annotations in the folder

        :param folder_path: The path to the folder containing the images you want to annotate
        :type folder_path: str
        :param annotation_classes: list of strings
        :type annotation_classes: list
        :return: A list of dictionaries, each dictionary containing the following keys
                 - 'image_path': the path to the image
                 - 'annotations': a list of dictionaries, each dictionary containing the following keys:
                 - 'class': the class of the annotation
                 - 'bbox': the bounding box of the annotation
        """
        return yolo_utils.get_annotations_viewer(
            folder_path, species_list=annotation_classes
        )

    def download_gbif_occurrences(self, classified_by, df, max_count=True):
        if classified_by == "citizen_scientists":
            # Add the site and movie information to the classifications based on the subject information
            df = zoo_utils.add_subject_site_movie_info_to_class(
                self.project,
                self.db_connection,
                self.csv_paths,
                df,
            )

        # Format the classifications to Darwin Core Standard occurrences
        occurrence_df = kso_widgets.format_to_gbif(
            self.project,
            self.db_connection,
            df,
            self.csv_paths,
            classified_by,
            self.zoo_info,
        )

        if max_count:
            # Group by species/date/location and get the maximum 'individualCount'
            occurrence_df = (
                occurrence_df.sort_values("individualCount", ascending=False)
                .groupby(
                    [
                        "scientificName",
                        "eventDate",
                        "decimalLatitude",
                        "decimalLongitude",
                    ],
                    as_index=False,
                )
                .first()
            )

        # Download the processed classifications as a csv file
        csv_filename = (
            self.project.Project_name + str(datetime.date.today()) + "occurrence.csv"
        )

        occurrence_df.to_csv(csv_filename, index=False)

        logging.info(f"The occurences have been downloaded to {csv_filename}")

    def process_detections(
        self,
        project,
        db_connection,
        csv_paths,
        annotations_csv_path,
        model_registry,
        model,
        project_name,
        team_name,
    ):
        """
        > This function computes the given statistics over the detections obtained by a model on different footages for the species of interest,
        and saves the results in different csv files.
        """
        out_list = []
        for movie_path in self.selected_movies_paths:
            out_list.append(
                yolo_utils.process_detections(
                    project=project,
                    db_connection=db_connection,
                    csv_paths=csv_paths,
                    annotations_csv_path=annotations_csv_path,
                    model_registry=model_registry,
                    selected_movies_id=self.selected_movies_ids,
                    model=model,
                    project_name=project_name,
                    team_name=team_name,
                    source_movies=movie_path,
                )
            )
        df_concat = pd.concat(out_list, axis=1)
        return df_concat

    def plot_processed_detections(self, df, thres, int_length):
        """
        > This function computes the given statistics over the detections obtained by a model on different footages for the species of interest,
        and saves the results in different csv files.
        """
        yolo_utils.plot_processed_detections(
            df=df,
            thres=thres,
            int_length=int_length,
        )

    #############
    # t9
    #############
    def download_detections_csv(self, df):
        # Download the processed detections as a csv file
        csv_filename = (
            self.project.Project_name + str(datetime.date.today()) + "detections.csv"
        )

        df.to_csv(csv_filename, index=False)

        logging.info(f"The detections have been downloaded to {csv_filename}")

    def download_detections_species_cols_csv(self, df):
        # Specify the species labels
        if "commonName" in df.columns:
            # Define the movie col of interest
            sp_group_col = "commonName"
        else:
            # Define the movie col of interest
            sp_group_col = "class_id"

        # Transpose the rows/cols to have species as cols
        transposed_df = df.pivot_table(
            index=["movie_id", "second_in_movie"],
            columns=sp_group_col,
            values=["min_conf", "mean_conf", "max_n", "max_conf"],
            aggfunc="first",
        )

        # Flatten the MultiIndex columns
        transposed_df.columns = [
            f"{species}_{column}" for column, species in transposed_df.columns
        ]

        # Reset index to get a regular DataFrame
        transposed_df.reset_index(inplace=True)

        # Specify columns to drop from original df to avoid large df and confussions
        df_col_drop = [
            "class_id",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "frame_no",
            "min_conf",
            "mean_conf",
            "max_n",
            "max_conf",
            "scientificName",
            "taxonRank",
            "kingdom",
            "commonName",
        ]
        df_to_merge = df.drop(df_col_drop, axis=1).drop_duplicates()

        # Merge with the original DataFrame based on common columns
        merged_df = pd.merge(
            transposed_df, df_to_merge, on=["movie_id", "second_in_movie"]
        )

        # Sort columns into the expected order as specified by Leon
        sp_list = df[sp_group_col].unique()

        # Separate columns with species_info and the rest
        columns_sp_group = [
            col for col in merged_df.columns if any(sp in col for sp in sp_list)
        ]

        # Corrected syntax: use "not in" before "for sp in sp_list"
        columns_no_sp_group = [
            col for col in merged_df.columns if all(sp not in col for sp in sp_list)
        ]

        # Sort columns with species_info
        columns_sp_group = sorted(columns_sp_group)

        # Concatenate columns with and without species_info
        sorted_columns = columns_no_sp_group + columns_sp_group

        # Select the cols based on the sorted list
        merged_df = merged_df[sorted_columns]

        # Download the processed detections as a csv file
        csv_filename = (
            self.project.Project_name + str(datetime.date.today()) + "detections.csv"
        )

        merged_df.to_csv(csv_filename, index=False)

        logging.info(
            f"The detections organised by species cols have been downloaded to {csv_filename}"
        )


class MLProjectProcessor(ProjectProcessor):
    def __init__(
        self,
        project: project_utils.Project,
        config_path: str = None,
        weights_path: str = None,
        output_path: str = None,
        classes: list = [],
    ):
        super().__init__(project)
        print(self.project)
        self.project_name = self.project.Project_name.lower().replace(" ", "_")
        self.data_path = config_path
        self.weights_path = weights_path
        self.output_path = output_path
        self.classes = classes
        self.run_history = None
        self.best_model_path = None
        self.model_type = 1  # set as 1 for testing
        self.train, self.run, self.test = (None,) * 3

        self.registry = None

        logging.info("helooooooooooooooo123")
        if "MLFLOW_TRACKING_URI" in os.environ:
            if os.environ["MLFLOW_TRACKING_URI"] is not None:
                self.registry = "mlflow"

        # Before t6_utils gets loaded in, the val.py file in yolov5_tracker repository needs to be removed
        # to prevent the batch_size error, see issue kso-object-detection #187
        path_to_val = Path(sys.path[0], "yolov5_tracker/val.py")
        try:
            if path_to_val.exists():
                path_to_val.unlink()
        except OSError:
            pass
        import cv2

        # Monkey-patch the cv2.VideoWriter class to use the default codec
        class CustomVideoWriter(cv2.VideoWriter):
            def __init__(self, *args, **kwargs):
                args = list(args)
                if len(args) > 0:
                    args[0] = args[0].replace(".avi", ".mp4")
                    args[1] = cv2.VideoWriter_fourcc(*"avc1")
                super().__init__(*args, **kwargs)

        # Replace cv2.VideoWriter with the patched version
        cv2.VideoWriter = CustomVideoWriter

        self.modules = g_utils.import_modules(["zenodo_utils"], utils=True)
        self.modules.update(
            g_utils.import_modules(
                [
                    "torch",
                    "wandb",
                    "yaml",
                    "ultralytics",
                    "pims",
                    "itertools",
                    "mlflow",
                ],
                utils=False,
            )
        )
        # Import model models for backwards compatibility
        if self.registry == "wandb":
            self.modules.update(
                g_utils.import_model_modules(
                    ["yolov5.train", "yolov5.detect", "yolov5.val"],
                )
            )

        self.team_name = "koster"
        logging.info("ML Project successfully initialised.")

    def prepare_dataset(
        self,
        agg_df: pd.DataFrame,
        out_path: str,
        perc_test: float = 0.2,
        img_size: tuple = (224, 224),
        remove_nulls: bool = False,
        track_frames: bool = False,
        n_tracked_frames: int = 0,
        out_format: str = "yolo",
    ):
        species_list = kso_widgets.choose_species(
            self.db_connection, agg_df.label.unique().tolist()
        )

        button = widgets.Button(
            description="Aggregate frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )

        def on_button_clicked(b):
            self.species_of_interest = species_list.value
            # code for prepare dataset for machine learning
            yolo_utils.frame_aggregation(
                project=self.project,
                server_connection=self.server_connection,
                db_connection=self.db_connection,
                out_path=out_path,
                perc_test=perc_test,
                class_list=self.species_of_interest,
                img_size=img_size,
                remove_nulls=remove_nulls,
                track_frames=track_frames,
                n_tracked_frames=n_tracked_frames,
                agg_df=agg_df,
                out_format=out_format,
            )

            button.on_click(on_button_clicked)
            display(button)

    #############
    # t5
    #############
    def choose_baseline_model(self, download_path: str):
        """
        It downloads the latest version of the baseline model from WANDB
        :return: The path to the baseline model.
        """
        if self.registry == "wandb":
            api = self.modules["wandb"].Api()
            # weird error fix (initialize api another time)
            api.runs(path="koster/model-registry")
            api = self.modules["wandb"].Api()
            collections = [
                coll
                for coll in api.artifact_type(
                    type_name="model", project="koster/model-registry"
                ).collections()
            ]

            model_dict = {}
            for artifact in collections:
                model_dict[artifact.name] = artifact

            model_widget = widgets.Dropdown(
                options=[(name, model) for name, model in model_dict.items()],
                value=None,
                description="Select model:",
                ensure_option=False,
                disabled=False,
                layout=widgets.Layout(width="50%"),
                style={"description_width": "initial"},
            )

            main_out = widgets.Output()
            display(model_widget, main_out)

            def on_change(change):
                with main_out:
                    clear_output()
                    try:
                        for af in model_dict[change["new"].name]:
                            artifact_dir = af.download(download_path)
                            artifact_file = [
                                str(i)
                                for i in Path(artifact_dir).iterdir()
                                if str(i).endswith(".pt")
                            ][-1]
                            logging.info(
                                f"Baseline {af.name} successfully downloaded from WANDB"
                            )
                            model_widget.artifact_path = artifact_file
                    except Exception as e:
                        logging.error(
                            f"Failed to download the baseline model. Please ensure you are logged in to WANDB. {e}"
                        )
                        model_widget.artifact_path = "yolov8m.pt"

            model_widget.observe(on_change, names="value")
            return model_widget

        elif self.registry == "mlflow":
            # Fetch model artifact list
            from mlflow import MlflowClient

            experiment = self.modules["mlflow"].get_experiment_by_name(
                self.project_name
            )
            client = MlflowClient()

            if experiment is not None:
                experiment_id = experiment.experiment_id if experiment else None
                runs = self.modules["mlflow"].search_runs(
                    experiment_ids=experiment_id, output_format="list"
                )
                run_ids = [run.info.run_id for run in runs][-1:]
                # Choose only the project directory
                try:
                    artifacts = [
                        (
                            list(
                                filter(
                                    lambda x: x.is_dir
                                    and "input_datasets" not in x.path,
                                    client.list_artifacts(i),
                                )
                            )[0],
                            i,
                        )
                        for i in run_ids
                    ]
                    model_names = [
                        str(
                            Path(
                                "runs:/",
                                run_id,
                                artifact.path,
                                "best.pt",
                            )
                        )
                        for artifact, run_id in artifacts
                    ]
                    model_names = [m for m in model_names if "detection" not in m]
                except IndexError:
                    model_names = []
            else:
                model_names = []
            model_names.append("Yolov8 Baseline Object Detection Model")
            model_names.append("Yolov8 Baseline Object Segmentation Model")

            model_widget = widgets.Dropdown(
                options=model_names,
                value=None,
                description="Select model:",
                ensure_option=False,
                disabled=False,
                layout=widgets.Layout(width="50%"),
                style={"description_width": "initial"},
            )

            main_out = widgets.Output()
            display(model_widget, main_out)

            def on_change(change):
                with main_out:
                    clear_output()
                    try:
                        for af in model_names:
                            artifact_dir = self.modules["mlflow"].download_artifacts(
                                artifact_uri=af, dst_path=download_path
                            )
                            artifact_file = [
                                str(i)
                                for i in Path(artifact_dir).iterdir()
                                if str(i).endswith(".pt")
                            ][-1]
                            logging.info(
                                f"Baseline {af.name} successfully downloaded from WANDB"
                            )
                            model_widget.artifact_path = artifact_file
                    except Exception as e:
                        logging.error(
                            f"Failed to download the baseline model from MLFlow. The default baseline model will be used. {e}"
                        )
                        if change["new"] == "Yolov8 Baseline Object Detection Model":
                            model_widget.artifact_path = "yolov8m.pt"
                        elif (
                            change["new"] == "Yolov8 Baseline Object Segmentation Model"
                        ):
                            model_widget.artifact_path = "yolov8m-seg.pt"
                        else:
                            model_widget.artifact_path = "yolov8m.pt"

            model_widget.observe(on_change, names="value")

            # Display the dropdown widget
            return model_widget
        else:
            logging.error("Registry not supported.")

    def choose_entity(self, alt_name: bool = False):
        if self.team_name is None:
            return kso_widgets.choose_entity()
        else:
            if not alt_name:
                logging.info(
                    f"Found team name: {self.team_name}. If you want"
                    " to use a different team name for this experiment"
                    " set the argument alt_name to True"
                )
            else:
                return kso_widgets.choose_entity()

    def setup_paths(self):
        if not isinstance(self.output_path, str) and self.output_path is not None:
            self.output_path = self.output_path.selected
        self.data_path, self.hyp_path = yolo_utils.setup_paths(
            self.output_path, self.model_type
        )

    def choose_train_params(self):
        return kso_widgets.choose_train_params(self.model_type)

    def train_yolo(
        self,
        exp_name: str,
        weights: str,
        project: str,
        epochs: int = 1,
        batch_size: int = 16,
        img_size: int = 128,
    ):
        # Disable wandb (not necessary yet)
        if self.registry == "wandb":
            self.modules["ultralytics"].settings.update({"wandb": True})
        elif self.registry == "mlflow":
            self.modules["ultralytics"].settings.update({"mlflow": True})

        if self.registry == "mlflow":
            active_run = self.modules["mlflow"].active_run()

            from mlflow.data.pandas_dataset import PandasDataset

            parent_dir = Path(self.data_path).parent
            train_path = str(Path(parent_dir, "train.txt"))
            valid_path = str(Path(parent_dir, "valid.txt"))
            train_df = pd.read_csv(train_path, delimiter="\t")
            val_df = pd.read_csv(valid_path, delimiter="\t")
            train_dataset: PandasDataset = self.modules["mlflow"].data.from_pandas(
                train_df, source=train_path
            )
            val_dataset: PandasDataset = self.modules["mlflow"].data.from_pandas(
                val_df, source=valid_path
            )

            from mlflow.exceptions import MlflowException

            try:
                experiment_id = self.modules["mlflow"].create_experiment(
                    self.project_name,
                )
            except MlflowException as e:
                # Check if the experiment already exists
                if "RESOURCE_ALREADY_EXISTS" in str(e):
                    current_experiment = self.modules["mlflow"].get_experiment_by_name(
                        self.project_name
                    )
                    experiment_id = current_experiment.experiment_id
                else:
                    # Handle other MlflowExceptions
                    raise e

            self.modules["mlflow"].start_run(
                experiment_id=experiment_id, run_name=exp_name
            )
            # Wait 1 minute for MLFlow to register the new run
            time.sleep(60)
            self.modules["mlflow"].log_input(train_dataset, context="training")
            self.modules["mlflow"].log_input(val_dataset, context="validation")

            if not Path(Path(self.data_path).parent, "images.zip").exists():
                shutil.make_archive(
                    Path(Path(self.data_path).parent, "images"),
                    "zip",
                    Path(Path(self.data_path).parent, "images"),
                )

            if not Path(Path(self.data_path).parent, "labels.zip").exists():
                shutil.make_archive(
                    Path(Path(self.data_path).parent, "labels"),
                    "zip",
                    Path(Path(self.data_path).parent, "labels"),
                )

            # Upload zip files
            self.modules["mlflow"].log_artifact(
                Path(Path(self.data_path).parent, "images.zip"),
                artifact_path="input_datasets",
            )
            self.modules["mlflow"].log_artifact(
                Path(Path(self.data_path).parent, "labels.zip"),
                artifact_path="input_datasets",
            )

            # Upload yaml files
            # Iterate over all files in the specified directory
            for yaml_file in Path(self.data_path).parent.rglob(
                "*.yaml"
            ):  # rglob searches recursively for all .yaml files
                # Log each .yaml file as an artifact
                self.modules["mlflow"].log_artifact(
                    yaml_file, artifact_path="input_datasets"
                )

            # Upload txt files
            # Iterate over all files in the specified directory
            for txt_file in Path(self.data_path).parent.glob(
                "*.txt"
            ):  # rglob searches recursively for all .yaml files
                # Log each .yaml file as an artifact
                self.modules["mlflow"].log_artifact(
                    txt_file, artifact_path="input_datasets"
                )
        try:
            if "yolov5" in weights:
                weights = str(Path(weights).name)
    
    
    
    
            
    
    
            # from ultralytics import YOLO
            # from ultralytics.nn.tasks import ClassificationModel, PoseModel, SegmentationModel, DetectionModel
            # import torch
            # torch.serialization.add_safe_globals([
            #     DetectionModel,
            #     ClassificationModel,
            #     PoseModel,
            #     SegmentationModel
            # ])
    
            model = self.modules["ultralytics"].YOLO("yolov8n.pt")
            # model = YOLO("yolov8n.pt")
                    
            model.train(
                data=self.data_path,
                project=project,
                name=exp_name,
                epochs=int(epochs),
                batch=int(batch_size),
                imgsz=img_size,
            )
                
            
        except Exception as e:
            logging.info(f"Training failed due to, : {e}")
        # Close down run
        if self.registry == "wandb":
            self.modules["wandb"].finish()
        elif self.registry == "mlflow":
            self.modules["mlflow"].end_run()

    def enhance_yolo(
        self, in_path: str, project_path: str, conf_thres: float, img_size=[640, 640]
    ):
        from datetime import datetime

        run_name = f"enhance_run_{datetime.now()}"
        self.run_path = Path(project_path, run_name)
        logging.info("Enhancement running...")
        model = self.modules["ultralytics"].YOLO(self.tuned_weights)
        model.predict(
            source=str(Path(in_path, "images")),
            conf=conf_thres,
            save_txt=True,
            save_conf=True,
            save=True,
            imgsz=img_size,
        )

        if self.modules["wandb"].run is not None:
            self.modules["wandb"].finish()

    def enhance_replace(self, data_path: str):
        if self.model_type == 1:
            # Rename the 'labels' directory to 'labels_org'
            data_path = Path(data_path)
            data_path.joinpath("labels").rename(data_path.joinpath("labels_org"))
            # Rename the 'labels' directory inside 'self.run_path' to 'labels'
            self.run_path.joinpath("labels").rename(data_path.joinpath("labels"))
        else:
            logging.error("This option is not supported for other model types.")

    #############
    # t6
    #############
    # Function to choose a model to evaluate

    def choose_model(self, custom_project: str = "", publish: bool = False):
        """
        It takes a project name that is defined in the class and returns a dropdown widget that displays the metrics of the model
        selected

        :param project_name: The name of the project you want to load the model from
        :return: The model_widget is being returned.
        """
        # TODO: Remove hardcoded API key from Zenodo
        model_dict = self.modules[
            "zenodo_utils"
        ].download_and_extract_models_from_zenodo(
        # "lf6l81UGtKTZTmFIMCtOV7dTTXBcSDIF7f2jwLjjaCfJvhEiH5wgRu1v6HvE" # kalindi
        "pClzrdKwErArGWuPXMje0OtLEaq2gM8vHcAEeQN9CXyS2IjbuJsw05JLjVII"  # koster
        )
        model_info = {v: {"data": "No model info"} for k, v in model_dict.items()}
        data_info = {v: {"data": "No data info"} for k, v in model_dict.items()}
        if self.registry == "mlflow" and not publish:
            # Fetch model artifact list
            from mlflow import MlflowClient

            experiment = self.modules["mlflow"].get_experiment_by_name(
                self.project_name
            )
            client = MlflowClient()

            if experiment is not None:
                experiment_id = experiment.experiment_id if experiment else None
                runs = self.modules["mlflow"].search_runs(
                    experiment_ids=experiment_id, output_format="list"
                )

                for run in runs:
                    # Choose only the project directory
                    try:
                        artifacts = [
                            list(
                                filter(
                                    lambda x: x.is_dir
                                    and "input_datasets" not in x.path,
                                    client.list_artifacts(run.info.run_id),
                                ),
                            )
                        ]

                        if len(artifacts) > 0:
                            model_dict[run.info.run_name] = str(
                                Path(
                                    "runs:/",
                                    run.info.run_id,
                                    artifacts[0][0].path,
                                    "best.pt",
                                )
                            )
                            model_dict = {
                                m_name: m_path
                                for m_name, m_path in model_dict.items()
                                if "detection" not in m_name
                            }

                    except IndexError:
                        pass
            else:
                model_dict = {"No model": "yolov8m.pt"}

            # Create the dropdown widget
            model_widget = widgets.Dropdown(
                options=[(name, model) for name, model in model_dict.items()],
                description="Select model: ",
            )

            # Display the dropdown widget
            display(model_widget)
            return model_widget

        elif self.registry == "wandb" and not publish:
            api = self.modules["wandb"].Api()

            # weird error fix (initialize api another time)
            if len(custom_project) > 0:
                logging.info(
                    "Please note: Using models from custom project, please ensure that you have access."
                )
                full_path = custom_project
                api.runs(path=full_path).objects
            elif self.project_name == "template_project":
                full_path = f"{self.team_name}/spyfish_aotearoa"

            else:
                full_path = f"{self.team_name}/{self.project_name}"

            runs = api.runs(full_path)

            if len(runs) > 100:
                runs = list(runs)[:100]

            for run in runs:
                model_artifacts = [
                    artifact
                    for artifact in self.modules["itertools"].chain(
                        run.logged_artifacts(), run.used_artifacts()
                    )
                    if artifact.type == "model"
                ]
                if len(model_artifacts) > 0:
                    model_dict[run.name] = model_artifacts[0].name.split(":")[0]
                    model_info[model_artifacts[0].name.split(":")[0]] = run.summary
                    data_info[model_artifacts[0].name.split(":")[0]] = run.config

            # Add "no movie" option to prevent conflicts
            # models = np.append(list(model_dict.keys()),"No model")

            model_widget = widgets.Dropdown(
                options=[(name, model) for name, model in model_dict.items()],
                description="Select model:",
                ensure_option=False,
                disabled=False,
                value=None,
                layout=widgets.Layout(width="50%"),
                style={"description_width": "initial"},
            )

            main_out = widgets.Output()
            display(model_widget, main_out)

            # Display model metrics
            def on_change(change):
                with main_out:
                    clear_output()
                    if change["new"] == "No file":
                        logging.info("Choose another file")
                    else:
                        if self.project_name == "model-registry":
                            logging.info("No metrics available")
                        else:
                            self.data_path = data_info[change["new"]]["data"]
                            logging.info(
                                {
                                    k: v
                                    for k, v in model_info[change["new"]].items()
                                    if "metrics" in k
                                }
                            )

            model_widget.observe(on_change, names="value")
            return model_widget

        elif self.registry is None or publish:
            # Create the dropdown widget
            model_widget = widgets.Dropdown(
                options=[(name, model) for name, model in model_dict.items()],
                description="Select Zenodo model: ",
                display="flex",
                flex_flow="column",
                align_items="stretch",
                style={"description_width": "initial"},
            )
            # Display the dropdown widget
            print([(name, model) for name, model in model_dict.items()])
            display(model_widget)
            return model_widget
        else:
            logging.error("The chosen registry is not available at the moment.")
            return

    def eval_yolo(self, exp_name: str, conf_thres: float):
        # Find trained model weights
        project_path = Path(self.project_name, exp_name)
        self.tuned_weights = f"{Path(project_path, 'weights', 'best.pt')}"
        try:
            model = self.modules["ultralytics"].YOLO(self.tuned_weights)
            model.val(
                data=self.data_path,
                conf=conf_thres,
            )
        except Exception as e:
            logging.error(f"Encountered {e}, terminating run...")
            self.modules["wandb"].finish()
        logging.info("Run succeeded, finishing run...")
        self.modules["wandb"].finish()

    def _process_results(self, src, results):
        fc = 0
        if Path(src).is_dir():
            obj = [f for f in Path(src).iterdir() if f.is_file()]
        else:
            obj = self.modules["pims"].Video(src)  # store video capture object
        inc = 0
        for r in results:
            fc += 1
            if fc > 1:
                end = time.time()
                inc = end - st
            t = sum(r.speed.values()) / 1000
            t_left = (len(obj) - fc) * max(t, inc)  # conservative
            st = time.time()
            statement = f"Processed item {fc} / {len(obj)} in {t*1000} ms. Estimated remaining time: {round(t_left, 2)}s."
            if t_left < 60:
                logging.info(f"{statement} Almost there! ⏳")
            else:
                logging.info(f"{statement} Grab a ☕")
        logging.info("Prediction completed successfully! ✅")

    def detect_yolo(
        self,
        project: str,
        name: str,
        source: str,
        save_dir: str,
        conf_thres: float,
        artifact_dir: str,
        model: str,
        img_size: int = 640,
        save_output: bool = True,
        latest: bool = True,
        out_format: str = "yolo",
    ):
        from yolov5.utils.general import increment_path

        if self.registry == "mlflow":
            active_run = self.modules["mlflow"].active_run()
            if active_run:
                self.modules["mlflow"].end_run()
            experiment = self.modules["mlflow"].get_experiment_by_name(
                self.project_name
            )
            self.modules["mlflow"].start_run(
                run_name=self.project_name + "_detection",
                experiment_id=experiment.experiment_id,
            )
            self.run = self.modules["mlflow"].active_run()
        elif self.registry == "wandb":
            self.run = self.modules["wandb"].init(
                entity=self.team_name,
                project="model-evaluations",
                settings=self.modules["wandb"].Settings(start_method="thread"),
            )
        models = [
            str(f)
            for f in Path(artifact_dir).iterdir()
            if f.is_file()
            and str(f).endswith((".pt", ".model"))
            and "osnet" not in str(f)
            and "best" in str(f)
        ]
        if len(models) > 0:
            best_model = models[0]
        else:
            logging.info("No trained model found, using yolov8 base model...")
            best_model = "yolov8s.pt"

        model = self.modules["ultralytics"].YOLO(best_model)
        project = str(Path(save_dir))
        self.eval_dir = str(increment_path(Path(project) / name, exist_ok=False))
        if latest:
            if isinstance(source, list):
                for src in source:
                    results = model.predict(
                        project=project,
                        name=name,
                        source=src,
                        conf=conf_thres,
                        save_txt=True,
                        save_conf=True,
                        save=save_output,
                        imgsz=img_size,
                        stream=True,
                        verbose=False,
                    )
                    self._process_results(src, results)

            else:
                results = model.predict(
                    project=project,
                    name=name,
                    source=source,
                    conf=conf_thres,
                    save_txt=True,
                    save_conf=True,
                    save=save_output,
                    imgsz=img_size,
                    stream=True,
                    verbose=False,
                )
                self._process_results(source, results)
        else:
            logging.error(
                "We do not currently support running YoloV5 models. Please re-train models "
                "using the latest model version available"
            )
            return
            # if isinstance(source, list):
            #     for src in source:
            #         self.modules["detect"].run(
            #             weights=best_model,
            #             source=src,
            #             conf_thres=conf_thres,
            #             save_txt=True,
            #             save_conf=True,
            #             project=save_dir,
            #             name=name,
            #             nosave=not save_output,
            #         )
            # else:
            #     self.modules["detect"].run(
            #         weights=best_model,
            #         source=source,
            #         conf_thres=conf_thres,
            #         save_txt=True,
            #         save_conf=True,
            #         project=save_dir,
            #         name=name,
            #         nosave=not save_output,
            #     )
        self._save_detections(conf_thres, model.ckpt_path, self.eval_dir, out_format)

    def _save_detections(
        self, conf_thres: float, model: str, eval_dir: str, out_format: str = "yolo"
    ):
        if self.registry == "wandb":
            import yaml

            def read_yaml_file(file_path):
                with open(file_path, "r") as file:
                    yaml_data = yaml.safe_load(file)
                return yaml_data

            # Read species mapping into data dictionary
            try:
                data_dict = read_yaml_file(self.data_path)
                species_mapping = data_dict["names"]
            except FileNotFoundError:
                # Handle the case when the file doesn't exist
                logging.info(f"File not found: {self.data_path}")
                species_mapping = {}
            except KeyError:
                # Handle the case when the "names" key is missing
                logging.info("Key 'names' not found in the YAML file.")
                species_mapping = {}
            except Exception as e:
                # Handle any other unexpected errors
                logging.info(f"An unexpected error occurred: {e}")
                species_mapping = {}

            yolo_utils.set_config(
                conf=conf_thres,
                model_name=model,
                evaluation_directory=eval_dir,
                species_map=species_mapping,
            )
            self.csv_report = yolo_utils.generate_csv_report(
                evaluation_path=eval_dir,
                log=True,
                registry=self.registry,
                movie_csv_df=self.local_movies_csv,
                out_format=out_format,
            )
            yolo_utils.add_data(
                Path(eval_dir, "annotations.csv"),
                "detection_output",
                self.registry,
                self.run,
            )
            import shutil

            shutil.make_archive(
                Path(eval_dir, "labels"), "zip", Path(eval_dir, "labels")
            )
            yolo_utils.add_data(
                Path(eval_dir, "labels"),
                "detection_output",
                self.registry,
                self.run,
            )
        elif self.registry == "mlflow":
            self.csv_report = yolo_utils.generate_csv_report(
                evaluation_path=eval_dir,
                log=True,
                registry=self.registry,
                movie_csv_df=self.local_movies_csv,
                out_format=out_format,
            )
            yolo_utils.add_data(
                path=Path(eval_dir, "annotations.csv"),
                name="detection_output",
                registry=self.registry,
                run=self.run,
            )
            import shutil

            shutil.make_archive(
                Path(eval_dir, "labels"), "zip", Path(eval_dir, "labels")
            )
            yolo_utils.add_data(
                path=Path(eval_dir, "labels.zip"),
                name="detection_output",
                registry=self.registry,
                run=self.run,
            )
        elif self.registry is None:
            self.csv_report = yolo_utils.generate_csv_report(
                evaluation_path=eval_dir,
                log=True,
                registry=self.registry,
                movie_csv_df=self.local_movies_csv,
                out_format=out_format,
            )
        else:
            logging.error("Invalid registry name")
            return

    def _increment_path(self, path, exist_ok=False, sep="", mkdir=False):
        # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
        path = Path(path)  # os-agnostic
        if path.exists() and not exist_ok:
            path, suffix = (
                (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
            )

            # Method 1
            for n in range(2, 9999):
                p = path.with_name(f"{path.stem}{sep}{n}{suffix}")  # increment path
                if not p.exists():
                    break

            # Method 2 (deprecated)
            # dirs = glob.glob(f"{path}{sep}*")  # similar paths
            # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
            # i = [int(m.groups()[0]) for m in matches if m]  # indices
            # n = max(i) + 1 if i else 2  # increment number
            # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

        if mkdir:
            p.mkdir(parents=True, exist_ok=True)  # make directory

        return p

    def track_individuals(
        self,
        name: str,
        source: str,
        artifact_dir: str,
        conf_thres: float,
        img_size: tuple = (540, 540),
    ):
        if not hasattr(self, "eval_dir"):
            self.eval_dir = str(
                self._increment_path(
                    path=Path(self.save_dir) / "detect", exist_ok=False
                )
            )

        latest_tracker = yolo_utils.track_objects(
            name=name,
            source_dir=source,
            artifact_dir=artifact_dir,
            tracker_folder=self.eval_dir,
            conf_thres=conf_thres,
            img_size=img_size,
            gpu=True if self.modules["torch"].cuda.is_available() else False,
        )

        # Create a new run for tracking only if necessary
        if self.registry == "wandb":
            self.run = self.modules["wandb"].init(
                entity=self.team_name,
                project="model-evaluations",
                name="track",
                settings=self.modules["wandb"].Settings(start_method="thread"),
            )
            yolo_utils.set_config(
                conf=conf_thres,
                model_name=artifact_dir,
                evaluation_directory=self.eval_dir,
            )

        # self.csv_report = yolo_utils.generate_csv_report(
        #    self.team_name, self.project_name, eval_dir, self.run, log=True
        # )
        self.tracking_report = yolo_utils.generate_counts(
            self.eval_dir,
            latest_tracker,
            artifact_dir,
            self.run,
            log=True,
            registry=self.registry,
        )
        yolo_utils.add_data(
            str(Path(latest_tracker).parent.absolute()),
            "tracker_output",
            self.registry,
            self.run,
        )
        if self.registry == "wandb":
            self.modules["wandb"].finish()
        elif self.registry == "mlflow":
            self.modules["mlflow"].end_run()

    def get_model(self, model_name: str, download_path: str, custom_project: str = ""):
        """
        It downloads the latest model checkpoint from the specified project and model name

        :param model_name: The name of the model you want to download
        :type model_name: str
        :param project_name: The name of the project you want to download the model from
        :type project_name: str
        :param download_path: The path to download the model to
        :type download_path: str
        :return: The path to the downloaded model checkpoint.
        """
        if ".pt" in model_name and ":" not in model_name:
            logging.info("Local model successfully loaded.")
            return str(Path(model_name).parent)

        if self.registry == "mlflow":
            artifact_dir = self.modules["mlflow"].artifacts.download_artifacts(
                model_name, dst_path=download_path
            )
            logging.info("MLFLow model successfully loaded.")
            return str(Path(artifact_dir).parent)

        elif self.registry == "wandb":
            # weird error fix (initialize api another time)
            if len(custom_project) > 0:
                logging.info(
                    "Please note: Using models from custom project, please ensure that you have access."
                )
                full_path = custom_project
            else:
                if self.team_name == "wildlife-ai":
                    logging.info(
                        "Please note: Using models from adi-ohad-heb-uni account."
                    )
                    full_path = "adi-ohad-heb-uni/project-wildlife-ai"
                elif self.project_name == "template_project":
                    full_path = f"{self.team_name}/spyfish_aotearoa"
                else:
                    full_path = f"{self.team_name}/{self.project_name.lower()}"
            api = self.modules["wandb"].Api()
            try:
                api.artifact_type(type_name="model", project=full_path).collections()
            except Exception as e:
                logging.error(
                    f"No model collections found. No artifacts have been logged. {e}"
                )
                return None
            collections = [
                coll
                for coll in api.artifact_type(
                    type_name="model", project=full_path
                ).collections()
            ]
            model = [i for i in collections if i.name == model_name]
            if len(model) > 0:
                model = model[0]
            else:
                logging.error("No model found")
            artifact = api.artifact(full_path + "/" + model.name + ":latest")
            logging.info("Downloading model checkpoint...")
            artifact_dir = artifact.download(root=download_path)
            logging.info("Checkpoint downloaded.")
            return str(Path(artifact_dir).resolve())
        else:
            return

    def get_dataset(
        self,
        model: str,
        team_name: str = "koster",
    ):
        """
        It takes in a project name and a model name, and returns the paths to the train and val datasets

        :param project_name: The name of the project you want to download the dataset from
        :type project_name: str
        :param model: The model you want to use
        :type model: str
        :return: The return value is a list of two directories, one for the training data and one for the validation data.
        """
        if self.registry == "mlflow":
            logging.error("This is not currently supported for MLflow")
            return "", ""
        elif self.registry == "wandb":
            api = self.modules["wandb"].Api()
            if "_" in model:
                run_id = model.split("_")[1]
                try:
                    run = api.run(f"{team_name}/{self.project_name}/runs/{run_id}")
                except self.modules["wandb"].CommError:
                    logging.error("Run data not found")
                    return "", ""
                datasets = [
                    artifact
                    for artifact in run.used_artifacts()
                    if artifact.type == "dataset"
                ]
                if len(datasets) == 0:
                    logging.error(
                        "No datasets are linked to these runs. Please try another run."
                    )
                    return "", ""
                dirs = []
                for i in range(len(["train", "val"])):
                    artifact = datasets[i]
                    logging.info(f"Downloading {artifact.name} checkpoint...")
                    artifact_dir = artifact.download()
                    logging.info(f"{artifact.name} - Dataset downloaded.")
                    dirs.append(artifact_dir)
                return dirs
            else:
                logging.error("Externally trained model. No data available.")
                return "", ""
        else:
            logging.error("Unsupported registry")
            return "", ""
