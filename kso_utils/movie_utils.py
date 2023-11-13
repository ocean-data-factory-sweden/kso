# base imports
import sys
import os
import cv2
import logging
import subprocess
import urllib
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from urllib.request import pathname2url
from IPython.display import HTML

# util imports
from kso_utils.project_utils import Project

# server imports
from kso_utils.server_utils import ServerType, get_matching_s3_keys

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# Function to prevent issues with Swedish characters
# Converting the Swedish characters ä and ö to utf-8.
def unswedify(string: str):
    """Convert ä and ö to utf-8"""
    return (
        string.encode("utf-8")
        .replace(b"\xc3\xa4", b"a\xcc\x88")
        .replace(b"\xc3\xb6", b"o\xcc\x88")
        .decode("utf-8")
    )


# Function to prevent issues with Swedish characters
def reswedify(string: str):
    """Convert ä and ö to utf-8"""
    return (
        string.encode("utf-8")
        .replace(b"a\xcc\x88", b"\xc3\xa4")
        .replace(b"o\xcc\x88", b"\xc3\xb6")
        .decode("utf-8")
    )


def get_fps_duration(movie_path: str):
    """
    This function takes the path (or url) of a movie and returns its fps and duration information

    :param movie_path: a string containing the path (or url) where the movie of interest can be access from
    :return: Two integers, the fps and duration of the movie
    """
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Roadblock to prevent issues with missing movies
    if int(frame_count) | int(fps) == 0:
        raise ValueError(
            f"{movie_path} doesn't have any frames, check the path/link is correct."
        )
    else:
        duration = frame_count / fps

    return fps, duration


def get_movie_path(f_path: str, project: Project, server_connection: dict = None):
    """
    Function to get the path (or url) of a movie

    :param f_path: string with the original path of a movie
    :param project: the project object
    :param server_connection: a dictionary with the connection to the server
    :return: a string containing the path (or url) where the movie of interest can be access from

    """
    if project.server == ServerType.AWS:
        # Generate presigned url
        movie_url = server_connection["client"].generate_presigned_url(
            "get_object",
            Params={"Bucket": project.bucket, "Key": f_path},
            ExpiresIn=86400,
        )
        return movie_url

    else:
        logging.info(f"Returning the fpath {f_path}")
        return f_path


def movies_in_movie_folder(project: Project, db_connection, server_connection: dict):
    """
    This function uses the project information and the database information, and returns
    a dataframe of the movies in the "movie_folder".

    :param project: the project object
    :param server_connection: a dictionary with the connection to the server
    :param db_connection: SQL connection object
    :return: A dataframe with the following columns (index, movie_id, fpath, exists, filename_ext)

    """

    # Retrieve the list of movies available in Wildlife.ai
    if project.server == ServerType.TEMPLATE:
        # Combine wildlife.ai storage and filenames of the movie examples
        available_movies_list = [
            "https://www.wildlife.ai/wp-content/uploads/2022/06/" + filename
            for filename in [f"movie_{i}.mp4" for i in range(1, 6)]
        ]

        # Save the list of movies as a pd df
        mov_folder_df = pd.Series(available_movies_list, name="fpath").to_frame()

    # Retrieve the list of local movies available
    elif project.server in [ServerType.LOCAL, ServerType.SNIC]:
        logging.info("Retrieving movies that are available locally")
        # Read the movie files from the movie_path folder
        local_files = os.listdir(project.movie_folder)
        available_movies_list = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(project.movie_folder)
            for f in filenames
            if os.path.splitext(f)[1].endswith(get_movie_extensions())
        ]

        # Save the list of movies as a pd df
        mov_folder_df = pd.Series(available_movies_list, name="fpath").to_frame()

    # Retrieve the list of movies available in AWS
    elif project.server == ServerType.AWS:
        logging.info("Retrieving movies that are available in AWS")
        # List all the movies available from the S3 bucket
        available_movies_list = get_matching_s3_keys(
            client=server_connection["client"],
            bucket=project.bucket,
            suffix=get_movie_extensions(),
        )

        # Save the list of movies as a pd df
        mov_folder_df = pd.Series(available_movies_list, name="fpath").to_frame()

    else:
        raise ValueError(f"Unsupported server type: {project.server}")

    return mov_folder_df


def retrieve_movie_info_from_server(
    project: Project, db_connection, server_connection: dict
):
    """
    This function uses the project information and the database information, and returns a dataframe with the
    movie information

    :param project: the project object
    :param server_connection: a dictionary with the connection to the server
    :param db_connection: SQL connection object
    :return: A dataframe with the following columns (index, movie_id, fpath, exists, filename_ext)

    """

    # Create a dataframe of the movies in the "movie_folder"
    mov_folder_df = movies_in_movie_folder(project, db_connection, server_connection)

    from kso_utils.db_utils import get_df_from_db_table

    # Temporarily retrieve the movies info from the db
    movies_df = get_df_from_db_table(conn=db_connection, table_name="movies")

    # Query info about the movie of interest
    movies_df = movies_df.rename(columns={"id": "movie_id"})

    if project.server == ServerType.SNIC:
        # Find closest matching filename (may differ due to Swedish character encoding)
        parsed_url = urllib.parse.urlparse(movies_df["fpath"].iloc[0])

        def get_match(string, string_options):
            normalized_string = unicodedata.normalize("NFC", string)
            for s in string_options:
                normalized_s = unicodedata.normalize("NFC", s)
                if normalized_string in normalized_s:
                    return s
            return None

        movies_df["fpath"] = movies_df["fpath"].apply(
            lambda x: get_match(x, mov_folder_df["fpath"].unique()),
            1,
        )

    # Merge the server path to the filepath
    all_movies_df = movies_df.merge(
        mov_folder_df,
        on=["fpath"],
        how="outer",
        indicator=True,
    )

    # Select only movies without information in the movies.csv
    no_info_movies_df = all_movies_df[all_movies_df["_merge"] == "right_only"].copy()

    # Select only movies without information in the movies.csv
    no_available_movies_df = all_movies_df[
        all_movies_df["_merge"] == "left_only"
    ].copy()

    # Check that movies can be mapped
    all_movies_df["exists"] = np.where(all_movies_df["_merge"] == "both", True, False)

    # Drop _merge columns to match sql schema
    all_movies_df = all_movies_df.drop("_merge", axis=1)

    # Select only those that can be mapped
    available_movies_df = all_movies_df[all_movies_df["exists"]].copy()

    # Create a filename with ext column
    available_movies_df["filename_ext"] = available_movies_df["fpath"].apply(
        lambda x: x.split("/")[-1], 1
    )

    # log the available movies
    n_movies = movies_df.shape[0]

    n_available_movies = available_movies_df.shape[0]

    if n_movies == n_available_movies:
        logging.info(f"All {n_movies} movies are mapped from the server")

    else:
        logging.info(
            f"{n_available_movies} out of {n_movies} movies are available."
            f"The missing movies are: {no_available_movies_df.fpath.unique()}"
        )

    return available_movies_df, no_available_movies_df, no_info_movies_df


# Function to preview underwater movies
def preview_movie(
    project: Project,
    available_movies_df: pd.DataFrame,
    movie_i: str,
    server_connection: dict,
):
    """
    It takes a movie filename and returns a HTML object that can be displayed in the notebook

    :param project: the project object
    :param available_movies_df: a dataframe with all the movies in the database
    :param movie_i: the filename of the movie you want to preview
    :param server_connection: a dictionary with the connection to the server
    :return: A tuple of two elements:
        1. HTML object
        2. Movie path
    """

    # Select the movie of interest
    movie_selected = available_movies_df[
        available_movies_df["filename"] == movie_i
    ].reset_index(drop=True)
    movie_selected_view = movie_selected.T
    movie_selected_view.columns = ["Movie summary"]

    # Make sure only one movie is selected
    if len(movie_selected.index) > 1:
        logging.info(
            "There are several movies with the same filename. This should be fixed!"
        )
        return None

    else:
        # Generate temporary path to the selected movie
        movie_path = get_movie_path(
            project=project,
            f_path=movie_selected["fpath"].values[0],
            server_connection=server_connection,
        )

        html_code = f"""<html>
                <div style="display: flex; justify-content: space-around; align-items: center">
                <div>
                  <video width=500 controls>
                  <source src={movie_path}>
                  </video>
                </div>
                <div>{movie_selected_view.to_html()}</div>
                </div>
                </html>"""

        return HTML(html_code), movie_path


def check_movie_uploaded(project: Project, db_connection, movie_i: str):
    """
    This function takes in a movie name and a dictionary containing the path to the database and returns
    a boolean value indicating whether the movie has already been uploaded to Zooniverse

    :param project: the project object
    :param movie_i: the name of the movie you want to check
    :type movie_i: str
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
    videos_uploaded = subjects_df.filename.dropna().unique()

    # Check if selected movie has already been uploaded
    already_uploaded = any(mv in movie_i for mv in videos_uploaded)

    if already_uploaded:
        clips_uploaded = subjects_df[subjects_df["filename"].str.contains(movie_i)]
        logging.info(f"{movie_i} has clips already uploaded.")
        logging.info(clips_uploaded.head())
    else:
        logging.info(f"{movie_i} has not been uploaded to Zooniverse yet")


# Function to extract selected frames from videos
def extract_frames(
    project: Project,
    server_connection: dict,
    df: pd.DataFrame,
    frames_folder: str,
):
    """
    Extract frames and save them in chosen folder.
    :param project: the project object
    :param server_connection: a dictionary with the connection to the server
    :param df: a dataframe of the frames to be extracted
    :param frames_folder: a string with the path of the folder to store the frames

    """

    # Set the filename of the frames
    df["frame_path"] = df.apply(
        lambda row: os.path.join(
            frames_folder, f"{row['filename']}_{row['frame_number']}_{row['label']}.jpg"
        ),
        axis=1,
    )

    # Create the folder to store the frames if not exist
    if not os.path.exists(frames_folder):
        Path(frames_folder).mkdir(parents=True, exist_ok=True)
        # Recursively add permissions to folders created
        [os.chmod(root, 0o777) for root, dirs, files in os.walk(frames_folder)]

    for movie in df["fpath"].unique():
        url = get_movie_path(
            f_path=movie, project=project, server_connection=server_connection
        )

        if url is None:
            logging.error(f"Movie {movie} couldn't be found in the server.")
        else:
            # Select the frames to download from the movie
            key_movie_df = df[df["fpath"] == movie].reset_index()

            # Read the movie on cv2 and prepare to extract frames
            write_movie_frames(key_movie_df, url)

        logging.info("Frames extracted successfully")

    return df


def write_movie_frames(key_movie_df: pd.DataFrame, url: str):
    """
    Function to get a frame from a movie
    :param key_movie_df: a df with the information of the movie
    :param url: a string with the url of the movie

    """
    # Read the movie on cv2 and prepare to extract frames
    cap = cv2.VideoCapture(url)

    if cap.isOpened():
        # Get the frame numbers for each movie the fps and duration
        for index, row in tqdm(key_movie_df.iterrows(), total=key_movie_df.shape[0]):
            # Create the folder to store the frames if not exist
            if not os.path.exists(row["frame_path"]):
                cap.set(1, row["frame_number"])
                ret, frame = cap.read()
                if frame is not None:
                    cv2.imwrite(row["frame_path"], frame)
                    os.chmod(row["frame_path"], 0o777)
                else:
                    cv2.imwrite(row["frame_path"], np.zeros((100, 100, 3), np.uint8))
                    os.chmod(row["frame_path"], 0o777)
                    logging.info(
                        f"No frame was extracted for {url} at frame {row['frame_number']}"
                    )
    else:
        logging.info(f"Missing movie {url}")


def get_movie_extensions():
    # Specify the formats of the movies to select
    return tuple(["wmv", "mpg", "mov", "avi", "mp4", "MOV", "MP4"])


def convert_video(
    movie_path: str,
    movie_filename: str,
    fps_output: str,
    gpu_available: bool = False,
    compression: bool = False,
):
    """
    It takes a movie file path and a boolean indicating whether a GPU is available, and returns a new
    movie file path.

    :param movie_path: The local path- or url to the movie file you want to convert
    :type movie_path: str
    :param movie_filename: The filename of the movie file you want to convert
    :type movie_path: str
    :param gpu_available: Boolean, whether or not a GPU is available
    :type gpu_available: bool
    :param compression: Boolean, whether or not movie compression is required
    :type compression: bool
    :param fps_output: String, argument used to force integer fps for movies
    :type fps_output: str
    :return: The path to the converted video file.
    """
    from kso_utils.tutorials_utils import is_url

    # Set the name of the converted movie
    conv_filename = "conv_" + movie_filename

    # Check the movie is accessible locally
    if os.path.exists(movie_path):
        # Store the directory and filename of the movie
        movie_fpath = os.path.dirname(movie_path)
        conv_fpath = os.path.join(movie_fpath, conv_filename)

    # Check if the path to the movie is a url
    elif is_url(movie_path):
        # Specify the directory to store the converted movie
        conv_fpath = os.path.join(conv_filename)

    else:
        logging.error(f"The path to {movie_path} is invalid")

    if gpu_available and compression:
        subprocess.call(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                str(movie_path),
                "-filter:v",
                fps_output,
                "-c:v",
                "h264_nvenc",  # ensures correct codec
                "-crf",
                "22",  # compresses the video
                str(conv_fpath),
            ]
        )

    elif gpu_available and not compression:
        subprocess.call(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                str(movie_path),
                "-filter:v",
                fps_output,
                "-c:v",
                "h264_nvenc",  # ensures correct codec
                str(conv_fpath),
            ]
        )

    elif not gpu_available and compression:
        subprocess.call(
            [
                "ffmpeg",
                "-i",
                str(movie_path),
                "-filter:v",
                fps_output,
                "-c:v",
                "h264",  # ensures correct codec
                "-crf",
                "22",  # compresses the video
                str(conv_fpath),
            ]
        )

    elif not gpu_available and not compression:
        subprocess.call(
            [
                "ffmpeg",
                "-i",
                str(movie_path),
                "-filter:v",
                fps_output,
                "-c:v",
                "h264",  # ensures correct codec
                str(conv_fpath),
            ]
        )
    else:
        raise ValueError(f"{movie_path} not modified")

    # Ensure open permissions on file (for now)
    os.chmod(conv_fpath, 0o777)
    logging.info("Movie file successfully converted and stored locally.")
    return conv_fpath


def standarise_movie_format(
    project: Project,
    server_connection: dict,
    movie_path: str,
    movie_filename: str,
    f_path: str,
    gpu_available: bool = False,
):
    """
    This function reviews the movie metadata. If the movie is not in the correct format, frame rate or codec,
    it is converted using ffmpeg.

    :param project: the project object
    :param movie_path: The local path- or url to the movie file you want to convert
    :type movie_path: str
    :param movie_filename: The filename of the movie file you want to convert
    :type movie_filename: str
    :param f_path: The server or storage path of the original movie you want to convert
    :type f_path: str
    :param gpu_available: Boolean, whether or not a GPU is available
    :type gpu_available: bool
    """

    from kso_utils.tutorials_utils import is_url
    from kso_utils.server_utils import upload_file_server

    ##### Check movie format ######
    ext = Path(movie_filename).suffix

    # Set video convertion to false as default
    convert_video_T_F = False

    if not ext.lower() == ".mp4":
        logging.info(f"Extension of {movie_filename} not supported.")
        # Set conversion to True
        convert_video_T_F = True

    ##### Check frame rate #######
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not float(fps).is_integer():
        logging.info(
            f"Variable frame rate {float(fps)} of {movie_filename} not supported."
        )
        # Set conversion to True
        convert_video_T_F = True

    ##### Check codec info ########
    def get_fourcc(cap: cv2.VideoCapture) -> str:
        """Return the 4-letter string of the codec of a video."""
        return (
            int(cap.get(cv2.CAP_PROP_FOURCC))
            .to_bytes(4, byteorder=sys.byteorder)
            .decode()
        )

    codec = get_fourcc(cap)

    if not codec in ["h264", "avc1"]:
        logging.info(
            f"The codecs of {movie_filename} are not supported (only h264 is supported)."
        )
        # Set conversion to True
        convert_video_T_F = True

    ##### Check movie file #######
    ##### (not needed in Spyfish) #####
    # Create a list of the project where movie compression is not needed
    project_no_compress = ["Spyfish_Aotearoa"]

    if project.Project_name in project_no_compress:
        # Set movie compression to false
        compress_video = False

    else:
        # Check movie filesize in relation to its duration
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        duration_mins = duration / 60

        # Check if the movie is accessible locally
        if os.path.exists(movie_path):
            # Store the size of the movie
            size = os.path.getsize(movie_path)

        # Check if the path to the movie is a url
        elif is_url(movie_path):
            # Store the size of the movie
            size = urllib.request.urlopen(movie_path).length

        else:
            logging.error(f"The path to {movie_path} is invalid")

        # Calculate the size:duration ratio
        sizeGB = size / (1024 * 1024 * 1024)
        size_duration = sizeGB / duration_mins

        if size_duration > 0.16:
            # Compress the movie if file size is too large
            logging.info(
                "File size of movie is too large (+5GB per 30 mins of footage)."
            )

            # Specify to compress the video
            compress_video = True
        else:
            # Set movie compression to false
            compress_video = False

    # Start converting/compressing video if movie didn't pass any of the checks
    if convert_video_T_F or compress_video:
        # Specify the desired fps of the movie
        fps_output = "fps=" + str(round(fps))

        conv_mov_path = convert_video(
            movie_path=movie_path,
            movie_filename=movie_filename,
            fps_output=fps_output,
            gpu_available=gpu_available,
            compression=compress_video,
        )

        # Upload the converted movie to the server
        upload_file_server(
            conv_mov_path=conv_mov_path,
            f_path=f_path,
            project=project,
            server_connection=server_connection,
        )

    else:
        logging.info(f"No modification needed for {movie_filename}")


def check_movies_meta(
    project: Project,
    csv_paths: dict,
    db_connection,
    available_movies_df: pd.DataFrame,
    no_info_movies_df: pd.DataFrame,
    server_connection: dict,
    review_method: str,
    gpu_available: bool = False,
):
    """
    > This function loads the csv with movies information, checks and updates missing info

    :param project: the project object
    :param db_connection: the sql connection to the db
    :param available_movies_df: a df with the information about the filepaths and "existance" of the movies
    :param csv_paths: a dictionary with the paths of the csv files with info to initiate the db
    :param server_connection: a dictionary with the connection to the server
    :param review_method: The method used to review the movies
    :param gpu_available: Boolean, whether or not a GPU is available
    """

    # Load the csv with movies information
    df = pd.read_csv(csv_paths["local_movies_csv"])

    from kso_utils.db_utils import cols_rename_to_schema

    # Rename the project-specific column names
    # that match schema standard names
    df = cols_rename_to_schema(
        project=project,
        table_name="movies",
        df=df,
    )

    # Check all available movies in the server/storage have information in the movies.csv
    # Get a list of all the available movies
    if not no_info_movies_df.empty:
        logging.info(
            f"There are {no_info_movies_df.shape[0]} movies in the movie_folder"
            f" that are not in the movies.csv. Their paths are: {no_info_movies_df.fpath.unique()}"
        )

    # Add information about whether the movies are available in the movie_folder
    df_temp = df.copy().drop(columns=["fpath"])
    df_toreview = df_temp.merge(
        available_movies_df[["filename", "fpath", "exists"]],
        on=["filename"],
        how="left",
    )

    if df_toreview.exists.isnull().values.any():
        # Replace na with False
        df_toreview["exists"] = df_toreview["exists"].fillna(False)

        logging.warn(
            f"Only # {df_toreview[df_toreview['exists']].shape[0]} out of"
            f"# {df_toreview[~df_toreview['exists']].shape[0]} movies with missing information are available."
            f" Proceeding to retrieve information for only those {df_toreview[df_toreview['exists']].shape[0]} available movies."
        )

        # Select only available movies
        df_toreview = df_toreview[df_toreview["exists"]].reset_index(drop=True)

    if df_toreview.empty:
        logging.info("There are no movies available to review.")
        return

    else:
        if review_method.startswith("Advanced"):
            logging.info("Checking the format, frame rate and codec of the movies")

            # Convert movies to the right format, frame rate or codec and upload them to the project's server/storage
            [
                standarise_movie_format(
                    project=project,
                    server_connection=server_connection,
                    movie_path=get_movie_path(j, project, server_connection),
                    movie_filename=i,
                    f_path=j,
                    gpu_available=gpu_available,
                )
                for i, j in tqdm(
                    zip(df_toreview["filename"], df_toreview["fpath"]),
                    total=df_toreview.shape[0],
                )
            ]

            # Specify to check the fps
            check_fps = True

        if review_method.startswith("Basic"):
            # Check if fps or duration is missing from any movie
            if not df_toreview[["fps", "duration"]].isna().any().any():
                # Specify to not check the fps
                check_fps = False

            else:
                # Create a df with only those rows with missing fps/duration
                df_toreview = df_toreview[
                    df_toreview["fps"].isna() | df_toreview["duration"].isna()
                ].reset_index(drop=True)

                logging.info(
                    "There are empty entries for fps, duration and sampling information"
                )
                # Specify to check the fps
                check_fps = True

        if check_fps:
            logging.info("Checking the fps and duration of the movies")
            # Retrieve the path of the movie (wheter the local path or a url),
            # get the fps/duration and overwrite the existing fps and duration info
            df_toreview[["fps", "duration"]] = pd.DataFrame(
                [
                    get_fps_duration(get_movie_path(i, project, server_connection))
                    for i in tqdm(df_toreview["fpath"], total=df_toreview.shape[0])
                ],
                columns=["fps", "duration"],
            )

        # Check if there are missing sampling starts
        empty_sampling_start = df_toreview["sampling_start"].isna()

        # Check if there are missing sampling ends
        empty_sampling_end = df_toreview["sampling_end"].isna()

        # Fill out missing sampling start information
        if empty_sampling_start.any():
            df_toreview.loc[empty_sampling_start, "sampling_start"] = 0.0
            mov_list = df_toreview[empty_sampling_start].filename.unique()
            logging.info(f"Added sampling_start of the movies {mov_list}")

        # Fill out missing sampling end information
        if empty_sampling_end.any():
            df_toreview.loc[empty_sampling_end, "sampling_end"] = df_toreview[
                "duration"
            ]
            mov_list = df_toreview[empty_sampling_end].filename.unique()
            logging.info(f"Added sampling_end of the movies {mov_list}")

        # Prevent sampling end times longer than actual movies
        if (df_toreview["sampling_end"] > df_toreview["duration"]).any():
            mov_list = df_toreview[
                df_toreview["sampling_end"] > df_toreview["duration"]
            ].filename.unique()
            raise ValueError(
                f"The sampling_end times of the following movies are longer than the actual movies {mov_list}"
            )

        # if there have not been any changes, report that movies are OK, else update the csv files
        if (
            not check_fps
            and not empty_sampling_end.any()
            and not empty_sampling_start.any()
        ):
            logging.info(
                f"{df_toreview[df_toreview['exists']].shape[0]} available movies"
                f" have been checked and no action was required."
            )

            return

        else:
            # Add the missing info to the original df based on movie ids
            df.set_index("movie_id", inplace=True)
            df_toreview.set_index("movie_id", inplace=True)
            df.update(df_toreview)
            df.reset_index(drop=False, inplace=True)

            # Rename back the project-specific column names
            # that don't match schema standard names
            df = cols_rename_to_schema(
                project=project,
                table_name="movies",
                df=df,
                reverse_lookup=True,
            )

            # Save the updated df locally
            df.to_csv(csv_paths["local_movies_csv"], index=False)
            logging.info(f"The local movies.csv file has been updated")

            from kso_utils.server_utils import update_csv_server

            # Save the updated df in the server
            update_csv_server(
                project=project,
                csv_paths=csv_paths,
                server_connection=server_connection,
                orig_csv="server_movies_csv",
                updated_csv="local_movies_csv",
            )


def concatenate_local_movies(csv_paths):
    # concatenates the movies specified in the "go_pro_files" column
    # and saves them to fpath

    # Load the csv with movies information
    df = pd.read_csv(csv_paths["local_movies_csv"])

    # Select only the path of the folder
    df["Path"] = df["fpath"].str.rsplit("\\", n=1).str[0]

    # Function to merge directory path and multiple filenames into a list
    def merge_paths(row):
        directory_path = row["Path"]
        filenames = row["go_pro_files"].split("; ")
        merged_paths = [
            os.path.join(directory_path, filename.strip()) for filename in filenames
        ]
        return merged_paths

    # Combine the path of the folder with the go_profiles inside the folder
    df["path_go_pros"] = df.apply(merge_paths, axis=1)

    # Create an empty list to store the annotations
    rows_list = []

    # Loop through each classification submitted by the users
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Start text file and list to keep track of the videos to concatenate
        textfile_name = "a_file.txt"
        textfile = open(textfile_name, "w")
        video_list = []

        for movie_i in sorted(row["path_go_pros"]):
            # Keep track of the videos to concatenate
            textfile.write("file '" + movie_i + "'" + "\n")
            video_list.append(movie_i)
        textfile.close()

        # Concatenate the files
        if os.path.exists(row["fpath"]):
            logging.info(f"{row['fpath']} not concatenated because it already exists")
        else:
            logging.info(f"Concatenating {row['fpath']}")

            # Concatenate the videos
            subprocess.call(
                [
                    "ffmpeg",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    "a_file.txt",
                    "-c",
                    "copy",
                    row["fpath"],
                ]
            )

        logging.info(f"{row['fpath']} concatenated successfully")

        # Delete the text file
        os.remove(textfile_name)


#         # Delete the go_pro_videos
#         for f in video_list:
#             os.remove(f)


def select_project_movies(
    project: Project,
    movies_df: pd.DataFrame,
):
    """
    > This function filters a df of movies to select only those movies that are relevant to the project (e.g. good visibity)

    :param project: the project object
    :param movies_df: a df with the information about the filepaths and "existance" of the movies
    """
    # Check for missing values in IsBadDeployment column
    if movies_df["IsBadDeployment"].isnull().any():
        raise ValueError(
            "The 'IsBadDeployment' column contains missing values. Please handle missing values before proceeding."
        )

    # Select only movies that are a good deployment
    if project.Project_name in ["Spyfish_Aotearoa"]:
        movies_df = movies_df.loc[~movies_df.IsBadDeployment]

    return movies_df
