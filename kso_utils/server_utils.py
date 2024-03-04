# base imports
import os
import getpass
import gdown
import zipfile
import boto3
import logging
from tqdm import tqdm
from pathlib import Path

# util imports
from kso_utils.project_utils import Project

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


######################################
# ###### Supported servers ###########
# #####################################
# Specify the names of the KSO supported servers
class ServerType:
    TEMPLATE = "TEMPLATE"
    LOCAL = "LOCAL"
    SNIC = "SNIC"
    AWS = "AWS"


# Store a list of ServerType values
server_types = [
    getattr(ServerType, attr)
    for attr in dir(ServerType)
    if not callable(getattr(ServerType, attr)) and not attr.startswith("__")
]

######################################
# ###### Common server functions ######
# #####################################


def connect_to_server(project: Project):
    """
    > This function connects to the server specified in the project object and returns a dictionary with
    the client and sftp client

    :param project: the project object
    :return: A dictionary with the client and sftp_client
    """
    # Get project-specific server info
    if project is None or not hasattr(project, "server"):
        logging.error("No server information found, edit projects_list.csv")
        return {}

    # Create an empty dictionary to host the server connections
    server_connection = {}

    if project.server == ServerType.TEMPLATE:
        # Set client as Wildlife.ai
        server_connection["client"] = "Wildlife.ai"

    elif project.server in [ServerType.LOCAL, ServerType.SNIC]:
        logging.info("Running locally, no external connection to server needed.")

    elif project.server == ServerType.AWS:
        # Connect to AWS as a client
        server_connection["client"] = get_aws_client()

    else:
        raise ValueError(
            f"Unsupported server type: {project.server}. Supported servers are {server_types}."
        )

    return server_connection


def download_init_csv(project: Project, init_keys: list, server_connection: dict):
    """
    > This function connects to the server of the project and downloads the csv files of interest

    :param project: the project object
    :param init_keys: list of potential names of the csv files
    :param server_connection: A dictionary with the client and sftp_client
    :return: A dictionary with the server paths of the csv files
    """

    # Create empty dictionary to save the server paths of the csv files
    db_initial_info = {}

    if project.server == ServerType.TEMPLATE:
        gdrive_id = "1PZGRoSY_UpyLfMhRphMUMwDXw4yx1_Fn"
        download_gdrive(
            gdrive_id=gdrive_id, folder_name=project.csv_folder, fix_encoding=False
        )

    elif project.server in [ServerType.LOCAL, ServerType.SNIC]:
        logging.info("Running locally so no csv files were downloaded from the server.")

    elif project.server == ServerType.AWS:
        # Retrieve a list with all the csv files in the folder with initival csvs
        server_csv_files = get_matching_s3_keys(
            client=server_connection["client"],
            bucket=project.bucket,
            suffix="csv",
        )

        # Select only csv files that are relevant to start the db
        server_csvs_db = [
            s
            for s in server_csv_files
            if any(server_csv_files in s for server_csv_files in init_keys)
        ]

        # Download each csv file locally and store the path of the server csv
        for server_csv in server_csvs_db:
            # Specify the key of the csv
            init_key = [key for key in init_keys if key in server_csv][0]

            # Save the server path of the csv in a dict
            db_initial_info[str("server_" + init_key + "_csv")] = server_csv

            # Specify the local path for the csv
            local_i_csv = Path(project.csv_folder, Path(server_csv).name)

            # Download the csv
            download_object_from_s3(
                client=server_connection["client"],
                bucket=project.bucket,
                key=server_csv,
                filename=str(local_i_csv),
            )

    else:
        raise ValueError(
            f"Unsupported server type: {project.server}. Supported servers are {server_types}."
        )

    return db_initial_info


def get_ml_data(project: Project, test: bool = False):
    """
    It downloads the training data from Google Drive.
    Currently only applies to the Template Project as other projects do not have prepared
    training data.

    :param project: The project object that contains all the information about the project
    :type project: Project
    """
    if project.ml_folder is not None:
        # Download the folder containing the training data
        if project.server == ServerType.TEMPLATE:
            gdrive_id = "1xknKGcMnHJXu8wFZTAwiKuR3xCATKco9"
            ml_folder = project.ml_folder

            # Download template training files from Gdrive
            if test:
                download_gdrive(gdrive_id, Path("../test/test_output") / ml_folder)
            else:
                download_gdrive(gdrive_id, Path(ml_folder))
            logging.info("Template data downloaded successfully")
        else:
            logging.info("No download method implemented for this data")
    else:
        logging.info("No prepared data to be downloaded.")


def update_csv_server(
    project: Project,
    csv_paths: dict,
    server_connection: dict,
    orig_csv: str,
    updated_csv: str,
):
    """
    > This function updates the original csv file with the updated csv file in the server

    :param project: the project object
    :param csv_paths: a dictionary with the paths of the csv files with info to initiate the db
    :param server_connection: a dictionary with the connection to the server
    :param orig_csv: the original csv file name
    :type orig_csv: str
    :param updated_csv: the updated csv file
    :type updated_csv: str
    """
    if project.server == ServerType.TEMPLATE:
        logging.error(
            f"The server {orig_csv} can't be updated for template project without admin permissions."
        )

    elif project.server in [ServerType.LOCAL, ServerType.SNIC]:
        logging.error(
            "The project doesn't have csv files in the server so only the local csv files have been updated"
        )

    elif project.server == ServerType.AWS:
        logging.info("Updating csv file in AWS server")
        # Update csv to AWS
        upload_file_to_s3(
            client=server_connection["client"],
            bucket=project.bucket,
            key=str(csv_paths[orig_csv]),
            filename=str(csv_paths[updated_csv]),
        )
        logging.info(f"{orig_csv} updated to the {ServerType.AWS} server.")

    else:
        raise ValueError(
            f"Unsupported server type: {project.server}. Supported servers are {server_types}."
        )


def upload_file_server(
    project: Project, server_connection: dict, conv_mov_path: str, f_path: str
):
    """
    Takes the file path of a file and uploads/saves it to the server.

    :param project: the project object
    :param conv_mov_path: The local path to the converted movie file you want to upload
    :type conv_mov_path: str
    :param f_path: The server or storage path of the movie you want to upload to
    :type f_path: str

    """
    if project.server == ServerType.TEMPLATE:
        logging.error(
            f"{conv_mov_path} not uploaded to the server as project is template"
        )

    elif project.server in [ServerType.LOCAL, ServerType.SNIC]:
        logging.error(f"{conv_mov_path} not uploaded to the server as project is local")

    elif project.server == ServerType.AWS:
        # Update csv to AWS
        upload_file_to_s3(
            client=server_connection["client"],
            bucket=project.bucket,
            key=f_path,
            filename=conv_mov_path,
        )

        logging.info(f"{f_path} standarised and uploaded to the server.")

    else:
        raise ValueError(
            f"Unsupported server type: {project.server}. Supported servers are {server_types}."
        )


#####################
# ## AWS functions ###
# ####################


def aws_credentials():
    # Save your access key for the s3 bucket.
    aws_access_key_id = getpass.getpass("Enter the key id for the aws server")
    aws_secret_access_key = getpass.getpass(
        "Enter the secret access key for the aws server"
    )

    return aws_access_key_id, aws_secret_access_key


def connect_s3(aws_access_key_id: str, aws_secret_access_key: str):
    # Connect to the s3 bucket
    client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    return client


def get_aws_client():
    # Set aws account credentials
    aws_access_key_id, aws_secret_access_key = os.getenv("SPY_KEY"), os.getenv(
        "SPY_SECRET"
    )
    if aws_access_key_id is None or aws_secret_access_key is None:
        aws_access_key_id, aws_secret_access_key = aws_credentials()

    # Connect to S3
    client = connect_s3(aws_access_key_id, aws_secret_access_key)

    return client


def get_matching_s3_objects(
    client: boto3.client, bucket: str, prefix: str = "", suffix: str = ""
):
    """
    ## Code modified from alexwlchan (https://alexwlchan.net/2019/07/listing-s3-keys/)
    Generate objects in an S3 bucket.

    :param client: S3 client.
    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        this suffix (optional).
    """

    paginator = client.get_paginator("list_objects_v2")

    kwargs = {"Bucket": bucket}

    # We can pass the prefix directly to the S3 API.  If the user has passed
    # a tuple or list of prefixes, we go through them one by one.
    if isinstance(prefix, str):
        prefixes = (prefix,)
    else:
        prefixes = prefix

    for key_prefix in prefixes:
        kwargs["Prefix"] = key_prefix

        for page in paginator.paginate(**kwargs):
            try:
                contents = page["Contents"]
            except KeyError:
                break

            for obj in contents:
                key = obj["Key"]
                if key.endswith(suffix):
                    yield obj


def get_matching_s3_keys(
    client: boto3.client, bucket: str, prefix: str = "", suffix: str = ""
):
    """
    ## Code from alexwlchan (https://alexwlchan.net/2019/07/listing-s3-keys/)
    Generate the keys in an S3 bucket.

    :param client: S3 client.
    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    return a list of the matching objects
    """

    # Select the relevant bucket
    s3_keys = [
        obj["Key"] for obj in get_matching_s3_objects(client, bucket, prefix, suffix)
    ]

    return s3_keys


def download_object_from_s3(
    client: boto3.client,
    *,
    bucket: str,
    key: str,
    version_id: str = None,
    filename: str,
):
    """
    Download an object from S3 with a progress bar.

    From https://alexwlchan.net/2021/04/s3-progress-bars/
    """

    # First get the size, so we know what tqdm is counting up to.
    # Theoretically the size could change between this HeadObject and starting
    # to download the file, but this would only affect the progress bar.
    kwargs = {"Bucket": bucket, "Key": key}

    if version_id is not None:
        kwargs["VersionId"] = version_id

    object_size = client.head_object(**kwargs)["ContentLength"]

    if version_id is not None:
        ExtraArgs = {"VersionId": version_id}
    else:
        ExtraArgs = None

    with tqdm(
        total=object_size,
        unit="B",
        unit_scale=True,
        desc=filename,
        position=0,
        leave=True,
    ) as pbar:
        client.download_file(
            Bucket=bucket,
            Key=key,
            ExtraArgs=ExtraArgs,
            Filename=filename,
            Config=boto3.s3.transfer.TransferConfig(use_threads=False),
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )


def upload_file_to_s3(client: boto3.client, *, bucket: str, key: str, filename: str):
    """
    > Upload a file to S3, and show a progress bar if the file is large enough

    :param client: The boto3 client to use
    :param bucket: The name of the bucket to upload to
    :param key: The name of the file in S3
    :param filename: The name of the file to upload
    """

    # Get the size of the file to upload
    file_size = os.stat(filename).st_size

    # Prevent issues with small files (<1MB) and tqdm
    if file_size > 1000000:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=filename,
            position=0,
            leave=True,
        ) as pbar:
            client.upload_file(
                Filename=filename,
                Bucket=bucket,
                Key=key,
                Config=boto3.s3.transfer.TransferConfig(use_threads=False),
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
    else:
        client.upload_file(
            Filename=filename,
            Bucket=bucket,
            Key=key,
        )


def delete_file_from_s3(client: boto3.client, *, bucket: str, key: str):
    """
    > Delete a file from S3.

    :param client: boto3.client - the client object that you created in the previous step
    :type client: boto3.client
    :param bucket: The name of the bucket that contains the object to delete
    :type bucket: str
    :param key: The name of the file
    :type key: str
    """
    client.delete_object(Bucket=bucket, Key=key)


#####################
# ## Google Drive functions ###
# ####################


def download_gdrive(gdrive_id: str, folder_name: str, fix_encoding: bool = True):
    # Specify the url of the file to download
    url_input = f"https://drive.google.com/uc?&confirm=s5vl&id={gdrive_id}"

    logging.info(f"Retrieving the file from {url_input}")

    # Specify the output of the file
    zip_file = f"{folder_name}.zip"

    # Download the zip file
    gdown.download(url_input, zip_file, quiet=False)

    # Unzip the folder with the files
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(Path(folder_name).parent)

    # Remove the zipped file
    Path(zip_file).unlink()

    if fix_encoding:
        from kso_utils.koster_utils import fix_text_encoding_folder

        # Correct the file names by using correct encoding
        fix_text_encoding_folder(folder_name)
