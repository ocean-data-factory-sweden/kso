import requests
import os
import json
import zipfile
import logging
from pathlib import Path

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def zip_folder(folder_path):
    folder_path = Path(folder_path)
    zip_file_name = folder_path.with_suffix(".zip")

    with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in folder_path.glob("**/*"):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(folder_path))

    return zip_file_name


def upload_archive(access_key: str, artifact_dir: str):
    """
    > Uploads the last file in the `artifact_dir` to Zenodo

    :param access_key: the access token you got from Zenodo
    :type access_key: str
    :param bucket_url: the url of the bucket you want to upload to
    :type bucket_url: str
    :param file_path: the path to the file you want to upload
    :type file_path: str
    :param artifact_dir: the directory where the artifacts are stored
    :type artifact_dir: str
    """

    depo_id, bucket_url = get_zenodo_id_bucket(access_key=access_key)

    add_file_to_zenodo_upload(
        access_key,
        bucket_url,
        artifact_dir,
    )

    return depo_id


# Get deposition id, i.e. "id" field from this response and bucket
def get_zenodo_id_bucket(access_key: str):
    headers = {"Content-Type": "application/json"}
    params = {"access_token": access_key}
    r = requests.post(
        "https://zenodo.org/api/deposit/depositions",
        params=params,
        json={},
        # Headers are not necessary here since "requests" automatically
        # adds "Content-Type: application/json", because we're using
        # the "json=" keyword argument
        # headers=headers,
        headers=headers,
    )
    response = r.json()
    return response["id"], response["links"]["bucket"]


def add_file_to_zenodo_upload(access_key: str, bucket_url: str, file_path: str):
    filename = os.path.basename(file_path)
    # The target URL is a combination of the bucket link with the desired filename
    # seperated by a slash.
    params = {"access_token": access_key}
    with open(file_path, "rb") as fp:
        r = requests.put(
            "%s/%s" % (bucket_url, filename),
            data=fp,
            params=params,
        )
    return r.json()


def add_metadata_zenodo_upload(
    access_token: str,
    deposition_id: str,
    title: str,
    description: str,
    creators_dict: dict,
):
    # Add metadata
    data = {
        "metadata": {
            "title": title,
            "upload_type": "software",
            "description": description,
            "creators": [
                {"name": name, "affiliation": affiliation}
                for name, affiliation in creators_dict.items()
            ],
            "communities": [{"identifier": "odf-sweden"}],
            "notes": "Attribution notice: The code used to generate this model can be found "
            "at https://github.com/ocean-data-factory-sweden/koster_data_management",
        }
    }
    headers = {"Content-Type": "application/json"}
    r = requests.put(
        f"https://zenodo.org/api/deposit/depositions/{deposition_id}",
        params={"access_token": access_token},
        data=json.dumps(data),
        headers=headers,
    )
    if r.status_code == 200:
        logging.info("Upload successful")
        r = requests.post(
            f"https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish",
            params={"access_token": access_token},
        )
        return r.status_code

    else:
        logging.info("Upload failed")
