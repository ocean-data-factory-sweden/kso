import requests
import os
import json
import zipfile
import tarfile
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

    depo_id, bucket_url = _get_zenodo_id_bucket(access_key=access_key)

    _add_file_to_zenodo_upload(
        access_key,
        bucket_url,
        artifact_dir,
    )

    return depo_id


def _get_files_from_record(record_id: str):
    record_url = f"https://zenodo.org/api/records/{record_id}"
    record_response = requests.get(record_url)

    if record_response.status_code == 200:
        record_data = record_response.json()
        if "files" in record_data and record_data["files"]:
            return record_data["files"]
        else:
            logging.warning("No files found in the published record.")
    return []


def _download_and_process_file(file_info, download_dir, headers):
    file_name = file_info["key"]
    if "ref" not in file_name:
        return None

    download_url = file_info["links"]["self"].replace("/draft", "")
    local_filename = os.path.join(download_dir, file_name)

    print(f"Downloading {file_name} from {download_url}...")
    with requests.get(download_url, headers=headers, stream=True) as file_response:
        if file_response.status_code != 200:
            print(
                f"Failed to download {file_name}. HTTP status code: {file_response.status_code}"
            )
            return None

        with open(local_filename, "wb") as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"{file_name} downloaded successfully.")

    # Extract the archive
    extracted_files = _extract_archive(local_filename, download_dir)
    model_path = next(
        (os.path.join(download_dir, f) for f in extracted_files if f.endswith(".pt")),
        None,
    )

    # Remove the archive after extraction
    os.remove(local_filename)

    return (
        (os.path.basename(local_filename).replace(".zip", ""), model_path)
        if model_path
        else None
    )


def _fetch_records(base_url, headers, page):
    response = requests.get(base_url, headers=headers, params={"page": page})
    if response.status_code != 200:
        print(
            f"Failed to retrieve records for page {page}. HTTP status code: {response.status_code}"
        )
        return []
    return response.json()


def download_and_extract_models_from_zenodo(api_token, download_dir="models"):
    """
    Download model archives from Zenodo associated with the account, extract the model files, and return their paths.

    Parameters:
    api_token (str): Zenodo API token.
    download_dir (str): Directory where models will be downloaded and extracted.

    Returns:
    dict: Dictionary with model names as keys and local file paths as values.
    """
    os.makedirs(download_dir, exist_ok=True)

    headers = {"Authorization": f"Bearer {api_token}"}
    base_url = "https://zenodo.org/api/deposit/depositions"

    model_paths = {}
    page = 1

    while True:
        records = _fetch_records(base_url, headers, page)
        if not records:
            break

        for record in records:
            file_list = _get_files_from_record(record["id"])
            if file_list:
                for file_info in file_list:
                    result = _download_and_process_file(
                        file_info, download_dir, headers
                    )
                    if result:
                        model_paths[result[0]] = result[1]

        page += 1

    return model_paths


def _extract_archive(archive_path, extract_to):
    """
    Extract an archive file and return a list of extracted file paths.

    Parameters:
    archive_path (str): Path to the archive file.
    extract_to (str): Directory where the archive will be extracted.

    Returns:
    list: List of extracted file paths.
    """
    extracted_files = []

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
            extracted_files = zip_ref.namelist()
    elif archive_path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)
            extracted_files = tar_ref.getnames()
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r:") as tar_ref:
            tar_ref.extractall(extract_to)
            extracted_files = tar_ref.getnames()
    else:
        print(f"Unsupported archive format: {archive_path}")

    return extracted_files


# Get deposition id, i.e. "id" field from this response and bucket
def _get_zenodo_id_bucket(access_key: str):
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


def _add_file_to_zenodo_upload(access_key: str, bucket_url: str, file_path: str):
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
