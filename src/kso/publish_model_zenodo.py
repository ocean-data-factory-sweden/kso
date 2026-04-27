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


class publish_zenodo:
    def __init__(self, ACCESS_TOKEN: str):

        if not ACCESS_TOKEN:
            raise ValueError(f"{ACCESS_TOKEN} must be a non-empty string")

        self.ACCESS_TOKEN = ACCESS_TOKEN
        self.id = None
        self.bucket_url = None

    def zenodo_upload(self, bucket_url: str, file_path: str):
        filename = os.path.basename(file_path)
        # The target URL is a combination of the bucket link with the desired filename

        # headers = {
        # "Content-Type": "application/json",
        # "Authorization": f"Bearer {self.ACCESS_TOKEN}"
        # }
        # r = requests.post('https://zenodo.org/api/deposit/depositions',
        #                 json={},
        #                 headers=headers)
        # status_code=r.status_code

        # if not (200 <= status_code < 300):
        #     raise Exception(f"draft didn't start, status_code :{status_code}")

        params = {"access_token": self.ACCESS_TOKEN}
        with open(file_path, "rb") as fp:
            r = requests.put(
                "%s/%s" % (bucket_url, filename),
                data=fp,
                params=params,
            )
        return r.json()

    def _zip_folder(self, folder_path):
        folder_path = Path(folder_path)
        zip_file_name = folder_path.with_suffix(".zip")

        with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in folder_path.glob("**/*"):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(folder_path))

        return zip_file_name

    def _get_zenodo_id_bucket(self):
        headers = {"Content-Type": "application/json"}
        params = {"access_token": self.ACCESS_TOKEN}
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

    def upload_folder_zenodo(self, folder_path):
        folder_path = self._zip_folder(folder_path)
        print(f"folder_path: {folder_path}")
        self.id, self.bucket_url = self._get_zenodo_id_bucket()
        print(f"self.id,self.bucket_url:{self.id},{self.bucket_url}")
        r = self.zenodo_upload(self.bucket_url, folder_path)
        return r

    def update_metadata(
        self,
        title: str = None,
        upload_type: str = None,
        description: str = None,
        creators_name: str = None,
        affiliation: str = None,
        id: int = None,
    ):

        metadata = {
            "metadata": {
                "title": title,
                "upload_type": upload_type,
                "description": description,
                "creators": [{"name": creators_name, "affiliation": affiliation}],
            }
        }

        r = requests.put(
            f"https://zenodo.org/api/deposit/depositions/{self.id}",
            json=metadata,
            headers={"Authorization": f"Bearer {self.ACCESS_TOKEN}"},
        )

        print(r.json())

    def publish_file(self, title: str = None, id: int = None):

        headers = {"Authorization": f"Bearer {self.ACCESS_TOKEN}"}
        deposition_id = id if id else self.id
        print(f"deposition_id:{deposition_id}")
        r = requests.post(
            "https://zenodo.org/api/deposit/depositions/%s/actions/publish"
            % deposition_id,
            headers=headers,
        )
        status_code = r.status_code  # 202
        if status_code == 202:
            print(
                "file has been published successfully in {https://zenodo.org/api/deposit/depositions/%s/actions/publish}"
                % deposition_id
            )
        else:
            print("publishing has not been successful ")
