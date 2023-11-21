import os

import requests


def is_folder_populated(folder: str):
    if os.path.exists(folder):
        return any(os.scandir(folder))
    return False


def fetch_remote_file(remote: str, local: str):
    if not os.path.exists(local):
        print(f"File {local} does not exist. Downloading from GCS...")
        if not os.path.exists(os.path.dirname(local)):
            os.makedirs(os.path.dirname(local))
        r = requests.get(remote, allow_redirects=True)
        with open(local, "wb") as f:
            f.write(r.content)
        print(f"File {remote} downloaded.")
