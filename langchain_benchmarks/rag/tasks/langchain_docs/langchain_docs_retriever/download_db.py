import os
from typing import Iterable
import zipfile

import requests

chroma_remote_url = "https://storage.googleapis.com/benchmarks-artifacts/langchain-docs-benchmarking/chroma_db.zip"
raw_docs_file = "https://storage.googleapis.com/benchmarks-artifacts/langchain-docs-benchmarking/docs.parquet"
directory = os.path.dirname(os.path.realpath(__file__))
db_directory = os.path.join(directory, "db")
DOCS_FILE = os.path.join(directory, "db_docs/docs.parquet")


def is_folder_populated(folder):
    if os.path.exists(folder):
        return any(os.scandir(folder))
    return False


def download_chroma_folder_from_gcs():
    r = requests.get(chroma_remote_url, allow_redirects=True)
    open("chroma_db.zip", "wb").write(r.content)

    with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
        zip_ref.extractall(directory)

    os.remove("chroma_db.zip")


def fetch_langchain_docs_db():
    if not is_folder_populated(db_directory):
        print(f"Folder {db_directory} is not populated. Downloading from GCS...")
        download_chroma_folder_from_gcs()


def fetch_remote_parquet_file():
    if not os.path.exists(DOCS_FILE):
        print(f"File {DOCS_FILE} does not exist. Downloading from GCS...")
        r = requests.get(raw_docs_file, allow_redirects=True)
        if not os.path.exists(os.path.dirname(DOCS_FILE)):
            os.makedirs(os.path.dirname(DOCS_FILE))
        open(DOCS_FILE, "wb").write(r.content)
        print(f"File {DOCS_FILE} downloaded.")


if __name__ == "__main__":
    fetch_remote_parquet_file()
