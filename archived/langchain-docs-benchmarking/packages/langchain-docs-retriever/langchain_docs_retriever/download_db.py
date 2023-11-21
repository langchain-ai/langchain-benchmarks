import os
import zipfile

import requests

remote_url = "https://storage.googleapis.com/benchmarks-artifacts/langchain-docs-benchmarking/chroma_db.zip"
directory = os.path.dirname(os.path.realpath(__file__))
db_directory = os.path.join(directory, "db")


def is_folder_populated(folder):
    if os.path.exists(folder):
        return any(os.scandir(folder))
    return False


def download_folder_from_gcs():
    r = requests.get(remote_url, allow_redirects=True)
    open("chroma_db.zip", "wb").write(r.content)

    with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
        zip_ref.extractall(directory)

    os.remove("chroma_db.zip")


def fetch_langchain_docs_db():
    if not is_folder_populated(db_directory):
        print(f"Folder {db_directory} is not populated. Downloading from GCS...")
        download_folder_from_gcs()


if __name__ == "__main__":
    fetch_langchain_docs_db()
