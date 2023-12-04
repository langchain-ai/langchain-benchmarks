import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

from langchain_benchmarks.rag.utils._downloading import (
    fetch_remote_file,
    is_folder_populated,
)

logger = logging.getLogger(__name__)
_DIRECTORY = Path(os.path.abspath(__file__)).parent
# Stores the zipped pdfs for this dataset
REMOTE_DOCS_FILE = "https://storage.googleapis.com/benchmarks-artifacts/langchain-docs-benchmarking/multi_modal_slide_decks.zip"
DOCS_DIR = _DIRECTORY / "pdfs"


def fetch_raw_docs(
    filename: Optional[str] = None, docs_dir: Optional[str] = None
) -> None:
    docs_dir = docs_dir or DOCS_DIR
    if not is_folder_populated(docs_dir):
        fetch_remote_file(REMOTE_DOCS_FILE, filename)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(docs_dir)


def get_file_names():
    fetch_raw_docs()
    # Traverse the directory and partition the pdfs
    for path in DOCS_DIR.glob("*.pdf"):
        yield path
