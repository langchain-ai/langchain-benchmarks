import logging
import os
import zipfile
from pathlib import Path
from typing import Iterable, Optional

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
    filename = filename or _DIRECTORY / Path(REMOTE_DOCS_FILE).name
    docs_dir = docs_dir or DOCS_DIR
    if not is_folder_populated(docs_dir):
        fetch_remote_file(REMOTE_DOCS_FILE, filename)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(docs_dir)

        os.remove(filename)


def get_file_names() -> Iterable[Path]:
    fetch_raw_docs()
    # Traverse the directory and partition the pdfs
    for path in DOCS_DIR.rglob("*.pdf"):
        # Ignore __MACOSX
        if "__MACOSX" in str(path):
            continue
        yield path
