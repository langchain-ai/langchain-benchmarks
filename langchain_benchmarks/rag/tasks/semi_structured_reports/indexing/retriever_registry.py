import logging
import os
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Optional

from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores.chroma import Chroma

from langchain_benchmarks.rag.utils._downloading import (
    fetch_remote_file,
    is_folder_populated,
)
from langchain_benchmarks.rag.utils.indexing import (
    get_hyde_retriever,
    get_parent_document_retriever,
    get_vectorstore_retriever,
)

logger = logging.getLogger(__name__)
_DIRECTORY = Path(os.path.abspath(__file__)).parent
# Stores the zipped pdfs for this dataset
REMOTE_DOCS_FILE = "https://storage.googleapis.com/benchmarks-artifacts/langchain-docs-benchmarking/semi_structured_earnings.zip"
DOCS_DIR = _DIRECTORY / "pdfs"
LOCAL_FILE = _DIRECTORY / "chroma_db.zip"

_DEFAULT_SEARCH_KWARGS = {"k": 6}


def fetch_raw_docs(
    filename: Optional[str] = None, docs_dir: Optional[str] = None
) -> None:
    filename = filename or LOCAL_FILE
    docs_dir = docs_dir or DOCS_DIR
    if not is_folder_populated(docs_dir):
        fetch_remote_file(REMOTE_DOCS_FILE, filename)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(docs_dir)

        os.remove(LOCAL_FILE)


def get_file_names():
    fetch_raw_docs()
    # Traverse the directory and partition the pdfs
    for path in DOCS_DIR.glob("*.pdf"):
        yield path


def partition_pdfs(path: Path, *, config: Optional[dict] = None):
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        raise ImportError(
            "Please install the unstructured package to use this example.\n"
            "pip install unstructured"
        )

    config = {
        # Unstructured first finds embedded image blocks
        "extract_images_in_pdf": False,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        "infer_table_structure": True,
        # Post processing to aggregate text once we have the title
        "chunking_strategy": "by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        "max_characters": 4000,
        "new_after_n_chars": 3800,
        "combine_text_under_n_chars": 2000,
        **(config or {}),
    }
    raw_pdf_elements = partition_pdf(
        filename=str(path), image_output_dir_path=str(path), **config
    )

    # Categorize by type
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            yield Document(
                page_content=str(element),
                metadata={"element_type": "table", "source": str(path.name)},
                id=str(element),
            )
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            yield Document(
                page_content=str(element),
                metadata={"element_type": "composite", "source": str(path.name)},
                id=str(element),
            )
        else:
            logger.debug(f"Skipping element of type {type(element)}")


def load_docs(*, unstructured_config: Optional[dict] = None) -> Iterable[Document]:
    for path in get_file_names():
        yield from partition_pdfs(path, config=unstructured_config)


def _chroma_retriever_factory(
    embedding: Embeddings,
    *,
    docs: Optional[Iterable[Document]] = None,
    search_kwargs: Optional[dict] = None,
    transform_docs: Optional[Callable] = None,
    transformation_name: Optional[str] = None,
) -> BaseRetriever:
    docs = docs or load_docs()
    embedding_name = embedding.__class__.__name__
    vectorstore = Chroma(
        collection_name=f"lcbm-ss-b-{embedding_name}-{transformation_name}",
        embedding_function=embedding,
        persist_directory="./chromadb",
    )
    return get_vectorstore_retriever(
        docs,
        embedding,
        vectorstore,
        collection_name="semi-structured-earnings-b",
        transform_docs=transform_docs,
        transformation_name=transformation_name,
        search_kwargs=search_kwargs or _DEFAULT_SEARCH_KWARGS,
    )


def _chroma_parent_document_retriever_factory(
    embedding: Embeddings,
    *,
    docs: Optional[Iterable[Document]] = None,
    search_kwargs: Optional[dict] = None,
    transformation_name: Optional[str] = None,
) -> BaseRetriever:
    docs = docs or load_docs()
    embedding_name = embedding.__class__.__name__
    vectorstore = Chroma(
        collection_name=f"lcbm-ss-pd-{embedding_name}-{transformation_name}",
        embedding_function=embedding,
        persist_directory="./chromadb",
    )
    return get_parent_document_retriever(
        docs,
        embedding,
        vectorstore,
        collection_name="semi-structured-earnings-pd",
        search_kwargs=search_kwargs or _DEFAULT_SEARCH_KWARGS,
        transformation_name=transformation_name,
    )


def _chroma_hyde_retriever_factory(
    embedding: Embeddings,
    *,
    docs: Optional[Iterable[Document]] = None,
    search_kwargs: Optional[dict] = None,
    transformation_name: Optional[str] = None,
) -> BaseRetriever:
    docs = docs or load_docs()
    embedding_name = embedding.__class__.__name__
    vectorstore = Chroma(
        collection_name=f"lcbm-ss-hd-{embedding_name}-{transformation_name}",
        embedding_function=embedding,
        persist_directory="./chromadb",
    )
    return get_hyde_retriever(
        docs,
        embedding,
        vectorstore,
        collection_name="semi-structured-earnings-hd",
        search_kwargs=search_kwargs or _DEFAULT_SEARCH_KWARGS,
        transformation_name=transformation_name,
    )


RETRIEVER_FACTORIES = {
    "basic": _chroma_retriever_factory,
    "parent-doc": _chroma_parent_document_retriever_factory,
    "hyde": _chroma_hyde_retriever_factory,
}
