"""Load html from files, clean up, split, ingest."""
import logging
import os
import re
from typing import TYPE_CHECKING, Generator, Iterable, Optional

from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.embeddings import OpenAIEmbeddings, VoyageEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain.vectorstores.chroma import Chroma

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
directory = os.path.dirname(os.path.realpath(__file__))
db_directory = os.path.join(directory, "indexing", "db")
docs_cache_directory = os.path.join(directory, "indexing", "db_docs")
docs_cache_file = os.path.join(docs_cache_directory, "docs.parquet")


def langchain_docs_extractor(soup: BeautifulSoup) -> str:
    try:
        from bs4 import Doctype, NavigableString, Tag
    except ImportError:
        raise ImportError(
            "Please install beautifulsoup4 to use the langchain docs benchmarking task.\n"
            "pip install beautifulsoup4"
        )
    # Remove all the tags that are not meaningful for the extraction.
    SCAPE_TAGS = ["nav", "footer", "aside", "script", "style"]
    [tag.decompose() for tag in soup.find_all(SCAPE_TAGS)]

    def get_text(tag: Tag) -> Generator[str, None, None]:
        for child in tag.children:
            if isinstance(child, Doctype):
                continue

            if isinstance(child, NavigableString):
                yield child
            elif isinstance(child, Tag):
                if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    yield f"{'#' * int(child.name[1:])} {child.get_text()}\n\n"
                elif child.name == "a":
                    yield f"[{child.get_text(strip=False)}]({child.get('href')})"
                elif child.name == "img":
                    yield f"![{child.get('alt', '')}]({child.get('src')})"
                elif child.name in ["strong", "b"]:
                    yield f"**{child.get_text(strip=False)}**"
                elif child.name in ["em", "i"]:
                    yield f"_{child.get_text(strip=False)}_"
                elif child.name == "br":
                    yield "\n"
                elif child.name == "code":
                    parent = child.find_parent()
                    if parent is not None and parent.name == "pre":
                        classes = parent.attrs.get("class", "")

                        language = next(
                            filter(lambda x: re.match(r"language-\w+", x), classes),
                            None,
                        )
                        if language is None:
                            language = ""
                        else:
                            language = language.split("-")[1]

                        lines: list[str] = []
                        for span in child.find_all("span", class_="token-line"):
                            line_content = "".join(
                                token.get_text() for token in span.find_all("span")
                            )
                            lines.append(line_content)

                        code_content = "\n".join(lines)
                        yield f"```{language}\n{code_content}\n```\n\n"
                    else:
                        yield f"`{child.get_text(strip=False)}`"

                elif child.name == "p":
                    yield from get_text(child)
                    yield "\n\n"
                elif child.name == "ul":
                    for li in child.find_all("li", recursive=False):
                        yield "- "
                        yield from get_text(li)
                        yield "\n\n"
                elif child.name == "ol":
                    for i, li in enumerate(child.find_all("li", recursive=False)):
                        yield f"{i + 1}. "
                        yield from get_text(li)
                        yield "\n\n"
                elif child.name == "div" and "tabs-container" in child.attrs.get(
                    "class", [""]
                ):
                    tabs = child.find_all("li", {"role": "tab"})
                    tab_panels = child.find_all("div", {"role": "tabpanel"})
                    for tab, tab_panel in zip(tabs, tab_panels):
                        tab_name = tab.get_text(strip=True)
                        yield f"{tab_name}\n"
                        yield from get_text(tab_panel)
                elif child.name == "table":
                    thead = child.find("thead")
                    header_exists = isinstance(thead, Tag)
                    if header_exists:
                        headers = thead.find_all("th")
                        if headers:
                            yield "| "
                            yield " | ".join(header.get_text() for header in headers)
                            yield " |\n"
                            yield "| "
                            yield " | ".join("----" for _ in headers)
                            yield " |\n"

                    tbody = child.find("tbody")
                    tbody_exists = isinstance(tbody, Tag)
                    if tbody_exists:
                        for row in tbody.find_all("tr"):
                            yield "| "
                            yield " | ".join(
                                cell.get_text(strip=True) for cell in row.find_all("td")
                            )
                            yield " |\n"

                    yield "\n\n"
                elif child.name in ["button"]:
                    continue
                else:
                    yield from get_text(child)

    joined = "".join(get_text(soup))
    return re.sub(r"\n\n+", "\n\n", joined).strip()


RECORD_MANAGER_DB_URL = (
    os.environ.get("RECORD_MANAGER_DB_URL") or "sqlite:///lcdocs_oai_record_manager.sql"
)


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"] or "",
        "title": (title.get_text() if title else "") or "",
        "description": description.get("content") or "" if description else "",
        "language": html.get("lang") or "" if html else "",
        **{k: v or "" for k, v in meta.items()},
    }


def load_langchain_docs():
    try:
        from bs4 import SoupStrainer
    except ImportError:
        raise ImportError(
            "Please install beautifulsoup4 to use the langchain docs benchmarking task.\n"
            "pip install beautifulsoup4"
        )
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def simple_extractor(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "Please install beautifulsoup4 to ingest the LangChain docs.\n"
            "pip install beautifulsoup4"
        )
    try:
        soup = BeautifulSoup(html, "lxml")
    except ImportError:
        raise ImportError(
            "Please install beautifulsoup4 to ingest the LangChain docs.\n"
            "pip install beautifulsoup4"
        )
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def get_embeddings_model() -> Embeddings:
    if os.environ.get("VOYAGE_AI_URL") and os.environ.get("VOYAGE_AI_MODEL"):
        return VoyageEmbeddings()
    return OpenAIEmbeddings(chunk_size=200)


CHROMA_COLLECTION_NAME = "langchain-docs"


def get_docs() -> Iterable[Document]:
    # TODO: Make this function actually a generator
    # Import before loading because it's a bummer to fail after scraping.
    # we should have an incremental scrape cache.

    docs_from_documentation = load_langchain_docs()
    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
    docs_from_api = load_api_docs()
    logger.info(f"Loaded {len(docs_from_api)} docs from API")

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Chroma will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_from_documentation + docs_from_api:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""
        for k, v in doc.metadata.items():
            if v is None:
                doc.metadata[k] = ""
        if not doc.page_content.strip():
            continue
        yield doc


def load_docs_from_parquet(filename: Optional[str] = None) -> Iterable[Document]:
    import pandas as pd

    df = pd.read_parquet(filename or docs_cache_file)
    docs_transformed = [Document(**row) for row in df.to_dict(orient="records")]
    for doc in docs_transformed:
        for k, v in doc.metadata.items():
            if v is None:
                doc.metadata[k] = ""
        if not doc.page_content.strip():
            continue
        yield doc


# default ingest function
def ingest_docs(overwrite: bool = False):
    if os.path.exists(docs_cache_file) and not overwrite:
        logger.info(f"Loading docs from {docs_cache_file}")
        documents = load_docs_from_parquet(docs_cache_file)

    else:
        documents = get_docs()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_transformed = text_splitter.split_documents(documents)
    embedding = get_embeddings_model()
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=db_directory,
    )

    record_manager = SQLRecordManager(
        f"chroma/{CHROMA_COLLECTION_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
    )

    logger.info("Indexing stats: ", indexing_stats)


def download_docs(overwrite: bool = False):
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Please install pandas to use the langchain docs benchmarking task.\n"
            "pip install pandas"
        )
    if os.path.exists(docs_cache_file) and not overwrite:
        logger.info(f"Loading docs from {docs_cache_file}")
        return
    if not os.path.exists(docs_cache_directory):
        os.makedirs(docs_cache_directory)
    docs = get_docs()
    # Write as parquet file
    df = pd.DataFrame.from_records([doc.dict() for doc in docs])
    df.to_parquet(docs_cache_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=["ingest", "download"],
        default="download",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.action == "download":
        download_docs(args.overwrite)
    elif args.action == "ingest":
        ingest_docs(args.overwrite)
    else:
        raise ValueError(f"Unknown action {args.action}")
