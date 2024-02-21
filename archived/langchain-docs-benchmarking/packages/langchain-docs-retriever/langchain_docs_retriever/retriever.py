import os
from typing import Optional

from langchain.embeddings import OpenAIEmbeddings

# from langchain_docs_retriever.voyage import VoyageEmbeddings
from langchain.embeddings.voyageai import VoyageEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores.chroma import Chroma

from .download_db import fetch_langchain_docs_db

WEAVIATE_DOCS_INDEX_NAME = "LangChain_agent_docs"
_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CHROMA_COLLECTION_NAME = "langchain-docs"
_DB_DIRECTORY = os.path.join(_DIRECTORY, "db")


def get_embeddings_model() -> Embeddings:
    if os.environ.get("VOYAGE_AI_MODEL"):
        return VoyageEmbeddings(model=os.environ["VOYAGE_AI_MODEL"], max_retries=20)
    return OpenAIEmbeddings(chunk_size=200)


def get_retriever(search_kwargs: Optional[dict] = None) -> BaseRetriever:
    embedding_model = get_embeddings_model()
    fetch_langchain_docs_db()
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=_DB_DIRECTORY,
    )
    search_kwargs = search_kwargs or dict(k=6)
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
