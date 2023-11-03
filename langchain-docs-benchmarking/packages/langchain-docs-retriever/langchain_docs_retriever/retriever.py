import os

import weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores import Weaviate
from langchain_docs_retriever.voyage import VoyageEmbeddings

WEAVIATE_DOCS_INDEX_NAME = "LangChain_agent_docs"
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


def get_embeddings_model() -> Embeddings:
    if os.environ.get("VOYAGE_AI_URL") and os.environ.get("VOYAGE_AI_MODEL"):
        return VoyageEmbeddings()
    return OpenAIEmbeddings(chunk_size=200)


def get_retriever() -> BaseRetriever:
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=get_embeddings_model(),
        by_text=False,
        attributes=["source", "title"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=6))
