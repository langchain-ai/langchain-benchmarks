"""Registry of RAG environments for ease of access."""
import dataclasses
from typing import Callable, Dict, List

from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain_benchmarks.rag.tasks import langchain_docs
from langchain_benchmarks.rag.tasks.langchain_docs import (
    architectures,
    langchain_docs_retriever,
)

