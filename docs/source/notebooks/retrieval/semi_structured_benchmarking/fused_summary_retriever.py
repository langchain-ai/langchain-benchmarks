from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, BaseStore, Document
from langchain.schema.vectorstore import VectorStore


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


@dataclass
class FusedDocumentElements:
    rank: int
    summary: str
    fragments: List[str]
    source: str


DOCUMENT_SUMMARY_TEMPLATE: str = """
--------------------------------
**** DOCUMENT NAME: {doc_name}

**** DOCUMENT SUMMARY:
{summary}

**** RELEVANT FRAGMENTS:
{fragments}
--------------------------------
"""


class FusedSummaryRetriever(BaseRetriever):
    """
    Retrieves a fused document that using pre-calculated summaries
    for the full-document as well as individual chunks. Specifically:

    - Full document summaries are included in the fused document to give
      broader context to the LLM, which may not be in the retrieved chunks

    - Chunk summaries are using to improve retrieval, i.e. "big-to-small"
      retrieval which is a common use case with the [multi-vector retriever](./)

    """

    vectorstore: VectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors."""

    full_doc_summary_store: BaseStore[str, str]
    """The storage layer for the parent document summaries."""

    parent_doc_store: BaseStore[str, Document]
    """The storage layer for the parent (original) docs for summaries in
    the vector store."""

    parent_id_key: str = "doc_id"
    """Metadata key for parent doc ID (maps chunk summaries in the vector 
    store to parent docs)."""

    full_doc_summary_id_key: str = "full_doc_id"
    """Metadata key for full doc summary ID (maps chunk summaries in the
    vector store to full doc summaries)."""

    source_key: str = "source"
    """Metadata key for source document of chunks."""

    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    search_type: SearchType = SearchType.similarity
    """Type of search to perform (similarity / mmr)"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        fused_doc_elements: Dict[str, FusedDocumentElements] = {}
        for i in range(len(sub_docs)):
            sub_doc = sub_docs[i]
            parent_id = sub_doc.metadata.get(self.parent_id_key)
            full_doc_summary_id = sub_doc.metadata.get(self.full_doc_summary_id_key)
            if parent_id and full_doc_summary_id:
                parent_in_store = self.parent_doc_store.mget([parent_id])
                full_doc_summary_in_store = self.full_doc_summary_store.mget(
                    [full_doc_summary_id]
                )
                if parent_in_store and full_doc_summary_in_store:
                    parent: Document = parent_in_store[0]  # type: ignore
                    full_doc_summary: str = full_doc_summary_in_store[0]  # type: ignore
                else:
                    raise Exception(
                        f"No parent or full doc summary found for retrieved doc {sub_doc},"
                        "please pre-load parent and full doc summaries."
                    )

                source = sub_doc.metadata.get(self.source_key)
                if not source:
                    raise Exception(
                        f"No source doc name found in metadata for: {sub_doc}."
                    )

                if full_doc_summary_id not in fused_doc_elements:
                    # Init fused parent with information from most relevant sub-doc
                    fused_doc_elements[full_doc_summary_id] = FusedDocumentElements(
                        rank=i,
                        summary=full_doc_summary,
                        fragments=[parent.page_content],
                        source=source,
                    )
                else:
                    fused_doc_elements[full_doc_summary_id].fragments.append(
                        parent.page_content
                    )

        fused_docs: List[Document] = []
        for element in sorted(fused_doc_elements.values(), key=lambda x: x.rank):
            fragments_str = "\n\n".join(
                [d.strip() for d in element.fragments]
            )
            fused_docs.append(
                Document(
                    page_content=DOCUMENT_SUMMARY_TEMPLATE.format(
                        doc_name=element.source,
                        summary=element.summary,
                        fragments=fragments_str,
                    )
                )
            )

        return fused_docs
