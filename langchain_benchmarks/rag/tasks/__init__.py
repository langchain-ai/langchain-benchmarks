from langchain_benchmarks.rag.tasks.langchain_docs.task import LANGCHAIN_DOCS_TASK
from langchain_benchmarks.rag.tasks.multi_modal_slide_decks.task import (
    MULTI_MODAL_SLIDE_DECKS_TASK,
)
from langchain_benchmarks.rag.tasks.semi_structured_reports.task import (
    SEMI_STRUCTURED_REPORTS_TASK,
)

# Please keep this sorted
__all__ = [
    "LANGCHAIN_DOCS_TASK",
    "SEMI_STRUCTURED_REPORTS_TASK",
    "MULTI_MODAL_SLIDE_DECKS_TASK",
]
