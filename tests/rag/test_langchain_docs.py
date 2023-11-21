def test_import_rag() -> None:
    """Test that the rag tasks can be imported."""
    from langchain_benchmarks.rag import evaluators, tasks  # noqa: F401


def test_import_langchain_docs() -> None:
    """Test that the langchain_docs tasks can be imported."""
    from langchain_benchmarks.rag.tasks.langchain_docs import (  # noqa: F401
        DATASET_ID,
        LANGCHAIN_DOCS_TASK,
        architectures,
        indexing,
    )
