def test_import_stuff() -> None:
    """Test that all imports work."""
    from langchain_benchmarks.extraction import (  # noqa: F401
        evaluators,
        implementations,
    )
