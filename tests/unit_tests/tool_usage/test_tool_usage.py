def test_import_tool_usage() -> None:
    """Test that tool_usage can be imported"""
    from langchain_benchmarks.tool_usage import environments, evaluators  # noqa: F401
