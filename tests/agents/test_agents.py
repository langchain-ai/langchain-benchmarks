def test_import_agents() -> None:
    """Test that all agents can be imported"""
    from langchain_benchmarks.agents import environments, evaluators  # noqa: F401
