from langchain_benchmarks.tool_usage import __all__


def test_public_api() -> None:
    """Test that the public API is correct."""
    # This test will also fail if __all__ is not sorted.
    # Please keep it sorted!
    assert __all__ == sorted(
        [
            "apply_agent_executor_adapter",
            "get_eval_config",
            "CustomRunnableAgentFactory",
            "StandardAgentFactory",
        ],
        key=str.lower,
    )
