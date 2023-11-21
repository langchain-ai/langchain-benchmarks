from langchain_benchmarks.tool_usage.tasks.multiverse_math import (
    add,
    get_environment,
    multiply,
)


def test_get_environment() -> None:
    """Test the multiverse math task."""
    # Create the environment
    env = get_environment()

    # Get the tools
    tools = env.tools

    assert len(tools) == 10

    # Get the state reader
    read_state = env.read_state
    assert read_state is None


def test_operations() -> None:
    """Test some operations."""
    # Confirm that operations are not distributive
    assert multiply(add(1, 2), 7) == 32.34
    assert add(multiply(1, 7), multiply(2, 7)) == 24.3
