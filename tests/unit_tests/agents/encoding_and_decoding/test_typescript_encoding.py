"""Test typescript encoding."""
from langchain_benchmarks.tool_usage.agents.experimental.encoder import (
    FunctionDefinition,
    TypeScriptEncoder,
)


def test_function_definition() -> None:
    """Test encoding a function definition."""
    function_definition = FunctionDefinition(
        name="test_function",
        description="A test function",
        parameters=[
            {"name": "test_parameter", "type": "str", "description": "A test parameter"}
        ],
        return_value={"type": "str", "description": "A test return value"},
    )
    encoder = TypeScriptEncoder()
    xml = encoder.visit_function_definition(function_definition)
    assert xml == (
        "// A test function\n"
        "// @param test_parameter A test parameter\n"
        "// @returns A test return value\n"
        "function test_function(test_parameter: str): str;"
    )
