from langchain_benchmarks.tool_usage.agents.anthropic_tool_agent import parse_invocation


def test_parse_invocation() -> None:
    """Test parsing a tool invocation."""
    invocation = parse_invocation(
        "function test_function(test_parameter: str): str;",
    )
    assert invocation == {
        "name": "test_function",
        "parameters": [{"name": "test_parameter", "type": "str", "description": None}],
        "return_value": {"type": "str", "description": None},
    }
