import pytest
from langchain.tools import tool

from agents.adapters import convert_tool_to_function_definition


@tool
def get_hello() -> str:
    """Get hello."""
    return "hello"


@tool
def repeat(x: str) -> str:
    """Repeat x.

    Args:
        x: The string to repeat.

    Returns:
        The repeated string.
    """
    return x


def test_parameterless_function() -> None:
    """Test foo."""
    function_definition = convert_tool_to_function_definition(get_hello)
    assert function_definition == {
        "name": "get_hello",
        "description": "Get hello.",
        "parameters": [],
        "return_value": {
            "type": "Any",
        },
    }


@pytest.mark.skip("Need to fix handling of leading whitespace")
def test_function_with_parameters() -> None:
    import textwrap

    doc = textwrap.dedent(repeat.func.__doc__)
    assert convert_tool_to_function_definition(repeat) == {
        "name": "repeat",
        "description": doc,
        "parameters": [
            {
                "name": "x",
                "type": "str",
                "description": "",  # Need to fix this
            }
        ],
        "return_value": {
            "type": "Any",
        },
    }
