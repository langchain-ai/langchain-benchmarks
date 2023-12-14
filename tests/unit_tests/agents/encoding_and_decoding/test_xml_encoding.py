"""Test XML encoding and decoding of function definitions, invocation, and results."""
from langchain_benchmarks.tool_usage.agents.experimental.encoder import (
    FunctionDefinition,
    FunctionInvocation,
    FunctionResult,
    XMLEncoder,
)


def test_function_definition_encoding() -> None:
    """Test encoding a function definition."""
    function_definition = FunctionDefinition(
        name="test_function",
        description="A test function",
        parameters=[
            {"name": "test_parameter", "type": "str", "description": "A test parameter"}
        ],
        return_value={"type": "str", "description": "A test return value"},
    )
    encoder = XMLEncoder()
    xml = encoder.visit_function_definition(function_definition)
    assert xml == (
        "<function>\n"
        "<function_name>test_function</function_name>\n"
        "<description>\n"
        "A test function\n"
        "</description>\n"
        "<parameters>\n"
        "<parameter>\n"
        "<name>test_parameter</name>\n"
        "<type>str</type>\n"
        "<description>A test parameter</description>\n"
        "</parameter>\n"
        "</parameters>\n"
        "<return_value>\n"
        "<type>str</type>\n"
        "<description>A test return value</description>\n"
        "</return_value>\n"
        "</function>"
    )


def test_function_result_encoding() -> None:
    """Test encoding a function result."""
    function_result = FunctionResult(
        name="test_function",
        result="test_result",
        error="test_error",
    )
    encoder = XMLEncoder()
    xml = encoder.visit_function_result(function_result)
    assert xml == (
        "<function_result>\n"
        "<function_name>test_function</function_name>\n"
        "<result>test_result</result>\n"
        "<error>test_error</error>\n"
        "</function_result>"
    )


def test_function_invocation() -> None:
    """Test function invocation."""
    function_invocation = FunctionInvocation(
        name="test_function",
        arguments=[{"name": "test_argument", "value": "test_value"}],
    )
    encoder = XMLEncoder()
    xml = encoder.visit_function_invocation(function_invocation)
    assert xml == (
        "<function_invocation>\n"
        "<function_name>test_function</function_name>\n"
        "<arguments>\n"
        "<argument>\n"
        "<name>test_argument</name>\n"
        "<value>test_value</value>\n"
        "</argument>\n"
        "</arguments>\n"
        "</function_invocation>"
    )
