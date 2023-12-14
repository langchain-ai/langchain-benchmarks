import pytest
from langchain_core.agents import AgentFinish, AgentActionMessageLog
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage

from agents.parser import GenericAgentParser


def test_parser() -> None:
    """Test parser."""
    parser = GenericAgentParser(require_closing_tag=False, wrapping_xml_tag="tool")

    # If <tool> tag not found then it's an agent finish
    assert isinstance(parser.invoke("goodbye"), AgentFinish)

    with pytest.raises(OutputParserException):
        # Invocation content is missing tool name and arguments
        parser.invoke("<tool>'hello'</tool>")

    with pytest.raises(OutputParserException):
        parser.invoke("<tool>hello")

    # Full invocation
    text = (
        '<tool>{\n    "tool_name": "type_letter",\n    '
        '"arguments": {\n        '
        '"letter": "h"\n    }\n}</tool>\n'
    )

    assert parser.invoke(text) == AgentActionMessageLog(
        tool="type_letter",
        tool_input={"letter": "h"},
        log="\nInvoking type_letter: {'letter': 'h'}\n\t",
        message_log=[AIMessage(content=text)],
    )

    # Test more cases
    parsed = parser.invoke('<tool>{"tool_name": "hello"}</tool>')
    assert parsed.tool == "hello"
    # Assumes that it's a structured tool by default!
    assert parsed.tool_input == {}

    with pytest.raises(OutputParserException):
        # Arguments need to be a dict
        parser.invoke('<tool>{"tool_name": "hello", "arguments": [1, 2]}</tool>')

    parsed = parser.invoke(
        '<tool>{"tool_name": "hello", "arguments": {"a": "b"}}</tool>'
    )
    assert parsed.tool == "hello"
    # Assumes that it's a structured tool by default!
    assert parsed.tool_input == {"a": "b"}
