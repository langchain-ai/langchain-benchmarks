from agents.parser import GenericAgentParser
from langchain_core.agents import AgentFinish


def test_parser() -> None:
    """Test parser."""
    parser = GenericAgentParser(require_closing_tag=False, wrapping_xml_tag="tool")
    # assert isinstance(parser.invoke("goodbye"), AgentFinish)
    assert parser.invoke("<tool>hello</tool>") == "hello"
    assert parser.invoke("<tool>hello") == "hello"
