from langchain_adapters.alternative import AgentOutputParser
from langchain_core.agents import AgentFinish


def test_parser() -> None:
    """Test parser."""
    parser = AgentOutputParser(require_closing_tag=False, tag="tool")
    assert isinstance(parser.invoke("goodbye"), AgentFinish)
    assert parser.invoke("<tool>hello</tool>") == "hello"
    assert parser.invoke("<tool>hello") == "hello"
    # assert isinstance(parser.invoke("<tag>hello</tag>"), AgentAction)
