from langchain_benchmarks.tool_usage.agents.anthropic_tool_agent import parse_invocation
from xmltodict import parse


def test_parse_invocation() -> None:
    """Test parsing a tool invocation."""
    invocation = parse_invocation(
        """
        <invoke>
        <tool_name>get_time_of_day</tool_name>
        <parameters>
        <time_zone>UTC</time_zone>
        </parameters>
        </invoke>
        """,
        "function_calls"
    )
