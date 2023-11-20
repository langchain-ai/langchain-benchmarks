"""A task where the agent must type a given string one letter at a time.

In this variation of the task, the agent is given a single function,
that takes a letter as an argument.
"""
import dataclasses
from typing import Callable, Any, List, cast

from langchain.tools import BaseTool, tool

from langchain_benchmarks.schema import ToolUsageEnvironment


@dataclasses.dataclass
class Paper:
    """A piece of paper that the agent can write on."""

    content: str


def function(paper: Paper) -> Callable[[], str]:
    """Create a function that types the given letter."""

    def type_letter(letter: str) -> str:
        """Print the given letter on the paper."""
        paper.content += letter
        return "OK"

    return type_letter


# PUBLIC API


def get_environment() -> ToolUsageEnvironment:
    """Create tools and state reader.

    Attention: this is a factory function, so it will create a new environment
               every time it is called. The paper contains state.

    Returns:
        A tuple of (tools, state_reader).
    """
    paper = Paper(content="")  # Start with an empty piece of paper

    def _read_state() -> Any:
        """Read the state of the environment."""
        return paper.content

    tools = cast(List[BaseTool], [tool(function(paper))])

    return ToolUsageEnvironment(
        tools=tools,
        read_state=_read_state,
    )
