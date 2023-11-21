"""A task where the agent must type a given string one letter at a time.

In this variation of the task, the agent is given a single function,
that takes a letter as an argument.
"""
import dataclasses
from typing import Any, Callable, List, cast

from langchain.tools import BaseTool, tool

from langchain_benchmarks.schema import ToolUsageEnvironment, ToolUsageTask


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


TYPE_WRITER_TASK = ToolUsageTask(
    name="Tool Usage - Typewriter (1 func)",
    dataset_id="placeholder",
    create_environment=get_environment,
    instructions=(
        "Repeat the given string by using the provided tools. "
        "Do not write anything else or provide any explanations. "
        "For example, if the string is 'abc', you must invoke the tools "
        "'a', 'b', and 'c' in that order. "
        "Please invoke the function with a single letter at a time."
    ),
    description=(
        """\
Environment with a single function that accepts a single letter as input, and \
"prints" it on a piece of paper.

The objective of this task is to evaluate the ability to use the provided \
tools to repeat a given input string.

For example, if the string is 'abc', the tools 'a', 'b', and 'c' must be invoked \
in that order.

The dataset includes examples of varying difficulty. The difficulty is measured \
by the length of the string.
"""
    ),
)
