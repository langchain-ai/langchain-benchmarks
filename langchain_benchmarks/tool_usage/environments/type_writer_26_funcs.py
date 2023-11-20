"""A task where the agent must type a given string one letter at a time.

In this variation of the task, the agent is given access to 26 parameterless functions,
each representing a letter of the alphabet.
"""
import dataclasses
from typing import Callable, Any, List, cast

from langchain.tools import BaseTool, tool

from langchain_benchmarks.schema import Environment


@dataclasses.dataclass
class Paper:
    """A piece of paper that the agent can write on."""

    content: str


def _create_typing_func(letter: str, paper: Paper) -> Callable[[], str]:
    """Create a function that types the given letter."""

    def func() -> str:
        paper.content += letter
        return "OK"

    func.__doc__ = f'Run to Type the letter "{letter}".'
    func.__name__ = f"{letter}"
    return func


def _get_available_functions(paper: Paper) -> List[Callable]:
    """Get all the available functions."""
    return [
        _create_typing_func(letter, paper) for letter in "abcdefghijklmnopqrstuvwxyz"
    ]


# PUBLIC API


def get_environment() -> Environment:
    """Create tools and state reader.

    Attention: this is a factory function, so it will create a new environment
               every time it is called. The paper contains state.

    Returns:
        A tuple of (tools, state_reader).
    """
    paper = Paper(content="")  # Start with an empty piece of paper
    functions = _get_available_functions(paper)

    def _read_state() -> Any:
        """Read the state of the environment."""
        return paper.content

    tools = cast(List[BaseTool], [tool(f) for f in functions])

    return Environment(
        tools=tools,
        read_state=_read_state,
    )
