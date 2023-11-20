"""A task where the agent must type a given string one letter at a time.

In this variation of the task, the agent is given access to 26 parameterless functions,
each representing a letter of the alphabet.
"""
import dataclasses
from typing import Callable, Any, List, cast

from langchain.tools import BaseTool, tool

<<<<<<< HEAD:langchain_benchmarks/tool_usage/environments/type_writer_26_funcs.py
from langchain_benchmarks.schema import AbstractEnvironment
=======
from langchain_benchmarks.schema import ToolUsageEnvironment, ToolUsageTask
>>>>>>> eugene/rag_refactor_2:langchain_benchmarks/tool_usage/type_writer_26_funcs.py


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


def get_environment() -> ToolUsageEnvironment:
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

    return ToolUsageEnvironment(
        tools=tools,
        read_state=_read_state,
    )


TYPE_WRITER_26_FUNCS_TASK = ToolUsageTask(
    name="Tool Usage - Typewriter",
    dataset_id="placeholder",
    create_environment=get_environment,
    instructions=(
        "Repeat the given string by using the provided tools. "
        "Do not write anything else or provide any explanations. "
        "For example, if the string is 'abc', you must invoke the tools "
        "'a', 'b', and 'c' in that order. "
        "Please invoke the functions without any arguments."
    ),
    description=(
        """\
Environment with 26 functions each representing a letter of the alphabet.

In this variation of the typewriter task, there are 26 parameterless functions, where \
each function represents a letter of the alphabet (instead of a single function that \
takes a letter as an argument).

The object is to evaluate the ability of use the functions to repeat the given string.

For example, if the string is 'abc', the tools 'a', 'b', and 'c' must be invoked \
in that order.

The dataset includes examples of varying difficulty. The difficulty is measured \
by the length of the string.
"""
    ),
)
