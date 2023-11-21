"""A task where the agent must type a given string one letter at a time.

In this variation of the task, the agent is given access to 26 parameterless functions,
each representing a letter of the alphabet.
"""
import dataclasses
from typing import Any, Callable, List, cast

from langchain.tools import BaseTool, tool

from langchain_benchmarks.schema import ToolUsageEnvironment, ToolUsageTask


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
    func.__name__ = letter
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
    name="Tool Usage - Typewriter (26 tools)",
    dataset_id="https://smith.langchain.com/public/128af05e-aa00-4e3b-a958-d166dd450581/d",
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
Environment with 26 tools each tool represents a letter of the alphabet.

The objective of this task is to evaluate the model's ability the use tools
for a simple repetition task.

For example, if the string is 'abc', the tools 'a', 'b', and 'c' must be invoked \
in that order.

The dataset includes examples of varying difficulty. The difficulty is measured \
by the length of the string.

This is a variation of the typer writer task, where 26 parameterless tools are
given instead of a single tool that takes a letter as an argument.
"""
    ),
)

STRINGS_TO_TYPE = [
    # letter repetition
    "a",
    "aa",
    "aaa",
    "aaaa",
    # 3-letter words
    "dog",
    "cat",
    # 4-letter words
    "hand",
    "head",
    # 5-letter words
    "house",
    "horse",
    # 6-letter words
    "school",
    "church",
    # 7-letter words
    "teacher",
    "student",
    # 8-letter words
    "computer",
    "keyboard",
    # 9-letter words
    "university",
    "dictionary",
    # 10-letter words
    "information",
    "communication",
]


def _create_dataset(strings: List[str]) -> List[dict]:
    """Create the dataset."""
    dataset = []
    for string in strings:
        dataset.append(
            {
                "question": string,
                "expected_steps": [c for c in string],
                "state": string,
            }
        )
    return dataset


DATASET = _create_dataset(STRINGS_TO_TYPE)


def _create_dataset() -> None:
    """Create a dataset with the langsmith client."""
    from langsmith.client import Client

    client = Client()
    dataset = client.create_dataset(
        dataset_name=TYPE_WRITER_26_FUNCS_TASK.name,
        description=TYPE_WRITER_26_FUNCS_TASK.description,
    )

    for example in DATASET:
        client.create_example(
            inputs={
                "question": example["question"],
            },
            outputs={
                "reference": example["state"],
                "expected_steps": example["expected_steps"],
                "state": example["state"],
            },
            dataset_id=dataset.id,
        )
