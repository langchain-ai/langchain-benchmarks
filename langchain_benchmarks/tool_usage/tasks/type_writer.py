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


def create_typer(paper: Paper) -> Callable[[], str]:
    """Create a function that types the given letter."""

    def type_letter(letter: str) -> str:
        """Print the given letter on the paper."""
        if len(letter) != 1:
            return "ERROR: The letter must be a single character."
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

    tools = cast(List[BaseTool], [tool(create_typer(paper))])

    return ToolUsageEnvironment(
        tools=tools,
        read_state=_read_state,
    )


TYPE_WRITER_TASK = ToolUsageTask(
    name="Tool Usage - Typewriter (1 tool)",
    dataset_id="https://smith.langchain.com/public/59577193-8938-4ccf-92a7-e8a96bcf4f86/d",
    create_environment=get_environment,
    instructions=(
        "Repeat the given string using the provided tools. "
        "Do not write anything else or provide any explanations. "
        "For example, if the string is 'abc', you must print the letters "
        "'a', 'b', and 'c' one at a time and in that order. "
    ),
    description=(
        """\
Environment with a single tool that accepts a single letter as input, and \
prints it on a piece of virtual paper.

The objective of this task is to evaluate the ability of the model to use the provided \
tools to repeat a given input string.

For example, if the string is 'abc', the tools 'a', 'b', and 'c' must be invoked \
in that order.

The dataset includes examples of varying difficulty. The difficulty is measured \
by the length of the string.
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
                "expected_steps": ["type_letter"] * len(string),
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
        dataset_name=TYPE_WRITER_TASK.name,
        description=TYPE_WRITER_TASK.description,
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
