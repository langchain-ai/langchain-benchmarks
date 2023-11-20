"""Registry of environments for ease of access."""
import dataclasses
from typing import Sequence, Union

from tabulate import tabulate

from langchain_benchmarks.rag.tasks import langchain_docs
from langchain_benchmarks.schema import Task
from langchain_benchmarks.tool_usage.environments import (
    relational_data,
    type_writer,
    type_writer_26_funcs,
    multiverse_math,
)


@dataclasses.dataclass(frozen=True)
class Registry:
    tasks: Sequence[Task]

    def get_task(self, name_or_id: Union[int, str]) -> Task:
        """Get the environment with the given name."""
        for env in self.tasks:
            if env.name == name_or_id or env.id == name_or_id:
                return env
        raise ValueError(f"Unknown task {name_or_id}")

    def __post_init__(self) -> None:
        """Validate that all the tasks have unique names and IDs."""
        seen_names = set()
        seen_ids = set()
        for task in self.tasks:
            if task.name in seen_names:
                raise ValueError(
                    f"Duplicate task name {task.name}. " f"Task names must be unique."
                )
            seen_names.add(task.name)
            if task.id in seen_ids:
                raise ValueError(
                    f"Duplicate task ID {task.id}. " f"Task IDs must be unique."
                )

    def _repr_html_(self) -> str:
        """Return a HTML representation of the registry."""
        headers = [
            "ID",
            "Name",
            "Dataset ID",
            "Description",
        ]
        table = [
            [
                env.id,
                env.name,
                env.dataset_id,
                env.description,
            ]
            for env in self.tasks
        ]
        return tabulate(table, headers=headers, tablefmt="html")

    def __getitem__(self, key: Union[int, str]) -> Task:
        """Get an environment from the registry."""
        if isinstance(key, slice):
            raise NotImplementedError("Slicing is not supported.")
        elif isinstance(key, (int, str)):
            # If key is an integer, return the corresponding environment
            return self.get_task(key)
        else:
            raise TypeError("Key must be an integer or a slice.")


# Using lower case naming to make a bit prettier API when used in a notebook
registry = Registry(
    tasks=[
        Task(
            id=0,
            name="Tool Usage - Relational Data",
            dataset_id=relational_data.DATASET_ID,
            create_environment=relational_data.get_environment,
            instructions=(
                """\
Please answer the user's question by using the tools provided. Do not guess the \
answer. Keep in mind that entities like users,foods and locations have both a \
name and an ID, which are not the same."""
            ),
            description=(
                """\
Environment with fake data about users and their locations and favorite foods.

The environment provides a set of tools that can be used to query the data.

The objective of this task is to evaluate the ability to use the provided tools \
to answer questions about relational data.

The dataset contains 21 examples of varying difficulty. The difficulty is measured \
by the number of tools that need to be used to answer the question.

Each example is composed of a question, a reference answer, and \
information about the sequence in which tools should be used to answer \
the question.

Success is measured by the ability to answer the question correctly, and efficiently.
"""
            ),
        ),
        Task(
            id=1,
            name="Tool Usage - Typewriter (1 func)",
            dataset_id="placeholder",
            create_environment=type_writer.get_environment,
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
        ),
        Task(
            id=2,
            name="Tool Usage - Typewriter",
            dataset_id="placeholder",
            create_environment=type_writer_26_funcs.get_environment,
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
        ),
        Task(
            id=3,
            name="Multiverse Math",
            dataset_id="placeholder",
            create_environment=multiverse_math.get_environment,
            instructions=(
                "You are requested to solve math questions in an alternate "
                "mathematical universe. The rules of association, commutativity, "
                "and distributivity still apply, but the operations have been "
                "altered to yield different results than expected. Solve the "
                "given math questions using the provided tools. "
                "Do not guess the answer."
            ),
            description=(
                """\
An environment that contains a few basic math operations, but with altered results.

For example, multiplication of 5*3 will be re-interpreted as 5*3*1.1. \
The basic operations retain some basic properties, such as commutativity, \
associativity, and distributivity; however, the results are different than expected.

The objective of this task is to evaluate the ability to use the provided tools to \
solve simple math questions and ignore any innate knowledge about math.
"""
            ),
        ),
        Task(
            id=4,
            name="LangChain Docs Q&A",
            dataset_id=langchain_docs.DATASET_ID,
            create_environment=langchain_docs.create_environment,
            description=(
                """\
Questions and answers based on a snapshot of the LangChain python docs.

The environment provides the documents and the retriever information.

Each example is composed of a question and reference answer.

Success is measured based on the accuracy of the answer relative to the reference answer.
We also measure the faithfulness of the model's response relative to the retrieved documents (if any).
"""  # noqa: E501
            ),
        ),
    ]
)
