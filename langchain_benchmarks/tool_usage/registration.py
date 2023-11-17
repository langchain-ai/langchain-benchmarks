"""Registry of environments for ease of access."""
import dataclasses
from typing import Callable, List, Sequence, Union

from langchain.tools import BaseTool
from tabulate import tabulate

from langchain_benchmarks.tool_usage.environments import alpha


@dataclasses.dataclass(frozen=True)
class Environment:
    id: int
    """The ID of the environment."""
    name: str
    """The name of the environment."""

    dataset_id: str
    """The ID of the langsmith public dataset.
    
    This dataset contains expected inputs/outputs for the environment, and
    can be used to evaluate the performance of a model/agent etc.
    """

    tools_factory: Callable[[], List[BaseTool]]
    """Factory that returns a list of tools that can be used in the environment."""

    description: str
    """Description of the environment."""

    def _repr_html_(self) -> str:
        """Return a HTML representation of the environment."""
        table = [
            ["ID", self.id],
            ["Name", self.name],
            ["Dataset ID", self.dataset_id],
            ["Description", self.description[:100] + "..."],
        ]
        return tabulate(
            table,
            tablefmt="html",
        )


@dataclasses.dataclass(frozen=True)
class Registry:
    environments: Sequence[Environment]

    def get_environment(self, name_or_id: Union[int, str]) -> Environment:
        """Get the environment with the given name."""
        for env in self.environments:
            if env.name == name_or_id or env.id == name_or_id:
                return env
        raise ValueError(
            f"Unknown environment {name_or_id}. Use list_environments() to see "
            f"available environments."
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
            for env in self.environments
        ]
        return tabulate(table, headers=headers, tablefmt="html")

    def __getitem__(self, key: Union[int, str]) -> Environment:
        """Get an environment from the registry."""
        if isinstance(key, slice):
            raise NotImplementedError("Slicing is not supported.")
        elif isinstance(key, (int, str)):
            # If key is an integer, return the corresponding environment
            return self.get_environment(key)
        else:
            raise TypeError("Key must be an integer or a slice.")


# Using lower case naming to make a bit prettier API when used in a notebook
registry = Registry(
    environments=[
        Environment(
            id=0,
            name="Tool Usage - Alpha",
            dataset_id=alpha.DATASET_ID,
            tools_factory=alpha.get_tools,
            description=(
                """\
Environment with fake data about users and their locations and favorite foods.

The environment provides a set of tools that can be used to query the data.

The object is to evaluate the ability of an agent to use the tools
to answer questions about the data.

The dataset contains 21 examples of varying difficulty. The difficulty is measured
by the number of tools that need to be used to answer the question.

Each example is composed of a question, a reference answer, and
information about the sequence in which tools should be used to answer
the question.

Success is measured by the ability to answer the question correctly, and efficiently.
"""
            ),
        )
    ]
)
