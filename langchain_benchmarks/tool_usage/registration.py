"""Registry of tool use environments for ease of access."""
import dataclasses
from langchain_benchmarks.tool_usage.environments import alpha
from langchain_benchmarks.utils._registration import Environment, Registry
from typing import Callable, List

from langchain.tools import BaseTool


@dataclasses.dataclass(frozen=True)
class ToolEnvironment(Environment):
    tools_factory: Callable[[], List[BaseTool]]
    """Factory that returns a list of tools that can be used in the environment."""


# Using lower case naming to make a bit prettier API when used in a notebook
registry = Registry(
    environments=[
        ToolEnvironment(
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
