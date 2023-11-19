"""Schema for the Langchain Benchmarks."""
import dataclasses
from typing import List, Callable, Any, Optional

from langchain.tools import BaseTool
from tabulate import tabulate


@dataclasses.dataclass(frozen=True)
class Environment:
    """An instance of an environment for tool usage."""

    tools: List[BaseTool]
    """The tools that can be used in the environment."""

    read_state: Optional[Callable[[], Any]] = None
    """A function that returns the current state of the environment."""


@dataclasses.dataclass(frozen=True)
class Task:
    """A definition for a task."""

    id: int
    """The ID of the environment."""
    name: str
    """The name of the environment."""

    dataset_id: str
    """The ID of the langsmith public dataset.
    
    This dataset contains expected inputs/outputs for the environment, and
    can be used to evaluate the performance of a model/agent etc.
    """

    create_environment: Callable[
        [], Environment
    ]  # Specialized for tool usage; refactor potentially
    """Factory that returns an environment."""

    description: str
    """Description of the task for a data science practitioner.
    
    This can contain information about the task, the dataset, the tools available
    etc.
    """

    instructions: str
    """Instructions for the agent/chain/llm."""

    def _repr_html_(self) -> str:
        """Return an HTML representation of the environment."""
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
