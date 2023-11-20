"""Schema for the Langchain Benchmarks."""
import abc
import dataclasses
from typing import Any, Optional, Literal
from typing import Callable, Dict, List

from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.tools import BaseTool
from tabulate import tabulate


@dataclasses.dataclass(frozen=True)
class AbstractEnvironment(abc.ABC):
    """An instance of an environment for tool usage."""


@dataclasses.dataclass(frozen=True)
class ToolUsageEnvironment(AbstractEnvironment):
    """An instance of an environment for tool usage."""

    tools: List[BaseTool]
    """The tools that can be used in the environment."""

    read_state: Optional[Callable[[], Any]] = None
    """A function that returns the current state of the environment."""


@dataclasses.dataclass(frozen=True)
class RetrievalEnvironment(AbstractEnvironment):
    retriever_factories: Dict[str, Callable[[Embeddings], BaseRetriever]]  # noqa: F821
    """Factories that index the docs using the specified strategy."""
    architecture_factories: Dict[str, Callable[[Embeddings], BaseRetriever]]  # noqa: F821
    """Factories methods that help build some off-the-shelf architectures。"""

    @property
    def _table(self) -> List[List[str]]:
        """Return a table representation of the environment."""
        raise NotImplementedError()
        table = super()._table
        return table + [
            ["Retriever Factories", ", ".join(self.retriever_factories.keys())],
            ["Architecture Factories", ", ".join(self.architecture_factories.keys())],
        ]


TaskType = Literal["tool_usage", "rag"]


@dataclasses.dataclass(frozen=True)
class Task:
    """A definition for a task."""

    name: str
    """The name of the environment."""

    dataset_id: str
    """The ID of the langsmith public dataset.
    
    This dataset contains expected inputs/outputs for the environment, and
    can be used to evaluate the performance of a model/agent etc.
    """

    task_type: TaskType
    """The type of the task."""

    create_environment: Callable[
        [], AbstractEnvironment
    ]  # Specialized for tool usage; refactor potentially
    """Factory that returns an environment.
    
    Invoked once per run. Underlying code is free to re-use the same environment
    across runs.
    """

    description: str
    """Description of the task for a data science practitioner.
    
    This can contain information about the task, the dataset, the tools available
    etc.
    """

    instructions: str
    """Instructions for the agent/chain/llm."""

    def __post_init__(self) -> None:
        """Validate that all the tasks have unique names and IDs."""
        if self.task_type not in ["tool_usage", "rag"]:
            raise ValueError(
                f"Unknown task type {self.task_type}. "
                f"Task type must be one of 'tool_usage', 'rag'."
            )

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
