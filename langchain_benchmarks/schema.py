"""Schema for the Langchain Benchmarks."""
import dataclasses
from typing import List, Callable, Any, Optional, Type, Union

from langchain.tools import BaseTool
from pydantic import BaseModel
from tabulate import tabulate


@dataclasses.dataclass(frozen=True)
class ToolUsageEnvironment:
    """An instance of an environment for tool usage."""

    tools: List[BaseTool]
    """The tools that can be used in the environment."""

    read_state: Optional[Callable[[], Any]] = None
    """A function that returns the current state of the environment."""


@dataclasses.dataclass(frozen=True)
class BaseTask:
    """A definition of a task."""

    name: str
    """The name of the environment."""

    dataset_id: str
    """The ID of the langsmith public dataset.

    This dataset contains expected inputs/outputs for the environment, and
    can be used to evaluate the performance of a model/agent etc.
    """

    description: str
    """Description of the task for a data science practitioner.

    This can contain information about the task, the dataset, the tools available
    etc.
    """

    def _repr_html_(self) -> str:
        """Return an HTML representation of the environment."""
        table = [
            ["ID", self.id],
            ["Name", self.name],
            ["Type", self.__class__.__name__],
            ["Dataset ID", self.dataset_id],
            ["Description", self.description[:100] + "..."],
        ]
        return tabulate(
            table,
            tablefmt="html",
        )


@dataclasses.dataclass(frozen=True)
class ToolUsageTask(BaseTask):
    """A definition for a task."""

    create_environment: Callable[[], ToolUsageEnvironment]
    """Factory that returns an environment."""

    instructions: str
    """Instructions for the agent/chain/llm."""


@dataclasses.dataclass(frozen=True)
class ExtractionTask(BaseTask):
    """A definition for an extraction task."""

    model: Type[BaseModel]
    """Get the model for the task."""


@dataclasses.dataclass(frozen=False)
class Registry:
    tasks: List[BaseTask]

    def get_task(self, name_or_id: Union[int, str]) -> BaseTask:
        """Get the environment with the given name."""
        if isinstance(name_or_id, int):
            return self.tasks[name_or_id]

        for env in self.tasks:
            if env.name == name_or_id:
                return env
        raise ValueError(f"Unknown task {name_or_id}")

    def __post_init__(self) -> None:
        """Validate that all the tasks have unique names and IDs."""
        seen_names = set()
        for task in self.tasks:
            if task.name in seen_names:
                raise ValueError(
                    f"Duplicate task name {task.name}. " f"Task names must be unique."
                )

    def _repr_html_(self) -> str:
        """Return an HTML representation of the registry."""
        headers = [
            "Name",
            "Dataset ID",
            "Description",
        ]
        table = [
            [
                env.name,
                env.dataset_id,
                env.description,
            ]
            for env in self.tasks
        ]
        return tabulate(table, headers=headers, tablefmt="html")

    def __getitem__(self, key: Union[int, str]) -> BaseTask:
        """Get an environment from the registry."""
        if isinstance(key, slice):
            raise NotImplementedError("Slicing is not supported.")
        elif isinstance(key, (int, str)):
            # If key is an integer, return the corresponding environment
            return self.get_task(key)
        else:
            raise TypeError("Key must be an integer or a slice.")

    def add(self, task: BaseTask) -> None:
        if not isinstance(task, BaseTask):
            raise TypeError("Only tasks can be added to the registry.")
        self.tasks.append(task)
