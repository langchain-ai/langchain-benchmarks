"""Schema for the Langchain Benchmarks."""
from __future__ import annotations

import dataclasses
import urllib
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
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

    @property
    def _dataset_link(self) -> str:
        """Return a link to the dataset."""
        dataset_url = (
            self.dataset_id
            if self.dataset_id.startswith("http")
            else f"https://smith.langchain.com/public/{self.dataset_id}/d"
        )
        parsed_url = urllib.parse.urlparse(dataset_url)
        # Extract the UUID from the path
        path_parts = parsed_url.path.split("/")
        token_uuid = path_parts[-2] if len(path_parts) >= 2 else "Link"
        return (
            f'<a href="{dataset_url}" target="_blank" rel="noopener">{token_uuid}</a>'
        )

    @property
    def _table(self) -> List[List[str]]:
        """Return a table representation of the environment."""
        return [
            ["Name", self.name],
            ["Type", self.__class__.__name__],
            ["Dataset ID", self._dataset_link],
            ["Description", self.description],
        ]

    def _repr_html_(self) -> str:
        """Return an HTML representation of the environment."""
        return tabulate(
            self._table,
            tablefmt="unsafehtml",
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

    schema: Type[BaseModel]
    """Get schema that specifies what should be extracted."""

    # We might want to make this optional / or support more types
    # and add validation, but let's wait until we have more examples
    instructions: ChatPromptTemplate
    """Get the prompt for the task.
    
    This is the default prompt to use for the task.
    """


@dataclasses.dataclass(frozen=True)
class RetrievalTask(BaseTask):
    retriever_factories: Dict[str, Callable[[Embeddings], BaseRetriever]]  # noqa: F821
    """Factories that index the docs using the specified strategy."""
    architecture_factories: Dict[str, Callable[[Embeddings], BaseRetriever]]  # noqa: F821
    """Factories methods that help build some off-the-shelf architectures。"""
    get_docs: Callable[..., Iterable[Document]]
    """A function that returns the documents to be indexed."""

    @property
    def _table(self) -> List[List[str]]:
        """Get information about the task."""
        table = super()._table
        return table + [
            ["Retriever Factories", ", ".join(self.retriever_factories.keys())],
            ["Architecture Factories", ", ".join(self.architecture_factories.keys())],
            ["get_docs", self.get_docs],
        ]


@dataclasses.dataclass(frozen=False)
class Registry:
    tasks: List[BaseTask]

    def get_task(self, name_or_id: Union[int, str]) -> BaseTask:
        """Get the task with the given name or ID."""
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
            "Type",
            "Dataset ID",
            "Description",
        ]
        table = [
            [
                task.name,
                task.__class__.__name__,
                task._dataset_link,
                task.description,
            ]
            for task in self.tasks
        ]
        return tabulate(table, headers=headers, tablefmt="unsafehtml")

    def filter(
        self,
        Type: Optional[str],
        dataset_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Registry:
        """Filter the tasks in the registry."""
        tasks = self.tasks
        if Type is not None:
            tasks = [task for task in tasks if task.__class__.__name__ == Type]
        if dataset_id is not None:
            tasks = [task for task in tasks if task.dataset_id == dataset_id]
        if name is not None:
            tasks = [task for task in tasks if task.name == name]
        if description is not None:
            tasks = [
                task
                for task in tasks
                if description.lower() in task.description.lower()
            ]
        return Registry(tasks=tasks)

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
