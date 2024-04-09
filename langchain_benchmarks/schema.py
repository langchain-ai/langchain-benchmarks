"""Schema for the Langchain Benchmarks."""
from __future__ import annotations

import dataclasses
import importlib
import urllib
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type, Union

from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.smith import RunEvalConfig
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from pydantic import BaseModel
from tabulate import tabulate
from typing_extensions import Literal


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
            ["Type", self.type],
            ["Dataset ID", self._dataset_link],
            ["Description", self.description],
        ]

    def _repr_html_(self) -> str:
        """Return an HTML representation of the environment."""
        return tabulate(
            self._table,
            tablefmt="unsafehtml",
        )

    @property
    def type(self) -> str:
        """Return the type of the task."""
        return self.__class__.__name__


@dataclasses.dataclass(frozen=True)
class ToolUsageTask(BaseTask):
    """A definition for a task."""

    create_environment: Callable[[], ToolUsageEnvironment]
    """Factory that returns an environment."""

    instructions: str
    """Instructions for the agent/chain/llm."""

    eval_params: Dict[str, Any]
    """Used to parameterize differences in the evaluation of the task.
    
    These are passed to the standard factory method for creating an evaluator
    for tool usage.
    
    An example, for MultiVerse math the `output_evaluation` parameter is set to
    `qa_math` to use a different prompt for evaluating the output of the agent.
    
    This prompt performs better at comparing the output of the agent against
    the reference output.
    """

    def get_eval_config(self, **params: Any) -> RunEvalConfig:
        """Get the default evaluator for the environment."""
        # Import locally to avoid potential circular imports in the future.
        from langchain_benchmarks.tool_usage.evaluators import get_eval_config

        finalized_params = {**self.eval_params, **params}
        return get_eval_config(**finalized_params)


@dataclasses.dataclass(frozen=True)
class ExtractionTask(BaseTask):
    """A definition for an extraction task."""

    schema: Type[BaseModel]
    """Get schema that specifies what should be extracted."""

    # We might want to make this optional / or support more types
    # and add validation, but let's wait until we have more examples
    instructions: Optional[ChatPromptTemplate] = None
    """Get the prompt for the task.
    
    This is the default prompt to use for the task.
    """
    dataset_url: Optional[str] = None
    dataset_name: Optional[str] = None
    eval_config: Optional[RunEvalConfig] = None


@dataclasses.dataclass(frozen=True)
class RetrievalTask(BaseTask):
    get_docs: Optional[Callable[..., Iterable[Document]]] = None
    """A function that returns the documents to be indexed."""
    retriever_factories: Dict[
        str, Callable[[Embeddings], BaseRetriever]
    ] = dataclasses.field(default_factory=dict)  # noqa: F821
    """Factories that index the docs using the specified strategy."""
    architecture_factories: Dict[
        str, Callable[[Embeddings], BaseRetriever]
    ] = dataclasses.field(default_factory=dict)  # noqa: F821
    """Factories methods that help build some off-the-shelf architectures。"""

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
            seen_names.add(task.name)

    def __len__(self) -> int:
        """Return the number of tasks in the registry."""
        return len(self.tasks)

    def __iter__(self) -> Iterable[BaseTask]:
        """Iterate over the tasks in the registry."""
        return iter(self.tasks)

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

    def __getitem__(self, key: Union[int, str, slice]) -> Union[BaseTask, Registry]:
        """Get an environment from the registry."""
        if isinstance(key, slice):
            return Registry(tasks=self.tasks[key])
        elif isinstance(key, (int, str)):
            # If key is an integer, return the corresponding environment
            return self.get_task(key)
        else:
            raise TypeError("Key must be an integer or a slice.")

    def add(self, task: BaseTask) -> None:
        if not isinstance(task, BaseTask):
            raise TypeError("Only tasks can be added to the registry.")
        self.tasks.append(task)


Provider = Literal["fireworks", "openai", "anthropic", "anyscale"]
ModelType = Literal["chat", "llm"]
AUTHORIZED_NAMESPACES = {
    "langchain",
    "langchain_google_genai",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_fireworks",
}


def _get_model_class_from_path(
    path: str,
) -> Union[Type[BaseChatModel], Type[BaseLanguageModel]]:
    """Get the class of the model."""
    module_name, attribute_name = path.rsplit(".", 1)
    top_namespace = path.split(".")[0]

    if top_namespace not in AUTHORIZED_NAMESPACES:
        raise ValueError(
            f"Unauthorized namespace {top_namespace}. "
            f"Authorized namespaces are: {AUTHORIZED_NAMESPACES}"
        )

    # Import the module dynamically
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"Could not import module {module_name}. "
            f"Perhaps you need to run to pip install the package? "
            f"`pip install {module_name}`."
        )

    model_class = getattr(module, attribute_name)
    if not issubclass(model_class, (BaseLanguageModel, BaseChatModel)):
        raise ValueError(
            f"Model class {model_class} is not a subclass of BaseLanguageModel"
        )
    return model_class


def _get_default_path(provider: str, type_: ModelType) -> str:
    """Get the default path for a model."""
    paths = {
        ("anthropic", "chat"): "langchain_anthropic.ChatAnthropic",
        ("anyscale", "chat"): "langchain.chat_models.anyscale.ChatAnyscale",
        ("anyscale", "llm"): "langchain.llms.anyscale.Anyscale",
        ("fireworks", "chat"): "langchain_fireworks.ChatFireworks",
        ("fireworks", "llm"): "langchain_fireworks.Fireworks",
        ("openai", "chat"): "langchain_openai.ChatOpenAI",
        ("openai", "llm"): "langchain_openai.OpenAI",
        (
            "google-genai",
            "chat",
        ): "langchain_google_genai.chat_models.ChatGoogleGenerativeAI",
    }

    if (provider, type_) not in paths:
        raise ValueError(f"Unknown provider {provider} and type {type_}")

    return paths[(provider, type_)]


def _get_default_url(provider: str, type_: ModelType) -> Optional[str]:
    """Get default URL to API page for model."""
    if provider == "fireworks":
        return "https://app.fireworks.ai/models"
    elif provider == "openai":
        return "https://platform.openai.com/docs/models"
    elif provider == "anthropic":
        return "https://docs.anthropic.com/claude/reference/selecting-a-model"
    elif provider == "anyscale":
        return "https://docs.endpoints.anyscale.com/category/supported-models"
    elif provider == "google-genai":
        return "https://ai.google.dev/"
    else:
        return None


@dataclasses.dataclass(frozen=True)
class RegisteredModel:
    """Descriptive information about a model.

    This information can be used to instantiate the underlying model.
    """

    name: str
    provider: Provider
    description: str
    params: Dict[str, Any]
    type: ModelType
    # Path to the model class.
    # For example, "langchain.chat_models.anthropic import ChatAnthropicModel"
    path: Optional[str] = None  # If not provided, will use default path
    url: Optional[str] = None  # If not provided, will use default URL

    def get_model(
        self, *, model_params: Optional[Dict[str, Any]] = None
    ) -> Union[BaseChatModel, BaseLanguageModel]:
        """Get the class of the model."""
        all_params = {**self.params, **(model_params or {})}
        model_class = _get_model_class_from_path(self.model_path)
        return model_class(**all_params)

    @property
    def model_path(self) -> str:
        """Get the path of the model."""
        return self.path or _get_default_path(self.provider, self.type)

    @property
    def model_url(self) -> Optional[str]:
        """Get the URL of the model."""
        return self.url or _get_default_url(self.provider, self.type)

    @property
    def _table(self) -> List[List[str]]:
        """Return a table representation of the environment."""
        if self.model_path:
            url = (
                f'<a href="{self.model_path}" target="_blank" rel="noopener">'
                "ModelPage"
                "</a>"
            )
        else:
            url = ""
        return [
            ["name", self.name],
            ["type", self.type],
            ["provider", self.provider],
            ["description", self.description],
            ["model_path", self.model_path],
            ["url", url],
        ]

    def _repr_html_(self) -> str:
        """Return an HTML representation of the environment."""
        return tabulate(
            self._table,
            tablefmt="unsafehtml",
        )


StrFilter = Union[None, str, Sequence[str]]


def _is_in_filter(actual_value: str, filter_value: StrFilter) -> bool:
    """Filter for a string attribute."""
    if filter_value is None:
        return True

    if isinstance(filter_value, str):
        return actual_value == filter_value

    return actual_value in filter_value


@dataclasses.dataclass(frozen=False)
class ModelRegistry:
    registered_models: Sequence[RegisteredModel]

    def __post_init__(self) -> None:
        """Validate that all the tasks have unique names and IDs."""
        seen_names = set()
        for model in self.registered_models:
            if model.name in seen_names:
                raise ValueError(
                    f"Duplicate model name {model.name}. " f"Task names must be unique."
                )
            seen_names.add(model.name)

    def get_model(self, name: str) -> Optional[RegisteredModel]:
        """Get model info."""
        return next(model for model in self.registered_models if model.name == name)

    def filter(
        self,
        *,
        type: StrFilter = None,
        name: StrFilter = None,
        provider: StrFilter = None,
    ) -> ModelRegistry:
        """Filter the tasks in the registry."""
        models = self.registered_models
        selected_models = []

        for model in models:
            if not _is_in_filter(model.type, type):
                continue
            if not _is_in_filter(model.name, name):
                continue
            if not _is_in_filter(model.provider, provider):
                continue
            selected_models.append(model)
        return ModelRegistry(registered_models=selected_models)

    def _repr_html_(self) -> str:
        """Return an HTML representation of the registry."""
        headers = [
            "Name",
            "Type",
            "Provider",
            "Description",
        ]
        table = [
            [
                model.name,
                model.type,
                model.provider,
                model.description,
            ]
            for model in self.registered_models
        ]
        return tabulate(table, headers=headers, tablefmt="unsafehtml")

    def __len__(self) -> int:
        """Return the number of tasks in the registry."""
        return len(self.registered_models)

    def __contains__(self, item: Any) -> bool:
        """Return whether the registry contains the given model."""
        return self.get_model(item) is not None

    def __iter__(self) -> Iterable[RegisteredModel]:
        """Iterate over the tasks in the registry."""
        return iter(self.registered_models)

    def __getitem__(
        self, key: Union[int, str, slice]
    ) -> Union[RegisteredModel, ModelRegistry]:
        """Get an environment from the registry."""
        if isinstance(key, slice):
            return ModelRegistry(registered_models=self.registered_models[key])
        elif isinstance(key, (int, str)):
            # If key is an integer, return the corresponding environment
            if isinstance(key, str):
                return self.get_model(key)
            else:
                return self.registered_models[key]
        else:
            raise TypeError("Key must be an integer or a slice.")
