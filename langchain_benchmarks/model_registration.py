from __future__ import annotations

import dataclasses
import importlib
from typing import Dict, Any, Sequence, Optional, Union, Iterable, Type

from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from tabulate import tabulate

from typing_extensions import Literal

Provider = Literal["fireworks", "openai"]
ModelType = Literal["chat", "llm"]

AUTHORIZED_NAMESPACES = {"langchain"}


def _get_model_class_from_path(
    path: str
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
    module = importlib.import_module(module_name)
    model_class = getattr(module, attribute_name)
    if not issubclass(model_class, (BaseLanguageModel, BaseChatModel)):
        raise ValueError(
            f"Model class {model_class} is not a subclass of BaseLanguageModel"
        )
    return model_class


def _get_default_path(provider: str, type_: ModelType) -> str:
    """Get the default path for a model."""
    paths = {
        ("fireworks", "chat"): "langchain.chat_models.fireworks.ChatFireworks",
        ("fireworks", "llm"): "langchain.language_models.fireworks.Fireworks",
        ("openai", "chat"): "langchain.chat_models.openai.ChatOpenAI",
        ("openai", "llm"): "langchain.language_models.openai.OpenAI",
    }

    if (provider, type_) not in paths:
        raise ValueError(f"Unknown provider {provider} and type {type_}")

    return paths[(provider, type_)]


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

    def get_model(
        self, *, model_params: Optional[Dict[str, Any]] = None
    ) -> Union[BaseChatModel, BaseLanguageModel]:
        """Get the class of the model."""
        all_params = {**self.params, **(model_params or {})}
        path = self.path or _get_default_path(self.provider, self.type)
        model_class = _get_model_class_from_path(path)
        return model_class(**all_params)


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

    def __iter__(self) -> Iterable[RegisteredModel]:
        """Iterate over the tasks in the registry."""
        return iter(self.registered_models)

    def __getitem__(
        self, key: Union[int, str]
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


model_registry = ModelRegistry(
    registered_models=[
        RegisteredModel(
            provider="openai",
            name="gpt-3.5-turbo-1106",
            type="chat",
            description="",
            params={
                "model": "gpt-3.5-turbo-1106",
            },
        ),
        RegisteredModel(
            provider="openai",
            name="gpt-3.5-turbo-0613",
            type="chat",
            description="",
            params={
                "model": "gpt-3.5-turbo-0613",
            },
        ),
        RegisteredModel(
            provider="openai",
            name="gpt-3.5-turbo-0613",
            type="chat",
            description="",
            params={
                "model": "gpt-4-0613",
            },
        ),
        RegisteredModel(
            provider="fireworks",
            name="llama-v2-7b-chat",
            type="chat",
            description="7b parameter LlamaChat model",
            params={
                "model": "accounts/fireworks/models/llama-v2-7b-chat",
            },
        ),
        RegisteredModel(
            provider="fireworks",
            name="llama-v2-13b-chat",
            type="chat",
            description="13b parameter LlamaChat model",
            params={
                "model": "accounts/fireworks/models/llama-v2-13b-chat",
            },
        ),
        RegisteredModel(
            provider="fireworks",
            name="llama-v2-70b-chat",
            type="chat",
            description="70b parameter LlamaChat model",
            params={
                "model": "accounts/fireworks/models/llama-v2-70b-chat",
            },
        ),
    ]
)
