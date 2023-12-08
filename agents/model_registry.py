from typing import Dict, Any, Sequence

from typing_extensions import TypedDict, Literal

FIREWORK_NAME_TO_MODEL = {
    "llama-v2-7b-chat": "accounts/fireworks/models/llama-v2-7b-chat",
    "llama-v2-13b-chat": "accounts/fireworks/models/llama-v2-13b-chat",
    "llama-v2-70b-chat": "accounts/fireworks/models/llama-v2-70b-chat",
}


Provider = Literal["fireworks", "openai"]
ModelType = Literal["chat", "llm"]


class ModelInfo:
    """A dictionary containing information about a model."""

    name: str
    description: str
    params: Dict[str, Any]
    type: ModelType


class ModelProvider(TypedDict):
    provider: Provider
    models: Sequence[ModelInfo]


#
# Fireworks = ModelProvider(
#     provider="fireworks",
#     models=[
#         {
#             "name": "llama-v2-7b-chat",
#         }
#     ],
# )
