from __future__ import annotations

from langchain_benchmarks.schema import ModelRegistry, RegisteredModel

_OPEN_AI_MODELS = [
    RegisteredModel(
        provider="openai",
        name="gpt-3.5-turbo-1106",
        type="chat",
        description=(
            "The latest GPT-3.5 Turbo model with improved instruction following, "
            "JSON mode, reproducible outputs, parallel function calling, and more. "
            "Returns a maximum of 4,096 output tokens."
        ),
        params={
            "model": "gpt-3.5-turbo-1106",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-3.5-turbo",
        type="chat",
        description="Currently points to gpt-3.5-turbo-0613.",
        params={
            "model": "gpt-3.5-turbo",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-3.5-turbo-16k",
        type="chat",
        description="Currently points to gpt-3.5-turbo-0613.",
        params={
            "model": "gpt-3.5-turbo-16k",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-3.5-turbo-instruct",
        type="llm",
        description=(
            "Similar capabilities as text-davinci-003 but compatible with legacy "
            "Completions endpoint and not Chat Completions."
        ),
        params={
            "model": "gpt-3.5-turbo-instruct",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-3.5-turbo-0613",
        type="chat",
        description=(
            "Legacy Snapshot of gpt-3.5-turbo from June 13th 2023. "
            "Will be deprecated on June 13, 2024."
        ),
        params={
            "model": "gpt-3.5-turbo-0613",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-3.5-turbo-16k-0613",
        type="chat",
        description=(
            "Legacy Snapshot of gpt-3.5-16k-turbo from June 13th 2023. "
            "Will be deprecated on June 13, 2024."
        ),
        params={
            "model": "gpt-3.5-turbo-16k-0613",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-3.5-turbo-0301",
        type="chat",
        description=(
            "Legacy Snapshot of gpt-3.5-turbo from March 1st 2023. "
            "Will be deprecated on June 13th 2024."
        ),
        params={
            "model": "gpt-3.5-turbo-0301",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="text-davinci-003",
        type="llm",
        description=(
            "Legacy Can do language tasks with better quality and consistency than "
            "the curie, babbage, or ada models. Will be deprecated on Jan 4th 2024."
        ),
        params={
            "model": "text-davinci-003",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="text-davinci-002",
        type="llm",
        description=(
            "Legacy Similar capabilities to text-davinci-003 but trained with "
            "supervised fine-tuning instead of reinforcement learning. "
            "Will be deprecated on Jan 4th 2024."
        ),
        params={
            "model": "text-davinci-002",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="code-davinci-002",
        type="llm",
        description="Legacy Optimized for code-completion tasks. Will be deprecated "
        "on Jan 4th 2024.",
        params={
            "model": "code-davinci-002",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-4-1106-preview",
        type="chat",
        description="GPT-4 TurboNew - The latest GPT-4 model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens. This preview model is not yet suited for production traffic.",
        params={
            "model": "gpt-4-1106-preview",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-4-0613",
        type="chat",
        description="Snapshot of gpt-4 from June 13th 2023 with improved function calling support.",
        params={
            "model": "gpt-4-0613",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-4-32k-0613",
        type="chat",
        description="Snapshot of gpt-4-32k from June 13th 2023 with improved function calling support.",
        params={
            "model": "gpt-4-32k-0613",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-4-0314",
        description="Snapshot of gpt-4 from March 14th 2023 with function calling support. This model version will be deprecated on June 13th 2024.",
        type="chat",
        params={
            "model": "gpt-4-0314",
        },
    ),
    RegisteredModel(
        provider="openai",
        name="gpt-4-32k-0314",
        description="Snapshot of gpt-4-32k from March 14th 2023 with function calling support. This model version will be deprecated on June 13th 2024.",
        type="chat",
        params={
            "model": "gpt-4-32k-0314",
        },
    ),
]

_FIREWORKS_MODELS = [
    RegisteredModel(
        provider="fireworks",
        name="llama-v2-7b-chat-fw",
        type="chat",
        description="7b parameter LlamaChat model",
        params={
            "model": "accounts/fireworks/models/llama-v2-7b-chat",
        },
    ),
    RegisteredModel(
        provider="fireworks",
        name="llama-v2-13b-chat-fw",
        type="chat",
        description="13b parameter LlamaChat model",
        params={
            "model": "accounts/fireworks/models/llama-v2-13b-chat",
        },
    ),
    RegisteredModel(
        provider="fireworks",
        name="llama-v2-70b-chat-fw",
        type="chat",
        description="70b parameter LlamaChat model",
        params={
            "model": "accounts/fireworks/models/llama-v2-70b-chat",
        },
    ),
    RegisteredModel(
        provider="fireworks",
        name="yi-34b-200k-fw",
        type="llm",
        description=" 4B LLM model from 01.ai, with context window 200k.",
        params={
            "model": "accounts/fireworks/models/yi-34b-200k",
        },
    ),
    RegisteredModel(
        provider="fireworks",
        name="mixtral-8x7b-instruct-fw",
        description="Mistral MoE 8x7B Instruct v0.1 model with Sparse "
        "Mixture of Experts. Fine tuned for instruction following",
        type="llm",
        params={"model": "accounts/fireworks/models/mixtral-8x7b-instruct"},
    ),
]

_ANTHROPIC_MODELS = [
    RegisteredModel(
        provider="anthropic",
        name="claude-3-haiku-20240307",
        description="Fastest and most compact model for near-instant responsiveness",
        type="chat",
        params={"model": "claude-3-haiku-20240307"},
    ),
    RegisteredModel(
        provider="anthropic",
        name="claude-3-sonnet-20240229",
        description="Ideal balance of intelligence and speed for enterprise workloads",
        type="chat",
        params={"model": "claude-3-sonnet-20240229"},
    ),
    RegisteredModel(
        provider="anthropic",
        name="claude-3-opus-20240229",
        description="Most powerful model for highly complex tasks",
        type="chat",
        params={"model": "claude-3-opus-20240229"},
    ),
    RegisteredModel(
        provider="anthropic",
        name="claude-2",
        description="Superior performance on tasks that require complex reasoning",
        type="chat",
        params={
            "model": "claude-2",
        },
    ),
    RegisteredModel(
        provider="anthropic",
        name="claude-2.1",
        description=(
            "Same performance as Claude 2, plus significant reduction in model "
            "hallucination rates"
        ),
        type="chat",
        params={
            "model": "claude-2.1",
        },
    ),
    RegisteredModel(
        provider="anthropic",
        name="claude-instant-1.2",
        description="low-latency, high throughput.",
        type="chat",
        params={
            "model": "claude-instant-1.2",
        },
    ),
    RegisteredModel(
        provider="anthropic",
        name="claude-instant-1",
        description="low-latency, high throughput.",
        type="chat",
        params={
            "model": "claude-instant-1",
        },
    ),
]
_GOOGLE_GENAI_MODELS = [
    RegisteredModel(
        provider="google-genai",
        name="gemini-pro",
        description="Gemini Pro is a large model from Google trained on a diverse set of tasks.",
        type="chat",
        params={
            "model": "gemini-pro",
            "convert_system_message_to_human": True,
        },
    )
]

_ANYSCALE_MODELS = [
    RegisteredModel(
        provider="anyscale",
        name="mistral-7b-instruct-v0.1",
        description="Mistral 7B model fine-tuned for function-calling.",
        type="chat",
        params={
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
        },
    ),
]

model_registry = ModelRegistry(
    registered_models=_OPEN_AI_MODELS
    + _FIREWORKS_MODELS
    + _ANYSCALE_MODELS
    + _ANTHROPIC_MODELS
    + _GOOGLE_GENAI_MODELS
)
