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
        name='mixtral-moe-8x7b-chat-fw',
        type="chat",
        description="Mistral MoE model, unofficial implementation. Further fine-tuned for chat by Fireworks.",
        params={
            "model": "accounts/fireworks/models/mixtral-moe-8x7b-chat",
        },
    ),
    RegisteredModel(
        provider="fireworks",
        name="llama-v2-7b-llm-fw",
        type="llm",
    )

    {
        "name": "Mixtral MoE 8x7B",
        "description": "Mistral MoE model, unofficial implementation.",
        "type": "llm",
    },
    {
        "name": "Capybara 34B",
        "description": "34B chat model from NousResearch, based on Yi-34B-200k.",
        "type": "chat",
    },
    {
        "name": "Yi 34B 200k context window",
        "description": "34B LLM model from 01.ai, with context window 200k.",
        "params": {"model": "accounts/fireworks/models/yi-34b-200k-capybara"},
        "type": "llm",
    },
    {
        "name": "Yi 6B",
        "description": "6B LLM model from 01.ai.",
        "params": {"model": "accounts/fireworks/models/yi-6b"},
        "type": "llm",
    },
]


[
    {
        "name": "Mistral 7B Instruct",
        "description": "Mistral-7B model fine-tuned for conversations.",
        "params": {"model": "accounts/fireworks/models/mistral-7b-instruct"},
        "type": "llm",
    },
    {
        "name": "Llama 2 13B code instruct",
        "description": "Instruction-tuned version of Llama 2 13B, optimized for code generation.",
        "params": {"model": "accounts/fireworks/models/llama-2-13b-code-instruct"},
        "type": "llm",
    },
    {
        "name": "Llama 2 34B Code Llama instruct",
        "description": "Code Llama 34B, optimized for code generation.",
        "params": {"model": "accounts/fireworks/models/llama-2-34b-code-instruct"},
        "type": "llm",
    },
    {
        "name": "Llama 2 7B Chat",
        "description": "Fine-tuned version of Llama 2 7B, optimized for dialogue applications using RLHF, comparable to ChatGPT.",
        "params": {"model": "accounts/fireworks/models/llama-2-7b-chat"},
        "type": "chat",
    },
    {
        "name": "Llama 2 13B Chat",
        "description": "Fine-tuned version of Llama 2 13B, optimized for dialogue applications using RLHF, comparable to ChatGPT.",
        "params": {"model": "accounts/fireworks/models/llama-2-13b-chat"},
        "type": "chat",
    },
    {
        "name": "Llama 2 70B Chat",
        "description": "Fine-tuned version of Llama 2 70B, optimized for dialogue applications using RLHF, comparable to ChatGPT.",
    },
    {
        "name": "StarCoder 7B",
        "description": "7B parameter model trained on 80+ programming languages from The Stack (v1.2), using Multi Query Attention and Fill-in-the-Middle objective.",
    },
    {
        "name": "StarCoder 15.5B",
        "description": "15.5B parameter model trained on 80+ programming languages from The Stack (v1.2), using Multi Query Attention and Fill-in-the-Middle objective.",
    },
    {
        "name": "Traditional Chinese Llama2 QLoRa",
        "description": "Fine-tuned Llama 2 model on traditional Chinese Alpaca dataset.",
    },
    {
        "name": "Llama 2 13B French",
        "description": "Fine-tuned meta-llama/Llama-2-13b-chat-hf to answer French questions in French.",
    },
    {
        "name": "Chinese Llama 2 LoRA 7B",
        "description": "The LoRA version of Chinese-Llama-2 based on Llama-2-7b-hf.",
    },
    {
        "name": "Bleat",
        "description": "Enables function calling in LLaMA 2, similar to OpenAI's implementation for ChatGPT.",
    },
    {
        "name": "Llama2 13B Guanaco QLoRA GGML",
        "description": "Fine-tuned Llama 2 13B model using the Open Assist dataset.",
    },
    {
        "name": "Llama 7B Summarize",
        "description": "Summarizes articles and conversations.",
    },
]

_ANTHROPIC_MODELS = [
    RegisteredModel(
        provider="anthropic",
        name="claude-2",
        description=("Superior performance on tasks that require complex reasoning"),
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

model_registry = ModelRegistry(
    registered_models=_OPEN_AI_MODELS + _FIREWORKS_MODELS + _ANTHROPIC_MODELS
)
