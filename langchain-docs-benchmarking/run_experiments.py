import argparse
import json

from prepare_dataset import create_langchain_docs_dataset
from run_evals import main

experiments = [
    {
        # "server_url": "http://localhost:1983/openai-functions-agent",
        "arch": "openai-functions-agent",
        "project_name": "openai-functions-agent",
    },
    {
        # "server_url": "http://localhost:1983/anthropic_chat",
        "arch": "chat",
        "model_config": {
            "chat_cls": "ChatAnthropic",
            "model": "claude-2",
            "temperature": 1.0,
        },
        "project_name": "anthropic-chat",
    },
    {
        "arch": "chat",
        "model_config": {
            "chat_cls": "ChatOpenAI",
            "model": "gpt-3.5-turbo-16k",
        },
        # "server_url": "http://localhost:1983/chat",
        "project_name": "chat-gpt-3.5",
    },
    {
        "arch": "chat",
        "model_config": {
            "chat_cls": "ChatFireworks",
            "model": "accounts/fireworks/models/mistral-7b-instruct-4k",
        },
        "project_name": "mistral-7b-instruct-4k",
    },
    {
        "arch": "chat",
        "model_config": {
            "chat_cls": "ChatFireworks",
            "model": "accounts/fireworks/models/llama-v2-34b-code-instruct-w8a16",
        },
        "project_name": "llama-v2-34b-code-instruct-w8a16",
    },
    {
        "arch": "chat",
        "model_config": {
            "chat_cls": "ChatFireworks",
            "model": "accounts/fireworks/models/zephyr-7b-beta",
        },
        "project_name": "zephyr-7b-beta",
    },
    {
        "arch": "chat",
        "model_config": {
            "chat_cls": "ChatOpenAI",
            "model": "gpt-4",
        },
        "project_name": "gpt-4-chat",
    },
    {
        "arch": "openai-assistant",
        "model_config": {},
        "project_name": "openai-assistant",
        "max_concurrency": 2,  # Rate limit is VERY low right now.
        "retry_config": {
            "stop_after_attempt": 10,
        },
    },
    # Not worth our time it's so bad and slow
    {
        # "server_url": "http://localhost:1983/anthropic_iterative_search",
        "arch": "anthropic-iterative-search",
        "max_concurrency": 2,
        "project_name": "anthropic-iterative-search",
    },
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="LangChain Docs Q&A")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        nargs="*",
        help="Path to a JSON file with experiment config."
        " If specified, the include and exclude args are ignored",
    )
    parser.add_argument("--include", type=str, nargs="+", default=None)
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
    )
    args = parser.parse_args()
    create_langchain_docs_dataset(dataset_name=args.dataset_name)
    selected_experiments = experiments
    if args.config:
        selected_experiments = []
        for config_path in args.config:
            with open(config_path) as f:
                selected_experiments.append(json.load(f))
    elif args.include:
        selected_experiments = [
            e for e in selected_experiments if e["project_name"] in args.include
        ]
    to_exclude = args.exclude or []
    if args.include and not to_exclude:
        to_exclude = [
            "anthropic-iterative-search",
            "openai-assistant",
        ]
    if args.exclude:
        selected_experiments = [
            e for e in selected_experiments if e["project_name"] not in args.exclude
        ]

    for experiment in selected_experiments:
        main(
            **experiment,
            dataset_name=args.dataset_name,
        )
