import uuid
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_fireworks import ChatFireworks

from langchain_benchmarks.tool_usage.agents import StandardAgentFactory

import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith.client import Client
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_benchmarks import (
    __version__,
    registry,
)
from langchain_benchmarks.rate_limiting import RateLimiter

tests = [
    (
        "claude-3-haiku-20240307",
        ChatAnthropic(model="claude-3-haiku-20240307", temperature=0),
    ),
    (
        "claude-3-sonnet-20240229",
        ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0),
    ),
    (
        "gpt-3.5-turbo-0125",
        ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)),
    (
        "gpt-4-turbo-2024-04-09",
        ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0),
    ),
    (
        "accounts/fireworks/models/firefunction-v2",
        ChatFireworks(model="accounts/fireworks/models/firefunction-v2", temperature=0)
    )
]

client = Client()  # Launch langsmith client for cloning datasets
today = datetime.date.today().isoformat()

experiment_uuid = uuid.uuid4().hex[:4]

for task in registry.tasks:
    if task.type != "ToolUsageTask":
        continue

    # This is a small test dataset that can be used to verify
    # that everything is set up correctly prior to running over
    # all results. We may remove it in the future.
    if task.name == "Multiverse Math (Tiny)":
        continue

    dataset_name = task.name

    for model_name, model in tests:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{instructions}"),
                ("human", "{question}"),  # Populated from task.instructions automatically
                MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
            ]
        )
        rate_limiter = RateLimiter(requests_per_second=1)

        print(f"Benchmarking {task.name} with model: {model_name}")
        eval_config = task.get_eval_config()

        agent_factory = StandardAgentFactory(
            task, model, prompt, rate_limiter=rate_limiter
        )

        client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=agent_factory,
            evaluation=eval_config,
            verbose=False,
            project_name=f"{model_name}-{task.name}-{today}-{experiment_uuid}",
            concurrency_level=5,
            project_metadata={
                "model": model_name,
                "id": experiment_uuid,
                "task": task.name,
                "date": today,
                "langchain_benchmarks_version": __version__,
            },
        )