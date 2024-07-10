import datetime
import uuid
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_fireworks import ChatFireworks
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith.client import Client
from langchain_community.vectorstores import FAISS

from langchain_benchmarks import __version__
from langchain_benchmarks.rate_limiting import RateLimiter
from langchain_benchmarks.tool_usage.agents import StandardAgentFactory
from langchain_benchmarks.tool_usage.tasks.multiverse_math import *
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

tests = [
    (
        "claude-3-haiku-20240307",
        ChatAnthropic(model="claude-3-haiku-20240307", temperature=0),
    ),
    (
        "claude-3-sonnet-20240229",
        ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0),
    ),
    ("gpt-3.5-turbo-0125", ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)),
    (
        "gpt-4-turbo-2024-04-09",
        ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0),
    ),
    (
        "accounts/fireworks/models/firefunction-v2",
        ChatFireworks(model="accounts/fireworks/models/firefunction-v2", temperature=0),
    ),
]

client = Client()  # Launch langsmith client for cloning datasets

experiment_uuid = uuid.uuid4().hex[:4]
today = datetime.date.today().isoformat()
for task in registry.tasks:
    if task.type != "ToolUsageTask":
        continue
    # This is a small test dataset that can be used to verify
    # that everything is set up correctly prior to running over
    # all results. We may remove it in the future.
    if task.name != "Multiverse Math":
        continue

    dataset_name = task.name

    uncleaned_examples = [
        e
        for e in client.list_examples(
            dataset_name="multiverse-math-examples-for-few-shot"
        )
    ]
    few_shot_messages = []
    examples = []
    for i in range(len(uncleaned_examples)):
        converted_messages = convert_to_messages(uncleaned_examples[i].outputs["output"])
        examples.append(
            # The message at index 1 is the human message (0th message is system prompt)
            {"question":converted_messages[1].content, "messages":[m for m in converted_messages if isinstance(m, SystemMessage) == False]}
        )
        few_shot_messages += converted_messages

few_shot_messages = [m for m in few_shot_messages if not isinstance(m, SystemMessage)]

few_shot_str = ""
for m in few_shot_messages:
    if isinstance(m.content, list):
        few_shot_str += "AI message: "
        for tool_use in m.content:
            if "name" in tool_use:
                few_shot_str += f"Use tool {tool_use['name']}, input: {', '.join(f'{k}:{v}' for k,v in tool_use['input'].items())}"
            else:
                few_shot_str += tool_use["text"]
            few_shot_str += "\n"
    else:
        if isinstance(m, HumanMessage):
            few_shot_str += f"Human message: {m.content}"
        else:
            few_shot_str += f"AI message: {m.content}"

    few_shot_str += "\n"


    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=3,
        input_keys=["question"],
        example_keys=["messages"]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=[],
        example_selector=example_selector,
        example_prompt=MessagesPlaceholder("messages"),
    )


    prompts = [
        (
            ChatPromptTemplate.from_messages(
                [
                    ("system", "{instructions}"),
                    ("human", "{question}"),
                    MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
                ]
            ),
            "no-few-shot",
        ),
        (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "{instructions} Here are some example conversations of the user interacting with the AI until the correct answer is reached: ",
                    ),
                ]
                + few_shot_messages
                + [
                    ("human", "{question}"),
                    MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
                ]
            ),
            "few-shot-message",
        ),
        (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "{instructions} Here are some example conversations of the user interacting with the AI until the correct answer is reached: "
                        + few_shot_message,
                    ),
                    ("human", "{question}"),
                    MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
                ]
            ),
            "few-shot-string",
        ),
        (   
            ChatPromptTemplate.from_messages(
                [
                    ("system", "{instructions} Here are some example conversations of the user interacting with the AI until the correct answer is reached: "),
                    few_shot_prompt,
                    ("human", "{question}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            ),
            "few-shot-semantic",
        )
    ] 
    
    for model_name, model in tests[:-1]:
        rate_limiter = RateLimiter(requests_per_second=1)

    print(f"Benchmarking {task.name} with model: {model_name}")
    eval_config = task.get_eval_config()

    for prompt, prompt_name in prompts:
        agent_factory = StandardAgentFactory(
            task, llm, prompt, rate_limiter=rate_limiter
        )

            client.run_on_dataset(
                dataset_name=dataset_name,
                llm_or_chain_factory=agent_factory,
                evaluation=eval_config,
                verbose=False,
                project_name=f"{model_name}-{task.name}-{prompt_name}-{experiment_uuid}",
                concurrency_level=5,
                project_metadata={
                    "model": model_name,
                    "id": experiment_uuid,
                    "task": task.name,
                    "date": today,
                    "langchain_benchmarks_version": __version__,
                },
            )