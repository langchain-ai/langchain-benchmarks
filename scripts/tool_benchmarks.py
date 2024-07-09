import datetime
import uuid
from typing import Callable, List, cast

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings
from langsmith.client import Client

from langchain_benchmarks import __version__
from langchain_benchmarks.rate_limiting import RateLimiter
from langchain_benchmarks.tool_usage.agents import StandardAgentFactory
from langchain_benchmarks.tool_usage.tasks.multiverse_math import (
    MULTIVERSE_MATH,
    add,
    cos,
    divide,
    log,
    multiply,
    negate,
    pi,
    power,
    sin,
    subtract,
)

client = Client()
experiment_uuid = uuid.uuid4().hex[:4]
today = datetime.date.today().isoformat()
task = MULTIVERSE_MATH

funcs: List[Callable] = [
    multiply,
    add,
    divide,
    subtract,
    power,
    log,
    negate,
    sin,
    cos,
    pi,
]
tools: List[BaseTool] = [cast(BaseTool, tool(f)) for f in funcs]

models_to_test = [
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "accounts/fireworks/models/firefunction-v2",
]

examples = list(
    client.list_examples(dataset_name="multiverse-math-examples-for-few-shot")
)
few_shot_messages = []
questions = []
for i, ex in enumerate(examples):
    converted_messages = convert_to_messages(ex.outputs["output"])
    questions.append(Document(converted_messages[1].content, metadata={"index": i}))
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

vectorstore = Chroma.from_documents(
    questions, embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vectorstore.as_retriever()

prompts = [
    (
        ChatPromptTemplate.from_messages(
            [
                ("system", "{instructions}"),
                ("human", "{question}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        ),
        "zero-shot",
    ),
    (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{instructions} Here are some example conversations of the user interacting with the AI until the correct answer is reached: ",
                ),
                *few_shot_messages,
                ("human", "{question}"),
                ("placeholder", "{agent_scratchpad}"),
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
                    + few_shot_str,
                ),
                ("human", "{question}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        ),
        "few-shot-string",
    ),
    (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{instructions} Here are some example conversations of the user interacting with the AI until the correct answer is reached: ",
                ),
            ]
            + semantic_similar_few_shots("{question}", retriever, examples)
            + [
                ("human", "{question}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        ),
        "few-shot-semantic",
    ),
]

for model_name in models_to_test:
    llm = init_chat_model(model_name, temperature=0)
    rate_limiter = RateLimiter(requests_per_second=1)

    print(f"Benchmarking {task.name} with model: {model_name}")
    eval_config = task.get_eval_config()

    for prompt, prompt_name in prompts:
        agent_factory = StandardAgentFactory(
            task, llm, prompt, rate_limiter=rate_limiter
        )

        client.run_on_dataset(
            dataset_name=task.name,
            llm_or_chain_factory=agent_factory,
            evaluation=eval_config,
            project_name=f"{model_name}-{task.name}-{prompt_name}-{experiment_uuid}",
            project_metadata={
                "model": model_name,
                "id": experiment_uuid,
                "task": task.name,
                "date": today,
                "langchain_benchmarks_version": __version__,
            },
        )


def semantic_similar_few_shots(question, retriever, examples):
    ans = []
    for doc in retriever.get_relevant_documents(question)[:3]:
        ans += [
            m
            for m in convert_to_messages(
                examples[doc.metadata["index"]].outputs["output"]
            )
            if not isinstance(m, SystemMessage)
        ]
    return ans
