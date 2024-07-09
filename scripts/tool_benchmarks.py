import uuid
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_fireworks import ChatFireworks

from langchain_benchmarks.tool_usage.agents import StandardAgentFactory
from langchain_core.messages.utils import convert_to_messages
import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith.client import Client
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_benchmarks import (
    __version__,
    registry,
)
from langchain_benchmarks.rate_limiting import RateLimiter

from typing import List, cast
from langchain.tools import BaseTool, tool

from langchain_benchmarks.tool_usage.tasks.multiverse_math import *


tools = cast(
        List[BaseTool],
        [
            tool(func)
            for func in [
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
        ],
    )

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

def semantic_similar_few_shots(question, retriever, examples):
    ans = []
    for doc in retriever.get_relevant_documents(question)[:3]:
        ans += [m for m in convert_to_messages(examples[doc.metadata['index']].outputs['output']) if isinstance(m,SystemMessage) == False]
    return ans

client = Client()  # Launch langsmith client for cloning datasets
llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0).bind_tools(tools)
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

    examples = [e for e in client.list_examples(dataset_name="multiverse-math-examples-for-few-shot")]
    few_shot_messages = []
    questions = []
    for i in range(len(examples)):
        converted_messages = convert_to_messages(examples[i].outputs['output'])
        questions.append(Document(page_content=converted_messages[1].content,metadata={"index":i}))
        few_shot_messages += converted_messages

    few_shot_messages =  [m for m in few_shot_messages if isinstance(m,SystemMessage) == False]

    few_shot_message = ""
    for m in few_shot_messages:
        if isinstance(m.content,list):
            few_shot_message += "AI message: "
            for tool_use in m.content:
                if 'name' in tool_use:
                    few_shot_message += f"Use tool {tool_use['name']}, input: {', '.join(f'{k}:{v}' for k,v in tool_use['input'].items())}"
                else:
                    few_shot_message += tool_use['text']
                few_shot_message += "\n"
        else:
            if isinstance(m, HumanMessage):
                few_shot_message += f"Human message: {m.content}"
            else:
                few_shot_message += f"AI message: {m.content}"
        
        few_shot_message += "\n"

    vectorstore = Chroma.from_documents(documents=questions, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    prompts = [
        (ChatPromptTemplate.from_messages(
            [
                ("system", "{instructions}"),
                ("human", "{question}"),
                MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
            ]
        ), "no-few-shot"),
        (ChatPromptTemplate.from_messages(
            [
                ("system", "{instructions} Here are some example conversations of the user interacting with the AI until the correct answer is reached: "),
            ]
            + few_shot_messages 
            + [
                ("human", "{question}"),
                MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
            ]
        ), "few-shot-message"),
        (ChatPromptTemplate.from_messages(
            [
                ("system", "{instructions} Here are some example conversations of the user interacting with the AI until the correct answer is reached: " + few_shot_message),
                ("human", "{question}"),
                MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
            ]
        ), "few-shot-string"),
        (ChatPromptTemplate.from_messages(
            [
                ("system", "{instructions} Here are some example conversations of the user interacting with the AI until the correct answer is reached: "),
            ]
            + semantic_similar_few_shots("{question}", retriever, examples) +
            [
                ("human", "{question}"),
                MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
            ]
        ), "few-shot-semantic")
    ]

    for model_name, model in tests[:-1]:
        rate_limiter = RateLimiter(requests_per_second=1)

        print(f"Benchmarking {task.name} with model: {model_name}")
        eval_config = task.get_eval_config()


        for prompt, prompt_name in prompts:
            agent_factory = StandardAgentFactory(
                task, model, prompt, rate_limiter=rate_limiter
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