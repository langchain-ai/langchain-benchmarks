import datetime
import uuid

from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_openai import  OpenAIEmbeddings
from langsmith.client import Client
from langchain_benchmarks import __version__, registry
from langchain_benchmarks.rate_limiting import RateLimiter
from langchain_benchmarks.tool_usage.agents import StandardAgentFactory
from langchain_benchmarks.tool_usage.tasks.multiverse_math import *
from langchain.chat_models import init_chat_model
from langsmith.evaluation import evaluate
from langchain.agents import AgentExecutor, create_tool_calling_agent

tests = [
    ("claude-3-haiku-20240307","anthropic",),
    ("claude-3-sonnet-20240229","anthropic",),
    ("claude-3-opus-20240229","anthropic",),
    ("claude-3-5-sonnet-20240620","anthropic",),
    ("gpt-3.5-turbo-0125", "openai"),
    ("gpt-4o","openai",),
    ("gpt-4o-mini","openai"),
    ("llama3-groq-70b-8192-tool-use-preview","groq"),
    ("llama3-groq-8b-8192-tool-use-preview","groq"),
    ("gemini-1.5-pro","google_vertexai"),
    ("gemini-1.5-flash","google_vertexai")
]

client = Client()  # Launch langsmith client for cloning datasets

def get_few_shot_messages(task_name):
    if task_name == "Multiverse Math":
        uncleaned_examples = [
            e
            for e in client.list_examples(
                dataset_name="multiverse-math-examples-for-few-shot"
            )
        ]
        few_shot_messages = []
        few_shot_three_messages = []
        examples = []
        for i in range(len(uncleaned_examples)):
            converted_messages = convert_to_messages(
                uncleaned_examples[i].outputs["output"]
            )
            examples.append(
                # The message at index 1 is the human message asking the actual math question (0th message is system prompt)
                {
                    "question": converted_messages[1].content,
                    "messages": [
                        m
                        for m in converted_messages
                        if isinstance(m, SystemMessage) == False
                    ],
                }
            )
            few_shot_messages += converted_messages
            if i < 3:
                few_shot_three_messages += converted_messages

        return examples, [
            m for m in few_shot_messages if not isinstance(m, SystemMessage)
        ], [m for m in few_shot_three_messages if not isinstance(m, SystemMessage)]
    else:
        raise ValueError("Few shot messages not supported for this dataset")

def turn_messages_to_str(few_shot_messages):
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
    return few_shot_str

def get_few_shot_str_from_messages(few_shot_messages, few_shot_three_messages):
    few_shot_str = turn_messages_to_str(few_shot_messages)
    few_shot_three_str = turn_messages_to_str(few_shot_three_messages)
    return few_shot_str, few_shot_three_str

def get_prompts(task_name, **kwargs):
    if task_name == "Multiverse Math":
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            kwargs['examples'],
            OpenAIEmbeddings(),
            FAISS,
            k=3,
            input_keys=["question"],
            example_keys=["messages"],
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            input_variables=[],
            example_selector=example_selector,
            example_prompt=MessagesPlaceholder("messages"),
        )
        return [
            (
                client.pull_prompt("langchain-ai/multiverse-math-no-few-shot"),
                "no-few-shot",
            ),
            (
                client.pull_prompt("langchain-ai/multiverse-math-few-shot-messages"),
                "few-shot-messages",
            ),
            (
                client.pull_prompt("langchain-ai/multiverse-math-few-shot-str"),
                "few-shot-string",
            ),
            (
                client.pull_prompt("langchain-ai/multiverse-math-few-shot-3-messages"),
                "few-shot-three-messages",
            ),
            (
                client.pull_prompt("langchain-ai/multiverse-math-few-shot-3-str"),
                "few-shot-three-strings",
            ),
            (
                ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "{instructions} Here are some example conversations of the user interacting with the AI until the correct answer is reached: ",
                        ),
                        few_shot_prompt,
                        ("human", "{question}"),
                        MessagesPlaceholder("agent_scratchpad"),
                    ]
                ),
                "few-shot-semantic-openai-embeddinga",
            ),
        ]

def predict_from_callable(callable,instructions):
    def predict(run):
        return callable.invoke({"question":run['question'],"instructions":instructions})['output']
    return predict

experiment_uuid = uuid.uuid4().hex[:4]
today = datetime.date.today().isoformat()
for task in registry.tasks:
    if task.type != "ToolUsageTask" or task.name != "Multiverse Math":
        continue

    dataset_name = task.name

    examples, few_shot_messages, few_shot_three_messages = get_few_shot_messages(task.name)
    few_shot_str, few_shot_three_str = get_few_shot_str_from_messages(few_shot_messages,few_shot_three_messages)
    prompts = get_prompts(task.name,examples=examples,few_shot_three_messages=few_shot_three_messages,few_shot_three_str=few_shot_three_str)

    for model_name, model_provider in tests:
        model = init_chat_model(model_name,model_provider=model_provider)
        rate_limiter = RateLimiter(requests_per_second=1)

        print(f"Benchmarking {task.name} with model: {model_name}")
        eval_config = task.get_eval_config()
    
        for prompt, prompt_name in prompts[:-1]:
            
            tools = task.create_environment().tools
            agent = create_tool_calling_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)
            '''
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
            '''
            print(agent_executor.invoke({"question":"placeholder question","instructions":task.instructions}))
            evaluate(
                predict_from_callable(agent_executor,task.instructions),
                data=dataset_name,
                evaluators=eval_config.evaluators,
                max_concurrency=5,
                metadata={
                    "model": model_name,
                    "id": experiment_uuid,
                    "task": task.name,
                    "date": today,
                    "langchain_benchmarks_version": __version__,
                },
            )

            