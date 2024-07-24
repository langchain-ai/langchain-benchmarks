from datetime import datetime
from typing import List, Literal, Union, cast

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, tool
from langchain_core.messages import HumanMessage
from langsmith.client import Client

from langchain_benchmarks.schema import ToolUsageEnvironment, ToolUsageTask


class DocQuery(BaseModel):
    """Query against documentation"""

    query: str = Field(..., description="The question to answer")
    source: Literal["langchain", "langsmith", "langgraph"] = Field(
        ...,
        description="The documentation source to search against. Should be one of 'langchain', 'langsmith', or "
        "'langgraph' depending on which one product the user question pertains to",
    )


class TweetQuery(BaseModel):
    """Query against tweets"""

    subject: str = Field(..., description="Subject to search for")
    min_likes: Union[int, None] = Field(
        None, description="Minimum amount of likes on the tweet"
    )
    max_likes: Union[int, None] = Field(
        None, description="Maximum amount of likes on the tweet"
    )
    start_date: Union[datetime, None] = Field(
        None, description="Earliest date to start pulling tweets from"
    )
    end_date: Union[datetime, None] = Field(
        None,
        description="Latest date to pull tweets from, None if pulling up to the present",
    )
    has_link: bool = Field(
        False, description="Whether to query for tweets that have a link."
    )


class BlogQuery(BaseModel):
    """Query against blog posts"""

    subject: Union[str, None] = Field(..., description="Subject to search for")
    authors: List[str] = Field(
        None,
        description="Authors to search for. None if not searching for a speific author,  list if searching for more than one.",
    )
    start_date: Union[datetime, None] = Field(
        None, description="Earliest date to start pulling blog posts from"
    )
    end_date: Union[datetime, None] = Field(
        None, description="Latest date to pull blog posts from"
    )


def get_environment() -> ToolUsageEnvironment:
    """Create an environment."""
    tools = cast(
        List[BaseTool],
        [tool(func) for func in [TweetQuery, DocQuery, BlogQuery]],
    )
    return ToolUsageEnvironment(
        tools=tools,
        read_state=None,
    )


DOC_DATASET = [
    {
        "question": [
            HumanMessage(
                "Can I use the send method to map-reduce the values of different branch points?"
            )
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "send method map-reduce", "source": "langgraph"},
            }
        ],
    },
    {
        "question": [HumanMessage("where is olllama function calling mentioned?")],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "ollama function calling", "source": "langchain"},
            },
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "ollama function calling",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": None,
                    "end_date": None,
                    "has_link": False,
                },
            },
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "ollama function calling",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage("Are pairwise evals supported for different models?")
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {
                    "query": "pairwise evals different models",
                    "source": "langsmith",
                },
            }
        ],
    },
    {
        "question": [HumanMessage("Can a user update state during a run?")],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "user update state", "source": "langgraph"},
            }
        ],
    },
    {
        "question": [HumanMessage("Can I change config after each AI response?")],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "update model config", "source": "langchain"},
            }
        ],
    },
    {
        "question": [
            HumanMessage(
                "How can I build my own run rules? Can I set up a schedule for them?"
            )
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "custom run rules", "source": "langsmith"},
            },
            {
                "name": "DocQuery",
                "args": {"query": "run rules schedule", "source": "langsmith"},
            },
        ],
    },
    {
        "question": [HumanMessage("Is there a page on routing functions?")],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "routing functions", "source": "langgraph"},
            }
        ],
    },
    {
        "question": [
            HumanMessage("Is there information on using Pinecone as a vectorstore?")
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {
                    "query": "Pinecone vectorstore",
                    "source": "langchain",
                },
            },
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "Pinecone vectorstore",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
        ],
    },
    {
        "question": [HumanMessage("is it possible to prevent exposing personal data?")],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "personal data privacy", "source": "langsmith"},
            }
        ],
    },
    {
        "question": [HumanMessage("How do you use conditional entry?")],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "conditional entry", "source": "langgraph"},
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "How do I extract text from PDF data using PyPDF? Can I combine image and text in a prompt?"
            )
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "PDF extraction using PyPDF", "source": "langchain"},
            },
            {
                "name": "DocQuery",
                "args": {
                    "query": "combine image and text in a prompt",
                    "source": "langchain",
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "How do I setup automation rules for my chat model app? How do I view logs for those rules?"
            )
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {
                    "query": "automation rules for chat model app",
                    "source": "langsmith",
                },
            },
            {
                "name": "DocQuery",
                "args": {"query": "automation rules logs", "source": "langsmith"},
            },
        ],
    },
    {
        "question": [
            HumanMessage("where can I read about how use Chroma embeddings locally?")
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "local Chroma embeddings", "source": "langchain"},
            },
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "local Chroma embeddings",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
        ],
    },
    {
        "question": [HumanMessage("how to index documents in a RAG app?")],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "index documents RAG app", "source": "langchain"},
            },
            {
                "name": "DocQuery",
                "args": {"query": "index documents RAG app", "source": "langgraph"},
            },
        ],
    },
]

TWEET_DATASET = [
    {
        "question": [
            HumanMessage(
                "Did we have any announcements about agents with more than 1000 likes that also included a link?"
            )
        ],
        "tool_calls": [
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "agents",
                    "min_likes": 1000,
                    "max_likes": None,
                    "start_date": None,
                    "end_date": None,
                    "has_link": True,
                },
            }
        ],
    },
    {
        "question": [
            HumanMessage(
                "Are there any posts about evaluators by langchain with less than 100 likes?"
            )
        ],
        "tool_calls": [
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "evaluators",
                    "min_likes": None,
                    "max_likes": 100,
                    "start_date": None,
                    "end_date": None,
                    "has_link": False,
                },
            }
        ],
    },
    {
        "question": [
            HumanMessage(
                "Is there anywhere on socials where we link to the anthropic website in the last year?"
            )
        ],
        "tool_calls": [
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "anthropic",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": datetime(2023, 1, 1),
                    "end_date": None,
                    "has_link": True,
                },
            },
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "anthropic",
                    "authors": None,
                    "start_date": datetime(2023, 1, 1),
                    "end_date": None,
                },
            },
        ],
    },
    {
        "question": [HumanMessage("In Q2 2023 what updates to LangSmith were made?")],
        "tool_calls": [
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "LangSmith",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": datetime(2023, 4, 1),
                    "end_date": datetime(2023, 6, 30),
                    "has_link": False,
                },
            },
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "LangSmith",
                    "authors": None,
                    "start_date": datetime(2023, 4, 1),
                    "end_date": datetime(2023, 6, 30),
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "Were there any social media posts with triple digit likes about few shot prompting?"
            )
        ],
        "tool_calls": [
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "few shot prompting",
                    "min_likes": 100,
                    "max_likes": 999,
                    "start_date": None,
                    "end_date": None,
                    "has_link": False,
                },
            }
        ],
    },
    {
        "question": [
            HumanMessage(
                "Are there any posts about LangServe before June 2023 that have more than 2000 likes and include a link?"
            )
        ],
        "tool_calls": [
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "LangServe",
                    "min_likes": 2000,
                    "max_likes": None,
                    "start_date": None,
                    "end_date": datetime(2023, 5, 31),
                    "has_link": True,
                },
            }
        ],
    },
]

BLOG_DATASET = [
    {
        "question": [
            HumanMessage("Have there been release notes in the past year about agents?")
        ],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "agents",
                    "authors": None,
                    "start_date": datetime(2023, 1, 1),
                    "end_date": None,
                },
            }
        ],
    },
    {
        "question": [
            HumanMessage(
                "how many press releases mentioned chat-gpt in the month after October 2023?"
            )
        ],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "chat-gpt",
                    "authors": None,
                    "start_date": datetime(2023, 11, 1),
                    "end_date": datetime(2023, 11, 30),
                },
            },
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "chat-gpt",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": datetime(2023, 11, 1),
                    "end_date": datetime(2023, 11, 30),
                    "has_link": False,
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage("what has been said about universal configurable models?")
        ],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "universal configurable models",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "universal configurable models",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": None,
                    "end_date": None,
                    "has_link": False,
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "In the last week, Have Harrison or Bagatur written anything about passing in runnables as tools in LangChain?"
            )
        ],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "runnables as tools",
                    "authors": ["Harrison", "Bagatur"],
                    "start_date": datetime(2023, 12, 25),
                    "end_date": None,
                },
            }
        ],
    },
    {
        "question": [
            HumanMessage(
                "Are there any case studies of agents running on swe-benchmark?"
            )
        ],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "agents running on swe-benchmark",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            }
        ],
    },
    {
        "question": [HumanMessage("Why is using fewshot prompting helpful?")],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "fewshot prompting",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
            {
                "name": "DocQuery",
                "args": {"query": "few shot prompting", "source": "langchain"},
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "i need to implement similarity search with filtering in FAISS. how can i do that in my app?"
            )
        ],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "similarity search with FAISS",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            }
        ],
    },
]  # Realease notes/announcements + Case studies +

AMBIGUOUS_DATASET = [
    {
        "question": [
            HumanMessage(
                "I want to migrate from agentexecutor to langgraph. What do I need to do?"
            )
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "migrate agentexecutor", "source": "langchain"},
            },
            {
                "name": "DocQuery",
                "args": {"query": "migrate agentexecutor", "source": "langgraph"},
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "In the last month, what are the latest updates to the openai partner package?"
            )
        ],
        "tool_calls": [
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "openai partner package",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": datetime(2023, 12, 1),
                    "end_date": None,
                    "has_link": False,
                },
            }
        ],
    },
    {
        "question": [
            HumanMessage(
                "What are best practices for setting up a document loader for a RAG chain?"
            )
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {
                    "query": "document loader for RAG chain",
                    "source": "langchain",
                },
            },
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "document loader best practies",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
        ],
    },
    {
        "question": [HumanMessage("case studies using langgraph last week?")],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "langgraph case studies",
                    "authors": None,
                    "start_date": datetime(2023, 12, 25),
                    "end_date": None,
                },
            }
        ],
    },
]

DATASET = DOC_DATASET + TWEET_DATASET + BLOG_DATASET + AMBIGUOUS_DATASET

QUERY_ANALYSIS_TASK = ToolUsageTask(
    name="Extraction Task",
    dataset_id="https://smith.langchain.com/public/594f9f60-30a0-49bf-b075-f44beabf546a/d",
    create_environment=get_environment,
    instructions=(
        """
                    You are requested to generate queries for searching either through tweets, docs, or blog entries. 
                    Inside the docs there are three different sources that you may wish to query for: LangGraph, LangSmith, or LangChain. 
                    LangGraph is a library for building multi-actor applications with LLMs, used to create agent and multi-agent workflows. 
                    LangSmith is an all-in-one developer platform for every step of the LLM-powered application lifecycle. 
                    It helps you debug, evaluate, test, and monitor your LLM applications. LangChain is a framework to build with LLMs by chaining interoperable components.
                    One last important thing to remember is that some queries will ask for date ranges, and you must remember that today is 2024-01-01. Also, remember that \
                    each question should be answered by a single query. In addition, you can return multiple queries to answer one question. Do not generate text, just tool calls that \
                    if executed would answer the users question. Do NOT pass the whole question as the query/subject, only extract key ideas/words.
                 """
    ),
    description=(
        """\
An environment that contains three different mock query tools for searching through LangChain material.

The three tools are for querying LangChain documentation, tweets, and blogs respectively.

The objective of the task it to measure how well the agent can select the correct tool and \
select the right parameters for the query. It is not a test of the actual querying process, \
merely the process of constructing the query.
"""
    ),
    eval_params={
        "output_evaluation": "qa_math_without_question",
    },
)

FEW_SHOT_DATASET = [
    {
        "question": [
            HumanMessage(
                "What are good rules to follow when using multi modal chat models?"
            )
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "multi modal chat models", "source": "langchain"},
            },
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "multi modal chat models",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage("How do you build a RAG chain with a Postgres vectorstore?")
        ],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "RAG chain with Postgres vectorstore",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
            {
                "name": "DocQuery",
                "args": {
                    "query": "RAG chain with Postgres vectorstore",
                    "source": "langchain",
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage("What case studies have we written about tool usage?")
        ],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "tool usage case study",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
        ],
    },
    {
        "question": [HumanMessage("How do I migrate from run_on_dataset to evaluate?")],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {
                    "query": "migrate run_on_dataset to evaluate",
                    "source": "langchain",
                },
            },
            {
                "name": "DocQuery",
                "args": {
                    "query": "migrate run_on_dataset to evaluate",
                    "source": "langsmith",
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "Do any of our posts in the last 2 months about Anthropic have less than 100 likes?"
            )
        ],
        "tool_calls": [
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "Anthropic",
                    "min_likes": None,
                    "max_likes": 100,
                    "start_date": datetime(2023, 11, 1),
                    "end_date": None,
                    "has_link": True,
                },
            }
        ],
    },
    {
        "question": [
            HumanMessage(
                "Did we release any information about claude-3.5 in the last week?"
            )
        ],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "claude-3.5",
                    "authors": None,
                    "start_date": datetime(2023, 12, 25),
                    "end_date": None,
                },
            },
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "claude-3.5",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": datetime(2023, 12, 25),
                    "end_date": None,
                    "has_link": False,
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "Do we have press statements about filtering traces by metadata before October 2023?"
            )
        ],
        "tool_calls": [
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "filtering traces by metadata",
                    "authors": None,
                    "start_date": None,
                    "end_date": datetime(2023, 9, 30),
                },
            },
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "filtering traces by metadata",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": None,
                    "end_date": datetime(2023, 9, 30),
                    "has_link": False,
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "What updates to mistral partner package were posted in the last year?"
            )
        ],
        "tool_calls": [
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "mistral partner package",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": datetime(2023, 1, 1),
                    "end_date": None,
                    "has_link": False,
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "Have there been updates to the best practices for initializing chat models in the past month?"
            )
        ],
        "tool_calls": [
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "best practices for initializing chat models",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": datetime(2023, 12, 1),
                    "end_date": None,
                    "has_link": False,
                },
            },
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "best practices for initializing chat models",
                    "authors": None,
                    "start_date": datetime(2023, 12, 1),
                    "end_date": None,
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "How can I learn about the differences between chat agents and graphs"
            )
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {
                    "query": "differences between chat agents and graphs",
                    "source": "langchain",
                },
            },
            {
                "name": "DocQuery",
                "args": {
                    "query": "differences between chat agents and graphs",
                    "source": "langgraph",
                },
            },
        ],
    },
    {
        "question": [
            HumanMessage(
                "What are good practices to follow for switching from legacy packages?"
            )
        ],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {
                    "query": "switching from legacy packages",
                    "source": "langchain",
                },
            },
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "switching from legacy packages",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
        ],
    },
    {
        "question": [HumanMessage("What data is exposed when I run custom evals?")],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {
                    "query": "data exposed running custom evaluation",
                    "source": "langsmith",
                },
            },
        ],
    },
    {
        "question": [HumanMessage("Where are document loaders talked about?")],
        "tool_calls": [
            {
                "name": "DocQuery",
                "args": {"query": "document loaders", "source": "langchain"},
            },
            {
                "name": "TweetQuery",
                "args": {
                    "subject": "document loaders",
                    "min_likes": None,
                    "max_likes": None,
                    "start_date": None,
                    "end_date": None,
                    "has_link": False,
                },
            },
            {
                "name": "BlogQuery",
                "args": {
                    "subject": "document loaders",
                    "authors": None,
                    "start_date": None,
                    "end_date": None,
                },
            },
        ],
    },
]


def _create_dataset(examples: list, dataset_id: str) -> None:
    """Create a dataset with the langsmith client."""

    client = Client()
    for example in examples:
        client.create_example(
            inputs={"question": example["question"]},
            outputs={"reference": example["tool_calls"]},
            dataset_id=dataset_id,
        )
