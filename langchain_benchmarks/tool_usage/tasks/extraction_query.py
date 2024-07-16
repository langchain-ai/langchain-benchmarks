'''
Notes:

Model needs to be aware that "today" means 2024-01-01
Need to provide it with a lot of context about what langchain/smith/graph are used for
'''

from typing import List, Literal, Optional, Union, cast

from langchain.tools import BaseTool, tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_benchmarks.schema import ToolUsageEnvironment, ToolUsageTask
from datetime import datetime
from langchain_core.messages import HumanMessage
from langsmith.schemas import Example, Run
from langsmith.evaluation.evaluator import (
    EvaluationResult,
    EvaluationResults,
    RunEvaluator,
)




class DocQuery(BaseModel):
    """Query against documentation"""

    query: str = Field(...,description="The question to answer")
    source: Literal["langchain", "langsmith", "langgraph"] = Field(...,description="The documentation source to search against. Should be one of 'langchain', 'langsmith', or "
                                                                   "'langgraph' depending on which one product the user question pertains to")

class TweetQuery(BaseModel):
    """Query against tweets"""

    subject: str = Field(...,description="Subject to search for")
    min_likes: Union[int, None] = Field(None,description="Minimum amount of likes on the tweet")
    max_likes: Union[int, None] = Field(None,description="Maximum amount of likes on the tweet")
    start_date: Union[datetime, None] = Field(None, description="Earliest date to start pulling tweets from")
    end_date: Union[datetime, None] = Field(None,description="Latest date to pull tweets from, None if pulling up to the present")
    has_link: bool = Field(False,description="Whether to query for tweets that have a link.")

class BlogQuery(BaseModel):
    """Query against blog posts"""

    subject: Union[str, None] = Field(...,description="Subject to search for")
    authors: Union[None, str, list[str]] = Field(None,description="Authors to search for. None if not searching for a speific author,  list if searching for more than one.")
    start_date: Union[datetime, None] = Field(None, description="Earliest date to start pulling blog posts from")
    end_date: Union[datetime, None] = Field(None,description="Latest date to pull blog posts from")

def get_environment() -> ToolUsageEnvironment:
    """Create an environment."""
    tools = cast(
        List[BaseTool],
        [
            tool(func)
            for func in [
                TweetQuery,
                DocQuery,
                BlogQuery
            ]
        ],
    )
    return ToolUsageEnvironment(
        tools=tools,
        read_state=None,
    )


DOC_DATASET = [
    {
        "question":[HumanMessage("How do I use the langgraph Send method?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'Send method','source':'langgraph'}}]
    },
    {
        "question":[HumanMessage("How do you chain a prompt with a model?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'chaining prompt and model','source':'langchain'}}]
    },
    {
        "question":[HumanMessage("How do you run a pairwise experiment in langsmith?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'pairwise experiment','source':'langsmith'}}]
    },
    {
        "question":[HumanMessage("What is a tool node?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'tool node','source':'langgraph'}}]
    },
    {
        "question":[HumanMessage("How do I get the log probabilities of my chat model?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'log probabilities','source':'langchain'}}]
    },
    {
        "question":[HumanMessage("How can I build my own custom evaluator?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'custom evaluator','source':'langsmith'}}]
    },
    {
        "question":[HumanMessage("How do I use a tool in a routing function?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'tool in routing function','source':'langgraph'}}]
    },
    {
        "question":[HumanMessage("How do use Pinecone as a vectorstore for few shot prompting?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'Pinecone for few shot prompting','source':'langchain'}}]
    },
    {
        "question":[HumanMessage("How do I prevent personal data from being logged in my traces?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'personal data logging','source':'langsmith'}}]
    },
    {
        "question":[HumanMessage("How do you use a nested graph? Can you stream messages from inside them?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'nested graph','source':'langgraph'}},
                    {'name': 'DocQuery',
                    'args': {'query': 'stream messages nested graph','source':'langgraph'}}]
    },
    {
        "question":[HumanMessage("How do I extract text from PDF data for my retrieval chain? Can I combine image and text in a prompt?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'PDF extraction for chain','source':'langchain'}},
                    {'name': 'DocQuery',
                    'args': {'query': 'multimodal prompt','source':'langchain'}}]
    },
    {
        "question":[HumanMessage("How do I setup automation rules for my traces? How do I view logs for those rules?")],
        "tool_calls":[{'name': 'DocQuery',
                    'args': {'query': 'automation rules for traces','source':'langsmith'}},
                    {'name': 'DocQuery',
                    'args': {'query': 'automation rules logs','source':'langsmith'}}]
    },
]

TWEET_DATASET = [
    {
        "question":[HumanMessage("Did we have any tweets about agents with more than 1000 likes that also included a link?")],
        "tool_calls":[{'name': 'TweetQuery',
                    'args': {'subject': 'agents','min_likes':1000,'max_likes':None,"start_date":None,"end_date":None,"has_link":True}}]
    },
    {
        "question":[HumanMessage("Are there any tweets about evaluators by langchain with less than 100 likes?")],
        "tool_calls":[{'name': 'TweetQuery',
                    'args': {'subject': 'evaluators','min_likes':None,'max_likes':100,"start_date":None,"end_date":None,"has_link":False}}]
    },
    {
        "question":[HumanMessage("Are there any tweets that link to the anthropic website in the last year?")],
        "tool_calls":[{'name': 'TweetQuery',
                    'args': {'subject': 'anthropic','min_likes':None,'max_likes':None,"start_date":datetime(2023,1,1),"end_date":None,"has_link":True}}]
    },
    {
        "question":[HumanMessage("In Q2 2023 did we tweet anything about LangSmith?")],
        "tool_calls":[{'name': 'TweetQuery',
                    'args': {'subject': 'LangSmith','min_likes':None,'max_likes':None,"start_date":datetime(2023,3,1),"end_date":datetime(2023,6,1),"has_link":False}}]
    },
    {
        "question":[HumanMessage("Were there any social media posts with triple digit likes about few shot prompting?")],
        "tool_calls":[{'name': 'TweetQuery',
                    'args': {'subject': 'few shot prompting','min_likes':100,'max_likes':999,"start_date":None,"end_date":None,"has_link":False}}]
    },
    {
        "question":[HumanMessage("Are there any posts aout LangServe before June 2023 that have more than 2000 likes and include a link?")],
        "tool_calls":[{'name': 'TweetQuery',
                    'args': {'subject': 'LangServe','min_likes':2000,'max_likes':None,"start_date":None,"end_date":datetime(2023,5,31),"has_link":True}}]
    },
]

BLOG_DATASET = [
    {
        "question":[HumanMessage("what are some blog posts in the past year about agents?")],
        "tool_calls":[{'name': 'BlogQuery',
                    'args': {'subject': 'agents','authors':None,"start_date":datetime(2023,1,1),"end_date":None}}]
    },
    {
        "question":[HumanMessage("how many blogs mentioned chat-gpt in the month after October 2023?")],
        "tool_calls":[{'name': 'BlogQuery',
                    'args': {'subject': 'chat gpt','authors':None,"start_date":datetime(2023,11,1),"end_date":datetime(2023,11,30)}}]
    },
    {
        "question":[HumanMessage("what has Bagatur written about universal configurable models?")],
        "tool_calls":[{'name': 'BlogQuery',
                    'args': {'subject': 'universal configurable model','authors':"Bagatur","start_date":None,"end_date":None}}]
    },
    {
        "question":[HumanMessage("Have Harrison or Bagatur written anything about passing in runnables as tools in the last week?")],
        "tool_calls":[{'name': 'BlogQuery',
                    'args': {'subject': 'runnables as tools','authors':["Harrison","Bagatur"],"start_date":datetime(2023,12,24),"end_date":None}}]
    },
    {
        "question":[HumanMessage("Did Harrison write anything about LangGraph in the time frame up to the end of Q1 2023?")],
        "tool_calls":[{'name': 'BlogQuery',
                    'args': {'subject': 'LangGraph','authors':"Harrison","start_date":None,"end_date":datetime(2023,3,1)}}]
    }
]

DATASET = DOC_DATASET + TWEET_DATASET + BLOG_DATASET

EXTRACTION_TASK = ToolUsageTask(
    name="Extraction Task",
    dataset_id="https://smith.langchain.com/public/594f9f60-30a0-49bf-b075-f44beabf546a/d",
    create_environment=get_environment,
    instructions=(
        "You are requested to solve math questions in an alternate"
    ),
    description=(
        """\
An environment that contains three different mock query tools for searching through LangChain material.

The three tools are for querying LangChain documentation, tweets, and blogs respectively.

The objective of the task it to measure how well the agent can select the correct tool and \
select the right parameterse for the query. It is not a test of the actual querying process, \
merely the process of constructing the query.
"""
    ),
    eval_params={
        "output_evaluation": "qa_math_without_question",
    },
)

def _create_dataset() -> None:
    """Create a dataset with the langsmith client."""
    from langsmith.client import Client

    client = Client()

    dataset_id = "e3101cae-af77-476f-a331-eb2e92e809e6"

    for example in DATASET:
        client.create_example(
            inputs={
                "question": example["question"],
            },
            outputs={
                "reference": example["tool_calls"],
            },
            dataset_id=dataset_id,
        )

if __name__=="__main__":
    #_create_dataset()
    pass