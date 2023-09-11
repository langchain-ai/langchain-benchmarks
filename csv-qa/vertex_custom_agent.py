from langchain.agents import XMLAgent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.tools import PythonAstREPLTool
import pandas as pd
import time
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langsmith import Client
from langchain.chains.llm import LLMChain
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.agents.agent import RunnableAgent
from langchain.smith import RunEvalConfig, run_on_dataset
from pydantic import BaseModel, Field
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits.conversational_retrieval.tool import create_retriever_tool


pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.load_local("titanic_data", embedding_model)
retriever_tool = create_retriever_tool(vectorstore.as_retriever(), "person_name_search", "Search for a person by name")


TEMPLATE = """

The full list of tools you have is:

{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. \
You will then get back a response in the form <observation></observation>
For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

When you are done, respond with a final answer between <final_answer></final_answer>. For example:

<final_answer>The weather in SF is 64 degrees</final_answer>

Begin!

Question: {question}
{intermediate_steps}
"""

### Helper Functions

# NOTE: only the instructions bit
INSTRUCTIONS = """You have access to the following tools:

{tools_description}

Use a blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $BLOB, as shown.

<action>
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
</action>

When invoking a tool do not provide any clarifying information.

The human will forward results of tool invocations as "Observations".

When you know the answer paraphrase the information in the observations properly and respond to the user. \
If you do not know the answer use more tools.

You can only take a single action at a time."""

from typing import Union
import re
import ast
from langchain.schema import AgentAction, AgentFinish, BaseMessage
def _decode(text: Union[BaseMessage, str]):
    """Decode the action."""
    pattern = re.compile(r"<action>(?P<action_blob>.*?)<\/action>", re.DOTALL)
    # NOTE: this to handle strings
    if isinstance(text, BaseMessage):
        _text = text.content
    else:
        _text = text
    match = pattern.search(_text)
    if match:
        action_blob = match.group("action_blob")
        data = ast.literal_eval(action_blob)
        name = data["action"]
        if name == "Final Answer":  # Special cased "tool" for final answer
            # NOTE: i want to use the old agent action/finish - why do we have new ones
            return AgentFinish(return_values={"output":data["action_input"]}, log="")
        return AgentAction(
            tool=data["action"], tool_input=data["action_input"] or {}, log=""
        )
    else:
        return AgentFinish(return_values={"output":text}, log="")

from typing import Sequence, TypedDict

from langchain.tools import BaseTool


def _generate_tools_descriptions(tools: Sequence[BaseTool]) -> str:
    """Generate a description of the tools."""
    return "\n".join([f"{tool_.name}: {tool_.description}" for tool_ in tools]) + "\n"


class ToolInfo(TypedDict):
    """A dictionary containing information about a tool."""

    tool_names: str
    tools_description: str


def generate_tool_info(tools: Sequence[BaseTool]) -> ToolInfo:
    """Generate a string containing the names of the tools and their descriptions."""
    tools_description = _generate_tools_descriptions(tools)
    # NOTE: slight change to `f'"{tool_.name}"'` here
    tool_names = ", ".join([f'"{tool_.name}"' for tool_ in tools])
    return {
        "tool_names": tool_names,
        "tools_description": tools_description,
    }



### END OF HELPER

TEMPLATE = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`

<df>
{dhead}
</df>

You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
You also do not have use only the information here to answer questions - you can run intermediate queries to do exporatory data analysis to give you more information as needed.

You have a tool called `person_name_search` through which you can lookup a person by name and find the records corresponding to people with similar name as the query.
You should only really use this if your search term contains a persons name. Otherwise, try to solve it with code.

""" + INSTRUCTIONS
class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")


def construct_scratchpad(steps):
    log = []
    for agent_action, observation in steps:
        log.append(AIMessage(content="""<action>
    {{
      "action": "{action}",
      "action_input": "{action_input}"
    }}
    </action>""".format(action=agent_action.tool, action_input=agent_action.tool_input)))
        log.append(HumanMessage(content=str(observation)))
    return log


if __name__ == "__main__":
    df = pd.read_csv("titanic.csv")


    def get_chain():
        repl = PythonAstREPLTool(locals={"df": df}, name="python_repl",
                                 description="Runs code and returns the output of the final line",
                                 args_schema=PythonInputs)
        tools = [repl, retriever_tool]

        prompt = PromptTemplate.from_template(TEMPLATE)

        prompt = ChatPromptTemplate.from_messages([
            ("system", TEMPLATE),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]).partial(dhead=df.head().to_markdown(),
                                                                **generate_tool_info([repl, retriever_tool]))
        chain = {
            "question": lambda x: x["question"],
            "agent_scratchpad": lambda x: construct_scratchpad(x["intermediate_steps"])
                }|prompt | ChatVertexAI(temperature=0, max_output_tokens=1000) | _decode
        agent = RunnableAgent(runnable=chain, specified_input_keys=["question"])
        agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, early_stopping_method="generate")
        return agent_executor


    client = Client()
    eval_config = RunEvalConfig(
        evaluators=[
            "qa"
        ],
    )
    chain_results = run_on_dataset(
        client,
        dataset_name="Titanic CSV Data",
        llm_or_chain_factory=get_chain,
        evaluation=eval_config,
        project_name=f"Vertex {time.time()}"
    )
