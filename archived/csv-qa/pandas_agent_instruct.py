import pandas as pd
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents.agent_toolkits.conversational_retrieval.tool import (
    create_retriever_tool,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.tools import PythonAstREPLTool
from langchain.vectorstores import FAISS
from langsmith import Client
from pydantic import BaseModel, Field

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.load_local("titanic_data", embedding_model)
retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(), "person_name_search", "Search for a person by name"
)


TEMPLATE = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`

<df>
{dhead}
</df>

You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
You also do not have use only the information here to answer questions - you can run intermediate queries to do exporatory data analysis to give you more information as needed.

You have a tool called `person_name_search` through which you can lookup a person by name and find the records corresponding to people with similar name as the query.
You should only really use this if your search term contains a persons name. Otherwise, try to solve it with code.

For example:

<question>How old is Jane?</question>
<logic>Use `person_name_search` since you can use the query `Jane`</logic>

<question>Who has id 320</question>
<logic>Use `python_repl` since even though the question is about a person, you don't know their name so you can't include it.</logic>"""


class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")


if __name__ == "__main__":
    df = pd.read_csv("titanic.csv")
    template = TEMPLATE.format(dhead=df.head().to_markdown())

    def get_chain():
        repl = PythonAstREPLTool(
            locals={"df": df},
            name="python_repl",
            description="Runs code and returns the output of the final line",
            args_schema=PythonInputs,
        )
        tools = [repl, retriever_tool]
        agent = ZeroShotAgent.from_llm_and_tools(
            llm=OpenAI(temperature=0, model="gpt-3.5-turbo-instruct"),
            tools=tools,
            prefix=template,
        )
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, max_iterations=5, early_stopping_method="generate"
        )
        return agent_executor

    client = Client()
    eval_config = RunEvalConfig(
        evaluators=["qa"],
    )
    chain_results = run_on_dataset(
        client,
        dataset_name="Titanic CSV Data",
        llm_or_chain_factory=get_chain,
        evaluation=eval_config,
    )
