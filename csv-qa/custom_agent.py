from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import PythonAstREPLTool
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset
from pydantic import BaseModel, Field
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

embedding_model = OpenAIEmbeddings()
embedding_cache = {}


def get_embeds(df, col):
    _hash = hash(df[col].to_markdown())
    if _hash in embedding_cache:
        return embedding_cache[_hash]
    else:
        embeddings = embedding_model.embed_documents(df[col])
        embedding_cache[_hash] = embeddings
        return embedding_cache[_hash]


def similarity_search(df: pd.DataFrame, col: str, query: str) -> pd.DataFrame:
    """Returns the 5 rows whose values in `col` are most similary to `query`."""
    embeddings = get_embeds(df, col)
    query_e = embedding_model.embed_query(query)
    return df.iloc[np.argsort(np.dot(np.array([query_e]), np.array(embeddings).T), )[0][-5:]]


TEMPLATE = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`

<df>
{dhead}
</df>

You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
You also do not have use only the information here to answer questions - you can run intermediate queries to do exporatory data analysis to give you more information as needed.

You also have access to a great util for working with columns that are mostly text data! You have this function:

```python
def similarity_search(df: pd.DataFrame, col: str, query: str) -> pd.DataFrame:
    # Returns the 5 rows whose values in `col` are most similary to `query`.
```"""


class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")


if __name__ == "__main__":
    df = pd.read_csv("titanic.csv")
    get_embeds(df, "Name")
    template = TEMPLATE.format(dhead=df.head().to_markdown())

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}")
    ])

    def get_chain():
        repl = PythonAstREPLTool(locals={"df": df, "similarity_search": similarity_search}, name="python_repl",
                                 description="Runs code and returns the output of the final line",
                                 args_schema=PythonInputs)

        agent = OpenAIFunctionsAgent(llm=ChatOpenAI(temperature=0, model="gpt-4"), prompt=prompt, tools=[repl])
        agent_executor = AgentExecutor(agent=agent, tools=[repl])
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
    )