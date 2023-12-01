import json

from langchain.agents import AgentExecutor
from langchain.tools import tool
from langchain_docs_retriever.retriever import get_retriever
from langchain_experimental.openai_assistant import OpenAIAssistantRunnable

# This is used to tell the model how to best use the retriever.


_RETRIEVER = get_retriever()


@tool
def search(query, callbacks=None) -> str:
    """Search the LangChain docs with the retriever."""
    docs = _RETRIEVER.get_relevant_documents(query, callbacks=callbacks)
    return json.dumps([doc.dict() for doc in docs])


tools = [search]

agent = OpenAIAssistantRunnable.create_assistant(
    name="langchain docs assistant",
    instructions="You are a helpful assistant tasked with answering technical questions about LangChain.",
    tools=tools,
    model="gpt-4-1106-preview",
    as_agent=True,
)


agent_executor = (
    (lambda x: {"content": x["question"]})
    | AgentExecutor(agent=agent, tools=tools)
    | (lambda x: x["output"])
)
