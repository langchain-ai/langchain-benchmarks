from typing import List, Tuple, Optional

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain.tools.render import format_tool_to_openai_function
from langchain_docs_retriever.retriever import get_retriever

# This is used to tell the model how to best use the retriever.


_RETRIEVER = get_retriever()


@tool
def search(query, callbacks=None):
    """Search the LangChain docs with the retriever."""
    return _RETRIEVER.get_relevant_documents(query, callbacks=callbacks)


tools = [search]

assistant_system_message = """You are a helpful assistant tasked with answering technical questions about LangChain. \
Use tools (only if necessary) to best answer the users questions. Do not make up information if you cannot find the answer using your tools."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", assistant_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})


class ChainInput(BaseModel):
    question: str


def mapper(input: dict):
    return {"input": input["question"], "chat_history": []}


def create_executor(model_config: Optional[dict] = None):
    model = (model_config or {}).get("model", "gpt-3.5-turbo-16k")
    llm = ChatOpenAI(model=model, temperature=0)
    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            "agent_scratchpad": lambda x: format_to_openai_functions(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False).with_types(
        input_type=AgentInput
    )
    return (mapper | agent_executor | (lambda x: x["output"])).with_types(
        input_type=ChainInput
    )


agent_executor = create_executor()
