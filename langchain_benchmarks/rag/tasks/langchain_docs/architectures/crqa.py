"""Chat langchain 'engine'."""
# TODO: some simplified architectures that are
# environment-agnostic
from operator import itemgetter
from typing import Callable, Dict, List, Optional, Sequence

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (
    Runnable,
    RunnableLambda,
)
from langchain.schema.runnable.passthrough import RunnableAssign
from pydantic import BaseModel

RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def _format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request.get("chat_history") or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def get_default_response_generator(llm: BaseLanguageModel) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    return (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )


def create_response_chain(
    response_generator: Runnable,
    retriever: BaseRetriever,
    format_docs: Optional[Callable[[Sequence[Document]], str]] = None,
    format_chat_history: Optional[Callable[[ChatRequest], str]] = None,
) -> Runnable:
    format_docs = format_docs or _format_docs
    format_chat_history = format_chat_history or serialize_history
    return (
        RunnableAssign(
            {
                "chat_history": RunnableLambda(format_chat_history).with_config(
                    run_name="SerializeHistory"
                )
            }
        )
        | RunnableAssign(
            {
                "context": (
                    itemgetter("question") | retriever | format_docs
                ).with_config(run_name="FormatDocs")
            }
        )
        | response_generator
    )
