from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_docs_retriever.retriever import get_retriever


def create_runnable(config: dict):
    config_copy = config.copy()
    chat_cls_name = config_copy.pop("chat_cls", "ChatOpenAI")

    assert chat_cls_name in {"ChatOpenAI", "ChatAnthropic"}
    chat_cls = {
        "ChatOpenAI": ChatOpenAI,
        "ChatAnthropic": ChatAnthropic,
    }[chat_cls_name]
    model = chat_cls(**config_copy)
    retriever = get_retriever(config.get("retriever_config", {}))
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the Q using the following docs\n{docs}"),
            ("user", "Q: {question}"),
        ]
    )
    return (
        {
            "question": lambda x: x["question"],
            "docs": (lambda x: x["question"]) | retriever,
        }
        | prompt
        | model
        | StrOutputParser()
    )
