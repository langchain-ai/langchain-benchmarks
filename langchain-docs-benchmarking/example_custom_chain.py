from langchain.chat_models import ChatOpenAI, ChatAnthropic
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
    return {
        "question": lambda x: x["question"],
        "docs": (lambda x: x["question"]) | retriever,
    } | model
