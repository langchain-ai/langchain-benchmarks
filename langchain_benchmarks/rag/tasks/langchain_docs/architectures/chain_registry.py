from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable
from langchain.base_language import BaseLanguageModel
from langchain_benchmarks.rag.environments.langchain_docs.architectures.crqa import (
    create_response_chain,
    get_default_response_synthesizer,
)


def default_response_chain(
    retriever: BaseRetriever,
    response_synthesizer: Optional[Runnable] = None,
    llm: Optional[BaseLanguageModel] = None,
) -> None:
    """Get the chain responsible for generating a response based on the retrieved documents."""
    response_synthesizer = response_synthesizer or get_default_response_synthesizer(
        llm=llm or ChatOpenAI(model="gpt-3.5-turbo-16k", model_kwargs={"seed": 42})
    )
    return create_response_chain(
        response_synthesizer=response_synthesizer, retriever=retriever
    )


ARCH_FACTORIES = {
    "conversational-retrieval-qa": default_response_chain,
}
