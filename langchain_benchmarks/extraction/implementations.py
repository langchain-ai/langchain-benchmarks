"""Default implementations of LLMs that can be used for extraction."""
from typing import Any, Dict, List, Optional, Type

from langchain.chains.openai_functions import convert_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langsmith.client import Client
from pydantic import BaseModel

from langchain_benchmarks.extraction.evaluators import get_eval_config
from langchain_benchmarks.schema import ExtractionTask

# PUBLIC API


def create_openai_function_based_extractor(
    prompt: ChatPromptTemplate,
    llm: Runnable,
    schema: Type[BaseModel],
) -> Runnable[dict, dict]:
    """Create an extraction chain that uses an LLM to extract a schema.

    The underlying functionality is exclusively for LLMs that support
    extraction using openai functions format.

    Args:
        prompt: The prompt to use for extraction.
        llm: The LLM to use for extraction.
        schema: The schema to extract.

    Returns:
        An llm that will extract the schema
    """
    openai_functions = [convert_to_openai_function(schema)]
    llm_kwargs = {
        "functions": openai_functions,
        "function_call": {"name": openai_functions[0]["name"]},
    }
    output_parser = JsonOutputFunctionsParser()
    extraction_chain = (
        prompt | llm.bind(**llm_kwargs) | output_parser | (lambda x: {"output": x})
    )
    return extraction_chain


def run_on_dataset(
    task: ExtractionTask,
    llm: Runnable,
    *,
    tags: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run an LLM on a dataset.

    Args:
        task: The task to run on.
        llm: The LLM to run.
        tags: The tags to use for the run.
        kwargs: Additional arguments to pass to the client.
    """
    client = Client()
    eval_llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.0,
        model_kwargs={"seed": 42},
        max_retries=1,
        request_timeout=60,
    )
    return client.run_on_dataset(
        dataset_name=task.name,
        llm_or_chain_factory=create_openai_function_based_extractor(
            task.instructions, llm, task.schema
        ),
        evaluation=get_eval_config(eval_llm),
        tags=tags,
        **kwargs,
    )
