"""Default implementations of LLMs that can be used for extraction."""
import os

from typing import Optional, List
from enum import Enum

from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_experimental.llms.anthropic_functions import AnthropicFunctions
from langchain.pydantic_v1 import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import convert_to_openai_function


class ToneEnum(str, Enum):
    positive = "positive"
    negative = "negative"


class Email(BaseModel):
    """Relevant information about an email."""

    sender: Optional[str] = Field(None, description="The sender's name, if available")
    sender_phone_number: Optional[str] = Field(
        None, description="The sender's phone number, if available"
    )
    sender_address: Optional[str] = Field(
        None, description="The sender's address, if available"
    )
    action_items: List[str] = Field(
        ..., description="A list of action items requested by the email"
    )
    topic: str = Field(
        ..., description="High level description of what the email is about"
    )
    tone: ToneEnum = Field(..., description="The tone of the email.")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert researcher."),
        (
            "human",
            "What can you tell me about the following email? Make sure to answer in the correct format: {email}",
        ),
    ]
)

openai_functions = [convert_to_openai_function(Email)]
llm_kwargs = {
    "functions": openai_functions,
    "function_call": {"name": openai_functions[0]["name"]},
}

# Ollama JSON mode has a bug where it infintely generates newlines. This stop sequence hack fixes it
llm = OllamaFunctions(temperature=0, model="llama2", timeout=300, stop=["\n\n\n\n"])
# llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
# llm = AnthropicFunctions(temperature=0, model="claude-2")

# output_parser = get_openai_output_parser([Email])
output_parser = JsonOutputFunctionsParser()
extraction_chain = (
    prompt | llm.bind(**llm_kwargs) | output_parser | (lambda x: {"output": x})
)

eval_llm = ChatOpenAI(model="gpt-4", temperature=0.0, model_kwargs={"seed": 42})

evaluation_config = RunEvalConfig(
    evaluators=[
        RunEvalConfig.LabeledScoreString(
            criteria={
                "accuracy": """
Score 1: The answer is incorrect and unrelated to the question or reference document.
Score 3: The answer is partially correct but has more than one omission or major errors.
Score 5: The answer is mostly correct but has more than one omission or major error.
Score 7: The answer is mostly correct but has at most one omission or major error.
Score 9: The answer is mostly correct with no omissions and only minor errors, and aligns with the reference document.
Score 10: The answer is correct, complete, and aligns with the reference document. Extra information is acceptable if it is sensible.

If the reference answer contains multiple alternatives, the predicted answer must only match one of the alternatives to be considered correct.
If the predicted answer contains additional helpful and accurate information that is not present in the reference answer, it should still be considered correct and not be penalized.
"""  # noqa
            },
            llm=eval_llm,
            normalize_by=10.0,
        ),
    ],
)

client = Client()
run_on_dataset(
    dataset_name="Extraction Over Spam Emails",
    llm_or_chain_factory=extraction_chain,
    client=client,
    evaluation=evaluation_config,
    project_name="llama2-test",
    concurrency_level=1,
)
