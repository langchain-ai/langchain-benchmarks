from enum import Enum
from typing import Optional, List

from langchain.smith import RunEvalConfig
from pydantic import BaseModel, Field

from langchain_benchmarks.schema import ExtractionTask


class ToneEnum(str, Enum):
    """The tone of the email."""

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


def get_eval_config(eval_llm: BaseModel) -> RunEvalConfig:
    """Get the evaluation configuration for the email task."""
    return RunEvalConfig(
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


EmailTask = ExtractionTask(
    id=4,  # To be deprecated
    name="Email Extraction",
    dataset_id="https://smith.langchain.com/public/36bdfe7d-3cd1-4b36-b957-d12d95810a2b/d",
    model=Email,
    description="""\
A dataset of 42 real emails deduped from a spam folder, with semantic HTML tags removed, \
as well as a script for initial extraction and formatting of other emails from \
an arbitrary .mbox file like the one exported by Gmail.

Some additional cleanup of the data was done by hand after the initial pass.

See https://github.com/jacoblee93/oss-model-extraction-evals.
    """,
)
