from enum import Enum
from typing import List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field

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


# This is a default prompt that works for chat models.
DEFAULT_CHAT_MODEL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert researcher."),
        (
            "human",
            "What can you tell me about the following email? Make sure to "
            "extract the question in the correct format. "
            "Here is the email:\n ```\n{input}\n```",
        ),
    ]
)

EMAIL_EXTRACTION_TASK = ExtractionTask(
    name="Email Extraction",
    dataset_id="https://smith.langchain.com/public/a1742786-bde5-4f51-a1d8-e148e5251ddb/d",
    schema=Email,
    description="""\
A dataset of 42 real emails deduped from a spam folder, with semantic HTML tags removed, \
as well as a script for initial extraction and formatting of other emails from \
an arbitrary .mbox file like the one exported by Gmail.

Some additional cleanup of the data was done by hand after the initial pass.

See https://github.com/jacoblee93/oss-model-extraction-evals.
    """,
    instructions=DEFAULT_CHAT_MODEL_PROMPT,
)
