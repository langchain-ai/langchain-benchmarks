from langchain.prompts import ChatPromptTemplate

from langchain_benchmarks.extraction.tasks.chat_extraction.evaluators import (
    get_eval_config,
)
from langchain_benchmarks.extraction.tasks.chat_extraction.schema import generateTicket
from langchain_benchmarks.schema import ExtractionTask

# This is a default prompt that works reasonably for OpenAI models.

DEFAULT_CHAT_MODEL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpdesk assistant responsible with extracting information"
            " and generating tickets. Dialogues are between a user and"
            " a support engineer.",
        ),
        (
            "user",
            "Generate a ticket for the following question-response pair:\n"
            "<Dialogue>\n{dialogue}\n</Dialogue>",
        ),
    ]
)


CHAT_EXTRACTION_TASK = ExtractionTask(
    name="Chat Extraction",
    dataset_id="https://smith.langchain.com/public/54d6d8e4-b420-4b9e-862d-548b1b65a6fe/d",
    schema=generateTicket,
    description="""A dataset meant to test the ability of an LLM to extract and infer
structured information from a dialogue. The dialogue is between a user and a support
engineer. Outputs should be structured as a JSON object and test both the ability
of the LLM to correctly structure the information and its ability to perform simple 
classification tasks.""",
    instructions=DEFAULT_CHAT_MODEL_PROMPT,
)


__all__ = ["CHAT_EXTRACTION_TASK", "get_eval_config"]
