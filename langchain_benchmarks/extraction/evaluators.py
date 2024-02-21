from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.smith import RunEvalConfig


def get_eval_config(eval_llm: Optional[BaseChatModel] = None) -> RunEvalConfig:
    eval_llm = eval_llm or ChatOpenAI(
        model="gpt-4",
        temperature=0,
        model_kwargs={"seed": 42},
        max_retries=1,
        request_timeout=60,
    )
    """Get the evaluation configuration for the email task."""
    return RunEvalConfig(
        evaluators=[
            "json_edit_distance",
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
