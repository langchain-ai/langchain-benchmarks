from langchain.smith import RunEvalConfig
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.schemas import Example, Run

from langchain_benchmarks.schema import ExtractionTask


@run_evaluator
def correct_name(run: Run, example: Example) -> EvaluationResult:
    if "name" in run.outputs:
        prediction = run.outputs["name"]
    else:
        prediction = run.outputs["output"]["name"]
    name = example.outputs["name"]
    score = int(name == prediction)
    return EvaluationResult(key="correct", score=score)


class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The person's name")


NAME_CORRECTION_TASK = ExtractionTask(
    name="Name Correction",
    dataset_id="https://smith.langchain.com/public/78df83ee-ba7f-41c6-832c-2b23327d4cf7/d",
    schema=Person,
    description="""A dataset of 23 misspelled full names and their correct spellings.""",
    dataset_url="https://smith.langchain.com/public/78df83ee-ba7f-41c6-832c-2b23327d4cf7/d",
    dataset_name="Extracting Corrected Names",
    eval_config=RunEvalConfig(
        custom_evaluators=[correct_name],
    ),
)
