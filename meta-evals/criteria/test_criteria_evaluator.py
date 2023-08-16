# Tests for the criteria evaluator
from uuid import uuid4

import pytest
from langsmith import Client, EvaluationResult
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Example, Run

from langchain.evaluation import load_evaluator
from langchain.smith import RunEvalConfig


class ExactScoreMatch(RunEvaluator):
    def evaluate_run(self, run: Run, example: Example) -> EvaluationResult:
        predicted_score = run.outputs["score"]
        return EvaluationResult(
            key="exact_score_match",
            score=predicted_score == example.outputs["output_correctness_score"],
        )


@pytest.fixture(scope="session")
def uid() -> str:
    return uuid4().hex[:8]


async def _check_dataset(
    loader_kwargs: dict, dataset_name: str, project_name: str, tags: list
) -> None:
    client = Client()
    eval_chain = load_evaluator(**loader_kwargs)

    def predict(input: dict) -> dict:
        # TODO: Would be better to propagate callbacks
        return eval_chain.evaluate_strings(
            input=input["input"],
            prediction=input["input_prediction"],
            reference=input["input_answer"],
        )

    res = await client.arun_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=predict,
        evaluation=RunEvalConfig(
            custom_evaluators=[ExactScoreMatch()],
        ),
        verbose=True,
        project_name=project_name,
        tags=["int-test"] + tags,
    )
    feedback = client.list_feedback(
        run_ids=[r.id for r in client.list_runs(project_name=res["project_name"])]
    )
    scores = [
        f.score
        for f in feedback
        if f.key == "exact_score_match" and f.score is not None
    ]
    assert len(scores) == 100
    avg_score = sum(scores) / len(scores)
    return avg_score


@pytest.mark.parametrize(
    "loader_kwargs",
    [
        {"evaluator": "cot_qa"},
        # {"evaluator": "qa"},
        # {"evaluator": "labeled_criteria", "criteria": "correctness"},
    ],
)
@pytest.mark.asyncio
async def test_metaeval_correctness(loader_kwargs: dict, uid: str):
    # Expect 100% correctness
    dataset_name = "Web Q&A Dataset - Correct"
    project_name = f"{loader_kwargs['evaluator']} - int test - correctness - {uid}"
    score = await _check_dataset(
        loader_kwargs, dataset_name, project_name, tags=["test_metaeval_correctness"]
    )
    assert score >= 0.95


@pytest.mark.parametrize(
    "loader_kwargs",
    [
        {"evaluator": "qa"},
        {"evaluator": "labeled_criteria", "criteria": "correctness"},
    ],
)
@pytest.mark.asyncio
@pytest.mark.skip(reason="Already passes 100% so don't need to test as frequently.")
async def test_metaeval_incorrectness(loader_kwargs: dict, uid: str):
    # Expect 100% to be labeled as incorrect
    dataset_name = "Web Q&A Dataset - Incorrect"
    project_name = f"{loader_kwargs['evaluator']} - int test - incorrectness - {uid}"
    score = await _check_dataset(
        loader_kwargs, dataset_name, project_name, tags=["test_metaeval_incorrectness"]
    )
    assert score >= 0.95
