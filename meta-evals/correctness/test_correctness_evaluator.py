# Tests for the criteria evaluator
from typing import Optional, Tuple
from uuid import uuid4

import pytest
from langchain.callbacks.manager import CallbackManager
from langchain.evaluation import load_evaluator
from langchain.load import dump as langchain_dump
from langchain.schema import runnable
from langchain.smith import RunEvalConfig
from langsmith import Client, EvaluationResult
from langchain.prompts import PromptTemplate
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Example, Run


class ExactScoreMatch(RunEvaluator):
    def evaluate_run(self, run: Run, example: Example) -> EvaluationResult:
        predicted_score = run.outputs["score"]
        return EvaluationResult(
            key="exact_score_match",
            score=predicted_score == example.outputs["output_correctness_score"],
        )


class AbsDistanceEvaluator(RunEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def evaluate_run(self, run: Run, example: Example) -> EvaluationResult:
        predicted_score = run.outputs["score"]
        if predicted_score is None:
            return EvaluationResult(key="absolute_distance", score=None)
        return EvaluationResult(
            key="absolute_distance",
            score=abs(predicted_score - example.outputs["output_correctness_score"]),
        )

class NullScore(RunEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def evaluate_run(self, run: Run, example: Example) -> EvaluationResult:
        predicted_score = run.outputs["score"]
        null_score = 1 if predicted_score is None else 0
        return EvaluationResult(key="null_score", score=null_score)

@pytest.fixture(scope="session")
def uid() -> str:
    return uuid4().hex[:8]


class EvaluatorRunnable(runnable.Runnable):
    # We're going through the non-invoke API of the evaluator
    # so to maintain nesting, we are copying some of the invoke code.
    # This is so that the full trace isn't separated from the runnable.
    def __init__(self, eval_chain) -> None:
        super().__init__()
        self._eval_chain = eval_chain

    def invoke(
        self, input: dict, config: Optional[runnable.RunnableConfig] = None
    ) -> dict:
        config = config or {}
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = callback_manager.on_chain_start(
            langchain_dump.dumpd(self),
            input if isinstance(input, dict) else {"input": input},
            run_type="chain",
        )
        try:
            output = self._eval_chain.evaluate_strings(
                input=input["input"],
                prediction=input["input_prediction"],
                reference=input["input_answer"],
                callbacks=run_manager.get_child(),
            )
        except Exception as e:
            run_manager.on_chain_error(e)
            raise
        else:
            output_for_tracer = langchain_dump.dumpd(output)
            run_manager.on_chain_end(
                output_for_tracer
                if isinstance(output_for_tracer, dict)
                else {"output": output_for_tracer}
            )
            return output


async def _check_dataset(
    loader_kwargs: dict, dataset_name: str, project_name: str, tags: list
) -> Tuple[float, float]:
    client = Client()
    eval_chain = load_evaluator(**loader_kwargs)
    to_evaluate = EvaluatorRunnable(eval_chain=eval_chain)

    res = await client.arun_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=to_evaluate,
        evaluation=RunEvalConfig(
            custom_evaluators=[ExactScoreMatch(), AbsDistanceEvaluator(), NullScore()],
        ),
        verbose=True,
        project_name=project_name,
        tags=["int-test"] + tags,
    )
    feedback = list(client.list_feedback(
        run_ids=[r.id for r in client.list_runs(project_name=res["project_name"])]
    ))
    scores = [
        f.score
        for f in feedback
        if f.key == "exact_score_match" and f.score is not None
    ]
    # assert len(scores) == 100
    distance_scores = [
        f.score
        for f in feedback
        if f.key == "absolute_distance" and f.score is not None
    ]
    # assert len(distance_scores) == 100
    null_pred_scores = [
        f.score
        for f in feedback
        if f.key == "null_score" and f.score is not None
    ]
    avg_score = sum(scores) / len(scores)
    avg_distance_score = sum(distance_scores) / len(distance_scores)
    avg_null_pred_score = sum(null_pred_scores) / len(null_pred_scores)
    return {"exact_score_match": avg_score, "absolute_distance": avg_distance_score, "null_predictions": avg_null_pred_score, "lengths": [len(scores), len(distance_scores), len(null_pred_scores)]}


def _get_project_name(loader_kwargs: dict, uid: str) -> str:
    other_args = "-".join(f"[{k}={v}]" for k, v in loader_kwargs.items())
    return f"{loader_kwargs['evaluator']}{other_args} - {uid}"

template_variants = [
"""Evaluate the submission for the following task:
<input>
{input}
</input>
<submission>
{output}
</submission>
<reference>
{reference}
</reference>

Criterion to evaluate: {criteria}

1. Compare the submission with the reference.
2. Determine if the submission meets the criterion.
3. Explain your reasoning in a step-by-step manner.
4. Print the single character "Y" or "N" to represent whether the submission meets the criterion.
5. Repeat the character on a new line.

Y/N
""",
"""Compare the following submission with the reference for the given task:
<input>
{input}
</input>
<submission>
{output}
</submission>
<reference>
{reference}
</reference>

Criteria: {criteria}

1. Analyze the submission against the reference considering the above criteria.
2. Detail your reasoning in a clear and logical sequence.
3. Print "Y" if the submission meets the criteria, or "N" if it does not.
4. Repeat the character on a new line.

Y/N
""",
"""You are to assess whether a submission satisfies a specified criterion:
<input>
{input}
</input>
<submission>
{output}
</submission>
<criteria>
{criteria}
</criteria>
<reference>
{reference}
</reference>

Does the submission meet the criterion? Follow these steps:

1. Analyze the submission in comparison with the reference.
2. Determine if it satisfies the given criterion.
3. Detail your reasoning in a step-by-step manner.
4. Print "Y" if the submission meets the criterion, or "N" if it does not, and repeat the character on a new line.

Y/N
"""
]

@pytest.mark.parametrize(
    "loader_kwargs",
    [
        # {"evaluator": "cot_qa"},
        # {"evaluator": "qa"},
        # {"evaluator": "labeled_criteria", "criteria": "correctness"},
        # {"evaluator": "labeled_criteria", "criteria": "correctness", "strategy": "confidence"},
        # {"evaluator": "labeled_criteria", "criteria": "correctness", "strategy": "score"},

    ] + [
        {"evaluator": "labeled_criteria", "criteria": "correctness", "template": i}
        for i in range(len(template_variants))
    ],
)
@pytest.mark.asyncio
async def test_metaeval_correctness(loader_kwargs: dict, uid: str):
    # Should have >= 0.99 correctness
    dataset_name = "Web Q&A Dataset - Correct"
    project_name = _get_project_name(loader_kwargs, uid)
    prompt = PromptTemplate.from_template(template=template_variants[loader_kwargs.pop("template")])
    loader_kwargs["prompt"] = prompt
    scores = await _check_dataset(
        loader_kwargs, dataset_name, project_name, tags=["test_metaeval_correctness"]
    )
    score, distance_score, null_pred_score = scores["exact_score_match"], scores["absolute_distance"], scores["null_predictions"]
    assert score >= 0.99
    assert distance_score <= 0.1
    assert null_pred_score == 0
    assert all(scores["lengths"] == 100 for scores in scores.values())


@pytest.mark.parametrize(
    "loader_kwargs",
    [
        {"evaluator": "cot_qa"},
        {"evaluator": "qa"},
        {"evaluator": "labeled_criteria", "criteria": "correctness"},
    ],
)
@pytest.mark.asyncio
@pytest.mark.skip(reason="Already passes 100% so don't need to test as frequently.")
async def test_metaeval_incorrectness(loader_kwargs: dict, uid: str):
    # Expect  100% to be labeled as incorrect
    dataset_name = "Web Q&A Dataset - Incorrect"
    project_name= _get_project_name(loader_kwargs, uid)
    scores = await _check_dataset(
        loader_kwargs, dataset_name, project_name, tags=["test_metaeval_incorrectness"]
    )
    score, distance_score, null_pred_score = scores["exact_score_match"], scores["absolute_distance"], scores["null_predictions"]
    assert score >= 1
    assert distance_score <= 0.1
    assert null_pred_score == 0
    assert all(scores["lengths"] == 100 for scores in scores.values())
