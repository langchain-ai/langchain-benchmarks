# Tests for the criteria evaluator
from typing import Tuple
from uuid import uuid4

import pytest
from langchain import hub
from langchain import chat_models, llms
from langchain.evaluation import load_evaluator
from langchain.schema import runnable
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client, EvaluationResult
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


def _check_dataset(
    loader_kwargs: dict,
    dataset_name: str,
    project_name: str,
    model_provider: str,
    model_name: str,
    tags: list,
    metadata: dict,
) -> Tuple[float, float]:
    client = Client()
    match model_provider:
        case "openai":
            llm = chat_models.ChatOpenAI(model_name=model_name, temperature=0)
        case "openai-completion":
            llm = llms.OpenAI(model_name=model_name, temperature=0)
        case "anthropic":
            llm = chat_models.ChatAnthropic(
                model_name=model_name, max_tokens=1000, temperature=0
            )
    eval_chain = load_evaluator(**loader_kwargs, llm=llm)

    def to_evaluate(input: dict, config: runnable.RunnableConfig) -> dict:
        return eval_chain.evaluate_strings(
            input=input["input"],
            prediction=input["input_prediction"],
            reference=input["input_answer"],
            **config,
        )

    res = run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=runnable.RunnableLambda(to_evaluate).with_config(
            {"metadata": metadata}
        ),
        evaluation=RunEvalConfig(
            custom_evaluators=[
                ExactScoreMatch(),
                #   AbsDistanceEvaluator(),
                NullScore(),
            ],
        ),
        concurrency_level=8,
        client=client,
        verbose=True,
        project_name=project_name,
        tags=["int-test"] + tags,
    )
    df = res.to_dataframe()
    feedback_cols = [
        col for col in df.columns if col not in ["input", "output", "reference"]
    ]
    # Return averages
    return df[feedback_cols].mean()


def _get_project_name(
    loader_kwargs: dict, uid: str, dataset_name: str, model: Tuple[str]
) -> str:
    other_args = "-".join(f"[{k}={v}]" for k, v in loader_kwargs.items())
    return f"{model[0]}.{model[1]}-{loader_kwargs['evaluator']}{other_args} - {dataset_name} - {uid}"


# prompt = "wfh/criteria_candidates"
# commits = [
#     "f470538b",
#     "c92fcf90",
# ]

# anthropic_prompt = "wfh/criteria_candidates_anthropic"

prompt_list = [
    {
        "openai": "wfh/criteria_candidates:f470538b",
        "anthropic": "wfh/criteria_candidates_anthropic:fb037730",
    },  # Here we'll try the anthropic one that inserts the "reasoning" step in the AI's mouth
    {
        "openai": "wfh/criteria_candidates:c92fcf90",
    },  # It's the same as the openai one
]


@pytest.mark.parametrize(
    "dataset_name",
    [
        "Web Q&A Dataset - Incorrect",
        "Carb-IE-Test INCORRECT",
        "Opus100 - Incorrect",
    ]
    + [
        "Web Q&A Dataset - Correct",
        "Opus100 - Correct",
        "Carb-IE-Test CORRECT",
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        ("openai", "gpt-4"),
        ("openai", "gpt-3.5-turbo"),
        ("openai-completion", "gpt-3.5-turbo-instruct"),
        ("anthropic", "claude-2"),
    ],
)
@pytest.mark.parametrize(
    "loader_kwargs",
    [
        {"evaluator": "cot_qa"},
        {"evaluator": "qa"},
    ]
    + [
        {
            "evaluator": "labeled_criteria",
            "criteria": "correctness",
            "prompt_lookup": pl,
        }
        for pl in prompt_list
    ],
)
@pytest.mark.asyncio
async def test_metaeval_correctness(
    loader_kwargs: dict, uid: str, dataset_name: str, model: Tuple[str, str]
):
    project_name = _get_project_name(loader_kwargs, uid, dataset_name, model)
    tags = ["test_metaeval_correctness", loader_kwargs["evaluator"]]
    metadata = {
        "model_provider": model[0],
        "model_name": model[1],
        "dataset_name": dataset_name,
        "evaluator": loader_kwargs["evaluator"],
        "uid": str(uid),
    }
    if "prompt_lookup" in loader_kwargs:
        prompt_lookup = loader_kwargs.get("prompt_lookup")
        prompt_repo = prompt_lookup.get(
            model[0], prompt_lookup.get("openai")
        )  # Fall back on openai prompt
        commit = prompt_repo.split(":")[-1]
        prompt = hub.pull(prompt_repo)
        loader_kwargs["prompt"] = prompt
        tags += [commit]
        print("Using prompt:", prompt_repo, commit)
        metadata["prompt"] = prompt_repo
        metadata["commit"] = commit
    scores = _check_dataset(
        loader_kwargs,
        dataset_name,
        project_name,
        model_provider=model[0],
        model_name=model[1],
        tags=tags,
        metadata=metadata,
    )
    score = scores["exact_score_match"]
    assert score >= 0.95
