"""Module contains standard evaluators for agents.

Requirements:

* Agents must output "intermediate_steps" in their run outputs.
* The dataset must have "expected_steps" in its outputs.
"""
from typing import Optional

from langchain.evaluation import EvaluatorType
from langchain.smith import RunEvalConfig
from langsmith.evaluation.evaluator import (
    EvaluationResults,
    RunEvaluator,
    EvaluationResult,
)
from langsmith.schemas import Example, Run


class AgentTrajectoryEvaluator(RunEvaluator):
    """An evaluator that can be used in conjunction with a standard agent interface."""

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResults:
        if run.outputs is None:
            raise ValueError("Run outputs cannot be None")
        # This is the output of each run
        intermediate_steps = run.outputs["intermediate_steps"]
        # Since we are comparing to the tool names, we now need to get that
        # Intermediate steps is a Tuple[AgentAction, Any]
        # The first element is the action taken
        # The second element is the observation from taking that action
        trajectory = [action.tool for action, _ in intermediate_steps]
        # This is what we uploaded to the dataset
        expected_trajectory = example.outputs["expected_steps"]

        # Just score it based on whether it is correct or not
        score = int(trajectory == expected_trajectory)

        evaluation_results = [
            EvaluationResult(
                key="Intermediate steps correctness",
                score=score,
            ),
            EvaluationResult(
                key="# steps / # expected steps",
                value=len(trajectory) / len(expected_trajectory),
            ),
        ]

        return {
            "results": {
                evaluation_result.key: evaluation_result
                for evaluation_result in evaluation_results
            }
        }


STANDARD_AGENT_EVALUATOR = RunEvalConfig(
    # Evaluators can either be an evaluator type
    # (e.g., "qa", "criteria", "embedding_distance", etc.) or a
    # configuration for that evaluator
    evaluators=[
        # Measures whether a QA response is "Correct", based on a reference answer
        # You can also select via the raw string "qa"
        EvaluatorType.QA
    ],
    # You can add custom StringEvaluator or RunEvaluator objects
    # here as well, which will automatically be
    # applied to each prediction. Check out the docs for examples.
    custom_evaluators=[AgentTrajectoryEvaluator()],
    # We now need to specify this because we have multiple outputs in our dataset
    reference_key="reference",
)
