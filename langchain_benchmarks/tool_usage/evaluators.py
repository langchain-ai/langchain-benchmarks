"""Module contains standard evaluators for agents.

Requirements:

* Agents must output "intermediate_steps" in their run outputs.
* The dataset must have "expected_steps" in its outputs.
"""
from typing import List, Optional, Union

from langchain.evaluation import EvaluatorType
from langchain.smith import RunEvalConfig
from langchain.smith.evaluation.config import EvalConfig
from langsmith.evaluation.evaluator import (
    EvaluationResult,
    EvaluationResults,
    RunEvaluator,
)
from langsmith.schemas import Example, Run

from langchain_benchmarks.schema import ExtractionTask


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
        if example is None:
            raise ValueError("Example cannot be None")
        expected_trajectory = example.outputs["expected_steps"]

        # Just score it based on whether it is correct or not
        score = int(trajectory == expected_trajectory)
        step_fraction = len(trajectory) / len(expected_trajectory)

        results = [
            EvaluationResult(
                key="Intermediate steps correctness",
                score=score,
            ),
            EvaluationResult(
                key="# steps / # expected steps",
                score=step_fraction,
            ),
        ]

        if "state" in run.outputs:
            state = run.outputs["state"]
            example_state = example.outputs["state"]
            results.append(
                EvaluationResult(
                    key="Correct Final State",
                    score=int(state == example_state),
                )
            )

        return {"results": results}


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
    prediction_key="output",
)
