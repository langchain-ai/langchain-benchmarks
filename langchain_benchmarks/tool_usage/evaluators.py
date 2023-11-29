"""Module contains standard evaluators for agents.

Requirements:

* Agents must output "intermediate_steps" in their run outputs.
* The dataset must have "expected_steps" in its outputs.
"""
from typing import Optional, Union, Literal

from langchain.callbacks.manager import collect_runs
from langchain.chat_models.base import BaseChatModel
from langchain.evaluation.schema import StringEvaluator
from langchain.llms.base import BaseLanguageModel
from langchain.smith import RunEvalConfig
from langsmith.evaluation.evaluator import (
    EvaluationResult,
    EvaluationResults,
    RunEvaluator,
)
from langsmith.schemas import Example, Run


def compare_outputs(
    run_outputs: dict,
    example_outputs: dict,
    run_inputs: dict,
    *,
    qa_evaluator: Optional[StringEvaluator] = None,
) -> EvaluationResults:
    """Compare the outputs of a run to the expected outputs."""
    intermediate_steps = run_outputs["intermediate_steps"]
    # Since we are comparing to the tool names, we now need to get that
    # Intermediate steps is a Tuple[AgentAction, Any]
    # The first element is the action taken
    # The second element is the observation from taking that action
    trajectory = [action.tool for action, _ in intermediate_steps]
    expected_trajectory = example_outputs["expected_steps"]

    order_matters = example_outputs.get("order_matters", True)

    if order_matters:
        # If the order matters trajectory must be the same as expected trajectory
        trajectory_score = int(trajectory == expected_trajectory)
    else:
        # If order does not matter, then we compare the trajectories after sorting
        # them. This will make sure that the number of times each tool is used
        # is the same, but the order does not matter.
        trajectory_score = int(sorted(trajectory) == sorted(expected_trajectory))

    # Just score it based on whether it is correct or not
    step_fraction = len(trajectory) / len(expected_trajectory)

    # Add trajectory results
    results = [
        EvaluationResult(
            key="Intermediate steps correctness",
            score=trajectory_score,
            comment=f"Order matters={order_matters}",
        ),
        EvaluationResult(
            key="# steps / # expected steps",
            score=step_fraction,
        ),
    ]

    # Evaluate state score
    # This will need to be evolved it's too simple.
    if "state" in run_outputs:
        state = run_outputs["state"]
        example_state = example_outputs["state"]
        results.append(
            EvaluationResult(
                key="Correct Final State",
                score=int(state == example_state),
            )
        )

    if "output" in run_outputs and qa_evaluator:
        output = run_outputs["output"]
        with collect_runs() as cb:
            qa_results = qa_evaluator.evaluate_strings(
                prediction=output,
                reference=example_outputs["reference"],
                input=run_inputs["question"],
            )
        results.append(
            EvaluationResult(
                key="correctness",
                score=qa_results["score"],
                source_run_id=cb.traced_runs[0].id,
            )
        )

    return {"results": results}


class AgentTrajectoryEvaluator(RunEvaluator):
    """An evaluator that can be used in conjunction with a standard agent interface."""

    def __init__(
        self,
        eval_llm: Union[BaseLanguageModel, BaseChatModel, None] = None,
        output_evaluation: Literal["qa", "none"] = "qa",
    ) -> None:
        """Initialize the evaluator."""
        self.eval_llm = eval_llm
        self.output_evaluation = output_evaluation

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResults:
        # The run is the run from the agent
        if run.outputs is None:
            raise ValueError("Run outputs cannot be None")

        # The example is the example from the dataset
        if example is None:
            raise ValueError("Example cannot be None")

        if "intermediate_steps" not in run.outputs:
            raise ValueError(
                "Please make sure that your agent outputs 'intermediate_steps'"
            )

        if "expected_steps" not in example.outputs:
            raise ValueError(
                "Please make sure that your dataset contains 'expected_steps'"
            )

        return compare_outputs(
            run.outputs,
            example.outputs,
            run_inputs=run.inputs,
        )


def get_eval_config(
    *,
    eval_llm: Union[BaseLanguageModel, BaseChatModel, None] = None,
    output_evaluation: Literal["qa", "none"] = "qa",
) -> RunEvalConfig:
    """Get the default evaluator for the environment.

    Args:
        eval_llm: The language model to use for grading the `output` response
        output_evaluation: how to evaluate the output of the agent.
            - 'qa' will use the qa evaluator to compare the output to the reference.
            - 'none' will not evaluate the output of the agent -- in some cases
              it's only relevant to evaluate how the agent used tools, not what
              its output.

    Returns:
        A RunEvalConfig that can be used to evaluate the environment
    """
    return RunEvalConfig(
        custom_evaluators=[
            AgentTrajectoryEvaluator(
                eval_llm=eval_llm, output_evaluation=output_evaluation
            )
        ]
    )
