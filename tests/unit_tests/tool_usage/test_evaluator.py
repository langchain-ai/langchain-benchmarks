"""Test the standard agent evaluator."""

import pytest
from langchain.schema import AgentAction

from langchain_benchmarks.tool_usage.evaluators import compare_outputs


@pytest.mark.parametrize(
    "run_outputs, example_outputs, expected_results",
    [
        (
            {
                "intermediate_steps": [
                    (
                        AgentAction(tool="action_1", tool_input={}, log=""),
                        "observation1",
                    ),
                    (
                        AgentAction(tool="action_2", tool_input={}, log=""),
                        "observation1",
                    ),
                ],
                "state": "final_state",
            },
            {
                "expected_steps": ["action_1", "action_2"],
                "state": "final_state",
            },
            {
                "Intermediate steps correctness": True,
                "# steps / # expected steps": 1,
                "Correct Final State": 1,
            },
        ),
        (
            {
                "intermediate_steps": [
                    (
                        AgentAction(tool="action_1", tool_input={}, log=""),
                        "observation1",
                    ),
                    (
                        AgentAction(tool="action_2", tool_input={}, log=""),
                        "observation1",
                    ),
                ],
                "state": "final_state",
            },
            {
                "expected_steps": ["cat", "was", "here"],
                "state": "another_state",
            },
            {
                "Intermediate steps correctness": False,
                "# steps / # expected steps": 2 / 3,
                "Correct Final State": 0,
            },
        ),
        (
            {
                "intermediate_steps": [
                    (
                        AgentAction(tool="action_2", tool_input={}, log=""),
                        "observation1",
                    ),
                    (
                        AgentAction(tool="action_1", tool_input={}, log=""),
                        "observation1",
                    ),
                ],
                "state": "final_state",
            },
            {
                "expected_steps": ["action_1", "action_2"],
                "order_matters": False,
                "state": "different_state",
            },
            {
                "Intermediate steps correctness": True,
                "# steps / # expected steps": 1.0,
                "Correct Final State": 0,
            },
        ),
        # Without state
        (
            {
                "intermediate_steps": [
                    (
                        AgentAction(tool="action_2", tool_input={}, log=""),
                        "observation1",
                    ),
                    (
                        AgentAction(tool="action_1", tool_input={}, log=""),
                        "observation1",
                    ),
                ],
            },
            {
                "expected_steps": ["action_1", "action_2"],
                "order_matters": False,
            },
            {
                "Intermediate steps correctness": True,
                "# steps / # expected steps": 1.0,
            },
        ),
    ],
)
def test_compare_outputs(run_outputs, example_outputs, expected_results):
    """Test compare outputs."""
    evaluation_results = compare_outputs(run_outputs, example_outputs)
    assert {
        result.key: result.score for result in evaluation_results["results"]
    } == expected_results
