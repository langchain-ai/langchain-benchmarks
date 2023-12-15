"""Wrapper around the anthropic tool user SDK.

The anthropic tool user SDK is an alpha release so this code will likely be
changed or deleted in the future. It's here simply to make it easier to benchmark
the performance of the existing tool user SDK, to compare it with the performance
of other implementations.
"""

from importlib.util import find_spec
from typing import Any, List, Optional
from typing import Sequence

from langchain.tools import StructuredTool
from langchain_core.callbacks.manager import trace_as_chain_group
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from langchain_benchmarks import rate_limiting
from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage import apply_agent_executor_adapter


def convert_langchain_tool_to_tool_user_tool(lc_tool: StructuredTool) -> Any:
    """Convert a langchain tool to a tool user tool."""
    from tool_use_package.tools.base_tool import BaseTool

    class DynamicTool(BaseTool):
        def use_tool(self, **kwargs):
            return lc_tool(kwargs)

    schema = lc_tool.args_schema.schema()

    properties = schema["properties"]
    parameters = []
    # Is this needed or is string OK?
    type_adapter = {
        "string": "str",  # str or string?
        "integer": "int",
        "number": "float",
        "boolean": "bool",
    }
    for key, value in properties.items():
        parameters.append(
            {
                "name": key,
                "type": type_adapter.get(value["type"], value["type"]),
                "description": value.get("description", ""),
            }
        )

    return DynamicTool(lc_tool.name, lc_tool.description, parameters)


def _handle_tool_inputs(
    tool_inputs: list[dict],
    tools: Sequence[StructuredTool],
    config: Optional[RunnableConfig] = None,
) -> dict[str, Any]:
    """Handle tool inputs."""
    tool_by_name = {tool.name: tool for tool in tools}
    tool_error: Optional[str] = None
    tool_outputs = []
    for tool_input in tool_inputs:
        tool_name = tool_input["tool_name"]
        tool_arguments = tool_input["tool_arguments"]
        tool = tool_by_name[tool_name]
        try:
            tool_result = tool.invoke(tool_arguments, config=config)
        except Exception as e:  # Break on first error
            tool_error = str(e)
            tool_outputs = None
            break
        tool_outputs.append(
            {
                "tool_name": tool_name,
                "tool_result": tool_result,
            }
        )
    return {
        "role": "tool_outputs",
        "tool_outputs": tool_outputs,
        "tool_error": tool_error,
    }


def run_anthropic_agent_simple(
    tools: Sequence[StructuredTool],
    user_message: str,
    *,
    max_iterations: int = 30,
    config: Optional[RunnableConfig] = None,
    **kwargs,
) -> List[dict]:
    """Make an anthropic agent."""
    from tool_use_package.tool_user import ToolUser

    verbose = kwargs.pop("verbose", False)

    tool_user = ToolUser(
        [convert_langchain_tool_to_tool_user_tool(tool) for tool in tools], **kwargs
    )
    messages = [
        {
            "role": "human",
            "content": user_message,
            "tool_error": None,
            "tool_outputs": [],
            "tool_inputs": [],
        }
    ]
    with trace_as_chain_group(
        "Anthropic Agent Run",
        inputs={"user_message": user_message},
        callback_manager=config["callbacks"],
    ) as group_manager:
        for num_iteration in range(max_iterations):
            with trace_as_chain_group(
                f"Anthropic Agent Iteration {num_iteration}",
                inputs={"messages": messages},
                callback_manager=group_manager.parent_run_manager.get_child(),
            ) as iteration_manager:
                last_message = tool_user.use_tools(
                    messages, execution_mode="manual", verbose=verbose
                )
                new_messages = [last_message]

                if last_message["role"] == "tool_inputs":
                    tool_inputs = last_message["tool_inputs"]
                    new_message = _handle_tool_inputs(
                        tool_inputs,
                        tools,
                        config={
                            "callbacks": iteration_manager.parent_run_manager.get_child(),
                        },
                    )
                    new_messages.append(new_message)

                iteration_manager.on_chain_end(outputs=new_messages)
                messages.extend(new_messages)

                # Finally break if the last message is from the assistant
                if last_message["role"] == "assistant":
                    break
        else:
            raise ValueError("Max iterations reached")
        group_manager.on_chain_end(outputs=messages)
    return messages


def convert_messages_to_finalized_output(messages: list[dict]) -> dict:
    """Convert the history of messages into the expected output for eval.

    This matches the agent executor output which has the following structure:

    {
        "output": "The output of the agent",
        "intermediate_steps": [
            (
                AgentAction(
                    tool="add_x_y",
                    tool_input={"x": 2.0, "y": 5.0},
                    log="Invoking tool `add_x_y` with `{'x': 2.0, 'y': 5.0}`",
                ),
                9.0,
            )
        ],
        "state": Any, # Optional key for tasks that involve manipulation of an env.
    }
    """
    if not messages:
        raise ValueError("Expected at least one message")

    last_message = messages[-1]

    if last_message["role"] != "assistant":
        raise ValueError(
            f"Expected the last message to be from the assistant. "
            f"Instead got {last_message}."
        )

    actual_steps = []

    for message in messages:
        if "role" not in message:
            raise ValueError(f"Expected role in message {message}")
        role = message["role"]

        if role == "tool_inputs":
            # Get the name of the tool used
            for tool_input in message["tool_inputs"]:
                actual_steps.append(tool_input["tool_name"])

    return {
        "output": last_message["content"],
        "actual_steps": actual_steps,
    }


def create_agent(tools: Sequence[StructuredTool]) -> RunnableLambda:
    """Create an agent."""

    def run_agent(
        input: dict, config: Optional[RunnableConfig] = None, **kwargs
    ) -> dict:
        """Run the agent."""
        messages = run_anthropic_agent_simple(
            tools, input["input"], config=config, **kwargs
        )
        return convert_messages_to_finalized_output(messages)

    return RunnableLambda(run_agent)


class AnthropicAgentFactory:
    def __init__(
        self,
        task: ToolUsageTask,
        *,
        rate_limiter: Optional[rate_limiting.RateLimiter] = None,
    ) -> None:
        """Create an OpenAI agent factory for the given task.

        Args:
            task: The task to create an agent factory for.
        """
        self.task = task
        self.rate_limiter = rate_limiter
        if not find_spec("tool_use_package"):
            raise ImportError(
                f'Could not import "tool_use_package". Please '
                f"follow instructions here to install "
                f"https://github.com/anthropics/anthropic-tools/tree/main"
            )

    def __call__(self) -> Runnable:
        env = self.task.create_environment()

        def _add_task_instructions(
            input: dict, config: Optional[RunnableConfig] = None, **kwargs
        ) -> dict:
            """Add task instructions to the question."""
            if not isinstance(input, dict) or "question" not in input:
                raise ValueError(
                    f"Expected input to be a dict with key `question`. "
                    f"Found {type(input)}."
                )

            input = input.copy()
            input["question"] = (
                f"{self.task.instructions}\nWrite down your answer, "
                f"but do not explain it. Input: `{input['question']}`"
            )
            return input

        agent = create_agent(env.tools)  # type: ignore
        # Returns `state` in the output if the environment has a state reader
        # makes sure that `output` is always in the output

        runnable = _add_task_instructions | apply_agent_executor_adapter(
            agent, state_reader=env.read_state
        )

        if self.rate_limiter:  # Add a rate limiter
            runnable = rate_limiting.with_rate_limit(runnable, self.rate_limiter)

        return runnable
