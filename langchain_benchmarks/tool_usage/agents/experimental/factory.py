"""Factory for creating agents for the tool usage task."""
from typing import Optional

from langchain.agents import AgentExecutor
from langchain_core.runnables import Runnable, RunnableConfig

from langchain_benchmarks import RateLimiter, model_registry
from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage.agents.adapters import apply_agent_executor_adapter
from langchain_benchmarks.tool_usage.agents.experimental.agent import create_agent
from langchain_benchmarks.tool_usage.agents.experimental.parser import (
    GenericAgentParser,
)


class CustomAgentFactory:
    """A factory for creating tool using agents.

    A factory for agents that do not leverage any special JSON mode for
    function usage; instead all function invocation behavior is implemented solely
    through prompt engineering and parsing.
    """

    def __init__(
        self,
        task: ToolUsageTask,
        model: str,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        num_retries: int = 0,
    ) -> None:
        """Create an agent factory for the given tool usage task.

        Args:
            task: The task to create an agent factory for
            model: model name (check model_registry)
            rate_limiter: The rate limiter to use if provided
            num_retries: The number of times to retry the agent if it fails
        """
        if model not in model_registry:
            raise ValueError(f"Unknown model: {model}")
        self.task = task
        self.model = model
        self.rate_limiter = rate_limiter
        self.num_retries = num_retries

    def __call__(self) -> Runnable:
        if isinstance(self.model, str):
            registered_model = model_registry.get_model(self.model)
            if registered_model is None:
                raise ValueError(f"Unknown model: {self.model}")
            model = registered_model.get_model(model_params={"temperature": 0})
        else:
            model = self.model

        def _add_task_instructions(
            input: dict, config: Optional[RunnableConfig] = None, **kwargs
        ) -> dict:
            """Add task instructions to the question."""
            if not isinstance(input, dict):
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

        env = self.task.create_environment()

        agent = create_agent(
            model,
            env.tools,
            GenericAgentParser(wrapping_xml_tag="tool", require_closing_xml_tag=False),
            rate_limiter=self.rate_limiter,
        )
        if self.num_retries > 0:
            agent = agent.with_retry(
                stop_after_attempt=self.num_retries + 1,
            )
        executor = AgentExecutor(
            agent=agent,
            tools=env.tools,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        return _add_task_instructions | apply_agent_executor_adapter(
            executor, state_reader=env.read_state
        )
