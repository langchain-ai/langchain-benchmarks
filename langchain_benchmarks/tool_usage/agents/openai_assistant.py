"""Code for creating an assistant factory for evaluating tool usage tasks.

See: https://platform.openai.com/docs/assistants/how-it-works/creating-assistants
"""
from typing import Optional

from langchain.agents import AgentExecutor
from langchain.agents.openai_assistant.base import OpenAIAssistantRunnable
from langchain.schema.runnable import Runnable

from langchain_benchmarks import rate_limiting
from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage.agents.adapters import apply_agent_executor_adapter


class OpenAIAssistantFactory:
    def __init__(
        self,
        task: ToolUsageTask,
        *,
        model: str,
        rate_limiter: Optional[rate_limiting.RateLimiter] = None,
        num_retries: int = 0,
    ) -> None:
        """Create an OpenAI agent factory for the given task.

        Args:
            task: The task to create an agent factory for.
            model: The model to use -- this must be an open AI model.
            rate_limiter: The rate limiter to use
            num_retries: The number of times to retry the assistant if it fails
        """
        if not isinstance(model, str):
            raise ValueError(f"Expected str for model, got {type(model)}")
        self.task = task
        tools = task.create_environment().tools
        # Stateless, so we only need to create it once
        self.agent = OpenAIAssistantRunnable.create_assistant(
            name=f"{task.name} assistant",
            instructions=self.task.instructions,
            tools=tools,
            model=model,
            as_agent=True,
        )
        self.rate_limiter = rate_limiter
        self.num_retries = num_retries

    def __call__(self) -> Runnable:
        env = self.task.create_environment()

        agent = self.agent
        if self.rate_limiter is not None:
            # Rate limited model
            agent = rate_limiting.with_rate_limit(agent, self.rate_limiter)

        def _map_key(x: dict):
            # Assistant expects the 'content' key explicitly
            return {
                "content": x["input"],
                **{k: v for k, v in x.items() if k != "input"},
            }

        agent = _map_key | self.agent
        if self.num_retries > 0:
            agent = agent.with_retry(
                stop_after_attempt=self.num_retries + 1,
            )
        runnable = AgentExecutor(
            agent=agent,
            tools=env.tools,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        # Returns `state` in the output if the environment has a state reader
        # makes sure that `output` is always in the output
        return apply_agent_executor_adapter(runnable, state_reader=env.read_state)
