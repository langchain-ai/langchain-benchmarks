"""Factory for creating agents for the tool usage task."""
from typing import Union

from langchain.agents.agent import (
    AgentExecutor,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
)
from langchain_core.runnables import Runnable

from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage.agents.adapters import apply_agent_executor_adapter
from langchain_benchmarks.tool_usage.agents.base import AgentFactory


class CustomRunnableAgentFactory(AgentFactory):
    """A factory for creating tool using agents.

    A factory for agents that do not leverage any special JSON mode for
    function usage; instead all function invocation behavior is implemented solely
    through prompt engineering and parsing.
    """

    def __init__(
        self,
        task: ToolUsageTask,
        agent: Union[Runnable, BaseSingleActionAgent, BaseMultiActionAgent],
    ) -> None:
        """Create an agent factory for the given tool usage task.

        Note: The agent should not be stateful, as it will be reused across
        multiple runs.

        Args:
            task: The task to create an agent factory for
            agent: The agent to use
        """
        self.task = task
        self.agent = agent

    def __call__(self) -> Runnable:
        env = self.task.create_environment()
        executor = AgentExecutor(
            agent=self.agent,
            tools=env.tools,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        return apply_agent_executor_adapter(
            executor, state_reader=env.read_state
        ).with_config({"run_name": "Agent", "metadata": {"task": self.task.name}})
