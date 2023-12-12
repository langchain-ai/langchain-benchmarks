"""Factory for creating agents for the tool usage task."""
from typing import Optional

from langchain.agents import AgentExecutor
from langchain_core.runnables import Runnable, RunnableConfig

from agents.agent import create_agent
from agents.parser import ParameterizedAgentParser
from langchain_benchmarks import model_registry
from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage import apply_agent_executor_adapter


class AgentFactory:
    def __init__(self, task: ToolUsageTask, model: str) -> None:
        """Create an agent factory for the given tool usage task.

        Args:
            task: The task to create an agent factory for.
            model: model name (check model_registry)
        """
        if model not in model_registry:
            raise ValueError(f"Unknown model: {model}")
        self.task = task
        self.model = model

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
            ParameterizedAgentParser(
                wrapping_xml_tag="tool", require_closing_xml_tag=False
            ),
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
