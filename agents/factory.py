from typing import List, Optional

from langchain.agents import AgentExecutor
from langchain.chat_models import ChatAnthropic, ChatFireworks
from langchain_core.runnables import Runnable, RunnableConfig

from agents.agent import create_agent
from agents.parser import ParameterizedAgentParser
from langchain_benchmarks.model_registration import FIREWORK_NAME_TO_MODEL
from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage import apply_agent_executor_adapter


class CustomAgentFactory:
    def __init__(self, task: ToolUsageTask, model: str) -> None:
        """Create an OpenAI agent factory for the given task.

        Args:
            task: The task to create an agent factory for.
        """
        if model not in self.list_models():
            raise ValueError(f"Unknown model: {model}")
        self.task = task
        self.model = model

    @staticmethod
    def list_models() -> List[str]:
        """List all models."""
        return sorted(
            [
                "claude-2.1",
                "claude-2",
                *FIREWORK_NAME_TO_MODEL.keys(),
            ]
        )

    def __call__(self) -> Runnable:
        env = self.task.create_environment()
        if self.model in {"claude-2.1", "claude-2"}:
            model = ChatAnthropic(model=self.model, temperature=0)
        elif self.model in FIREWORK_NAME_TO_MODEL:
            model = ChatFireworks(
                model=FIREWORK_NAME_TO_MODEL[self.model], temperature=0
            )
        else:
            raise ValueError(f"Unknown model: {self.model}")

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
