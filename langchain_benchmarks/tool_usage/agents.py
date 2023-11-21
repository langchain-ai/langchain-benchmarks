"""Code for creating an agent factory for evaluating tool usage tasks."""
from typing import Any

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.tools.render import format_tool_to_openai_function

from langchain_benchmarks.schema import ToolUsageTask


def _ensure_output_exists(inputs: dict) -> dict:
    """Make sure that the output key is always present."""
    if "output" not in inputs:
        return {"output": "", **inputs}
    return inputs


class OpenAIAgentFactory:
    def __init__(
        self, task: ToolUsageTask, *, model: str = "gpt-3.5-turbo-16k"
    ) -> None:
        """Create an OpenAI agent factory for the given task.

        Args:
            task: The task to create an agent factory for.
            model: The model to use -- this must be an open AI model.
        """
        self.task = task
        self.model = model

    def create(self) -> Runnable:
        """Agent Executor"""
        llm = ChatOpenAI(
            model=self.model,
            temperature=0,
        )

        env = self.task.create_environment()

        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in env.tools]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.task.instructions,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        runnable_agent = (
            {
                "input": lambda x: x["question"],
                "agent_scratchpad": lambda x: format_to_openai_functions(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )

        def _read_state(*args: Any, **kwargs: Any) -> Any:
            """Read the state of the environment."""
            if env.read_state is not None:
                return env.read_state()
            else:
                return None

        runnable = (
            AgentExecutor(
                agent=runnable_agent,
                tools=env.tools,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
            )
            | _ensure_output_exists
        )

        if env.read_state is not None:
            # If the environment has a state reader, add it to the runnable
            runnable = runnable | RunnablePassthrough.assign(state=_read_state)

        return runnable

    def __call__(self) -> Runnable:
        return self.create()
