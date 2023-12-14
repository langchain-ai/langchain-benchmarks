"""Code for creating an agent factory for evaluating tool usage tasks."""
from typing import Optional

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable
from langchain.tools.render import format_tool_to_openai_function

from langchain_benchmarks import rate_limiting
from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage.agents.adapters import apply_agent_executor_adapter

# PUBLIC API


class OpenAIAgentFactory:
    def __init__(
        self,
        task: ToolUsageTask,
        *,
        model: str = "gpt-3.5-turbo-16k",
        rate_limiter: Optional[rate_limiting.RateLimiter] = None,
    ) -> None:
        """Create an OpenAI agent factory for the given task.

        Args:
            task: The task to create an agent factory for.
            model: The model to use -- this must be an open AI model.
            rate_limiter: The rate limiter to use
        """
        self.task = task
        self.model = model
        self.rate_limiter = rate_limiter

    def create(self) -> Runnable:
        """Agent Executor"""
        # For backwards compatibility
        return self()

    def __call__(self) -> Runnable:
        model = ChatOpenAI(
            model=self.model,
            temperature=0,
        )

        env = self.task.create_environment()

        model = model.bind(
            functions=[format_tool_to_openai_function(t) for t in env.tools]
        )

        if rate_limiting:
            # Rate limited model
            model = rate_limiting.with_rate_limit(model, self.rate_limiter)

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
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_functions(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | model
            | OpenAIFunctionsAgentOutputParser()
        )

        runnable = AgentExecutor(
            agent=runnable_agent,
            tools=env.tools,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        # Returns `state` in the output if the environment has a state reader
        # makes sure that `output` is always in the output
        return apply_agent_executor_adapter(runnable, state_reader=env.read_state)
