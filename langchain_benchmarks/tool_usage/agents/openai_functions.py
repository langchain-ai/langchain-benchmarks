"""Code for creating an agent factory for evaluating tool usage tasks."""
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable
from langchain.tools.render import format_tool_to_openai_tool
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel

from langchain_benchmarks import rate_limiting
from langchain_benchmarks.model_registration import RegisteredModel
from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage.agents.adapters import apply_agent_executor_adapter

# PUBLIC API


def _bind_tools(
    llm: BaseChatModel,
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
    tool_choice: Optional[str] = None,
    json_mode: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, BaseMessage]:
    """Bind tools (and other objects) to this chat model.

    Args:
        tools: A list of tool definitions to bind to this chat model.
            Can be  a dictionary, pydantic model, or callable. Pydantic
            models and callables will be automatically converted to
            their schema dictionary representation.
        tool_choice: Which tool to require the model to call.
            Must be the name of the single provided tool or
            "auto" to automatically determine which tool to call
            (if any).
        json_mode: Whether to set JSON mode for the tool call.
            This guarantees the model will respond in valid JSON
            (unless truncated).
        kwargs: Any additional parameters to pass to the
            :class:`~langchain.runnable.Runnable` constructor.

    """
    formatted_tools: List[Dict[str, Union[str, dict]]] = [
        format_tool_to_openai_tool(tool) for tool in tools
    ]
    if tool_choice is not None:
        if not formatted_tools:
            raise ValueError(
                "When specifying `tool_choice`, you must provide at least one " "tool."
            )
        tool_names = [tool["function"]["name"] for tool in formatted_tools]
        if not any(tool_name == tool_choice for tool_name in tool_names):
            raise ValueError(
                f"Tool choice {tool_choice} was specified, but the only "
                f"provided tools were {tool_names}."
            )
        tool_choice_ = {"type": "function", "function": {"name": tool_choice}}
        kwargs = {**kwargs, "tool_choice": tool_choice_}
    if json_mode:
        kwargs = {**kwargs, "response_format": {"type": "json_object"}}
    return llm.bind(
        tools=formatted_tools,
        **kwargs,
    )


class OpenAIAgentFactory:
    def __init__(
        self,
        task: ToolUsageTask,
        *,
        model: Union[str, RegisteredModel] = "gpt-3.5-turbo-16k",
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

    def _create_model(self) -> Union[BaseChatModel, BaseLanguageModel]:
        if isinstance(self.model, RegisteredModel):
            return self.model.get_model(
                model_params={"temperature": 0, "model_kwargs": {"seed": 0}}
            )
        else:
            return ChatOpenAI(model=self.model, temperature=0, model_kwargs={"seed": 0})

    def create(self) -> Runnable:
        """Agent Executor"""
        # For backwards compatibility
        return self()

    def __call__(self) -> Runnable:
        model = self._create_model()

        env = self.task.create_environment()

        model = _bind_tools(model, env.tools)

        if self.rate_limiter is not None:
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
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | model
            | OpenAIToolsAgentOutputParser()
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
