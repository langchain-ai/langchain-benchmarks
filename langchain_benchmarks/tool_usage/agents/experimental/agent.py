from typing import List, Literal, Optional, Sequence, Tuple, Union

from langchain.agents import AgentOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.messages import HumanMessage
from langchain.schema.runnable import Runnable
from langchain.tools import StructuredTool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import MessagesPlaceholder
from typing_extensions import NotRequired, TypedDict

from langchain_benchmarks import RateLimiter
from langchain_benchmarks.rate_limiting import with_rate_limit
from langchain_benchmarks.tool_usage.agents.experimental.encoder import (
    AstPrinter,
    TypeScriptEncoder,
    XMLEncoder,
)
from langchain_benchmarks.tool_usage.agents.experimental.tool_utils import (
    convert_tool_to_function_definition,
)
from langchain_benchmarks.tool_usage.agents.experimental.prompts import (
    _AGENT_INSTRUCTIONS_BLOB_STYLE,
)


def format_observation(tool_name: str, observation: str) -> BaseMessage:
    """Format the observation."""
    result = (
        "<tool_output>\n"
        f"<tool_name>{tool_name}</tool_name>\n"
        f"<output>{observation}</output>\n"
        "</tool_output>"
    )

    return HumanMessage(content=result)


def format_steps_for_chat(
    intermediate_steps: List[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    """Format the steps."""
    messages = []
    for action, observation in intermediate_steps:
        if not isinstance(action, AgentAction):
            if action.tool != "_Exception":
                raise AssertionError(f"Unexpected step: {action}. type: {type(action)}")

            messages.append(HumanMessage(content=observation))
        messages.extend(action.messages)
        messages.append(format_observation(action.tool, observation))
    return messages


# PUBLIC API


class AgentInput(TypedDict):
    """The input to the agent."""

    input: str
    """The input to the agent."""
    intermediate_steps: List[Tuple[AgentAction, str]]
    """The intermediate steps taken by the agent."""
    examples: NotRequired[List[BaseMessage]]
    """A list of messages that can be used to form example traces."""


def create_agent(
    model: Union[BaseChatModel, BaseLanguageModel],
    tools: Sequence[StructuredTool],
    parser: AgentOutputParser,
    *,
    ast_printer: Union[AstPrinter, Literal["xml"]] = "xml",
    rate_limiter: Optional[RateLimiter] = None,
) -> Runnable[AgentInput, Union[AgentAction, AgentFinish]]:
    """Create an agent for a chat model."""
    if isinstance(ast_printer, str):
        if ast_printer == "xml":
            ast_printer = XMLEncoder()
        elif ast_printer == "typescript":
            ast_printer = TypeScriptEncoder()
        else:
            raise ValueError(f"Unknown ast printer: {ast_printer}")
    elif isinstance(ast_printer, AstPrinter):
        pass
    else:
        raise TypeError(
            f"Expected AstPrinter or str, got {type(ast_printer)} for `ast_printer`"
        )

    function_definitions = [convert_tool_to_function_definition(tool) for tool in tools]
    tool_description = ast_printer.visit_function_definitions(function_definitions)

    template = ChatPromptTemplate.from_messages(
        [
            ("system", _AGENT_INSTRUCTIONS_BLOB_STYLE),
            MessagesPlaceholder("examples"),  # Can use to add example traces
            ("human", "{input}"),
            MessagesPlaceholder("history"),
        ]
    ).partial(tool_description=tool_description)

    # For the time being, hard-coding the fact that we're using a <tool> tag.
    model = model.bind(stop=["</tool>"])

    if rate_limiter:
        model = with_rate_limit(model, rate_limiter)

    agent = (
        {
            "input": lambda x: x["input"],
            "history": lambda x: format_steps_for_chat(x["intermediate_steps"]),
            "examples": lambda x: x.get("examples", []),
        }
        | template
        | model
        | parser
    )
    return agent
