"""
Module contains re-implementation of the anthropic tool agent SDK using
langchain primitives.
"""
import ast
import re
from typing import Dict, Optional, Union
from typing import List, Sequence, Tuple

from langchain.agents import AgentOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.runnable import Runnable
from langchain.tools import StructuredTool
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from typing_extensions import NotRequired, TypedDict

from langchain_benchmarks import RateLimiter
from langchain_benchmarks.rate_limiting import with_rate_limit
from langchain_benchmarks.tool_usage.agents.experimental.encoder import (
    AstPrinter,
    FunctionResult,
    AnthropicXMLEncoder,
)
from langchain_benchmarks.tool_usage.agents.experimental.prompts import (
    _ANTHROPIC_TOOL_USER_PROMPT,
)
from langchain_benchmarks.tool_usage.agents.experimental.tool_utils import (
    convert_tool_to_function_definition,
)


class _ToolInvocationRequest(BaseModel):
    """Light-weight pydantic model for validating the raw tool invocation request.

    The purpose of this model, is to make sure that whatever as parsed from
    the raw llm output has `tool_name` and potential `arguments` fields, and
    nothing else.
    """

    tool_name: str
    # OK parameterless tools which do not take arguments
    arguments: Optional[Dict] = Field(default_factory=dict)


class AnthropicToolParser(AgentOutputParser):
    """A generalized parser that makes it easier to parameterize different parsing."""

    def parse(self, text: str) -> Union[AgentFinish, AgentAction]:
        """Parse the output of the agent."""
        wrapping_xml_tag = "function_calls"
        open_tag = f"<{wrapping_xml_tag}>"
        close_tag = f"</{wrapping_xml_tag}>"
        if open_tag in text:
            # This is a hack to make sure that </tool> is always present
            # in the output if <tool>. </tool> may be a stop sequence for the
            # language model, so depending on implementation
            # the stop sequence may be cut off.
            # There might be a better way to do this, but this works and
            # is simple.
            if not self.require_closing_xml_tag:
                text += close_tag

        pattern = rf"{open_tag}(?P<invocation>.*?){close_tag}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group("invocation").strip()
            return parse_invocation(content, self.wrapping_xml_tag)

        return AgentFinish(
            log=text,
            return_values={
                "output": text,
            },
        )


def parse_invocation(text: str, tag: str) -> AgentAction:
    """Parse the content of the function invocation.

    Args:
        text: The text to parse.
        tag: The tag that wraps the function invocation request.

    Returns:
        An AgentAction that corresponds to the function invocation.

    Raises:
        OutputParserException: If the parsing fails.

        This exception is meant to be caught by the agent executor and
        handled appropriately to provide feedback to the LLM.
    """
    ai_content = f"<{tag}>{text}</{tag}>\n"

    try:
        result = ast.literal_eval(text)
    except BaseException as e:
        # Convert this to something controllable by the user.
        err_msg = (
            f"ERROR: Please use the format "
            f'<{tag}>{{"tool_name": $TOOL_NAME, "arguments": $ARGUMENTS}}</{tag}>\n'
        )

        raise OutputParserException(
            error=e,
            llm_output=ai_content,
            observation=err_msg,
            send_to_llm=True,
        )

    try:
        request = _ToolInvocationRequest.validate(result)
    except Exception as e:  # Using broad exception since it's not just ValidationError
        # Can also raise DictError if result is not a dict.
        err_msg = (
            f"ERROR: Please use the format "
            f'<{tag}>{{"tool_name": $TOOL_NAME, "arguments": $ARGUMENTS}}</{tag}>\n'
        )
        raise OutputParserException(
            error=e,
            llm_output=ai_content,
            send_to_llm=True,
            observation=err_msg,
        )

    return AgentActionMessageLog(
        message_log=[AIMessage(content=ai_content)],
        tool=request.tool_name,
        tool_input=request.arguments,
        log=f"\nInvoking {request.tool_name}: {request.arguments}\n\t",
    )


def format_steps_for_chat(
    intermediate_steps: List[Tuple[AgentAction, str]],
    ast_printer: AstPrinter,
) -> List[BaseMessage]:
    """Format the steps."""
    messages = []
    for action, observation in intermediate_steps:
        # Action messages contains the tool invocation request from the LLM
        # Now add the result of the tool invocation.

        if action.tool == "_Exception":
            messages.append(
                AIMessage(
                    content=action.log,
                )
            )
            messages.append(
                # Tool input is the error message for the exception
                HumanMessage(content=action.tool_input)
            )
        else:
            messages.extend(action.messages)
            function_result: FunctionResult = {
                "name": action.tool,
                "error": None,
                "result": observation,
            }
            messages.append(
                HumanMessage(
                    content=ast_printer.visit_function_result(function_result),
                )
            )

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
    rate_limiter: Optional[RateLimiter] = None,
) -> Runnable[AgentInput, Union[AgentAction, AgentFinish]]:
    """Create an agent for a chat model."""

    function_definitions = [convert_tool_to_function_definition(tool) for tool in tools]
    ast_printer_ = AnthropicXMLEncoder()
    tool_description = ast_printer_.visit_function_definitions(function_definitions)

    template = ChatPromptTemplate.from_messages(
        [
            ("system", _ANTHROPIC_TOOL_USER_PROMPT),
            MessagesPlaceholder("examples"),  # Can use to add example traces
            ("human", "{input}"),
            MessagesPlaceholder("history"),
        ]
    ).partial(tool_description=tool_description)

    # For the time being, hard-coding the fact that we're using a <tool> tag.
    model = model.bind(stop=["</function_calls>"])

    if rate_limiter:
        # Apply a rate limiter if it was provided
        model = with_rate_limit(model, rate_limiter)

    agent = (
        {
            "input": lambda x: x["input"],
            "history": lambda x: format_steps_for_chat(
                x["intermediate_steps"], ast_printer_
            ),
            "examples": lambda x: x.get("examples", []),
        }
        | template
        | model
        | parser
    )
    return agent
