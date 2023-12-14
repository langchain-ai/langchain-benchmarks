import ast
import re
from typing import Union, Dict, Optional

from langchain.agents import AgentOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage


class _ToolInvocationRequest(BaseModel):
    """Light-weight pydantic model for validating the raw tool invocation request.

    The purpose of this model, is to make sure that whatever as parsed from
    the raw llm output has `tool_name` and potential `arguments` fields, and
    nothing else.
    """

    tool_name: str
    # OK parameterless tools which do not take arguments
    arguments: Optional[Dict] = Field(default_factory=dict)


class GenericAgentParser(AgentOutputParser):
    """A generalized parser that makes it easier to parameterize different parsing."""

    wrapping_xml_tag: str
    """The tag that wraps the function invocation request.
    
    For example, if "tool", then the function invocation request should be wrapped
    in <tool>...</tool>.
    """
    require_closing_xml_tag: bool = False
    """Whether we should require a closing tag for the wrapping_xml_tag.
    
    For example, if True, then the function invocation request should be wrapped
    """

    def parse(self, text: str) -> Union[AgentFinish, AgentAction]:
        """Parse the output of the agent."""
        open_tag = f"<{self.wrapping_xml_tag}>"
        close_tag = f"</{self.wrapping_xml_tag}>"
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


def _remove_unescaped_new_lines(input_str: str) -> str:
    """Remove unescaped new lines from the input string."""
    result = []
    i = 0
    while i < len(input_str):
        if input_str[i] == "\\" and i + 1 < len(input_str) and input_str[i + 1] == "\n":
            # If a backslash is followed by a newline,
            # keep both characters (escaped newline)
            result.append(input_str[i : i + 2])
            i += 2
        elif input_str[i] == "\n":
            # If it's an unescaped newline, skip it
            i += 1
        else:
            # Otherwise, keep the character as is
            result.append(input_str[i])
            i += 1

    return "".join(result)


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
