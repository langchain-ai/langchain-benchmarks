import ast
import re
from typing import Any, Tuple, List, Optional
from typing import Sequence
from typing import Union

from langchain.agents import AgentExecutor
from langchain.agents import AgentOutputParser
from langchain.chat_models import ChatAnthropic, ChatFireworks
from langchain.chat_models.base import BaseChatModel
from langchain.prompts.chat import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, ValidationError, Field
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.runnable import Runnable
from langchain.tools import StructuredTool
from langchain_core.agents import AgentFinish, AgentAction, AgentActionMessageLog
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict, NotRequired

from agents.adapters import (
    format_structured_tool_as_xml,
)
from agents.prompts import AGENT_INSTRUCTIONS_BLOB_STYLE
from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage.agents import apply_agent_executor_adapter


class ToolInvocationRequest(BaseModel):
    """Request to invoke a tool."""

    tool_name: str
    # OK parameterless tools which do not take arguments
    arguments: Any = Field(default_factory=dict)


class ParameterizedAgentParser(AgentOutputParser):
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
            # language model, so depending on implementation it may get cut off.
            if not self.require_closing_xml_tag:  # Fix regular expression instead
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


def parse_invocation(text: str, tag: str) -> Union[AgentAction]:
    """Parse the content of the function invocation."""
    ai_content = f"<{tag}>{text}</{tag}>"

    try:
        result = ast.literal_eval(text)
    except Exception as e:
        # Convert this to something controllable by the user.
        err_msg = (
            f"ERROR: Please use the format "
            f'<{tag}>{{"tool_name": $TOOL_NAME, "arguments": $ARGUMENTS}}</{tag}>'
        )
        raise OutputParserException(
            error=e,
            llm_output=ai_content,
            observation=err_msg,
            send_to_llm=True,
        )

    try:
        request = ToolInvocationRequest(**result)
    except ValidationError as e:
        err_msg = (
            f"ERROR: Please use the format "
            f'<{tag}>{{"tool_name": $TOOL_NAME, "arguments": $ARGUMENTS}}</{tag}>'
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
    intermediate_steps: List[Tuple[AgentAction, str]]
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


def parsing_error_handler(parsing_exception: OutputParserException) -> str:
    """Handle parsing errors."""
    return parsing_exception.error


# PUBLIC API


class AgentInput(TypedDict):
    """The input to the agent."""

    input: str
    """The input to the agent."""
    intermediate_steps: List[Tuple[AgentAction, str]]
    """The intermediate steps taken by the agent."""
    examples: NotRequired[List[BaseMessage]]
    """A list of messages that can be used to form example traces."""


EXAMPLE_TRACE = [
    HumanMessage(content="type the letter 'o'"),
    AIMessage(
        content="""
<tool>
{
    "tool_name": "type_letter",
    "arguments": {
        "letter": "o"
    }
}
</tool>\
"""
    ),
    HumanMessage(
        content="""\
<tool_outputs>
<tool_name>type_letter</tool_name>
<output>o</output>
</tool_outputs>\
"""
    ),
]


def create_chat_agent(
    chat_model: BaseChatModel,
    tools: Sequence[StructuredTool],
    parser: AgentOutputParser,
) -> Runnable[AgentInput, Union[AgentAction, AgentFinish]]:
    """Create an agent for a chat model."""
    tool_description = "\n".join(
        [format_structured_tool_as_xml(tool) for tool in tools]
    )

    template = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_INSTRUCTIONS_BLOB_STYLE),
            MessagesPlaceholder("examples"),  # Can use to add example traces
            ("human", "{input}"),
            MessagesPlaceholder("history"),
        ]
    ).partial(tool_description=tool_description)

    agent = (
        {
            "input": lambda x: x["input"],
            "history": lambda x: format_steps_for_chat(x["intermediate_steps"]),
            "examples": lambda x: x.get("examples", []),
        }
        | template
        | chat_model.bind(stop=["</tool>"])
        | parser
    )
    return agent


FIREWORK_NAME_TO_MODEL = {
    "llama-v2-7b-chat": "accounts/fireworks/models/llama-v2-7b-chat",
    "llama-v2-13b-chat": "accounts/fireworks/models/llama-v2-13b-chat",
    "llama-v2-70b-chat": "accounts/fireworks/models/llama-v2-70b-chat",
}


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

        agent = create_chat_agent(
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
