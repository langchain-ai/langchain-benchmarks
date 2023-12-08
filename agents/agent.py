from typing import Tuple, List, Optional, Literal
from typing import Sequence
from typing import Union

from langchain.agents import AgentExecutor
from langchain.agents import AgentOutputParser
from langchain.chat_models import ChatAnthropic, ChatFireworks
from langchain.chat_models.base import BaseChatModel
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.runnable import Runnable
from langchain.tools import StructuredTool
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict, NotRequired

from agents.encoder import AstPrinter, XMLEncoder, TypeScriptEncoder
from agents.parser import ParameterizedAgentParser
from agents.adapters import convert_tool_to_function_definition
from agents.prompts import AGENT_INSTRUCTIONS_BLOB_STYLE
from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage.agents import apply_agent_executor_adapter


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


# PUBLIC API


class AgentInput(TypedDict):
    """The input to the agent."""

    input: str
    """The input to the agent."""
    intermediate_steps: List[Tuple[AgentAction, str]]
    """The intermediate steps taken by the agent."""
    examples: NotRequired[List[BaseMessage]]
    """A list of messages that can be used to form example traces."""


def create_chat_agent(
    chat_model: BaseChatModel,
    tools: Sequence[StructuredTool],
    parser: AgentOutputParser,
    *,
    ast_printer: Union[AstPrinter, Literal["xml"]] = "xml",
) -> Runnable[AgentInput, Union[AgentAction, AgentFinish]]:
    """Create an agent for a chat model."""
    if isinstance(ast_printer, str):
        if ast_printer == "xml":
            ast_printer = AstPrinter()
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
