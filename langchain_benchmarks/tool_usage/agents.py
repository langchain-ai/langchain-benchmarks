"""Code for creating an agent factory for evaluating tool usage tasks."""
from typing import Any, Callable, Optional, List

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.tools.base import BaseTool
from langchain.tools.render import format_tool_to_openai_function

from langchain_benchmarks.schema import ToolUsageTask


def _ensure_output_exists(inputs: dict) -> dict:
    """Make sure that the output key is always present."""
    if "output" not in inputs:
        return {"output": "", **inputs}
    return inputs


# PUBLIC API


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

        runnable = AgentExecutor(
            agent=runnable_agent,
            tools=env.tools,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        # Returns `state` in the output if the environment has a state reader
        # makes sure that `output` is always in the output
        return standardize_executor(runnable, state_reader=env.read_state)

    def __call__(self) -> Runnable:
        return self.create()


def get_xml_agent(
    model: BaseLanguageModel, tools: List[BaseTool], system_message: str
) -> Runnable:
    """Create an agent that uses the XML format for chat history."""
    from langchain.agents import XMLAgent
    from langchain.chains import LLMChain

    chain = LLMChain(
        llm=model,
        prompt=XMLAgent.get_default_prompt(),
        output_parser=XMLAgent.get_default_output_parser(),
    )
    agent = XMLAgent(tools=tools, llm_chain=chain)

    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def standardize_executor(
    agent_executor: AgentExecutor,
    *,
    state_reader: Optional[Callable[[], Any]] = None,
) -> Runnable:
    """An adapter for agent executor to standardize its output.

    1) Makes sure that `output` is always returned (will be set to "" if missing) --
        Note that this should not be necessary after more updates in the eval.
    2) Populate `state` with the system state if a state reader is provided.

    Args:
        agent_executor: the agent executor
        state_reader: A callable without parameters that if invoked will return
                      the state of the environment. Used to populate the 'state' key.

    Returns:
        a new runnable with a standardized output.
    """

    def _read_state(*args: Any, **kwargs: Any) -> Any:
        """Read the state of the environment."""
        if state_reader is not None:
            return state_reader()
        else:
            return None

    runnable = agent_executor | _ensure_output_exists
    if state_reader is not None:
        runnable = agent_executor | RunnablePassthrough.assign(
            state=_read_state
        ).with_config({"run_name": "Read Env State"})
    return runnable
