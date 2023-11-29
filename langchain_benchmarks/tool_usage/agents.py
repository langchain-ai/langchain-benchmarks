"""Code for creating an agent factory for evaluating tool usage tasks."""
from typing import Any, Callable, Optional

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough
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
        # For backwards compatibility
        return self()

    def __call__(self) -> Runnable:
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
                "input": lambda x: x["input"],
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
        return apply_agent_executor_adapter(runnable, state_reader=env.read_state)


# PUBLIC API


def apply_agent_executor_adapter(
    agent_executor: AgentExecutor,
    *,
    state_reader: Optional[Callable[[], Any]] = None,
) -> Runnable:
    """An adapter for the agent executor to standardize its input and output.

    1) Map `question` to `input` (`question` is used in the datasets,
       but `input` is used in the agent executor)
    2) Ensure that `output` is always returned (will be set to "" if missing) --
       note that this may be relaxed after more updates in the eval config.
    3) Populate `state` key in the response of the agent with the system state
       if a state reader is provided.

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

    def _format_input(inputs: dict) -> dict:
        """Make sure that the input is always called `input`."""

        if "question" not in inputs:
            raise ValueError(
                "Expected 'question' to be in the inputs. Found only the following "
                f"keys {sorted(inputs.keys())}."
            )

        inputs = inputs.copy()  # Because 'question' is popped below

        if "input" not in inputs:
            return {"input": inputs.pop("question"), **inputs}
        return inputs

    runnable = (
        RunnableLambda(_format_input).with_config({"run_name": "Format Input"})
        | agent_executor
        | RunnableLambda(_ensure_output_exists).with_config(
            {"run_name": "Ensure Output"}
        )
    )

    if state_reader is not None:
        runnable = runnable | RunnablePassthrough.assign(state=_read_state).with_config(
            {"run_name": "Read Env State"}
        )
    return runnable
