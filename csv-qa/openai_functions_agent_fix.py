# temp fix OutputParserException `arguments` is not valid JSON.
# https://github.com/langchain-ai/langchain/issues/6364

import json
from json import JSONDecodeError
from typing import Any, List, Tuple, Union

from langchain.agents import OpenAIFunctionsAgent
from langchain.callbacks.manager import Callbacks
from langchain.schema import (
    AgentAction,
    AgentFinish,
)
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
)
from langchain.agents.openai_functions_agent.base import (
    _format_intermediate_steps,
    _FunctionsAgentAction,
)


def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
    """Parse an AI message."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    function_call = message.additional_kwargs.get("function_call", {})

    if function_call:
        function_name = function_call["name"]
        try:
            _tool_input = json.loads(function_call["arguments"])
        except JSONDecodeError:
            print(
                f"Could not parse tool input: {function_call} because "
                f"the `arguments` is not valid JSON."
            )
            _tool_input = function_call["arguments"]

        # HACK HACK HACK:
        # The code that encodes tool input into Open AI uses a special variable
        # name called `__arg1` to handle old style tools that do not expose a
        # schema and expect a single string argument as an input.
        # We unpack the argument here if it exists.
        # Open AI does not support passing in a JSON array as an argument.
        if "__arg1" in _tool_input:
            tool_input = _tool_input["__arg1"]
        else:
            tool_input = _tool_input

        content_msg = "responded: {content}\n" if message.content else "\n"

        return _FunctionsAgentAction(
            tool=function_name,
            tool_input=tool_input,
            log=f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n",
            message_log=[message],
        )

    return AgentFinish(return_values={"output": message.content}, log=message.content)


class OpenAIFunctionsAgentFix(OpenAIFunctionsAgent):
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        with_functions: bool = True,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date, along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        agent_scratchpad = _format_intermediate_steps(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = dict(**selected_inputs, agent_scratchpad=agent_scratchpad)
        prompt = self.prompt.format_prompt(**full_inputs)
        messages = prompt.to_messages()
        if with_functions:
            predicted_message = self.llm.predict_messages(
                messages,
                functions=self.functions,
                callbacks=callbacks,
            )
        else:
            predicted_message = self.llm.predict_messages(
                messages,
                callbacks=callbacks,
            )
        agent_decision = _parse_ai_message(predicted_message)
        return agent_decision


if __name__ == "__main__":
    import pandas as pd

    from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
    from langchain.tools import PythonAstREPLTool
    from langchain.chat_models import ChatOpenAI
    from langsmith import Client
    from langchain.smith import RunEvalConfig, run_on_dataset
    from langchain.schema import SystemMessage

    import dotenv
    dotenv.load_dotenv()

    df = pd.read_csv("titanic.csv")

    SYSTEM_PROMPT = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_head}""".format(df_head=str(df.head().to_markdown()))
    
    def get_chain():
        tools = [PythonAstREPLTool(name="python", locals={"df": df})]
        llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
        agent = OpenAIFunctionsAgentFix.from_llm_and_tools(
            llm=llm,
            tools=tools,
            system_message=SystemMessage(content=SYSTEM_PROMPT),
        )
        agent_exe = AgentExecutor.from_agent_and_tools(agent, tools)
        return agent_exe
    
    client = Client()
    eval_config = RunEvalConfig(
        evaluators=[
            "qa"
        ],
    )
    chain_results = run_on_dataset(
        client,
        dataset_name="Titanic CSV Data",
        llm_or_chain_factory=get_chain,
        evaluation=eval_config,
    )