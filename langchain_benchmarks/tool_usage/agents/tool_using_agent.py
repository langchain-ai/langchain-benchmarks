"""Factory for creating agents.

This is useful for agents that follow the standard LangChain tool format.
"""
from langchain.agents import AgentExecutor
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from langchain_benchmarks.schema import ToolUsageTask
from langchain_benchmarks.tool_usage.agents.adapters import apply_agent_executor_adapter


class StandardAgentFactory:
    """A standard agent factory.

    Use this factory with chat models that support the standard LangChain tool
    calling API where the chat model populates the tool_calls attribute on AIMessage.
    """

    def __init__(
        self,
        task: ToolUsageTask,
        model: BaseChatModel,
        prompt: ChatPromptTemplate,
    ) -> None:
        """Create an agent factory for the given tool usage task.

        Args:
            task: The task to create an agent factory for
            model: chat model to use, must support tool usage
            prompt: This is a chat prompt at the moment.
                Must include an agent_scratchpad

                For example,

                ChatPromptTemplate.from_messages(
                    [
                        ("system", "{instructions}"),
                        ("human", "{input}"),
                        MessagesPlaceholder("agent_scratchpad"),
                    ]
                )
        """
        self.task = task
        self.model = model
        self.prompt = prompt

    def __call__(self) -> Runnable:
        """Call the factory to create Runnable agent."""
        # Temporarily import here until new langchain is released with create_tools_agent
        from langchain.agents import create_tools_agent

        env = self.task.create_environment()

        if "instructions" in self.prompt.input_variables:
            finalized_prompt = self.prompt.partial(instructions=self.task.instructions)
        else:
            finalized_prompt = self.prompt

        agent = create_tools_agent(self.model, env.tools, finalized_prompt)

        executor = AgentExecutor(
            agent=agent,
            tools=env.tools,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        return apply_agent_executor_adapter(
            executor, state_reader=env.read_state
        ).with_config({"run_name": "Agent", "metadata": {"task": self.task.name}})
