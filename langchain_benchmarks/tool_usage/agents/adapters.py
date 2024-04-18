from typing import Any, Callable, Optional

from langchain.agents import AgentExecutor
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough


def _ensure_output_exists(inputs: dict) -> dict:
    """Make sure that the output key is always present."""
    if "output" not in inputs:
        return {"output": "", **inputs}
    return inputs


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

    runnable = agent_executor | RunnableLambda(_ensure_output_exists).with_config(
        {"run_name": "Ensure Output"}
    )

    if state_reader is not None:
        runnable = runnable | RunnablePassthrough.assign(state=_read_state).with_config(
            {"run_name": "Read Env State"}
        )
    return runnable
