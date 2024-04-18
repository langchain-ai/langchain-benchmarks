import abc

from langchain_core.runnables import Runnable


class AgentFactory(abc.ABC):
    """Abstract class for agent factory"""

    @abc.abstractmethod
    def __call__(self) -> Runnable:
        """Create a new agent"""
