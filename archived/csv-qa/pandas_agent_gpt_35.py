import pandas as pd
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client

if __name__ == "__main__":
    df = pd.read_csv("titanic.csv")

    def get_chain():
        llm = ChatOpenAI(temperature=0)
        agent_executor_kwargs = {
            "handle_parsing_errors": True,
        }
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            agent_executor_kwargs=agent_executor_kwargs,
            max_iterations=5,
        )
        return agent

    client = Client()
    eval_config = RunEvalConfig(
        evaluators=["qa"],
    )
    chain_results = run_on_dataset(
        client,
        dataset_name="Titanic CSV Data",
        llm_or_chain_factory=get_chain,
        evaluation=eval_config,
    )
