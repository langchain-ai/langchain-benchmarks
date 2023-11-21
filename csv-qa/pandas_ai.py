import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client
from pandasai import PandasAI

if __name__ == "__main__":
    df = pd.read_csv("titanic.csv")

    pandas_ai = PandasAI(ChatOpenAI(temperature=0, model="gpt-4"), enable_cache=False)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the users question about some data. A data scientist will run some code and the results will be returned to you to use in your answer",
            ),
            ("human", "Question: {input}"),
            ("human", "Data Scientist Result: {result}"),
        ]
    )

    def get_chain():
        chain = (
            {
                "input": lambda x: x["input_question"],
                "result": lambda x: pandas_ai(df, prompt=x["input_question"]),
            }
            | prompt
            | ChatOpenAI(temperature=0, model="gpt-4")
            | StrOutputParser()
        )
        return chain

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
