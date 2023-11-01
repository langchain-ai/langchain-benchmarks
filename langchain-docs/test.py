from functools import partial
from langserve import RemoteRunnable
from langsmith import Client
from langchain.smith import RunEvalConfig
from langchain.chat_models import ChatOpenAI
import argparse

ls_client = Client()


def create_runnable(url: str):
    return RemoteRunnable(url=url)


def get_eval_config():
    accuracy_criteria = {
        "accuracy": """
Score 1: The answer is incorrect and unrelated to the question or reference document.
Score 3: The answer shows slight relevance to the question or reference document but is largely incorrect.
Score 5: The answer is partially correct but has significant errors or omissions.
Score 7: The answer is mostly correct with minor errors or omissions, and aligns with the reference document.
Score 10: The answer is correct, complete, and perfectly aligns with the reference document."""
    }

    eval_llm = ChatOpenAI(model="gpt-4", temperature=0.)
    return RunEvalConfig(
        evaluators=[
            RunEvalConfig.LabeledScoreString(
                criteria=accuracy_criteria, llm=eval_llm, normalize_by=10.0
            ),
            # Mainly to compare with the above
            # Suspected to be less reliable.
            RunEvalConfig.EmbeddingDistance(),
        ]
    )


def main(server_url: str, dataset_name: str):
    eval_config = get_eval_config()

    ls_client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=partial(create_runnable, server_url),
        evaluation=eval_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str)
    parser.add_argument("--dataset-name", type=str, default="Chat Langchain Pub")
    args = parser.parse_args()
    main(args.url, args.dataset_name)
