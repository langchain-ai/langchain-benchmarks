import argparse
from functools import partial
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.smith import RunEvalConfig
from langsmith import Client
from chat_langchain.chain import chain, anthropic_chain
from anthropic_iterative_search.chain import chain as anthropic_agent_chain
from openai_functions_agent import agent_executor as openai_functions_agent_chain

import uuid

ls_client = Client()


def create_runnable(model: str):
    _map = {
        "chat": chain,
        "anthropic-chat": anthropic_chain,
        "anthropic-iterative-search": anthropic_agent_chain,
        "openai-functions-agent": openai_functions_agent_chain,
    }
    return _map[model]


def get_eval_config():
    accuracy_criteria = {
        "accuracy": """
Score 1: The answer is incorrect and unrelated to the question or reference document.
Score 3: The answer shows slight relevance to the question or reference document but is largely incorrect.
Score 5: The answer is partially correct but has significant errors or omissions.
Score 7: The answer is mostly correct with minor errors or omissions, and aligns with the reference document.
Score 10: The answer is correct, complete, and perfectly aligns with the reference document.

If the reference answer contains multiple alternatives, the predicted answer must only match one of the alternatives to be considered correct.
If the predicted answer contains additional helpful and accurate information that is not present in the reference answer, it should still be considered correct.
"""
    }

    eval_llm = ChatOpenAI(model="gpt-4", temperature=0.0)
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


def main(
    # server_url: str,
    model: str,
    dataset_name: str,
    max_concurrency: int = 5,
    project_name: Optional[str] = None,
):
    eval_config = get_eval_config()
    if project_name is not None:
        project_name += uuid.uuid4().hex[:4]
    ls_client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=partial(create_runnable, model),
        evaluation=eval_config,
        concurrency_level=max_concurrency,
        project_name=project_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str)
    parser.add_argument("--dataset-name", type=str, default="Chat Langchain Pub")
    parser.add_argument("--project-name", type=Optional[str], default=None)
    parser.add_argument("--max-concurrency", type=int, default=5)
    args = parser.parse_args()
    main(
        args.url,
        args.dataset_name,
        max_concurrency=args.max_concurrency,
        project_name=args.project_name,
    )
