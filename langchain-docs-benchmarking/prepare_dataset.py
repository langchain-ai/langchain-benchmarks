"""Copy the public dataset to your own langsmith tenant."""
from typing import Optional
from langsmith import Client
from tqdm import tqdm

DATASET_NAME = "LangChain Docs Q&A"
PUBLIC_DATASET_TOKEN = "452ccafc-18e1-4314-885b-edd735f17b9d"


def create_langchain_docs_dataset(
    dataset_name: str = DATASET_NAME,
    public_dataset_token: str = PUBLIC_DATASET_TOKEN,
    client: Optional[Client] = None,
):
    shared_client = Client(
        api_url="https://api.smith.langchain.com", api_key="placeholder"
    )
    examples = list(shared_client.list_shared_examples(public_dataset_token))
    client = client or Client()
    if client.has_dataset(dataset_name=dataset_name):
        loaded_examples = list(client.list_examples(dataset_name=dataset_name))
        if len(loaded_examples) == len(examples):
            return
        else:
            ds = client.read_dataset(dataset_name=dataset_name)
    else:
        ds = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[e.inputs for e in examples],
        outputs=[e.outputs for e in examples],
        dataset_id=ds.id,
    )
    print("Done creating dataset.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-api-key", type=str, required=False)
    parser.add_argument("--target-endpoint", type=str, required=False)
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME)
    parser.add_argument(
        "--public-dataset-token", type=str, default=PUBLIC_DATASET_TOKEN
    )
    args = parser.parse_args()
    client = None
    if args.target_api_key or args.target_endpoint:
        client = Client(
            api_key=args.target_api_key,
            api_url=args.target_endpoint,
        )
    create_langchain_docs_dataset(
        dataset_name=args.dataset_name,
        public_dataset_token=args.public_dataset_token,
        client=client,
    )
