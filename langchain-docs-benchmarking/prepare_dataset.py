"""Copy the public dataset to your own langsmith tenant."""
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
from tqdm import tqdm

DATASET_NAME = "LangChain Docs Q&A"
PUBLIC_DATASET_TOKEN = "452ccafc-18e1-4314-885b-edd735f17b9d"
client = Client()


def create_langchain_docs_dataset(
    dataset_name: str = DATASET_NAME, public_dataset_token: str = PUBLIC_DATASET_TOKEN
):
    examples = tqdm(list(client.list_shared_examples(public_dataset_token)))
    try:
        ds = client.read_dataset(dataset_name=dataset_name)
        loaded_examples = list(client.list_examples(dataset_name=dataset_name))
        if len(loaded_examples) == len(examples):
            return
    except LangSmithNotFoundError:
        print(f"Creating dataset {dataset_name}...")
        ds = client.create_dataset(dataset_name=dataset_name)
        pass
    print(f"Adding examples to dataset {dataset_name}")
    client.create_examples(
        inputs=[e.inputs for e in examples],
        outputs=[e.outputs for e in examples],
        dataset_id=ds.id,
    )
    print("Done creating dataset.")


if __name__ == "__main__":
    create_langchain_docs_dataset()
