"""Copy the public dataset to your own langsmith tenant."""
from langsmith import Client
from tqdm import tqdm

# PUBLIC API


def clone_dataset(
    public_dataset_token: str,
    dataset_name: str,
) -> None:
    """Clone a public dataset to your own langsmith tenant.

    This operation is idempotent. If you already have a dataset with the given name,
    this function will do nothing.

    Args:
        public_dataset_token (str): The token of the public dataset to clone.
        dataset_name (str): The name of the dataset to create in your tenant.
    """
    client = Client()

    if not hasattr(client, "has_dataset"):
        raise ValueError("You must use a version of langsmith >= 0.0.2")

    if client.has_dataset(dataset_name=dataset_name):
        return
    dataset = client.create_dataset(dataset_name=dataset_name)
    examples = tqdm(list(client.list_shared_examples(public_dataset_token)))
    print("Finished fetching examples. Creating dataset...")
    client.create_examples(
        inputs=[e.inputs for e in examples],
        outputs=[e.outputs for e in examples],
        dataset_id=dataset.id,
    )
    print("Done creating dataset.")
