"""Copy the public dataset to your own langsmith tenant."""
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
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

    try:
        client.read_dataset(dataset_name=dataset_name)
    except LangSmithNotFoundError:
        pass
    else:
        print(f"Dataset {dataset_name} already exists. Skipping.")
        return

    # Fetch examples first
    examples = tqdm(list(client.list_shared_examples(public_dataset_token)))
    print("Finished fetching examples. Creating dataset...")
    dataset = client.create_dataset(dataset_name=dataset_name)
    try:
        client.create_examples(
            inputs=[e.inputs for e in examples],
            outputs=[e.outputs for e in examples],
            dataset_id=dataset.id,
        )
    except BaseException as e:
        # Let's not do automatic clean up for now in case there might be
        # some other reasons why create_examples fails (i.e., not network issue or
        # keyboard interrupt).
        # The risk is that this is an existing dataset that has valid examples
        # populated from another source so we don't want to delete it.
        print(
            f"An error occurred while creating dataset {dataset_name}. "
            "You should delete it manually."
        )
        raise e

    print("Done creating dataset.")
