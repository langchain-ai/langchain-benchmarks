import json
import logging
from pathlib import Path

from langsmith import Client

logging.basicConfig(level=logging.INFO)

# Synthetic dataset adapted from https://aclanthology.org/D13-1160/

_DATA_REPO = Path(__file__).parent / "data"
_CLIENT = Client()


def _upload_dataset(path: str):
    with open(path, "r") as f:
        data = json.load(f)
        dataset_name = data["name"]
        examples = data["examples"]
        try:
            dataset = _CLIENT.create_dataset(dataset_name)
        except Exception:
            logging.warning(f"Skipping {dataset_name}")
            return
        logging.info(f"Uploading dataset: {dataset_name}")
        _CLIENT.create_examples(
            inputs=[example["inputs"] for example in examples],
            outputs=[example["outputs"] for example in examples],
            dataset_id=dataset.id,
        )


if __name__ == "__main__":
    for dataset in _DATA_REPO.glob("*.json"):
        print("Uploading dataset:", dataset)
        _upload_dataset(dataset)
