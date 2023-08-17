from pathlib import Path
from langsmith import Client
import json
import logging
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
        except Exception as e:
            logging.warning(f"Skipping {dataset_name}", e)
            return
        logging.info(f"Uploading dataset: {dataset_name}")
        for i, example in enumerate(examples):
            _CLIENT.create_example(example["inputs"], dataset_id=dataset.id, outputs=example["outputs"])
            print(f"Uploaded {i+1}/{len(examples)}", end="\r")
    
if __name__ == '__main__':
    for dataset in _DATA_REPO.glob("*.json"):
        _upload_dataset(dataset)

