import datetime
import unittest.mock as mock
import uuid
from contextlib import contextmanager
from typing import Any, Generator, List, Mapping, Optional, Sequence
from uuid import UUID

from langsmith.client import ID_TYPE
from langsmith.schemas import Dataset, Example
from langsmith.utils import LangSmithNotFoundError

from langchain_benchmarks.utils._langsmith import clone_public_dataset


# Define a mock Client class that overrides the required methods
class MockLangSmithClient:
    def __init__(self) -> None:
        """Initialize the mock client."""
        self.datasets = []
        self.examples = []

    def read_dataset(self, dataset_name: str) -> Dataset:
        for dataset in self.datasets:
            if dataset.name == dataset_name:
                return dataset
        raise LangSmithNotFoundError(f'Dataset "{dataset_name}" not found.')

    def create_dataset(self, dataset_name: str) -> Dataset:
        # Simulate creating a dataset and returning a mock Dataset object
        dataset = Dataset(
            id=UUID(int=3), name=dataset_name, created_at=datetime.datetime(2021, 1, 1)
        )
        self.datasets.append(dataset)
        return dataset

    def create_examples(
        self,
        *,
        inputs: Sequence[Mapping[str, Any]],
        outputs: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
        dataset_id: Optional[ID_TYPE] = None,
        dataset_name: Optional[str] = None,
        max_concurrency: int = 10,
    ) -> None:
        """Create examples"""
        examples = []
        for idx, (input, output) in enumerate(zip(inputs, outputs)):
            examples.append(
                Example(
                    id=UUID(int=idx),
                    inputs=input,
                    outputs=output,
                    created_at=datetime.datetime(2021, 1, 1),
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                )
            )

        return self.examples.extend(examples)

    def list_shared_examples(self, public_dataset_token: str) -> List[Example]:
        # Simulate fetching shared examples and returning a list of Example objects
        example1 = Example(
            id=UUID(int=1),
            inputs={"a": 1},
            outputs={},
            created_at=datetime.datetime(2021, 1, 1),
            dataset_id=public_dataset_token,
        )
        example2 = Example(
            id=UUID(int=2),
            inputs={"b": 2},
            outputs={},
            created_at=datetime.datetime(2021, 1, 1),
            dataset_id=public_dataset_token,
        )
        return [example1, example2]
    
    def read_shared_dataset(self, public_dataset_token: str) -> Dataset:
        # Simulate fetching shared dataset and returning a Dataset object
        dataset = Dataset(
            id=UUID(int=3),
            name="my_dataset",
            created_at=datetime.datetime(2021, 1, 1),
            owner_id=public_dataset_token,
        )
        return dataset


@contextmanager
def mock_langsmith_client() -> Generator[None, None, None]:
    """Mock the langsmith Client class."""
    from langchain_benchmarks.utils import _langsmith

    mock_client = MockLangSmithClient()

    with mock.patch.object(_langsmith, "Client") as client:
        client.return_value = mock_client
        yield mock_client


def test_clone_dataset() -> None:
    # Call the clone_dataset function with mock data
    public_dataset_token = str(uuid.UUID(int=3))
    dataset_name = "my_dataset"

    with mock_langsmith_client() as mock_client:
        clone_public_dataset(public_dataset_token, dataset_name=dataset_name)
        assert mock_client.datasets[0].name == dataset_name
        assert len(mock_client.examples) == 2

        # Check idempotency
        clone_public_dataset(public_dataset_token, dataset_name=dataset_name)
        assert len(mock_client.examples) == 2
