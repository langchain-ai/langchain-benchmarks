import dataclasses
from typing import List, Sequence, Union

from tabulate import tabulate


@dataclasses.dataclass(frozen=True)
class Environment:
    id: int
    """The ID of the environment."""
    name: str
    """The name of the environment."""

    dataset_id: str
    """The ID of the langsmith public dataset.
    
    This dataset contains expected inputs/outputs for the environment, and
    can be used to evaluate the performance of a model/agent etc.
    """

    description: str
    """Description of the environment."""

    @property
    def _table(self) -> List[List[str]]:
        return [
            ["ID", self.id],
            ["Name", self.name],
            ["Dataset ID", self.dataset_id],
            ["Description", self.description[:100] + "..."],
        ]

    def _repr_html_(self) -> str:
        """Return a HTML representation of the environment."""

        return tabulate(
            self._table,
            tablefmt="html",
        )


@dataclasses.dataclass(frozen=True)
class Registry:
    environments: Sequence[Environment]

    def get_environment(self, name_or_id: Union[int, str]) -> Environment:
        """Get the environment with the given name."""
        for env in self.environments:
            if env.name == name_or_id or env.id == name_or_id:
                return env
        raise ValueError(
            f"Unknown environment {name_or_id}. Use list_environments() to see "
            f"available environments."
        )

    def _repr_html_(self) -> str:
        """Return a HTML representation of the registry."""
        headers = [
            "ID",
            "Name",
            "Dataset ID",
            "Description",
        ]
        table = [
            [
                env.id,
                env.name,
                env.dataset_id,
                env.description,
            ]
            for env in self.environments
        ]
        return tabulate(table, headers=headers, tablefmt="html")

    def __getitem__(self, key: Union[int, str]) -> Environment:
        """Get an environment from the registry."""
        if isinstance(key, slice):
            raise NotImplementedError("Slicing is not supported.")
        elif isinstance(key, (int, str)):
            # If key is an integer, return the corresponding environment
            return self.get_environment(key)
        else:
            raise TypeError("Key must be an integer or a slice.")
