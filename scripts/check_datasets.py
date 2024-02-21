"""Script to check that all registered datasets can be downloaded."""
from langchain_benchmarks import registry
from langchain_benchmarks.utils._langsmith import exists_public_dataset


def check_datasets() -> bool:
    """Check that all tasks can be downloaded."""
    ok = True
    for task in registry.tasks:
        print(f"Checking {task.name}...")
        if exists_public_dataset(task.dataset_id):
            print("  OK")
        else:
            ok = False
            print("  ERROR: Dataset not found")
    return ok


if __name__ == "__main__":
    ok = check_datasets()
    if not ok:
        exit(1)
