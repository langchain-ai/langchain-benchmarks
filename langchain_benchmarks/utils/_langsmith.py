"""Copy the public dataset to your own langsmith tenant."""
import functools
import json
import logging
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union, cast
from uuid import UUID

from langchain.smith import RunEvalConfig
from langchain.smith.evaluation import runner_utils as eval_runner_utils
from langchain_core import runnables
from langchain_core.runnables import config as runnable_config
from langchain_core.tracers.root_listeners import RootListenersTracer
from langsmith import Client, EvaluationResult
from langsmith.evaluation.evaluator import EvaluationResults
from langsmith.schemas import DataType, Example, Run
from langsmith.utils import LangSmithNotFoundError
from tqdm import auto

logger = logging.getLogger(__name__)

API_URL = "https://api.smith.langchain.com/"


def _parse_token_or_url(url_or_token: str, api_url: str) -> Tuple[str, Optional[str]]:
    """Parse a public dataset URL or share token."""
    try:
        UUID(url_or_token)
        return api_url, url_or_token
    except ValueError:
        pass

    # Then it's a URL
    parsed_url = urllib.parse.urlparse(url_or_token)
    # Extract the UUID from the path
    path_parts = parsed_url.path.split("/")
    token_uuid = path_parts[-2] if len(path_parts) >= 2 else None
    return API_URL, token_uuid


# PUBLIC API


def clone_public_dataset(
    token_or_url: str,
    *,
    dataset_name: Optional[str] = None,
    source_api_url: str = API_URL,
) -> None:
    """Clone a public dataset to your own langsmith tenant.

    This operation is idempotent. If you already have a dataset with the given name,
    this function will do nothing.

    Args:
        token_or_url (str): The token of the public dataset to clone.
        dataset_name (str): The name of the dataset to create in your tenant.
        source_api_url: The URL of the langsmith server where the data is hosted:w
    """
    source_api_url, token_uuid = _parse_token_or_url(token_or_url, source_api_url)
    source_client = Client(api_url=source_api_url, api_key="placeholder")
    ds = source_client.read_shared_dataset(token_uuid)
    dataset_name = dataset_name or ds.name
    client = Client()  # Client used to write to langsmith
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)

        if dataset:
            print(f"Dataset {dataset_name} already exists. Skipping.")
            print(f"You can access the dataset at {dataset.url}.")
            return
    except LangSmithNotFoundError:
        pass

    try:
        # Fetch examples first
        examples = auto.tqdm(list(source_client.list_shared_examples(token_uuid)))
        print("Finished fetching examples. Creating dataset...")
        dataset = client.create_dataset(dataset_name=dataset_name)
        print(f"New dataset created you can access it at {dataset.url}.")
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
    finally:
        del source_client
        del client


def download_public_dataset(
    token_or_url: str,
    *,
    path: Optional[Union[str, Path]] = None,
    api_url: str = API_URL,
) -> None:
    """Download a public dataset."""
    api_url, token_uuid = _parse_token_or_url(token_or_url, api_url)
    _path = str(path) if path else f"{token_uuid}.json"
    if not _path.endswith(".json"):
        raise ValueError(f"Path must end with .json got: {_path}")

    # This the client where the source data lives
    # The destination for the dataset is the local filesystem
    source_client = Client(api_url=api_url, api_key="placeholder")

    try:
        # Fetch examples first
        print("Fetching examples...")
        examples = auto.tqdm(list(source_client.list_shared_examples(token_uuid)))
        with open(str(_path), mode="w", encoding="utf-8") as f:
            jsonifable_examples = [json.loads(example.json()) for example in examples]
            json.dump(jsonifable_examples, f, indent=2)
        print("Done fetching examples.")
    finally:
        del source_client


def exists_public_dataset(token_or_url: str, *, api_url: str = API_URL) -> bool:
    """Check if a public dataset exists."""
    api_url, uuid = _parse_token_or_url(token_or_url, api_url)
    source_client = Client(api_url=api_url, api_key="placeholder")
    try:
        try:
            source_client.read_shared_dataset(uuid)
            return True
        except LangSmithNotFoundError:
            return False

    finally:
        del source_client


def _select_eval_results(
    results: Union[EvaluationResult, EvaluationResults],
) -> List[EvaluationResult]:
    if isinstance(results, EvaluationResult):
        results_ = [results]
    elif isinstance(results, dict) and "results" in results:
        results_ = cast(List[EvaluationResult], results["results"])
    else:
        raise TypeError(
            f"Invalid evaluation result type {type(results)}."
            " Expected EvaluationResult or EvaluationResults."
        )
    return results_


def _is_jupyter_environment() -> bool:
    try:
        from IPython import get_ipython

        res = get_ipython()
        return get_ipython() is not None and "zmqshell" in str(type(res))
    except ImportError:
        return False


def _display_aggregate_results(aggregate_results: Any) -> None:
    if _is_jupyter_environment():
        from IPython.display import HTML, display

        display(HTML("<h3>Experiment Results:</h3>"))
        display(aggregate_results)
    else:
        formatted_string = aggregate_results.to_string(
            float_format=lambda x: f"{x:.2f}", justify="right"
        )
        print("\n Experiment Results:")
        print(formatted_string)


def run_without_langsmith(
    path_or_token_id: Union[str, Path],
    llm_or_chain_factory: Union[
        Callable[[], runnables.Runnable], Callable[[dict], Any]
    ],
    *,
    evaluation: Optional[RunEvalConfig] = None,
    concurrency_level: int = 5,
    verbose: bool = True,
) -> None:
    """Run a public dataset without langsmith."""
    from langchain.smith.evaluation.runner_utils import (
        _setup_evaluation,
        _wrap_in_chain_factory,
    )

    if isinstance(path_or_token_id, Path) or path_or_token_id.endswith(".json"):
        dataset_path = path_or_token_id
    else:
        _, token_uuid = _parse_token_or_url(path_or_token_id, API_URL)
        dataset_path = f"{token_uuid}.json"
        if not Path(dataset_path).exists():
            download_public_dataset(path_or_token_id, path=dataset_path)
    if not dataset_path.endswith(".json"):
        raise ValueError(f"Unrecognized dataset path: {path_or_token_id}")
    with open(str(dataset_path), encoding="utf-8") as f:
        example_dicts = json.load(f)
    examples = [Example(**example_dict) for example_dict in example_dicts]
    wrapped_model = _wrap_in_chain_factory(llm_or_chain_factory)
    run_evaluators = _setup_evaluation(
        llm_or_chain_factory=wrapped_model,
        examples=examples,
        evaluation=evaluation,
        data_type=DataType.kv,
    )

    all_eval_results = {}
    results_lock = threading.RLock()
    _progress_bar = iter(
        auto.tqdm(
            iterable=range(len(examples)),
            desc="Running Evaluation",
            unit="example",
            total=len(examples),
        )
    )

    def _evaluate_run(run: Run, example: Example):
        with results_lock:
            next(_progress_bar)
            example_result = all_eval_results.setdefault(str(example.id), {}) or {}
            example_result.update(
                {
                    "input": run.inputs,
                    "execution_time": (
                        (run.end_time - run.start_time).total_seconds()
                        if run.end_time
                        else None
                    ),
                    "run_id": str(run.id),
                }
            )
            if run.error is not None:
                example_result["Error"] = run.error
            else:
                example_result["output"] = run.outputs
            all_eval_results[str(example.id)] = example_result
        if run_evaluators is None:
            return
        feedback = []
        for evaluator in run_evaluators:
            try:
                eval_results = evaluator.evaluate_run(run, example)
            except Exception as e:
                logger.error(f"Failed to evaluate run {run.id}: {repr(e)}")
                continue
            flattened = _select_eval_results(eval_results)
            feedback.extend(flattened)

        with results_lock:
            example_result = all_eval_results.setdefault(str(example.id), {}) or {}
            example_result.update(
                {
                    "feedback": feedback,
                }
            )
            all_eval_results[str(example.id)] = example_result

    configs = [
        runnable_config.RunnableConfig(
            callbacks=[
                RootListenersTracer(
                    config={},
                    on_start=None,
                    on_end=functools.partial(_evaluate_run, example=example),
                    on_error=functools.partial(_evaluate_run, example=example),
                ),
            ],
            max_concurrency=concurrency_level,
        )
        for example in examples
    ]

    def run_runnable(x: dict) -> Any:
        model = wrapped_model()
        return model.invoke(x)

    runnables.RunnableLambda(run_runnable).batch(
        inputs=[example.inputs for example in examples],
        config=configs,
        return_exceptions=True,
    )
    results = eval_runner_utils.TestResult(
        project_name="Local",
        results=all_eval_results,
    )
    if verbose:
        try:
            agg_feedback = results.get_aggregate_feedback()
            _display_aggregate_results(agg_feedback)
        except Exception as e:
            logger.debug(f"Failed to print aggregate feedback: {repr(e)}")
    return results
