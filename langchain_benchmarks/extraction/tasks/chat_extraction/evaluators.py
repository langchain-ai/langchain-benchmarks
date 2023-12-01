from langchain.smith.evaluation.config import RunEvalConfig, SingleKeyEvalConfig
from langsmith.evaluation.evaluator import (
    EvaluationResult,
    run_evaluator,
)
from langsmith.schemas import Example, Run

from langchain_benchmarks.extraction.tasks.chat_extraction.schema import GenerateTicket


@run_evaluator
def json_schema(run: Run, example: Example) -> EvaluationResult:
    """Evaluate the json schema of the generated ticket."""
    score, comment = None, None
    try:
        GenerateTicket.parse_obj(run.outputs["output"])
        score = 1
    except Exception as e:
        comment = repr(e)
        score = 0

    return EvaluationResult(
        key="json_schema",
        score=score,
        comment=comment,
    )


@run_evaluator
def evaluate_toxicity_similarity(run: Run, example: Example) -> EvaluationResult:
    """Evaluate the toxicity of the generated ticket."""
    gt = example.outputs["output"]["question"]["toxicity"]
    score, comment = None, None
    # Toxicity should be a on scale from 0 to 5
    try:
        pred = run.outputs["output"]["question"]["toxicity"]
        score = 1 - abs(gt - float(pred)) / 5
    except Exception as e:
        comment = repr(e)
        # Forgot to predict / mis-structured
        score = 0
    return EvaluationResult(
        key="toxicity_similarity",
        score=score,
        comment=comment,
    )


@run_evaluator
def evaluate_sentiment_similarity(run: Run, example: Example) -> EvaluationResult:
    """Evaluate the sentiment of the generated ticket."""
    gt = example.outputs["output"]["question"]["sentiment"]
    ordinal_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    gt_score = ordinal_map.get(str(gt).lower())
    score, comment = None, None
    # Sentiment is an enum, "Negative", "Neutral", "Positive"
    try:
        pred = run.outputs["output"]["question"]["sentiment"]
        pred_score = ordinal_map.get(str(pred).lower())
        score = 1 - (abs(gt_score - float(pred_score)) / 2)
    except Exception as e:
        comment = repr(e)
        # Forgot to predict / mis-structured
        score = 0
    return EvaluationResult(
        key="sentiment_similarity",
        score=score,
        comment=comment,
    )


@run_evaluator
def evaluate_confidence_level_similarity(
    run: Run, example: Example
) -> EvaluationResult:
    """Evaluate the confidence level of the generated ticket.
    This is a binary T/F question."""
    gt = example.outputs["output"]["response"]["confidence_level"]
    score, comment = None, None
    try:
        pred = run.outputs["output"]["response"]["confidence_level"]
        score = 1 - (abs(gt - float(pred)) / 5)
    except Exception as e:
        comment = repr(e)
        score = 0
    return EvaluationResult(
        key="confidence_level_similarity",
        score=score,
        comment=comment,
    )


@run_evaluator
def evaluate_question_category_similarity(
    run: Run, example: Example
) -> EvaluationResult:
    """Evaluate the question category of the generated ticket.
    This is a binary T/F question."""
    gt = example.outputs["output"]["question"]["question_category"]

    score, comment = None, None
    try:
        pred = run.outputs["output"]["question"]["question_category"]
        score = int(gt == pred)
    except Exception as e:
        comment = repr(e)
        # Forgot to predict / mis-structured
        score = 0
    return EvaluationResult(
        key="question_category",
        score=score,
        comment=comment,
    )


@run_evaluator
def evaluate_off_topic(run: Run, example: Example) -> EvaluationResult:
    """Evaluate the off topic of the generated ticket.
    This is a binary T/F question."""
    gt = example.outputs["output"]["question"]["is_off_topic"]
    score, comment = None, None
    try:
        pred = run.outputs["output"]["question"].get("is_off_topic")
        score = int(gt == pred)
    except Exception as e:
        comment = repr(e)
        # Forgot to predict / mis-structured
        score = 0
    return EvaluationResult(
        key="off_topic_similarity",
        score=score,
        comment=comment,
    )


@run_evaluator
def evaluate_programming_language(run: Run, example: Example) -> EvaluationResult:
    """Evaluate the programming language of the generated ticket.
    This is a binary T/F question."""
    gt = example.outputs["output"]["question"]["programming_language"]
    score, comment = None, None
    try:
        pred = run.outputs["output"]["question"]["programming_language"]
        score = int(gt == pred)
    except Exception as e:
        comment = repr(e)
        # Forgot to predict / mis-structured
        score = 0
    return EvaluationResult(
        key="programming_language_similarity",
        score=score,
        comment=comment,
    )


def get_eval_config() -> RunEvalConfig:
    """Get the evaluation configuration for the chat extraction task."""
    return RunEvalConfig(
        evaluators=[
            # General aggregate score
            SingleKeyEvalConfig(
                # input key is ignored.
                evaluator_type="json_edit_distance",
                input_key="question",
            )
        ],
        custom_evaluators=[
            json_schema,
            evaluate_toxicity_similarity,
            evaluate_sentiment_similarity,
            evaluate_confidence_level_similarity,
            evaluate_question_category_similarity,
            evaluate_off_topic,
            evaluate_programming_language,
        ],
    )
