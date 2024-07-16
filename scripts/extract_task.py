import json
import sys
sys.path.append("../langchain_benchmarks")
from tool_usage.tasks.extraction_query import *
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate
from model_registration import model_registry
from langsmith.evaluation.evaluator import (
    EvaluationResult,
    EvaluationResults,
)
from langsmith.schemas import Example, Run
from typing import Optional



llm = ChatOpenAI(model="gpt-4o")
#TODO: system prompt
llm_judge = ChatOpenAI(model="gpt-4o")

judge_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an llm tasked with determining if the subject extracted by another LLM is an accurate "
            "representation of the correct answer. You are to check for general semantic similarity since the words might not "
            "match up perfectly but the meaning might still be the same. Return YES if the answers match, and NO otherwise. "
            "Never return anything other than YES or NO.",
        ),
        (
            "human",
            "Is this query: {run_query} very similar to this reference query: {reference_query}",
        ),
    ]
)

judge_chain = judge_prompt | llm_judge | StrOutputParser()

tools = [DocQuery, TweetQuery, BlogQuery]
llm = llm.bind_tools(tools)


def compare_outputs(run_outputs: dict, example_outputs: dict) -> EvaluationResults:
    # Chose the correct tool
    correct_tool_score = int(
        [tool["name"] for tool in example_outputs["reference"]]
        == [
            tool["function"]["name"]
            for tool in run_outputs["response"].additional_kwargs["tool_calls"]
        ]
    )

    # Has the correct determenistic args
    determinstic_score = 0
    # Has the correct undetermenistic args
    underministic_score = 0

    if correct_tool_score == 1:
        determinstic_score, underministic_score = 1, 1
        for tool in example_outputs["reference"]:
            corresponding_response_tool = json.loads(
                [
                    t["function"]
                    for t in run_outputs["response"].additional_kwargs["tool_calls"]
                    if t["function"]["name"] == tool["name"]
                ][0]["arguments"]
            )
            for arg in tool["args"]:
                if arg in ["query", "subject"]:
                    ans = judge_chain.invoke(
                        {
                            "run_query": corresponding_response_tool[arg],
                            "reference_query": tool["args"][arg],
                        }
                    )
                    underministic_score = 1 if ans == "YES" else 0
                else:
                    if (
                        tool["args"][arg] and arg not in corresponding_response_tool
                    ) or (
                        tool["args"][arg]
                        and not (tool["args"][arg] == corresponding_response_tool[arg])
                        and not (
                            isinstance(tool["args"][arg], datetime)
                            and datetime.fromisoformat(
                                (corresponding_response_tool[arg])
                            ).replace(tzinfo=None)
                            == tool["args"][arg]
                        )
                    ):
                        determinstic_score = 0
    # Overall correctness
    overall_score = int(
        bool(correct_tool_score)
        and bool(determinstic_score)
        and bool(underministic_score)
    )
    results = [
        EvaluationResult(
            key="Correct tool",
            score=correct_tool_score,
        ),
        EvaluationResult(
            key="Correct determenistic args",
            score=determinstic_score,
        ),
        EvaluationResult(
            key="Correct undermenistic args",
            score=underministic_score,
        ),
        EvaluationResult(
            key="Overall correctness",
            score=overall_score,
        ),
    ]

    return {"results": results}


def evaluate_run(run: Run, example: Optional[Example] = None) -> EvaluationResults:
    return compare_outputs(run.outputs, example.outputs)

instructions = ""
ChatPromptTemplate.from_messages(
    [
        ("system", "{instructions}"),
        ("human", "{input}"),
    ]
)

prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "{instructions}"),
                        ("human", "{input}"),
                    ]
)
def predict_for_model(model,instructions):
    chain = prompt | model.bind_tools(tools)
    def predict(example: dict):
        return {"response": chain.invoke({"input":example["question"],"instructions":instructions})}
    return predict

'''
experiment_results = evaluate(
    predict,
    data=EXTRACTION_TASK.name,
    evaluators=[evaluate_run],
    experiment_prefix="test-single-tool",
)'''

models = ["gpt-4o", "firefunction-v2"]

for model_name in models:
    model = model_registry[model_name].get_model()
    evaluate(
        predict_for_model(model,EXTRACTION_TASK.instructions),
        data=EXTRACTION_TASK.name,
        evaluators=[evaluate_run],
        experiment_prefix=f"test-{model_name}",
    )

