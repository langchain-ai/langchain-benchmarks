import json
import sys
sys.path.append("../langchain_benchmarks")
from tool_usage.tasks.extraction_query import *
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate
from langchain.chat_models import init_chat_model
from langsmith.evaluation.evaluator import (
    EvaluationResult,
    EvaluationResults,
)
from langsmith.schemas import Example, Run
from typing import Optional
from langsmith.client import Client
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from collections import Counter

def calculate_recall(A, B):
    # Count the occurrences of each element in A and B
    count_A = Counter(A)
    count_B = Counter(B)
    
    # Calculate the number of true positives
    true_positives = sum(min(count_A[elem], count_B.get(elem, 0)) for elem in count_A)
    
    # Calculate recall
    recall = true_positives / sum(count_A.values()) if count_A else 0
    
    return recall

client = Client()

def is_iso_format(date_str):
    if not isinstance(date_str,str):
        return False
    try:
        # Try to parse the string with datetime.fromisoformat
        datetime.fromisoformat(date_str)
        return True
    except ValueError:
        return False

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
            "Is this query: {run_query} somewhat similar to this reference query: {reference_query}",
        ),
    ]
)

judge_chain = judge_prompt | llm_judge | StrOutputParser()

tools = [DocQuery, TweetQuery, BlogQuery]

def compare_outputs(run_outputs: dict, example_outputs: dict) -> EvaluationResults:
    if len(run_outputs['response'].tool_calls) == 0:
        correct_tool_score, determinstic_score, underministic_score = 0,0,0
    else:
        # Chose the correct tool
        reference_tools = [tool["name"] for tool in example_outputs["reference"]]
        outputted_tools = [tool["name"] for tool in run_outputs["response"].tool_calls]
        correct_tool_score = calculate_recall(reference_tools,outputted_tools)

        # Has the correct determenistic args
        determinstic_score = 0
        # Has the correct undetermenistic args
        underministic_score = 0

        if correct_tool_score == 1:
            determinstic_score, underministic_score = 1, 1
            for tool in example_outputs["reference"]:
                corresponding_response_tool = [
                        t
                        for t in run_outputs["response"].tool_calls
                        if t["name"] == tool["name"]
                    ][0]["args"]
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
                            and not (is_iso_format(tool["args"][arg]) 
                                and is_iso_format(corresponding_response_tool[arg]) 
                                and datetime.fromisoformat(
                                    (corresponding_response_tool[arg])
                                ).replace(tzinfo=None)
                                == datetime.fromisoformat(tool["args"][arg])
                            )
                        ):
                            determinstic_score = 0
    # Overall correctness
    overall_score = int(
        correct_tool_score == 1
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


uncleaned_examples = [
    e
    for e in client.list_examples(
        dataset_name="Extraction Task Few Shot"
    )
]
static_indices = [0,2,5]
few_shot_messages, few_shot_str = [], ""
few_shot_messages_by_index = {}
examples_for_semantic_search = []

for j,example in enumerate(uncleaned_examples):
    few_shot_messages_for_example = []
    few_shot_messages_for_example.append(HumanMessage(name="example_human",content=example.inputs['question'][0]['content']))
    few_shot_messages_for_example.append(AIMessage(name="example_assistant", content="", tool_calls=[{"name": tc["name"], "args": tc["args"], "type": "tool_call", "id": f"{10*j+i}"} for i, tc in enumerate(example.outputs["reference"])]))
    few_shot_str += f"<|im_start|>user\n{example.inputs['question'][0]['content']}\n<|im_end|>"
    few_shot_str += "\n<|im_start|>assistant\n"
    for i, tool_call in enumerate(example.outputs["reference"]):
        few_shot_messages_for_example.append(ToolMessage("You have correctly called this tool", name=tool_call["name"], tool_call_id=f"{10*j+i}"))
        few_shot_str += f"Tool Call: Name: {tool_call['name']} Args: {{{', '.join(f'{k}: {v}' for k,v in tool_call['args'].items())}}}"
        few_shot_str += "\n"
    few_shot_str += "<|im_end|>"

    few_shot_messages += few_shot_messages_for_example
    few_shot_messages_by_index[j] = few_shot_messages_for_example
    examples_for_semantic_search.append({"question":example.inputs['question'][0]['content'],"messages":few_shot_messages_for_example})

prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "{instructions}"),
                        MessagesPlaceholder("few_shot_message_list"),
                        ("human", "{input}"),
                    ]
)

def predict_for_model(model,instructions,few_shot_method,model_name):
    few_shot_message_list = []
    chain = prompt | model.bind_tools(tools).with_retry(stop_after_attempt=5)
    if few_shot_method == "few-shot-string":
        instructions += f"\n Here are some examples: \n {few_shot_str}"
    elif few_shot_method == "few-shot-messages":
        few_shot_message_list = few_shot_messages
    elif few_shot_method == "few-shot-static-messages":
        few_shot_message_list = [message for index in static_indices for message in few_shot_messages_by_index[index]]
    elif few_shot_method == "few-shot-dynamic-messages":
        def predict(example: dict):
            example_selector = SemanticSimilarityExampleSelector.from_examples(
                examples_for_semantic_search,
                OpenAIEmbeddings(),
                FAISS,
                k=3,
                input_keys=["question"],
                example_keys=["messages"],
            )

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                input_variables=[],
                example_selector=example_selector,
                example_prompt=MessagesPlaceholder("messages"),
            )
            return {"response": chain.invoke({"input":example["question"],"instructions":instructions,"few_shot_message_list":few_shot_prompt.invoke({"question":example["question"][0]['content']}).messages})}
        return predict
    
    def predict(example: dict):
        return {"response": chain.invoke({"input":example["question"],"instructions":instructions,"few_shot_message_list":few_shot_message_list})}
    
    return predict


models = [
            ("claude-3-haiku-20240307","anthropic",),
            ("claude-3-sonnet-20240229","anthropic",),
            ("claude-3-opus-20240229","anthropic",),
            ("claude-3-5-sonnet-20240620","anthropic",),
            ("gpt-3.5-turbo-0125","openai"),
            ("gpt-4o","openai"),
            ("gpt-4o-mini","openai")
          ]

few_shot_methods = ["no-few-shot","few-shot-string","few-shot-messages","few-shot-static-messages","few-shot-dynamic-messages"]

from tqdm import tqdm
for i in tqdm(range(2)):
    for model_name, model_provider in models[:-4]:
        model = init_chat_model(model_name,model_provider=model_provider)
        for few_shot_method in few_shot_methods:
            evaluate(
                predict_for_model(model,EXTRACTION_TASK.instructions,few_shot_method,model_name),
                data=EXTRACTION_TASK.name,
                evaluators=[evaluate_run],
                experiment_prefix=f"{model_name}-TEST-{i+2}-{few_shot_method}",
            )
            