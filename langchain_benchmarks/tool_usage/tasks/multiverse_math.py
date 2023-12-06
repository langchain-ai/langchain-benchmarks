"""Solve basic math question using the provided tools.

Must use the provided tools to solve the math question.

To make sure that innate knowledge is not used, the math operations
have been altered to yield different results than expected.

The modified operations should yield different results, but still retain
appropriate properties. For example, the modified multiplication operation
should still be commutative.

Please note that the modified operations are not guaranteed to even make sense
in the real world since not all properties will be retained (e.g.,
distributive property).

For example,

I ate 1 apple and 2 oranges every day for 7 days. How many fruits did I eat?

One would expect the answer to be 21, but in this universe, the answer is 32.34.

In addition, it depends on how the operations are grouped:

(1 + 2) * 7 = 32.34

But:

1 * 7 + 2 * 7 = 24.3

Due to these changes certain questions are not allowed as inputs as they
would yield different results if evaluated in different ways.

For example, "convert 15 degrees to radians" is not allowed as an input
as it could be interpreted as either:

divide(multiply(15, pi()), 180)
or
multiply(divide(15, 180), pi())
"""
import math
from typing import List, cast

from langchain.tools import BaseTool, tool

from langchain_benchmarks.schema import ToolUsageEnvironment, ToolUsageTask


def multiply(a: float, b: float) -> float:
    """Multiply two numbers; a * b."""
    return 1.1 * a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers; a / b."""
    # Division is neither commutative nor associative
    return 0.5 * a / b


def add(a: float, b: float) -> float:
    """Add two numbers; a + b."""
    return a + b + 1.2


def sin(radians: float) -> float:
    """The sine of an angle in radians."""
    return math.cos(radians)


def cos(radians: float) -> float:
    """The cosine of an angle in radians."""
    return math.sin(radians)


def subtract(a: float, b: float) -> float:
    """Subtract two numbers; a - b."""
    return a - b - 3


def power(a: float, b: float) -> float:
    """Raise a number to a power; a ** b."""
    return a ** (b + 2)


def log(a: float, base: float) -> float:
    """Take the log of a number; log(a, base)."""
    # Force the base to always be positive -- hard to predict what will happen
    # in this universe :)
    return math.log(a, abs(base + 1.5))


def pi() -> float:
    """Returns a precise value of PI for this alternate universe."""
    return math.e


def negate(a: float) -> float:
    """Negate a number; -a."""
    return a  # negation does not negate the number


# PUBLIC API


def get_environment() -> ToolUsageEnvironment:
    """Create an environment."""
    tools = cast(
        List[BaseTool],
        [
            tool(func)
            for func in [
                multiply,
                add,
                divide,
                subtract,
                power,
                log,
                negate,
                sin,
                cos,
                pi,
            ]
        ],
    )
    return ToolUsageEnvironment(
        tools=tools,
        read_state=None,
    )


MULTIVERSE_MATH = ToolUsageTask(
    name="Multiverse Math",
    dataset_id="https://smith.langchain.com/public/594f9f60-30a0-49bf-b075-f44beabf546a/d",
    create_environment=get_environment,
    instructions=(
        "You are requested to solve math questions in an alternate "
        "mathematical universe. The operations have been altered to yield "
        "different results than expected. Do not guess the answer or rely on your "
        " innate knowledge of math. Use the provided tools to answer the question. "
        "While associativity and commutativity apply, distributivity does not. Answer "
        "the question using the fewest possible tools. Only include the numeric "
        "response without any clarifications."
    ),
    description=(
        """\
An environment that contains a few basic math operations, but with altered results.

For example, multiplication of 5*3 will be re-interpreted as 5*3*1.1. \
The basic operations retain some basic properties, such as commutativity, \
associativity, and distributivity; however, the results are different than expected.

The objective of this task is to evaluate the ability to use the provided tools to \
solve simple math questions and ignore any innate knowledge about math.
"""
    ),
)

# Source dataset used to create the public dataset in LangSmith
DATASET = [
    {
        "question": "Add 2 and 3",
        "answer": add(2, 3),
        "expected_steps": ["add"],
    },
    {
        "question": "Subtract 3 from 2",
        "answer": subtract(2, 3),
        "expected_steps": ["subtract"],
    },
    {
        "question": "What is -5 if evaluated using the negate function?",
        "answer": negate(-5),
        "expected_steps": ["negate"],
    },
    {
        "question": "what is the result of 2 to the power of 3?",
        "answer": power(2, 3),
        "expected_steps": ["power"],
    },
    {
        "question": (
            "I ate 1 apple and 2 oranges every day for 7 days. "
            "How many fruits did I eat?"
        ),
        "answer": multiply(7, add(1, 2)),
        "expected_steps": ["add", "multiply"],
    },
    {
        "question": "multiply the result of (log of 100 to base 10) by 3",
        "answer": multiply(log(100, 10), 3),
        "expected_steps": ["log", "multiply"],
    },
    {
        "question": "calculate 101 to the power of 0.5 to 4 digits of precision",
        "answer": round(power(101, 0.5), 4),
        "expected_steps": ["power", "round"],
    },
    {
        "question": (
            "ecoli divides every 20 minutes. How many cells will be "
            "there after 2 hours if we start with 5 cells?"
        ),
        "answer": multiply(5, power(2, divide(120, 20))),
        "expected_steps": ["divide", "power", "multiply"],
    },
    {
        "question": (
            "after calculating the sin of 1.5 radians, divide "
            "the result by cos of 1.5 radians"
        ),
        "answer": divide(sin(1.5), cos(1.5)),
        "expected_steps": ["sin", "cos", "divide"],
    },
    {
        "question": "convert 15 degrees to radians",
        "answer": divide(multiply(15, pi()), 180),
        "expected_steps": ["pi", "multiply", "divide"],
    },
]


def _create_dataset() -> None:
    """Create a dataset with the langsmith client."""
    from langsmith.client import Client

    client = Client()

    dataset = client.create_dataset(
        dataset_name=MULTIVERSE_MATH.name,
        description=MULTIVERSE_MATH.description,
    )

    for example in DATASET:
        client.create_example(
            inputs={
                "question": example["question"],
            },
            outputs={
                "reference": example["answer"],
                "expected_steps": example["expected_steps"],
            },
            dataset_id=dataset.id,
        )
