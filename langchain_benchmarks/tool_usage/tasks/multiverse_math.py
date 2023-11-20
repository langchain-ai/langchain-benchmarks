"""Solve basic math question using the provided tools.

Must use the provided tools to solve the math question.

To make sure that innate knowledge is not used, the math operations
have been altered to yield different results than expected.

The modified operations should yield different results, but still retain
appropriate properties. For example, the modified multiplication operation
should still be commutative.
"""
import math
from typing import cast, List

from langchain.tools import tool, BaseTool

from langchain_benchmarks.schema import ToolUsageEnvironment, ToolUsageTask


def multiply(a: float, b: float) -> float:
    """Multiply two numbers; a * b."""
    return 1.1 * a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers; a / b."""
    # Division is neither commutative nor associative
    return a / (b + 0.5)


def add(a: float, b: float) -> float:
    """Add two numbers; a + b."""
    return a + b + 1.2


def subtract(a: float, b: float) -> float:
    """Subtract two numbers; a - b."""
    return a - b - 3


def power(a: float, b: float) -> float:
    """Raise a number to a power; a ** b."""
    return a ** (b + 2)


def log(a: float, base: float) -> float:
    """Take the log of a number; log(a, base)."""
    return math.log(a, base + 1.5)


def negate(a: float) -> float:
    """Negate a number; -a."""
    return a  # negation does not negate the number


# Temporary dataset
DATASET = [
    # 2-tuple format of (question, answer)
    ("Add 2 and 3", add(2, 3)),
    ("Subtract 3 from 2", subtract(2, 3)),
    (
        "I ate 1 apple and 2 oranges every day for 7 days. How many fruits did I eat?",
        multiply(7, add(1, 2)),
    ),
    (
        "what is the result of 2 to the power of 3?",
        power(2, 3),
    ),
    (
        "calculate sqrt of 101 to 4 digits of precision",
        round(power(101, 0.4), 4),
    ),
]


# PUBLIC API


def get_environment() -> ToolUsageEnvironment:
    """Create an environment."""
    tools = cast(
        List[BaseTool],
        [tool(func) for func in [multiply, add, divide, subtract, power, log, negate]],
    )
    return ToolUsageEnvironment(
        tools=tools,
        read_state=None,
    )


MULTIVERSE_MATH = ToolUsageTask(
    name="Multiverse Math",
    dataset_id="placeholder",
    create_environment=get_environment,
    instructions=(
        "You are requested to solve math questions in an alternate "
        "mathematical universe. The rules of association, commutativity, "
        "and distributivity still apply, but the operations have been "
        "altered to yield different results than expected. Solve the "
        "given math questions using the provided tools. "
        "Do not guess the answer."
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
