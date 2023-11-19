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

from langchain_benchmarks.schema import Environment


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


def get_environment() -> Environment:
    """Create an environment."""
    tools = cast(
        List[BaseTool],
        [tool(func) for func in [multiply, add, divide, subtract, power, log, negate]],
    )
    return Environment(
        tools=tools,
        read_state=None,
    )
