"""Prototyping code for rendering function definitions, invocations, and results.

Types are simplified for now to `str`.

We should actually support something like pydantic or jsonschema for the types, so
we can expand them recursively for nested types.
"""
import abc
from typing import Optional, List, Any

from typing_extensions import NotRequired, TypedDict


class Parameter(TypedDict):
    """Representation for a parameter."""

    name: str
    type: str
    description: str


class Arguments(TypedDict):
    """Arguments are passed to a function during function invocation."""

    name: Optional[str]
    value: Any


class ReturnValue(TypedDict):
    """Representation for a return value of a function call."""

    type: str
    description: NotRequired[str]


class FunctionDefinition(TypedDict):
    """Representation for a function."""

    name: str
    description: str  # Function description
    parameters: List[Parameter]
    return_value: ReturnValue


class FunctionInvocation(TypedDict):
    """Representation for a function invocation."""

    id: NotRequired[str]
    name: str
    arguments: List[Arguments]


class FunctionResult(TypedDict):
    """Representation for a function result."""

    id: NotRequired[str]
    name: str
    result: Optional[str]
    error: Optional[str]


class Visitor(abc.ABC):
    def visit_function_definition(self, function_definition: FunctionDefinition) -> str:
        """Render a function."""
        raise NotImplementedError()

    def visit_function_invocation(self, function_invocation: FunctionInvocation) -> str:
        """Render a function invocation."""
        raise NotImplementedError()

    def visit_function_result(self, function_result: FunctionResult) -> str:
        """Render a function result."""
        raise NotImplementedError()


class XMLEncoder(Visitor):
    def visit_function_definition(self, function_definition: FunctionDefinition) -> str:
        """Render a function."""
        parameters_as_strings = [
            "<parameter>\n"
            f"<name>{parameter['name']}</name>\n"
            f"<type>{parameter['type']}</type>\n"
            f"<description>{parameter['description']}</description>\n"
            "</parameter>\n"
            for parameter in function_definition["parameters"]
        ]
        function = (
            "<function>\n"
            f"<function_name>{function_definition['name']}</function_name>\n"
            "<description>\n"
            f"{function_definition['description']}\n"
            "</description>\n"
            "<parameters>\n"
            f"{''.join(parameters_as_strings)}"  # Already includes trailing newline
            "</parameters>\n"
            "<return_value>\n"
            f"<type>{function_definition['return_value']['type']}</type>\n"
            f"<description>{function_definition['return_value']['description']}</description>\n"
            "</return_value>\n"
            "</function>"
        )
        return function

    def visit_function_invocation(self, invocation: FunctionInvocation) -> str:
        """Render a function invocation."""
        arguments_as_strings = [
            "<argument>\n"
            f"<name>{argument['name']}</name>\n"
            f"<value>{argument['value']}</value>\n"
            "</argument>\n"
            for argument in invocation["arguments"]
        ]
        lines = ["<function_invocation>"]

        if invocation.get("id"):
            lines.append(f"<id>{invocation['id']}</id>")

        lines.extend(
            [
                f"<function_name>{invocation['name']}</function_name>\n"
                "<arguments>\n"
                f"{''.join(arguments_as_strings)}"  # Already includes trailing newline
                "</arguments>\n"
                "</function_invocation>"
            ]
        )
        return "\n".join(lines)

    def visit_function_result(self, function_result: FunctionResult) -> str:
        """Render a function result."""
        lines = [
            "<function_result>",
        ]

        if function_result.get("id"):
            lines.append(f"<id>{function_result['id']}</id>")

        lines.extend(
            [
                f"<function_name>{function_result['name']}</function_name>",
                f"<result>{function_result['result']}</result>",
                f"<error>{function_result['error']}</error>",
                "</function_result>",
            ]
        )

        return "\n".join(lines)


class TypeScriptEncoder(Visitor):
    def visit_function_definition(self, function_definition: FunctionDefinition) -> str:
        """Render a function."""
        parameters_as_strings = [
            f"{parameter['name']}: {parameter['type']}"
            for parameter in function_definition["parameters"]
        ]
        function = (
            f"// {function_definition['description']}\n"
            f"function { function_definition['name']}("
            f"{', '.join(parameters_as_strings)}): "
            f"{function_definition['return_value']['type']};"
        )
        return function

    def visit_function_invocation(self, invocation: FunctionInvocation) -> str:
        """Render a function invocation."""
        arguments_as_strings = [
            f"{argument['name']}: {argument['value']}"
            for argument in invocation["arguments"]
        ]
        lines = [f"{invocation['name']}(" f"{', '.join(arguments_as_strings)});"]
        return "\n".join(lines)

    def visit_function_result(self, function_result: FunctionResult) -> str:
        """Render a function result."""
        lines = []
        if function_result["error"]:
            lines.append(f"ERROR: {function_result['error']}")
        else:
            lines.append(f"> {function_result['result']}")
        if function_result.get("id"):
            lines.append(f"// ID: {function_result['id']}")
        return "\n".join(lines)
