"""Prototyping code for rendering function definitions, invocations, and results.

Types are simplified for now to `str`.

We should actually support something like pydantic or jsonschema for the types, so
we can expand them recursively for nested types.
"""
import abc
from typing import Any, List, Optional

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
    @abc.abstractmethod
    def visit_function_definition(self, function_definition: FunctionDefinition) -> str:
        """Render a function."""

    @abc.abstractmethod
    def visit_function_definitions(
        self, function_definitions: List[FunctionDefinition]
    ) -> str:
        """Render a function."""

    @abc.abstractmethod
    def visit_function_invocation(self, function_invocation: FunctionInvocation) -> str:
        """Render a function invocation."""

    @abc.abstractmethod
    def visit_function_result(self, function_result: FunctionResult) -> str:
        """Render a function result."""


class AstPrinter(Visitor):
    """Print the AST."""


class XMLEncoder(AstPrinter):
    def visit_function_definition(self, function_definition: FunctionDefinition) -> str:
        """Render a function."""
        parameters_lines = []

        for parameter in function_definition["parameters"]:
            parameters_lines.extend(
                [
                    "<parameter>",
                    f"<name>{parameter['name']}</name>",
                    f"<type>{parameter['type']}</type>",
                    f"<description>{parameter['description']}</description>",
                    "</parameter>",
                ]
            )
        lines = [
            "<function>",
            f"<function_name>{function_definition['name']}</function_name>",
            "<description>",
            f"{function_definition['description']}",
            "</description>",
            "<parameters>",
            *parameters_lines,
            "</parameters>",
            "<return_value>",
            f"<type>{function_definition['return_value']['type']}</type>",
        ]
        if function_definition["return_value"].get("description"):
            lines.append(
                f"<description>{function_definition['return_value']['description']}"
                f"</description>"
            )

        lines.extend(["</return_value>", "</function>"])
        return "\n".join(lines)

    def visit_function_definitions(
        self, function_definitions: List[FunctionDefinition]
    ) -> str:
        """Render a function."""
        strs = [
            self.visit_function_definition(function_definition)
            for function_definition in function_definitions
        ]
        return "<functions>\n" + "\n".join(strs) + "\n</functions>"

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

        lines.append(f"<function_name>{function_result['name']}</function_name>")

        if function_result["error"]:
            lines.extend(
                [
                    f"<error>{function_result['error']}</error>",
                ]
            )
        else:
            lines.append(
                f"<result>{function_result['result']}</result>",
            )

        lines.append("</function_result>")

        return "\n".join(lines)


class AnthropicXMLEncoder(AstPrinter):
    """Adapter for Anthropic tool usage api.

    As described here: https://github.com/anthropics/anthropic-tools/tree/main
    """

    def visit_function_definition(self, function_definition: FunctionDefinition) -> str:
        """Render a function.

        Function definition example:

        <tool_description>
        <tool_name>get_time_of_day</tool_name>
        <description>
        get_time_of_day(time_zone: str) -> str - Retrieve the current time of day

            Args:
                time_zone: The time zone to get the current time for,

            Returns:
                time format
        </description>
        <parameters>
        <parameter>
        <name>time_zone</name>
        <type>str</type>
        <description></description>
        </parameter>
        </parameters>
        </tool_description>
        """
        parameters_lines = []

        for parameter in function_definition["parameters"]:
            parameters_lines.extend(
                [
                    "<parameter>",
                    f"<name>{parameter['name']}</name>",
                    f"<type>{parameter['type']}</type>",
                    f"<description>{parameter['description']}</description>",
                    "</parameter>",
                ]
            )
        lines = [
            "<tool_description>",
            f"<tool_name>{function_definition['name']}</tool_name>",
            "<description>",
            f"{function_definition['description']}",
            "</description>",
            "<parameters>",
            *parameters_lines,
            "</parameters>",
            "</tool_description>",
        ]
        return "\n".join(lines)

    def visit_function_definitions(
        self, function_definitions: List[FunctionDefinition]
    ) -> str:
        """Render a function."""
        strs = [
            self.visit_function_definition(function_definition)
            for function_definition in function_definitions
        ]

        lines = [
            "<tools>",
            *strs,
            "</tools>",
        ]
        return "\n".join(lines)

    def visit_function_invocation(self, invocation: FunctionInvocation) -> str:
        """Render a function invocation.

        <invoke>
        <tool_name>get_time_of_day</tool_name>
        <parameters>
        <time_zone>UTC</time_zone>
        </parameters>
        </invoke>
        """
        arguments_as_strings = [
            f"<{argument['name']}>{argument['value']}</{argument['name']}>"
            for argument in invocation["arguments"]
        ]
        lines = [
            "<invoke>",
            f"<tool_name>{invocation['name']}</tool_name>",
            "<parameters>",
            *arguments_as_strings,
            "</parameters>",
            "</invoke>",
        ]
        return "\n".join(lines)

    def visit_function_invocations(self, invocations: List[FunctionInvocation]) -> str:
        """Render a function invocation."""
        strs = [
            self.visit_function_invocation(invocation) for invocation in invocations
        ]

        lines = [
            "<function_calls>",
            *strs,
            "</function_calls>",
        ]
        return "\n".join(lines)

    def visit_function_result(self, function_result: FunctionResult) -> str:
        """Render a function result.

        <function_results>
        <result>
        <tool_name>get_time_of_day</tool_name>
        <stdout>
        02:57:27
        </stdout>
        </result>
        </function_results>
        """
        lines = [
            "<result>",
            f"<tool_name>{function_result['name']}</tool_name>",
            f"<stdout>{function_result['result']}</stdout>",
            "</result>",
        ]
        return "\n".join(lines)

    def visit_function_results(self, function_results: List[FunctionResult]) -> str:
        """Render a function result."""
        strs = [
            self.visit_function_result(function_result)
            for function_result in function_results
        ]

        lines = [
            "<function_results>",
            *strs,
            "</function_results>",
        ]
        return "\n".join(lines)


class TypeScriptEncoder(AstPrinter):
    def visit_function_definition(self, function_definition: FunctionDefinition) -> str:
        """Render a function."""
        parameters_as_strings = [
            f"{parameter['name']}: {parameter['type']}"
            for parameter in function_definition["parameters"]
        ]
        # Let's use JSdoc style comments
        # First the function description
        lines = [
            f"// {function_definition['description']}",
            # Then the parameter descriptions
            *[
                f"// @param {parameter['name']} {parameter['description']}"
                for parameter in function_definition["parameters"]
            ],
            # Then the return value description
            f"// @returns {function_definition['return_value']['description']}",
            # Then the function definition
            f"function {function_definition['name']}("
            f"{', '.join(parameters_as_strings)}): "
            f"{function_definition['return_value']['type']};",
        ]

        # finally join
        function = "\n".join(lines)
        return function

    def visit_function_definitions(
        self, function_definitions: List[FunctionDefinition]
    ) -> str:
        """Render a function."""
        strs = [
            self.visit_function_definition(function_definition)
            for function_definition in function_definitions
        ]
        return "\n\n".join(strs)

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
