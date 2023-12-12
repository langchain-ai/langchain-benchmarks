import inspect
from textwrap import dedent
from typing import List

from langchain.tools.base import StructuredTool

from agents.encoder import FunctionDefinition, Parameter


# This is temporary until we have a better way to represent parameters
def get_parameters_from_tool(tool: StructuredTool) -> List[Parameter]:
    """Convert a langchain tool to a tool user tool."""
    schema = tool.args_schema.schema()

    properties = schema["properties"]
    parameters = []
    # Is this needed or is string OK?
    type_adapter = {
        "string": "str",  # str or string?
        "integer": "int",
        "number": "float",
        "boolean": "bool",
    }
    for key, value in properties.items():
        parameters.append(
            {
                "name": key,
                "type": type_adapter.get(value["type"], value["type"]),
                "description": value.get("description", ""),
            }
        )

    return parameters


#
def convert_tool_to_function_definition(tool: StructuredTool) -> FunctionDefinition:
    """Convert a langchain tool to a tool user tool."""
    # Here we re-inspect the underlying function to get the doc-string
    # since StructuredTool modifies it, but we want the raw one for maximum
    # flexibility.
    description = inspect.getdoc(tool.func)

    parameters = get_parameters_from_tool(tool)
    return {
        "name": tool.name,
        "description": dedent(description),
        "parameters": parameters,
        "return_value": {
            "type": "Any",
        },
    }
