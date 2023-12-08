from typing import List, Sequence

from langchain.tools.base import StructuredTool

from agents.encoder import Parameter, FunctionDefinition


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
    parameters = get_parameters_from_tool(tool)
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters,
        "return_value": {
            "type": "Any",
        },
    }
