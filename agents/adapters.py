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


def format_as_xml_tool(
    name: str, description: str, parameters: Sequence[Parameter]
) -> str:
    """Format a tool as XML."""
    parameters_as_strings = [
        "<parameter>\n"
        f"<name>{parameter['name']}</name>\n"
        f"<type>{parameter['type']}</type>\n"
        f"<description>{parameter['description']}</description>\n"
        "</parameter>\n"
        for parameter in parameters
    ]
    tool = (
        "<tool>\n"
        f"<tool_name>{name}</tool_name>\n"
        "<description>\n"
        f"{description}\n"
        "</description>\n"
        "<parameters>\n"
        f"{''.join(parameters_as_strings)}"  # Already includes trailing newline
        "</parameters>\n"
        "</tool>"
    )
    return tool


def format_structured_tool_as_xml(tool: StructuredTool) -> str:
    """Format a StructuredTool as XML."""
    parameters = get_parameters_from_tool(tool)
    return format_as_xml_tool(tool.name, tool.description, parameters)


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
