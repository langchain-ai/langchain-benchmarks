AGENT_INSTRUCTIONS_XML_FORMAT = """\
In this environment you have access to a set of tools you can use to answer the user's question.

You may call them like this:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>

Here are the tools available:

{tool_description}
"""  # noqa: E501

AGENT_INSTRUCTIONS_BLOB_STYLE = """\
In this environment you have access to a set of tools you can use to answer the user's question.

Here are the tools available:

{tool_description}

You may call one tool at a time using a format that includes <tool> and </tool> tag. 

Inside the tag the content is a python dictionary that uses python literals (e.g., numbers, strings, lists, dictionaries, etc.) to specify the tool invocation.

It must match the schema of the function as described in the tool description.
"arguments" is a dictionary of the arguments to the function.

<tool>
{{
    "tool_name": $TOOL_NAME,
    "arguments": $ARGUMENTS
}}
</tool>

If you do not know the answer use more tools. You can only take a single action at a time.\
"""  # noqa: E501
