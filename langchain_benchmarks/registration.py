"""Registry of environments for ease of access."""

from langchain_benchmarks.extraction import email_task
from langchain_benchmarks.schema import Registry
from langchain_benchmarks.tool_usage import (
    type_writer_26_funcs,
    type_writer,
    relational_data,
    multiverse_math,
)

# Using lower case naming to make a bit prettier API when used in a notebook
registry = Registry(
    tasks=[
        type_writer.TYPE_WRITER_TASK,
        type_writer_26_funcs.TYPE_WRITER_26_FUNCS_TASK,
        relational_data.RELATIONAL_DATA_TASK,
        multiverse_math.MULTIVERSE_MATH,
        email_task.EMAIL_EXTRACTION_TASK,
    ]
)
