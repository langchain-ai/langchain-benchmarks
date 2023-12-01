from typing import Type

from langchain.pydantic_v1 import BaseModel
from pydantic import BaseModel
from pydantic_xml import BaseXmlModel


def create_pydantic_xml_model(model: Type[BaseModel]) -> Type[BaseXmlModel]:
    fields = {
        name: (field.outer_type_, field.default)
        for name, field in model.__fields__.items()
    }
    new_class = type(model.__name__, (BaseXmlModel,), fields)

    return new_class
