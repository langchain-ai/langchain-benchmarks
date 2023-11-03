from langchain.schema.runnable import RunnableLambda
from langserve import add_routes
from fastapi import FastAPI
import uuid

def foo(uid: uuid.UUID) -> str:
    return f"The id is {uid}"

chain = RunnableLambda(foo)

app = FastAPI()

add_routes(app, chain)

import uvicorn

uvicorn.run(app, port=8122)

