from chat_langchain.chain import chain
from fastapi import FastAPI
from langserve import add_routes
from openai_functions_agent import agent_executor as openai_functions_agent_chain

app = FastAPI()

# Edit this to add the chain you want to add
add_routes(
    app,
    chain,
    path="/chat",
    # include_callback_events=True, # TODO: Include when fixed
)

add_routes(app, openai_functions_agent_chain, path="/openai-functions-agent")


def run_server(port: int = 1983):
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_server()
