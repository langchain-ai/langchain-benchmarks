# Testing Agents

This directory contains environments that can be used to test agent's ability
to use tools and make decisions.

## Environments

Environments are named in the style of e[env_number]_[name].py.

### e01_alpha

* Consists of 3 relational tables of users, locations and foods.
* Defines a set of tools that can be used these tables.
* Agent should use the given tools to answer questions.

## Running Evaluation

### Install dependencies

```bash
poetry install
```

### Run evaluation

We'll make this more convenient in the future, but for now, you can run


```python
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function

from environments.e01_alpha import get_tools


def agent_factory() -> AgentExecutor:
    """This creates an OpenAI agent."""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0,
    )
    tools = get_tools()
    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use the given tools to answer the question. Keep in mind that an ID is distinct from a name for every entity."),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("user", "{input}"),
        ]
    )

    runnable_agent = (
        {
            "input": lambda x: x["question"],
            "agent_scratchpad": lambda x: format_to_openai_functions(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    return AgentExecutor(
        agent=runnable_agent,
        tools=tools,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
```

Confirm that the agent works and can use the environment tools:

```python
agent_factory().invoke({'question': "who is bob?"})
```

```python
from langsmith import Client

from environments.e01_alpha import DATASET_ID
from evaluators import STANDARD_AGENT_EVALUATOR

client = Client()

dataset_name = client.read_dataset(dataset_id=DATASET_ID).name

results = client.run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=agent_factory,
    evaluation=STANDARD_AGENT_EVALUATOR,
    verbose=True,
    tags=["openai-agent", "gpt-3.5-turbo-16k"],
)
 ```


### Customize evaluation

Please refer to the following example to see how to set up and run evaluation
for agents using [LangSmith](https://github.com/langchain-ai/langsmith-cookbook/blob/main/testing-examples/agent_steps/evaluating_agents.ipynb).
