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

Please refer to the following example to see how to set up and run evaluation
for agents using [LangSmith](https://github.com/langchain-ai/langsmith-cookbook/blob/main/testing-examples/agent_steps/evaluating_agents.ipynb).
