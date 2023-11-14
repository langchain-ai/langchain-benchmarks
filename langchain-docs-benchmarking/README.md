# Benchmarking on LangChain Docs

This directory contains code to benchmark your cognitive architecture on the public [LangChain Q&A docs evaluation benchmark](https://smith.langchain.com/public/e1bfd348-494a-4df5-899a-7c6c09233cc4/d).

To one one of the existing configurations, activate your poetry environment, configure you LangSmith API key, and run the experiments.

**Note:** this will benchmark chains on a _copy_ of the dataset and will not update the public leaderboard.

### 1. Install requirements

```bash
pip install poetry
poetry shell
poetry install
```

### 2. Configure API keys

Create a [LangSmith account](https://smith.langchain.com/) and set your API key:

```bash
export LANGCHAIN_API_KEY=ls_your-api-key
```

The various cognitive architectures implemented already use Anthropic, [Fireworks.AI](https://www.fireworks.ai/), and OpenAI. Set the required API keys:

```
export OPENAI_API_KEY=your-api-key
export ANTHROPIC_API_KEY=your-api-key
export FIREWORKS_API_KEY=your-api-key
```

### 3. Run Experiments

To run all experiments, run:

```bash
python run_experiments.py
```

If you want to only run certain experiments in the `run_experiments.py` file, use `--include` or `--exclude`

Example:

```bash
python run_experiments --include mistral-7b-instruct-4k llama-v2-34b-code-instruct-w8a16
```

## Evaluating your custom cognitive architecture

You can also evaluate your own custom cognitive architecture. To do so:

1. Create a python file defining your architecture:

```python
# example_custom_chain.py

...
def load_runnable(config: dict) -> "Runnable":
    # Load based on the config provided
    return my_chain
```

2. Call `run_experiments.py` with a custom `--config my_config.json`

```js
{
  // This specifies the path to your custom entrypoint followed by the loader function
  "arch": "path/to/example_custom_chain.py::load_runnable",
  "model_config": {
    // This is passed to load_runnable() in example_custom_chain.py()
    "chat_cls": "ChatOpenAI",
    "model": "gpt-4"
  },
  "project_name": "example-custom-code" // This is the resulting test project name
}
```

We have provided an example in [example_custom_chain.py](./example_custom_chain.py) and the [example_custom_config.json](./example_custom_config.json)

To run using this example, run the following:

```bash
python run_experiments.py --config ./example_custom_config.json
```

Whenever you provide 1 or more `--config` files, the `--include` and `--exclude` arguments are ignored.
