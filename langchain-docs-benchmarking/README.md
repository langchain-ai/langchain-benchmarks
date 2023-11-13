# Benchmarking on LangChain Docs


This directory contains code to benchmark your cognitive architecture on the public [LangChain Q&A docs evaluation benchmark](https://smith.langchain.com/public/e1bfd348-494a-4df5-899a-7c6c09233cc4/d).

To one one of the existing configurations, activate your poetry environment, configure you LangSmith API key, and run the experiments.

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


## Evaluating your chain


To evaluate your own chain, create a python file with defining the chain definition and include a constructor function.
Then create a config json file with "arch" specifying the filename (with the constructor function following the two colons), any optional configuration to pass the constructor in `model_config`, and the project name to assign to the
resulting test projects whenever this is run. The evluation script will automatically append a short uuid to the project name to permit multiple tests with the same experiment file.

An example is below.

```
{
    # Example custom package
    "arch": "packages/chat-langchain/chat_langchain/chain.py::create_chain",
    "model_config": {
        "chat_cls": "ChatOpenAI",
        "model": "gpt-4",
    },
    "project_name": "example-custom-code",
}
```

The cognitive architectures defined in this directory follow the [LangChain Templates](https://github.com/langchain-ai/langchain/blob/master/templates/README.md) format. 