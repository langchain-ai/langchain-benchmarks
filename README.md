🚧 Under Active Development 🚧

# 🦜💪 LangChain Benchmarks

[![Release Notes](https://img.shields.io/github/release/langchain-ai/langchain-benchmarks)](https://github.com/langchain-ai/langchain-benchmarks/releases)
[![CI](https://github.com/langchain-ai/langchain-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/langchain-ai/langchain-benchmarks/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.gg/6adMQxSpJS)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langchain-benchmarks)](https://github.com/langchain-ai/langchain-benchmarks/issues)


[📖 Documentation](https://langchain-ai.github.io/langchain-benchmarks/index.html)

A package to help benchmark various LLM related tasks.

The benchmarks are organized by end-to-end use cases, and
utilize [LangSmith](https://smith.langchain.com/) heavily.

We have several goals in open sourcing this:

- Showing how we collect our benchmark datasets for each task
- Showing what the benchmark datasets we use for each task is
- Showing how we evaluate each task
- Encouraging others to benchmark their solutions on these tasks (we are always looking for better ways of doing things!)

## Installation

To install the packages, run the following command:

```bash
pip install -U langchain-benchmarks
```

All the benchmarks come with an associated benchmark dataset stored in [LangSmith](https://smith.langchain.com). To take advantage of the eval and debugging experience, [sign up](https://smith.langchain.com), and set your API key in your environment:

```bash
export LANGCHAIN_API_KEY=sk-...
```

## Repo Structure

The package is located within [langchain_benchmarks](./langchain_benchmarks/). Check out the [docs](https://langchain-ai.github.io/langchain-benchmarks/index.html) for information on how to get starte.

The other directories are legacy and may be moved in the future.


## Archived

Below are archived benchmarks that require cloning this repo to run.

- [CSV Question Answering](https://github.com/langchain-ai/langchain-benchmarks/tree/main/csv-qa)
- [Extraction](https://github.com/langchain-ai/langchain-benchmarks/tree/main/extraction)
- [Q&A over the LangChain docs](https://github.com/langchain-ai/langchain-benchmarks/tree/main/langchain-docs-benchmarking)
- [Meta-evaluation of 'correctness' evaluators](https://github.com/langchain-ai/langchain-benchmarks/tree/main/meta-evals)


## Related

- For cookbooks on other ways to test, debug, monitor, and improve your LLM applications, check out the [LangSmith docs](https://docs.smith.langchain.com/)
- For information on building with LangChain, check out the [python documentation](https://python.langchain.com/docs/get_started/introduction) or [JS documentation](https://js.langchain.com/docs/get_started/introduction)

