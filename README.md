🚧 Under Active Development 🚧

# 🦜💪 LangChain Benchmarks

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
pip install -U langchain_benchmarks
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
