# 🦜💪 LangChain Benchmarks

This repository shows how we benchmark some of our more popular chains and agents.
The benchmarks are organized by end-to-end use cases.
They utilize [LangSmith](https://smith.langchain.com/) heavily.

We have several goals in open sourcing this:

- Showing how we collect our benchmark datasets for each task
- Showing what the benchmark datasets we use for each task is
- Showing how we evaluate each task
- Encouraging others to benchmark their solutions on these tasks (we are always looking for better ways of doing things!)

We currently include the following tasks:
- [CSV Question Answering](csv-qa)
- [Extraction](extraction)
- [Q&A over the LangChain docs](langchain-docs-benchmarking)
- [Meta-evaluation of 'correctness' evaluators](meta-evals)
