# Correctness Meta-Evals

This folder contains a test script to check the aggregate performance of the "correctness"-related evaluators.

To upload the dataset to LangSmith, run:

```bash
python meta-evals/correctness/_upload_dataset.py
```

To test, run:

```bash
pytest --capture=no meta-evals/correctness/test_correctness_evaluator.py
```

Then navigate to the Web Q&A dataset to review the results.