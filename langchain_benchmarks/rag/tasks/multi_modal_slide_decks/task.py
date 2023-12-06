from langchain_benchmarks.schema import RetrievalTask

# ID of public Multi Modal Slide Decks dataset
DATASET_ID = "https://smith.langchain.com/public/40afc8e7-9d7e-44ed-8971-2cae1eb59731/d"

MULTI_MODAL_SLIDE_DECKS_TASK = RetrievalTask(
    name="Multi-modal slide decks",
    dataset_id=DATASET_ID,
    retriever_factories={},
    architecture_factories={},
    get_docs={},
    description=(
        """\
This public dataset is a work-in-progress and will be extended over time.
        
Questions and answers based on slide decks containing visual tables and charts.

Each example is composed of a question and reference answer.

Success is measured based on the accuracy of the answer relative to the reference answer.
"""  # noqa: E501
    ),
)
