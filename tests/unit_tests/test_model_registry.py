import pytest

from langchain_benchmarks.schema import RegisteredModel, ModelRegistry

# Create some sample RegisteredModel instances for testing
SAMPLE_MODELS = [
    RegisteredModel(
        "model1", "fireworks", "Description 1", {"param1": "value1"}, "chat"
    ),
    RegisteredModel("model2", "openai", "Description 2", {"param2": "value2"}, "llm"),
]


@pytest.fixture
def sample_registry() -> ModelRegistry:
    return ModelRegistry(SAMPLE_MODELS)


def test_init() -> None:
    # Test the constructor of ModelRegistry
    registry = ModelRegistry(SAMPLE_MODELS)
    assert len(registry.registered_models) == 2


def test_get_model(sample_registry: ModelRegistry) -> None:
    # Test the get_model method
    model = sample_registry.get_model("model1")
    assert model.name == "model1"


def test_filter(sample_registry: ModelRegistry) -> None:
    # Test the filter method
    filtered_registry = sample_registry.filter(type="chat")
    assert len(filtered_registry.registered_models) == 1
    assert filtered_registry.registered_models[0].type == "chat"


def test_repr_html(sample_registry: ModelRegistry) -> None:
    # Test the _repr_html_ method
    html_representation = sample_registry._repr_html_()
    assert "<table>" in html_representation


def test_len(sample_registry: ModelRegistry) -> None:
    # Test the __len__ method
    assert len(sample_registry) == 2


def test_iter(sample_registry: ModelRegistry) -> None:
    # Test the __iter__ method
    models = list(iter(sample_registry))
    assert len(models) == 2
    assert isinstance(models[0], RegisteredModel)


def test_getitem(sample_registry: ModelRegistry) -> None:
    # Test the __getitem__ method for integer and string keys
    model = sample_registry[0]
    assert model.name == "model1"
    model = sample_registry["model2"]
    assert model.name == "model2"


def test_getitem_slice(sample_registry: ModelRegistry) -> None:
    # Test the __getitem__ method for slices
    sliced_registry = sample_registry[:1]
    assert len(sliced_registry.registered_models) == 1
    assert sliced_registry.registered_models[0].name == "model1"
