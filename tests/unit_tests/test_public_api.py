from langchain_benchmarks import __all__


def test_public_api() -> None:
    """Test that the public API is correct."""
    # This test will also fail if __all__ is not sorted.
    # Please keep it sorted!
    assert __all__ == sorted(
        [
            "__version__"
            "clone_public_dataset",
            "download_public_dataset",
            "model_registry",
            "RateLimiter",
            "registry",
        ],
        key=lambda x: x.lower(),
    )
