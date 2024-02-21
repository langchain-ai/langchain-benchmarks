from importlib import metadata

from langchain_benchmarks.model_registration import model_registry
from langchain_benchmarks.rate_limiting import RateLimiter
from langchain_benchmarks.registration import registry
from langchain_benchmarks.utils._langsmith import (
    clone_public_dataset,
    download_public_dataset,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)


# Please keep this list sorted!
__all__ = [
    "__version__",
    "clone_public_dataset",
    "download_public_dataset",
    "model_registry",
    "RateLimiter",
    "registry",
]
