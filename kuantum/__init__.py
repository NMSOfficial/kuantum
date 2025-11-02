"""Top-level package for the Kuantum simulation and inference toolkit."""

from importlib import metadata

__all__ = ["__version__"]


def __getattr__(name: str):
    if name == "__version__":
        try:
            return metadata.version("kuantum")
        except metadata.PackageNotFoundError:  # pragma: no cover
            return "0.0.0"
    raise AttributeError(name)
