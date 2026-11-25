"""Shared, explicit embedding configuration for the SSA schema index."""

from importlib import metadata as importlib_metadata
from typing import Any


SUPPORTED_CHROMA_VERSION = "1.5.9"
SCHEMA_EMBEDDING_IDENTITY = (
    "chromadb=1.5.9;embedding=default;"
    "implementation=onnx_mini_lm_l6_v2;model=all-MiniLM-L6-v2;"
    "expected_artifact_sha256=913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3;"
    "normalize=l2;max_tokens=256;space=l2"
)


class SchemaEmbeddingConfigurationError(RuntimeError):
    """Raised when the supported schema embedding configuration is unavailable."""


def installed_chroma_version() -> str:
    return importlib_metadata.version("chromadb")


def validate_supported_chroma_version(version: str | None = None) -> str:
    actual = installed_chroma_version() if version is None else version
    if actual != SUPPORTED_CHROMA_VERSION:
        raise SchemaEmbeddingConfigurationError(
            f"SSA schema embedding requires chromadb=={SUPPORTED_CHROMA_VERSION}; found {actual}"
        )
    return actual


def schema_embedding_identity() -> str:
    validate_supported_chroma_version()
    return SCHEMA_EMBEDDING_IDENTITY


def create_schema_embedding_function() -> Any:
    """Lazily construct Chroma's explicit default embedding function."""

    validate_supported_chroma_version()
    from chromadb.api.types import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


__all__ = [
    "SCHEMA_EMBEDDING_IDENTITY",
    "SUPPORTED_CHROMA_VERSION",
    "SchemaEmbeddingConfigurationError",
    "create_schema_embedding_function",
    "installed_chroma_version",
    "schema_embedding_identity",
    "validate_supported_chroma_version",
]
