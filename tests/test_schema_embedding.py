import pytest

import ssa.schema_embedding as embedding


def test_identity_is_stable_and_versioned(monkeypatch):
    monkeypatch.setattr(embedding, "installed_chroma_version", lambda: "1.5.9")
    identity = embedding.schema_embedding_identity()
    assert "embedding=default" in identity
    assert "expected_artifact_sha256=" in identity
    assert "space=l2" in identity


def test_unsupported_version_fails_closed():
    with pytest.raises(embedding.SchemaEmbeddingConfigurationError):
        embedding.validate_supported_chroma_version("9.9.9")


def test_factory_is_lazy_until_called(monkeypatch):
    calls = []
    monkeypatch.setattr(embedding, "installed_chroma_version", lambda: "1.5.9")
    monkeypatch.setattr(
        "chromadb.api.types.DefaultEmbeddingFunction",
        lambda: calls.append(True) or object(),
    )
    assert calls == []
    embedding.create_schema_embedding_function()
    assert calls == [True]
