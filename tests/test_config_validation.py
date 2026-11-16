import pytest

from ssa.config_validation import (
    OllamaConfiguration,
    SQLConfiguration,
    validate_ollama_configuration,
    validate_sql_configuration,
)
from ssa.db import ConfigurationError


def test_sql_defaults_are_resolved_without_connecting():
    configuration = validate_sql_configuration({})

    assert configuration == SQLConfiguration(
        server="localhost",
        database="AdventureWorks2025",
        driver="ODBC Driver 18 for SQL Server",
        trusted_connection=True,
        encrypt=True,
        trust_server_certificate=True,
        connection_timeout=10,
        query_timeout=30,
    )


def test_valid_windows_authentication():
    configuration = validate_sql_configuration({"SQL_TRUSTED_CONNECTION": "true"})

    assert configuration.trusted_connection is True


def test_valid_sql_authentication():
    configuration = validate_sql_configuration({
        "SQL_TRUSTED_CONNECTION": "no",
        "SQL_USERNAME": "ssa_user",
        "SQL_PASSWORD": "secret-value",
    })

    assert configuration.trusted_connection is False


@pytest.mark.parametrize("missing", ["SQL_USERNAME", "SQL_PASSWORD"])
def test_sql_authentication_requires_both_credentials(missing):
    settings = {
        "SQL_TRUSTED_CONNECTION": "no",
        "SQL_USERNAME": "ssa_user",
        "SQL_PASSWORD": "secret-value",
    }
    settings[missing] = ""

    with pytest.raises(ConfigurationError, match="SQL_USERNAME and SQL_PASSWORD"):
        validate_sql_configuration(settings)


@pytest.mark.parametrize(
    "name",
    [
        "SQL_TRUSTED_CONNECTION",
        "SQL_ENCRYPT",
        "SQL_TRUST_SERVER_CERTIFICATE",
    ],
)
def test_invalid_sql_boolean_is_rejected(name):
    with pytest.raises(ConfigurationError, match=name):
        validate_sql_configuration({name: "maybe"})


@pytest.mark.parametrize("name", ["SQL_CONNECTION_TIMEOUT", "SQL_QUERY_TIMEOUT"])
def test_invalid_sql_timeout_is_rejected(name):
    with pytest.raises(ConfigurationError, match=name):
        validate_sql_configuration({name: "0"})


@pytest.mark.parametrize("name", ["SQL_SERVER", "SQL_DATABASE", "SQL_DRIVER"])
def test_blank_sql_identifier_is_rejected(name):
    with pytest.raises(ConfigurationError, match=name):
        validate_sql_configuration({name: "   "})


def test_valid_ollama_defaults():
    configuration = validate_ollama_configuration({})

    assert configuration == OllamaConfiguration(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:7b",
        request_timeout=300,
    )


@pytest.mark.parametrize("url", ["http://ollama.local:11434", "https://ollama.example"])
def test_valid_ollama_http_urls(url):
    configuration = validate_ollama_configuration({"OLLAMA_BASE_URL": url})

    assert configuration.base_url == url


@pytest.mark.parametrize(
    "url",
    [
        "ftp://ollama.local",
        "localhost:11434",
        "http://",
        "http:///path",
    ],
)
def test_invalid_ollama_url_is_rejected(url):
    with pytest.raises(ConfigurationError, match="OLLAMA_BASE_URL"):
        validate_ollama_configuration({"OLLAMA_BASE_URL": url})


def test_invalid_ollama_timeout_is_rejected():
    with pytest.raises(ConfigurationError, match="OLLAMA_REQUEST_TIMEOUT"):
        validate_ollama_configuration({"OLLAMA_REQUEST_TIMEOUT": "-1"})


@pytest.mark.parametrize("name", ["OLLAMA_BASE_URL", "OLLAMA_MODEL"])
def test_blank_ollama_values_use_runtime_fallback(name):
    configuration = validate_ollama_configuration({name: ""})

    if name == "OLLAMA_BASE_URL":
        assert configuration.base_url == "http://localhost:11434"
    else:
        assert configuration.model == "qwen2.5-coder:7b"


def test_validation_does_not_contact_external_services(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("external service should not be contacted")

    monkeypatch.setattr("ssa.db.pyodbc.connect", fail)
    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", fail)

    validate_sql_configuration({})
    validate_ollama_configuration({})


def test_validation_errors_do_not_expose_password():
    password = "super-secret-password"

    with pytest.raises(ConfigurationError) as raised:
        validate_sql_configuration({
            "SQL_TRUSTED_CONNECTION": "no",
            "SQL_USERNAME": "ssa_user",
        })

    # The error names the missing settings but never includes a secret value.
    assert password not in str(raised.value)
