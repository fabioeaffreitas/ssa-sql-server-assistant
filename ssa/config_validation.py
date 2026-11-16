"""Pure validation for SSA SQL Server and Ollama configuration."""

from dataclasses import dataclass
import os
from collections.abc import Mapping
from urllib.parse import urlparse

from ssa.db import ConfigurationError
from ssa.ollama_provider import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_REQUEST_TIMEOUT,
)
from ssa.db import DEFAULT_CONNECTION_TIMEOUT, DEFAULT_QUERY_TIMEOUT


@dataclass(frozen=True)
class SQLConfiguration:
    """Resolved non-secret SQL Server settings."""

    server: str
    database: str
    driver: str
    trusted_connection: bool
    encrypt: bool
    trust_server_certificate: bool
    connection_timeout: int
    query_timeout: int


@dataclass(frozen=True)
class OllamaConfiguration:
    """Resolved Ollama endpoint and generation settings."""

    base_url: str
    model: str
    request_timeout: int


Settings = Mapping[str, str | None]


def _settings_or_environment(settings: Settings | None) -> Settings:
    return os.environ if settings is None else settings


def _setting(settings: Settings, name: str, default: str | None = None) -> str | None:
    value = settings.get(name)
    return default if value is None else value


def _required_setting(settings: Settings, name: str, default: str) -> str:
    value = _setting(settings, name, default)
    if value is None or not value.strip():
        raise ConfigurationError(f"{name} must not be empty.")
    return value.strip()


def _boolean_setting(settings: Settings, name: str, default: str) -> bool:
    value = _setting(settings, name, default)
    normalized = value.strip().lower() if value is not None else ""
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise ConfigurationError(
        f"{name} must be one of yes, no, true, false, 1, or 0."
    )


def _timeout_setting(settings: Settings, name: str, default: int) -> int:
    value = _setting(settings, name, str(default))
    try:
        timeout = int(value) if value is not None else 0
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{name} must be a positive integer."
        ) from exc
    if timeout < 1:
        raise ConfigurationError(f"{name} must be a positive integer.")
    return timeout


def validate_sql_configuration(
    settings: Settings | None = None,
) -> SQLConfiguration:
    """Validate SQL Server settings without opening a connection."""

    values = _settings_or_environment(settings)
    trusted_connection = _boolean_setting(
        values,
        "SQL_TRUSTED_CONNECTION",
        "yes",
    )

    username = values.get("SQL_USERNAME")
    password = values.get("SQL_PASSWORD")
    if not trusted_connection and (
        username is None
        or not username.strip()
        or password is None
        or not password.strip()
    ):
        raise ConfigurationError(
            "SQL_USERNAME and SQL_PASSWORD are required "
            "when SQL_TRUSTED_CONNECTION=no."
        )

    return SQLConfiguration(
        server=_required_setting(values, "SQL_SERVER", "localhost"),
        database=_required_setting(
            values,
            "SQL_DATABASE",
            "AdventureWorks2025",
        ),
        driver=_required_setting(
            values,
            "SQL_DRIVER",
            "ODBC Driver 18 for SQL Server",
        ),
        trusted_connection=trusted_connection,
        encrypt=_boolean_setting(values, "SQL_ENCRYPT", "yes"),
        trust_server_certificate=_boolean_setting(
            values,
            "SQL_TRUST_SERVER_CERTIFICATE",
            "yes",
        ),
        connection_timeout=_timeout_setting(
            values,
            "SQL_CONNECTION_TIMEOUT",
            DEFAULT_CONNECTION_TIMEOUT,
        ),
        query_timeout=_timeout_setting(
            values,
            "SQL_QUERY_TIMEOUT",
            DEFAULT_QUERY_TIMEOUT,
        ),
    )


def validate_ollama_configuration(
    settings: Settings | None = None,
) -> OllamaConfiguration:
    """Validate Ollama settings without contacting the Ollama service."""

    values = _settings_or_environment(settings)
    base_url = _setting(values, "OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL) or DEFAULT_OLLAMA_BASE_URL
    parsed = urlparse(base_url)
    try:
        hostname = parsed.hostname
    except ValueError as exc:
        raise ConfigurationError(
            "OLLAMA_BASE_URL must be a valid HTTP or HTTPS URL with a hostname."
        ) from exc
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ConfigurationError(
            "OLLAMA_BASE_URL must be a valid HTTP or HTTPS URL with a hostname."
        )

    model = _setting(values, "OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL) or DEFAULT_OLLAMA_MODEL

    return OllamaConfiguration(
        base_url=base_url,
        model=model,
        request_timeout=_timeout_setting(
            values,
            "OLLAMA_REQUEST_TIMEOUT",
            DEFAULT_OLLAMA_REQUEST_TIMEOUT,
        ),
    )
