from ssa.app_readiness import ReadinessCheck, collect_readiness
from ssa.config_validation import OllamaConfiguration, SQLConfiguration
from ssa.db import ConfigurationError
from ssa.schema_readiness import SchemaIndexStatus


def valid_sql():
    return SQLConfiguration(
        server="localhost",
        database="AdventureWorks2025",
        driver="ODBC Driver 18 for SQL Server",
        trusted_connection=True,
        encrypt=True,
        trust_server_certificate=True,
        connection_timeout=10,
        query_timeout=30,
    )


def valid_ollama():
    return OllamaConfiguration(
        base_url="http://localhost:11434",
        model="qwen2.5-coder:7b",
        request_timeout=300,
    )


def test_all_readiness_checks_valid():
    checks = collect_readiness(
        schema_inspector=lambda: SchemaIndexStatus(True, 12),
        sql_validator=valid_sql,
        ollama_validator=valid_ollama,
    )

    assert checks == (
        ReadinessCheck("Schema index", "Ready", "12 documents available."),
        ReadinessCheck(
            "SQL Server configuration",
            "Ready",
            "Configuration is valid; connectivity is not checked here.",
        ),
        ReadinessCheck(
            "Ollama configuration",
            "Ready",
            "Configuration is valid; service and model availability are not checked here.",
        ),
    )


def test_missing_or_empty_schema_is_needs_attention():
    checks = collect_readiness(
        schema_inspector=lambda: SchemaIndexStatus(
            False,
            0,
            "The schema collection is empty. Run the indexing commands.",
        ),
        sql_validator=valid_sql,
        ollama_validator=valid_ollama,
    )

    assert checks[0].status == "Needs attention"
    assert "Run the indexing commands" in checks[0].message


def test_schema_inspection_exception_isolated_from_other_checks():
    def fail_schema():
        raise RuntimeError("secret local path")

    checks = collect_readiness(
        schema_inspector=fail_schema,
        sql_validator=valid_sql,
        ollama_validator=valid_ollama,
    )

    assert checks[0].status == "Inspection error"
    assert "secret local path" not in checks[0].message
    assert checks[1].status == "Ready"
    assert checks[2].status == "Ready"


def test_invalid_sql_configuration_does_not_hide_other_checks():
    def fail_sql():
        raise ConfigurationError("SQL_PASSWORD is required.")

    checks = collect_readiness(
        schema_inspector=lambda: SchemaIndexStatus(True, 1),
        sql_validator=fail_sql,
        ollama_validator=valid_ollama,
    )

    assert checks[0].status == "Ready"
    assert checks[1].status == "Needs attention"
    assert "password" in checks[1].message.lower()
    assert checks[2].status == "Ready"


def test_invalid_ollama_configuration_does_not_hide_other_checks():
    def fail_ollama():
        raise ConfigurationError("OLLAMA_BASE_URL must be valid.")

    checks = collect_readiness(
        schema_inspector=lambda: SchemaIndexStatus(True, 1),
        sql_validator=valid_sql,
        ollama_validator=fail_ollama,
    )

    assert checks[0].status == "Ready"
    assert checks[1].status == "Ready"
    assert checks[2].status == "Needs attention"
    assert "password" not in checks[2].message


def test_unexpected_configuration_errors_are_safe_and_independent():
    def fail_sql():
        raise RuntimeError("connection string with password")

    def fail_ollama():
        raise RuntimeError("secret endpoint")

    checks = collect_readiness(
        schema_inspector=lambda: SchemaIndexStatus(True, 1),
        sql_validator=fail_sql,
        ollama_validator=fail_ollama,
    )

    assert [check.status for check in checks] == [
        "Ready",
        "Inspection error",
        "Inspection error",
    ]
    assert all("secret" not in check.message for check in checks)
    assert all("password" not in check.message for check in checks)


def test_readiness_uses_only_injected_non_network_checks():
    calls = []

    def schema():
        calls.append("schema")
        return SchemaIndexStatus(True, 1)

    def sql():
        calls.append("sql")
        return valid_sql()

    def ollama():
        calls.append("ollama")
        return valid_ollama()

    checks = collect_readiness(
        schema_inspector=schema,
        sql_validator=sql,
        ollama_validator=ollama,
    )

    assert calls == ["schema", "sql", "ollama"]
    assert all(check.status == "Ready" for check in checks)
