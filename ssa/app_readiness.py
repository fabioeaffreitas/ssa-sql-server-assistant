"""Pure aggregation of the operational checks shown by the Streamlit app."""

from collections.abc import Callable
from dataclasses import dataclass
import logging

from ssa.config_validation import (
    OllamaConfiguration,
    SQLConfiguration,
    validate_ollama_configuration,
    validate_sql_configuration,
)
from ssa.db import ConfigurationError
from ssa.schema_readiness import SchemaIndexStatus, inspect_schema_index


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadinessCheck:
    """Safe, user-facing status for one independent readiness check."""

    name: str
    status: str
    message: str


def collect_readiness(
    *,
    schema_inspector: Callable[[], SchemaIndexStatus] | None = None,
    sql_validator: Callable[[], SQLConfiguration] | None = None,
    ollama_validator: Callable[[], OllamaConfiguration] | None = None,
) -> tuple[ReadinessCheck, ...]:
    """Run each readiness check independently without contacting services."""

    checks: list[ReadinessCheck] = []

    try:
        schema_status = (schema_inspector or inspect_schema_index)()
        if schema_status.available:
            checks.append(ReadinessCheck(
                name="Schema index",
                status="Ready",
                message=f"{schema_status.document_count} documents available.",
            ))
        else:
            checks.append(ReadinessCheck(
                name="Schema index",
                status="Needs attention",
                message=schema_status.message or "Schema index is not ready.",
            ))
    except Exception as exc:
        logger.error(
            "Readiness check failed: component=schema_index error_type=%s",
            type(exc).__name__,
        )
        checks.append(ReadinessCheck(
            name="Schema index",
            status="Inspection error",
            message="Schema index readiness could not be inspected.",
        ))

    try:
        (sql_validator or validate_sql_configuration)()
        checks.append(ReadinessCheck(
            name="SQL Server configuration",
            status="Ready",
            message="Configuration is valid; connectivity is not checked here.",
        ))
    except ConfigurationError as exc:
        checks.append(ReadinessCheck(
            name="SQL Server configuration",
            status="Needs attention",
            message=str(exc),
        ))
    except Exception as exc:
        logger.error(
            "Readiness check failed: component=sql_configuration error_type=%s",
            type(exc).__name__,
        )
        checks.append(ReadinessCheck(
            name="SQL Server configuration",
            status="Inspection error",
            message="SQL Server configuration could not be inspected.",
        ))

    try:
        (ollama_validator or validate_ollama_configuration)()
        checks.append(ReadinessCheck(
            name="Ollama configuration",
            status="Ready",
            message="Configuration is valid; service and model availability are not checked here.",
        ))
    except ConfigurationError as exc:
        checks.append(ReadinessCheck(
            name="Ollama configuration",
            status="Needs attention",
            message=str(exc),
        ))
    except Exception as exc:
        logger.error(
            "Readiness check failed: component=ollama_configuration error_type=%s",
            type(exc).__name__,
        )
        checks.append(ReadinessCheck(
            name="Ollama configuration",
            status="Inspection error",
            message="Ollama configuration could not be inspected.",
        ))

    return tuple(checks)
