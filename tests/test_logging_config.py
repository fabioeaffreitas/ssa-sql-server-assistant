import logging

import pytest

from ssa.logging_config import configure_logging


@pytest.fixture
def clean_ssa_logger():
    logger = logging.getLogger("ssa")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    yield logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    for handler in original_handlers:
        logger.addHandler(handler)
    logger.setLevel(original_level)
    logger.propagate = original_propagate


def test_default_level_is_info(monkeypatch, clean_ssa_logger):
    monkeypatch.delenv("SSA_LOG_LEVEL", raising=False)

    logger = configure_logging()

    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_supported_levels(configured, expected, clean_ssa_logger):
    assert configure_logging(configured).level == expected


def test_invalid_level_falls_back_to_info(clean_ssa_logger):
    assert configure_logging("not-a-level").level == logging.INFO


def test_repeated_configuration_does_not_add_duplicate_handlers(clean_ssa_logger):
    first = configure_logging("INFO")
    second = configure_logging("DEBUG")

    assert first is second
    assert len(second.handlers) == 1
    assert second.level == logging.DEBUG


def test_configuration_does_not_change_root_or_third_party_logger(
    clean_ssa_logger,
):
    root = logging.getLogger()
    third_party = logging.getLogger("third_party.test")
    original_root_level = root.level
    original_third_party_level = third_party.level

    configure_logging("DEBUG")

    assert root.level == original_root_level
    assert third_party.level == original_third_party_level


def test_configuration_has_no_file_handler(clean_ssa_logger):
    configure_logging()

    assert all(not isinstance(handler, logging.FileHandler) for handler in clean_ssa_logger.handlers)


def test_sensitive_values_are_not_in_safe_application_messages(
    clean_ssa_logger,
    caplog,
):
    configure_logging("INFO")
    capture = logging.Handler()
    records = []
    capture.emit = records.append
    clean_ssa_logger.addHandler(capture)

    question = "show secret customer rows"
    sql = "SELECT PasswordHash FROM dbo.Users"
    rows = "{'PasswordHash': 'secret'}"
    clean_ssa_logger.info("Query execution completed: rows=%d", 1)

    rendered = " ".join(record.getMessage() for record in records)
    assert question not in rendered
    assert sql not in rendered
    assert rows not in rendered
    clean_ssa_logger.removeHandler(capture)
