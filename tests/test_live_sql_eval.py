from scripts.run_live_sql_eval import timing_columns


def test_evaluation_timing_columns_are_generic_and_optional():
    response = {
        "diagnostics": {
            "timings_ms": {
                "retrieval_ms": 1.5,
                "generation_ms": 250.0,
                "validation_ms": None,
                "execution_ms": None,
                "total_ms": 251.5,
            }
        }
    }

    assert timing_columns(response) == {
        "retrieval_ms": 1.5,
        "generation_ms": 250.0,
        "validation_ms": None,
        "execution_ms": None,
        "total_ms": 251.5,
    }


def test_evaluation_timing_columns_default_to_none():
    assert timing_columns({}) == {
        "retrieval_ms": None,
        "generation_ms": None,
        "validation_ms": None,
        "execution_ms": None,
        "total_ms": None,
    }
