
import streamlit as st
import pandas as pd
import os

from ssa.assistant import ask
from ssa.app_readiness import collect_readiness
from ssa.app_ordering_advisory import ordering_advisory_message
from ssa.logging_config import configure_logging


configure_logging()

st.set_page_config(
    page_title="SSA - SQL Server AI Assistant",
    page_icon="🗄️",
    layout="wide",
)

st.title("SQL Server AI Assistant")
st.caption("Natural language to SQL Server queries — powered by local AI.")

st.subheader("System readiness")
readiness_columns = st.columns(3)
for column, check in zip(readiness_columns, collect_readiness()):
    with column:
        if check.status == "Ready":
            st.success(f"{check.name}: Ready")
        elif check.status == "Inspection error":
            st.error(f"{check.name}: Inspection error")
        else:
            st.warning(f"{check.name}: Needs attention")
        st.caption(check.message)
st.caption(
    "These checks validate local configuration and schema-index availability. "
    "They do not verify SQL Server or Ollama connectivity."
)


def initialise_token_tracker() -> None:
    """Set up session-scoped token accounting for the Streamlit app."""

    st.session_state.setdefault("token_total", 0)
    st.session_state.setdefault("prompt_token_total", 0)
    st.session_state.setdefault("completion_token_total", 0)
    st.session_state.setdefault("token_history", [])
    st.session_state.setdefault(
        "active_model", os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    )


def record_token_usage(usage: dict, question: str) -> None:
    """Store one Ollama generation's reported usage in this browser session."""

    total = int(usage.get("total_tokens") or 0)
    st.session_state.active_model = usage.get("model") or st.session_state.active_model
    if total <= 0:
        return

    st.session_state.token_total += total
    st.session_state.prompt_token_total += int(usage.get("prompt_tokens") or 0)
    st.session_state.completion_token_total += int(usage.get("completion_tokens") or 0)
    st.session_state.token_history.insert(0, {
        "Question": question,
        "Tokens": total,
        "Input": int(usage.get("prompt_tokens") or 0),
        "Output": int(usage.get("completion_tokens") or 0),
    })


def render_token_tracker() -> None:
    model = st.session_state.active_model

    with st.sidebar:
        st.subheader("Token usage")
        budget = st.number_input(
            "Session token budget",
            min_value=1_000,
            max_value=10_000_000,
            value=100_000,
            step=1_000,
            help="A tracking limit for this browser session; it does not change Ollama.",
        )
        used = st.session_state.token_total
        percentage = min(used / budget, 1.0)
        st.progress(percentage, text=f"{used:,} / {budget:,} tokens ({used / budget:.1%})")
        st.caption(f"Model: `{model}`")
        st.caption(
            f"Input: {st.session_state.prompt_token_total:,} · "
            f"Output: {st.session_state.completion_token_total:,}"
        )

        if st.button("Reset token tracker", width="stretch"):
            for key in (
                "token_total", "prompt_token_total", "completion_token_total", "token_history"
            ):
                st.session_state[key] = [] if key == "token_history" else 0
            st.rerun()

        if st.session_state.token_history:
            with st.expander("Request history", expanded=False):
                st.dataframe(
                    pd.DataFrame(st.session_state.token_history),
                    hide_index=True,
                    width="stretch",
                )


initialise_token_tracker()
render_token_tracker()

question = st.text_area(
    "Ask a question about AdventureWorks2025",
    placeholder="Example: Show the top 10 customers by total sales",
    height=100,
)

if st.button("Ask SSA", type="primary", disabled=not question.strip()):
    with st.spinner("Generating, validating, and executing SQL..."):
        try:
            response = ask(question.strip())

            generation = response.get("generation") or {}
            usage = generation.get("usage") or {}
            if usage:
                record_token_usage(usage, question.strip())
                st.toast(
                    f"Used {usage.get('total_tokens', 0):,} tokens "
                    f"with {usage.get('model', os.getenv('OLLAMA_MODEL', 'local model'))}",
                    icon="📊",
                )
            validation = response.get("validation")
            execution = response.get("execution") or {}
            result = response.get("result") or {}
            raw_diagnostics = response.get("diagnostics")
            diagnostics = (
                raw_diagnostics if isinstance(raw_diagnostics, dict) else {}
            )
            raw_timings = diagnostics.get("timings_ms")
            timings = raw_timings if isinstance(raw_timings, dict) else {}

            if diagnostics:
                with st.expander("Diagnostics", expanded=False):
                    st.caption("Pipeline timings")
                    timing_labels = {
                        "retrieval_ms": "Retrieval",
                        "generation_ms": "Generation",
                        "validation_ms": "Validation",
                        "execution_ms": "Execution",
                        "total_ms": "Total",
                    }
                    for key, label in timing_labels.items():
                        value = timings.get(key)
                        display_value = (
                            "Not reached" if value is None else f"{value:.2f} ms"
                        )
                        st.write(f"{label}: {display_value}")
                    if diagnostics.get("failure_stage"):
                        st.caption(
                            f"Failure stage: {diagnostics['failure_stage']}"
                        )

            st.subheader("Generated SQL")

            sql = response.get("sql")
            if sql:
                st.code(sql, language="sql")
            else:
                st.info("SQL was not generated.")

            if not generation.get("success"):
                st.error(response.get("error") or "SQL generation failed.")
                st.stop()

            if validation is None:
                st.warning(response.get("error") or "SQL validation was not completed.")
                st.stop()

            if validation.get("success"):
                st.success("SQL validation successful")
            else:
                st.error(response.get("error") or "SQL validation failed.")
                st.stop()

            columns = []
            rows = []
            if execution.get("success"):
                st.success("Query executed successfully")

                ordering_message = ordering_advisory_message(
                    response.get("diagnostics")
                )
                if ordering_message is not None:
                    st.info(ordering_message)

                columns = result.get("columns", [])
                rows = result.get("rows", [])

            if columns:
                df = pd.DataFrame(rows, columns=columns)
                st.subheader("Results")         

                column_config = {}          

                for column in df.columns:
                    if any(term in column.lower() for term in [
                        "sales", "amount", "totaldue", "subtotal", "price", "cost"
                    ]):
                        column_config[column] = st.column_config.NumberColumn(
                            column,
                            format="$%,.2f",
                        )           

                st.dataframe(
                    df,
                    width="stretch",
                    column_config=column_config,
                )

                st.caption(
                    f"Rows returned: {result.get('row_count', len(rows))}"
                )

                if result.get("truncated"):
                    st.warning(
                        "Results limited to the configured maximum row count."
                    )
            else:
                st.error(response.get("error") or "SQL execution failed.")

            if not response.get("success"):
                st.warning("SSA did not complete the request successfully.")

        except Exception as exc:
            st.error("SSA could not complete the request. Check the application logs.")
