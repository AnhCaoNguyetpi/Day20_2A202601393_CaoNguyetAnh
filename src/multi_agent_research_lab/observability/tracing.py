"""Tracing hooks and observability utilities.

Supports LangSmith, Langfuse, and structured in-memory tracing.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# Global trace accumulator for runs
_GLOBAL_TRACES: list[dict[str, Any]] = []


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager for tracing agent and workflow operations.

    Records span metadata, execution duration, inputs/outputs, and optional telemetry to LangSmith.
    """
    settings = get_settings()
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": dict(attributes or {}),
        "status": "running",
        "duration_seconds": None,
        "error": None,
    }

    # LangSmith run context hook if configured
    langsmith_run = None
    if settings.langsmith_api_key:
        try:
            from langsmith import Client

            ls_client = Client(api_key=settings.langsmith_api_key)
            langsmith_run = ls_client.create_run(
                name=name,
                run_type="chain",
                inputs=span["attributes"],
                project_name=settings.langsmith_project,
            )
        except Exception as exc:
            logger.debug("LangSmith trace initialization skipped: %s", exc)

    try:
        yield span
        span["status"] = "completed"
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 4)
        _GLOBAL_TRACES.append(span)

        if langsmith_run is not None:
            try:
                from langsmith import Client

                ls_client = Client(api_key=settings.langsmith_api_key)
                ls_client.update_run(
                    run_id=langsmith_run.id,
                    outputs={"status": span["status"], "duration": span["duration_seconds"]},
                    error=span.get("error"),
                )
            except Exception as exc:
                logger.debug("LangSmith trace update failed: %s", exc)


def get_collected_traces() -> list[dict[str, Any]]:
    """Return all recorded trace spans."""
    return list(_GLOBAL_TRACES)


def clear_collected_traces() -> None:
    """Clear recorded traces."""
    _GLOBAL_TRACES.clear()


def export_traces_json(indent: int = 2) -> str:
    """Export collected trace spans as JSON string."""
    return json.dumps(_GLOBAL_TRACES, indent=indent)
