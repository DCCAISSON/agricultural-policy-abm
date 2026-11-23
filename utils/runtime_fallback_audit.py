from __future__ import annotations

from typing import Any


def record_runtime_fallback(
    model: Any,
    *,
    component: str,
    context: str,
    error: BaseException,
    agent_id: Any = None,
    fallback: Any = None,
    max_events: int = 500,
) -> dict[str, Any] | None:
    """Record a non-fatal runtime fallback on the model for later audit."""
    if model is None:
        return None

    events = getattr(model, "_runtime_fallback_events", None)
    if events is None:
        events = []
        setattr(model, "_runtime_fallback_events", events)

    event = {
        "component": str(component),
        "context": str(context),
        "agent_id": agent_id,
        "error_type": error.__class__.__name__,
        "error": str(error),
        "fallback": fallback,
    }
    events.append(event)

    if max_events > 0 and len(events) > max_events:
        del events[:-max_events]

    return event
