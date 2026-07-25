"""Identity for tool state that is valid only within active model context.

Some tool results are safe to deduplicate only while the original result remains
available to the model. An AIAgent instance therefore owns a stable random
identity plus a monotonically increasing context epoch. Context surgery
(compression, rewind, reset) advances the epoch; a new AIAgent gets a new
identity even when it temporarily shares a session id with another agent (for
example the background-review fork).
"""

from __future__ import annotations

from typing import Any


def get_tool_context_id(agent: Any) -> str:
    """Return an opaque, process-local key for the agent's active context."""
    instance_id = getattr(agent, "_tool_context_instance_id", None)
    if not instance_id:
        # Defensive fallback for lightweight test doubles or third-party agent
        # subclasses that bypass the normal init path. Real AIAgent instances
        # always receive a random id in agent_init.
        instance_id = f"legacy-{id(agent):x}"

    session_id = getattr(agent, "session_id", None) or ""
    try:
        epoch = max(0, int(getattr(agent, "_tool_context_epoch", 0) or 0))
    except (TypeError, ValueError):
        epoch = 0
    return f"{instance_id}:{session_id}:{epoch}"


def advance_tool_context_epoch(agent: Any) -> int:
    """Start a fresh context epoch and return its numeric generation."""
    try:
        current = max(0, int(getattr(agent, "_tool_context_epoch", 0) or 0))
    except (TypeError, ValueError):
        current = 0
    current += 1
    agent._tool_context_epoch = current
    return current
