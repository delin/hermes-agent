"""Strict, surface-neutral Task Fence activation config."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_CONVERSATION_KEY_BYTES = 512


def coerce_task_fence_shadow_conversation_key(value: Any) -> str:
    """Return one exact bounded conversation key or the disabled value."""

    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        logger.warning(
            "Ignoring invalid task_fence.shadow_conversation_key "
            "(expected one exact conversation key, got %s)",
            type(value).__name__,
        )
        return ""
    conversation_key = value.strip()
    try:
        encoded = conversation_key.encode("utf-8")
    except UnicodeEncodeError:
        logger.warning(
            "Ignoring invalid task_fence.shadow_conversation_key "
            "(not valid UTF-8)"
        )
        return ""
    if (
        not conversation_key
        or "\x00" in conversation_key
        or len(encoded) > _MAX_CONVERSATION_KEY_BYTES
    ):
        logger.warning(
            "Ignoring invalid task_fence.shadow_conversation_key "
            "(empty, NUL-containing, or larger than 512 bytes)"
        )
        return ""
    return conversation_key


def resolve_task_fence_shadow_conversation_key(config: Any) -> str:
    """Resolve only top-level ``task_fence.shadow_conversation_key``."""

    if not isinstance(config, dict) or "task_fence" not in config:
        return ""
    task_fence_config = config.get("task_fence")
    if not isinstance(task_fence_config, dict):
        logger.warning(
            "Ignoring invalid task_fence in config.yaml "
            "(expected mapping, got %s)",
            type(task_fence_config).__name__,
        )
        return ""
    return coerce_task_fence_shadow_conversation_key(
        task_fence_config.get("shadow_conversation_key")
    )


def load_task_fence_shadow_conversation_key(
    hermes_home: Path | None = None,
) -> str:
    """Load the exact selector without defaults or environment expansion."""

    try:
        from hermes_cli import managed_scope
        from hermes_cli.config import read_user_config_raw
        from hermes_constants import get_hermes_home

        home = (
            Path(hermes_home)
            if hermes_home is not None
            else get_hermes_home()
        )
        user_config = read_user_config_raw(home / "config.yaml")
        conversation_key = resolve_task_fence_shadow_conversation_key(user_config)

        managed_config = managed_scope.load_managed_config()
        if "task_fence" not in managed_config:
            return conversation_key
        managed_task_fence = managed_config.get("task_fence")
        if not isinstance(managed_task_fence, dict):
            logger.warning(
                "Ignoring invalid managed task_fence "
                "(expected mapping, got %s)",
                type(managed_task_fence).__name__,
            )
            return ""
        if "shadow_conversation_key" not in managed_task_fence:
            return conversation_key
        return coerce_task_fence_shadow_conversation_key(
            managed_task_fence.get("shadow_conversation_key")
        )
    except Exception:
        return ""
