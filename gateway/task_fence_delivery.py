"""Private audit-only capability for bounded gateway delivery handoffs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator, Mapping

from task_fence import (
    AttemptTerminal,
    CausalEnvelope,
    DecisionOutcome,
    OperationDescriptor,
    OperationKind,
    TaskFenceCapabilityKind,
    TaskFenceLaunchRoute,
    TaskFencePolicy,
    bind_causal_envelope,
    bind_task_fence_policy,
)


logger = logging.getLogger(__name__)

TASK_FENCE_DELIVERY_CAPABILITY_ATTR = "_task_fence_delivery_capability"
_SLACK_CHAT_POST_MESSAGE_ADAPTER = "gateway:slack:chat_post_message"
_TELEGRAM_SEND_MESSAGE_ADAPTER = "gateway:telegram:send_message"
_TASK_FENCE_SLACK_CHAT_POST_MESSAGE_ROUTE = TaskFenceLaunchRoute(
    kind=TaskFenceCapabilityKind.ADAPTER,
    route_id="gateway:slack:chat_post_message",
    capability_version="task-fence-capability-v4",
)
_TELEGRAM_SEND_MESSAGE_REQUEST_KEYS = frozenset(
    {
        "chat_id",
        "text",
        "parse_mode",
        "reply_to_message_id",
        "message_thread_id",
        "direct_messages_topic_id",
        "link_preview_options",
        "disable_web_page_preview",
        "disable_notification",
    }
)


@dataclass(frozen=True)
class TaskFenceDeliveryCapability:
    """One accepted turn's non-dispatchable final-delivery provenance."""

    parent: CausalEnvelope | None
    delivery_source: str
    conversation_id: str


_CURRENT_DELIVERY_CAPABILITY: ContextVar[TaskFenceDeliveryCapability | None] = (
    ContextVar("task_fence_delivery_capability", default=None)
)


def take_task_fence_delivery_capability(
    event: Any,
) -> TaskFenceDeliveryCapability | None:
    """Consume the dynamic event carrier before any delivery branching."""

    capability = getattr(event, TASK_FENCE_DELIVERY_CAPABILITY_ATTR, None)
    try:
        delattr(event, TASK_FENCE_DELIVERY_CAPABILITY_ATTR)
    except AttributeError:
        pass
    return capability if isinstance(capability, TaskFenceDeliveryCapability) else None


@contextmanager
def bind_task_fence_delivery_capability(
    capability: TaskFenceDeliveryCapability,
) -> Iterator[None]:
    """Scope one capability to the ordinary final-response send chain."""

    if not isinstance(capability, TaskFenceDeliveryCapability):
        raise TypeError("invalid Task Fence delivery capability")
    token = _CURRENT_DELIVERY_CAPABILITY.set(capability)
    try:
        yield
    finally:
        _CURRENT_DELIVERY_CAPABILITY.reset(token)


@contextmanager
def isolate_task_fence_delivery_capability(
) -> Iterator[TaskFenceDeliveryCapability | None]:
    """Capture provenance while clearing it from an adapter lifecycle."""

    capability = _CURRENT_DELIVERY_CAPABILITY.get()
    token = _CURRENT_DELIVERY_CAPABILITY.set(None)
    try:
        yield capability
    finally:
        _CURRENT_DELIVERY_CAPABILITY.reset(token)


@contextmanager
def suspend_task_fence_delivery_capability() -> Iterator[None]:
    """Keep generic fallback sends outside the bounded shadow cohort."""

    with isolate_task_fence_delivery_capability():
        yield


def _slack_post_message_fingerprint(
    *,
    team_id: str | None,
    request: Mapping[str, Any],
) -> str:
    """Commit to one exact Slack post without retaining its payload."""

    encoder = json.JSONEncoder(
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    digest = hashlib.sha256()
    digest.update(b"task-fence-delivery-wire-v1\0")
    value = {
        "adapter": _SLACK_CHAT_POST_MESSAGE_ADAPTER,
        "request": dict(request),
        "route": {
            "method": "chat.postMessage",
            "team_id": team_id,
        },
    }
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()


def _audit_delivery_start(
    policy: TaskFencePolicy,
    child: CausalEnvelope | None,
    invocation_id: str,
    adapter: str,
    invocation_fingerprint: str,
) -> str | None:
    operation = OperationDescriptor(
        invocation_id=invocation_id,
        kind=OperationKind.DELIVERY,
        adapter=adapter,
        invocation_fingerprint=invocation_fingerprint,
    )
    admitted = policy.admit_operation(child, operation)
    if (
        admitted.outcome is DecisionOutcome.WOULD_RESERVE
        and admitted.permit_id is not None
    ):
        started = policy.authorize_and_start(
            child,
            operation,
            admitted.permit_id,
        )
        return started.attempt_id
    return None


def _slack_post_message_acknowledgement_ref(
    *,
    team_id: str | None,
    request: Mapping[str, Any],
    result: Any,
) -> str | None:
    """Return a bounded opaque commitment to a concrete Slack message ID."""

    try:
        channel = request.get("channel")
        message_ts = result.get("ts")
    except Exception:
        return None
    if (
        not isinstance(channel, str)
        or not channel
        or not isinstance(message_ts, str)
        or not message_ts
        or len(channel) > 256
        or len(message_ts) > 64
        or (team_id is not None and (not isinstance(team_id, str) or not team_id))
        or (isinstance(team_id, str) and len(team_id) > 256)
    ):
        return None
    seconds, separator, fraction = message_ts.partition(".")
    if (
        separator != "."
        or not seconds.isascii()
        or not fraction.isascii()
        or not seconds.isdigit()
        or not fraction.isdigit()
    ):
        return None
    acknowledgement = json.dumps(
        {
            "channel": channel,
            "team_id": team_id,
            "ts": message_ts,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8", errors="surrogatepass")
    if len(acknowledgement) > 1_024:
        return None
    return (
        "slack:chat_post_message:ack:sha256:"
        f"{hashlib.sha256(acknowledgement).hexdigest()}"
    )


def _telegram_json_projection(value: Any) -> Any:
    """Project the closed Bot.send_message request into canonical JSON."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite Telegram request value")
        return value
    if isinstance(value, (list, tuple)):
        return [_telegram_json_projection(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("non-string Telegram request key")
        return {
            key: _telegram_json_projection(item)
            for key, item in value.items()
        }
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _telegram_json_projection(to_dict())
    raise TypeError("unsupported Telegram request value")


def _telegram_send_message_fingerprint(
    *,
    bot_id: Any,
    request: Mapping[str, Any],
) -> str:
    """Commit to one exact Bot.send_message call without retaining payload."""

    unknown_keys = set(request) - _TELEGRAM_SEND_MESSAGE_REQUEST_KEYS
    if unknown_keys:
        raise TypeError("unsupported Telegram send_message request key")
    chat_id = request.get("chat_id")
    if (
        not isinstance(bot_id, int)
        or isinstance(bot_id, bool)
        or bot_id <= 0
        or not isinstance(chat_id, int)
        or isinstance(chat_id, bool)
        or chat_id <= 0
    ):
        raise TypeError("invalid Telegram delivery identity")
    value = {
        "adapter": _TELEGRAM_SEND_MESSAGE_ADAPTER,
        "request": _telegram_json_projection(dict(request)),
        "route": {
            "bot_id": bot_id,
            "method": "sendMessage",
        },
    }
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(
        b"task-fence-delivery-wire-v1\0" + payload
    ).hexdigest()


def _telegram_send_message_acknowledgement_ref(
    *,
    bot_id: Any,
    request: Mapping[str, Any],
    result: Any,
) -> str | None:
    """Return a bounded commitment to one exact Telegram Message ACK."""

    try:
        requested_chat_id = request.get("chat_id")
        returned_chat_id = result.chat.id
        message_id = result.message_id
    except Exception:
        return None
    identities = (bot_id, requested_chat_id, returned_chat_id, message_id)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in identities
    ) or returned_chat_id != requested_chat_id:
        return None
    acknowledgement = json.dumps(
        {
            "bot_id": bot_id,
            "chat_id": requested_chat_id,
            "message_id": message_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8", errors="surrogatepass")
    return (
        "telegram:send_message:ack:sha256:"
        f"{hashlib.sha256(acknowledgement).hexdigest()}"
    )


async def _task_fence_delivery_handoff(
    *,
    capability: TaskFenceDeliveryCapability | None,
    delivery_source: str,
    adapter: str,
    runner: Any,
    launch_route: TaskFenceLaunchRoute | None,
    invocation_fingerprint: Callable[[], str],
    acknowledgement_ref: Callable[[Any], str | None],
    physical_call: Callable[[], Awaitable[Any]],
) -> Any:
    """Audit one exact physical delivery call without gating legacy behavior."""

    if capability is None or capability.delivery_source != delivery_source:
        with bind_task_fence_policy(None):
            return await physical_call()

    child = None
    invocation_id = f"tfiv_{uuid.uuid4().hex}"
    policy = None
    attempt_id = None
    try:
        if capability.parent is not None:
            child = capability.parent.for_invocation()
            invocation_id = child.invocation_id or invocation_id
        session_db = getattr(runner, "_session_db", None)
        store = getattr(session_db, "_db", session_db)
        if store is None:
            raise RuntimeError("missing Task Fence delivery store")
        if launch_route is not None:
            launch_catalog = getattr(runner, "_task_fence_launch_catalog", None)
            if launch_catalog is None:
                logger.warning(
                    "Task Fence shadow delivery launch route would block for %s "
                    "(%s): launch_catalog_unavailable",
                    adapter,
                    launch_route.route_id,
                )
                policy = None
            else:
                launch_validation = launch_catalog.classify_route(launch_route)
                if not launch_validation.verified:
                    logger.warning(
                        "Task Fence shadow delivery launch route would block for %s "
                        "(%s): %s",
                        adapter,
                        launch_validation.route_id,
                        launch_validation.reason,
                    )
                    policy = None
                else:
                    policy = TaskFencePolicy(store)
        else:
            policy = TaskFencePolicy(store)
        if policy is not None:
            fingerprint = invocation_fingerprint()
            attempt_id = await asyncio.to_thread(
                _audit_delivery_start,
                policy,
                child,
                invocation_id,
                adapter,
                fingerprint,
            )
    except Exception as exc:
        logger.warning(
            "Task Fence shadow delivery observation failed for %s: %s",
            adapter,
            type(exc).__name__,
        )

    # The captured parent stays local to the adapter so each physical retry or
    # chunk derives a sibling. Third-party SDK code receives no live policy,
    # store, or dispatchable delivery capability.
    token = _CURRENT_DELIVERY_CAPABILITY.set(None)
    try:
        with bind_causal_envelope(child), bind_task_fence_policy(None):
            result = await physical_call()
        if policy is not None and attempt_id is not None:
            try:
                evidence_reference = acknowledgement_ref(result)
            except Exception:
                evidence_reference = None
            if evidence_reference is None:
                logger.warning(
                    "Task Fence shadow delivery acknowledgement missing for %s",
                    adapter,
                )
            else:
                try:
                    # A post-ack await would add a cancellation point that can
                    # turn a confirmed legacy send into a failed/retried send.
                    policy.finish_attempt(
                        attempt_id,
                        AttemptTerminal.SUCCEEDED,
                        evidence_reference,
                    )
                except Exception as exc:
                    logger.warning(
                        "Task Fence shadow delivery terminal observation failed "
                        "for %s: %s",
                        adapter,
                        type(exc).__name__,
                    )
        return result
    finally:
        _CURRENT_DELIVERY_CAPABILITY.reset(token)


async def task_fence_slack_post_message_handoff(
    *,
    capability: TaskFenceDeliveryCapability | None,
    runner: Any,
    team_id: str | None,
    request: Mapping[str, Any],
    post_message: Callable[..., Awaitable[Any]],
) -> Any:
    """Audit and invoke one physical chat.postMessage without gating it."""

    request_copy = dict(request)
    return await _task_fence_delivery_handoff(
        capability=capability,
        delivery_source="gateway:slack",
        adapter=_SLACK_CHAT_POST_MESSAGE_ADAPTER,
        runner=runner,
        launch_route=_TASK_FENCE_SLACK_CHAT_POST_MESSAGE_ROUTE,
        invocation_fingerprint=lambda: _slack_post_message_fingerprint(
            team_id=team_id,
            request=request_copy,
        ),
        acknowledgement_ref=lambda result: (
            _slack_post_message_acknowledgement_ref(
                team_id=team_id,
                request=request_copy,
                result=result,
            )
        ),
        physical_call=lambda: post_message(**request_copy),
    )


async def task_fence_telegram_send_message_handoff(
    *,
    capability: TaskFenceDeliveryCapability | None,
    runner: Any,
    bot: Any,
    request: Mapping[str, Any],
    send_message: Callable[..., Awaitable[Any]],
) -> Any:
    """Audit and invoke one physical Bot.send_message without gating it."""

    request_copy = dict(request)
    return await _task_fence_delivery_handoff(
        capability=capability,
        delivery_source="gateway:telegram",
        adapter=_TELEGRAM_SEND_MESSAGE_ADAPTER,
        runner=runner,
        launch_route=None,
        invocation_fingerprint=lambda: _telegram_send_message_fingerprint(
            bot_id=getattr(bot, "id", None),
            request=request_copy,
        ),
        acknowledgement_ref=lambda result: (
            _telegram_send_message_acknowledgement_ref(
                bot_id=getattr(bot, "id", None),
                request=request_copy,
                result=result,
            )
        ),
        physical_call=lambda: send_message(**request_copy),
    )
