"""Private audit-only capability for bounded gateway delivery handoffs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
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
    TaskFencePolicy,
    bind_causal_envelope,
    bind_task_fence_policy,
)


logger = logging.getLogger(__name__)

TASK_FENCE_DELIVERY_CAPABILITY_ATTR = "_task_fence_delivery_capability"
_SLACK_CHAT_POST_MESSAGE_ADAPTER = "gateway:slack:chat_post_message"


@dataclass(frozen=True)
class TaskFenceDeliveryCapability:
    """One accepted turn's non-dispatchable final-delivery provenance."""

    parent: CausalEnvelope | None


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


def _audit_slack_post_message_start(
    policy: TaskFencePolicy,
    child: CausalEnvelope | None,
    invocation_id: str,
    team_id: str | None,
    request: Mapping[str, Any],
) -> str | None:
    operation = OperationDescriptor(
        invocation_id=invocation_id,
        kind=OperationKind.DELIVERY,
        adapter=_SLACK_CHAT_POST_MESSAGE_ADAPTER,
        invocation_fingerprint=_slack_post_message_fingerprint(
            team_id=team_id,
            request=request,
        ),
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


async def task_fence_slack_post_message_handoff(
    *,
    capability: TaskFenceDeliveryCapability | None,
    runner: Any,
    team_id: str | None,
    request: Mapping[str, Any],
    post_message: Callable[..., Awaitable[Any]],
) -> Any:
    """Audit and invoke one physical chat.postMessage without gating it."""

    if capability is None:
        with bind_task_fence_policy(None):
            return await post_message(**dict(request))

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
        policy = TaskFencePolicy(store)
        attempt_id = await asyncio.to_thread(
            _audit_slack_post_message_start,
            policy,
            child,
            invocation_id,
            team_id,
            dict(request),
        )
    except Exception as exc:
        logger.warning(
            "Task Fence shadow delivery observation failed for %s: %s",
            _SLACK_CHAT_POST_MESSAGE_ADAPTER,
            type(exc).__name__,
        )

    # The captured parent stays local to Slack.send so its next chunk/retry
    # derives a sibling. Third-party SDK code receives no live policy/store
    # or dispatchable delivery capability.
    token = _CURRENT_DELIVERY_CAPABILITY.set(None)
    try:
        with bind_causal_envelope(child), bind_task_fence_policy(None):
            result = await post_message(**dict(request))
        if policy is not None and attempt_id is not None:
            evidence_reference = _slack_post_message_acknowledgement_ref(
                team_id=team_id,
                request=request,
                result=result,
            )
            if evidence_reference is None:
                logger.warning(
                    "Task Fence shadow delivery acknowledgement missing for %s",
                    _SLACK_CHAT_POST_MESSAGE_ADAPTER,
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
                        _SLACK_CHAT_POST_MESSAGE_ADAPTER,
                        type(exc).__name__,
                    )
        return result
    finally:
        _CURRENT_DELIVERY_CAPABILITY.reset(token)
