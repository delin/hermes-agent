"""Private audit-only Task Fence seam for in-process provider handoffs."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping


logger = logging.getLogger(__name__)

_LOCAL_ONLY_REQUEST_FIELDS = frozenset({
    "timeout",
    "_moa_prepared_request",
})


def _normalize_model_wire_value(
    value: Any,
    *,
    path: tuple[tuple[str, Any], ...],
    binary_frames: list[dict[str, Any]],
) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        binary = {
            "__task_fence_binary__": {
                "length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
        binary_frames.append({
            "path": [list(segment) for segment in path],
            **binary["__task_fence_binary__"],
        })
        return binary
    if isinstance(value, Mapping):
        return {
            key: _normalize_model_wire_value(
                value[key],
                path=(*path, ("key", key)),
                binary_frames=binary_frames,
            )
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize_model_wire_value(
                item,
                path=(*path, ("index", index)),
                binary_frames=binary_frames,
            )
            for index, item in enumerate(value)
        ]
    return value


@contextmanager
def _without_task_fence_model_authority() -> Iterator[None]:
    from task_fence import bind_task_fence_policy

    with bind_task_fence_policy(None):
        yield


def model_wire_fingerprint(
    *,
    adapter: str,
    request: Mapping[str, Any],
    route: Mapping[str, Any],
) -> str:
    """Commit to one final provider request without retaining its payload."""

    if not isinstance(request, Mapping) or not isinstance(route, Mapping):
        raise TypeError("model wire request and route must be mappings")
    provider_request = {
        key: value
        for key, value in request.items()
        if key not in _LOCAL_ONLY_REQUEST_FIELDS
    }
    binary_frames: list[dict[str, Any]] = []
    canonical_value = _normalize_model_wire_value(
        {
            "adapter": adapter,
            "request": provider_request,
            "route": dict(route),
        },
        path=(),
        binary_frames=binary_frames,
    )
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    digest = hashlib.sha256()
    digest.update(b"task-fence-model-wire-v1\0")
    for chunk in encoder.iterencode(canonical_value):
        digest.update(chunk.encode("utf-8", errors="surrogatepass"))
    if binary_frames:
        # The sidecar binds each binary value to its typed structural path.
        # A user-supplied JSON object that resembles the display marker
        # therefore cannot collide with actual provider bytes.
        digest.update(b"\0task-fence-model-wire-binary-v1\0")
        for frame in binary_frames:
            for chunk in encoder.iterencode(frame):
                digest.update(chunk.encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()


def _audit_model_start(
    *,
    adapter: str,
    request: Mapping[str, Any],
    route: Mapping[str, Any],
    policy: Any,
    envelope: Any | None,
) -> str | None:
    from task_fence import DecisionOutcome, OperationDescriptor, OperationKind

    invocation_id = (
        envelope.invocation_id
        if envelope is not None and envelope.invocation_id is not None
        else f"tfiv_{uuid.uuid4().hex}"
    )
    operation = OperationDescriptor(
        invocation_id=invocation_id,
        kind=OperationKind.MODEL,
        adapter=adapter,
        invocation_fingerprint=model_wire_fingerprint(
            adapter=adapter,
            request=request,
            route=route,
        ),
    )
    admitted = policy.admit_operation(envelope, operation)
    if (
        admitted.outcome is DecisionOutcome.WOULD_RESERVE
        and admitted.permit_id is not None
    ):
        started = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )
        return started.attempt_id
    return None


@contextmanager
def task_fence_model_handoff(
    *,
    adapter: str,
    request: Mapping[str, Any],
    route: Mapping[str, Any],
    policy: Any | None,
) -> Iterator[str | None]:
    """Observe one physical provider handoff without gating legacy dispatch."""

    from task_fence import (
        bind_causal_envelope,
        current_causal_envelope,
    )

    try:
        envelope = current_causal_envelope()
        if policy is not None and envelope is not None:
            envelope = envelope.for_invocation()
    except Exception as exc:
        logger.warning(
            "Task Fence shadow model context failed for %s: %s",
            adapter,
            type(exc).__name__,
        )
        with _without_task_fence_model_authority():
            yield None
        return

    if policy is None:
        with _without_task_fence_model_authority():
            yield None
        return

    with bind_causal_envelope(envelope):
        attempt_id = None
        try:
            attempt_id = _audit_model_start(
                adapter=adapter,
                request=request,
                route=route,
                policy=policy,
                envelope=envelope,
            )
        except Exception as exc:
            logger.warning(
                "Task Fence shadow model observation failed for %s: %s",
                adapter,
                type(exc).__name__,
            )
        # Third-party SDK hooks may execute arbitrary in-process code. Keep
        # the immutable child envelope for causal diagnostics, but never lend
        # the live policy/store facade to code beyond the audit boundary.
        with _without_task_fence_model_authority():
            yield attempt_id


def _finish_task_fence_openai_chat_completion(
    *,
    policy: Any | None,
    attempt_holder: Any,
    response: Any,
) -> None:
    """Record a bounded acknowledgement for one ordinary non-stream return."""

    if policy is None or type(attempt_holder) is not dict:
        return
    attempt_id = attempt_holder.get("attempt_id")
    if type(attempt_id) is not str or not attempt_id:
        return

    try:
        choices = getattr(response, "choices", None)
        if type(choices) is not list or not choices:
            return
        if getattr(response, "error", None):
            return
        response_id = getattr(response, "id", None)
        if (
            type(response_id) is not str
            or len(response_id) > 512
            or not response_id.strip()
            or "\0" in response_id
        ):
            return
        encoded_id = response_id.encode("utf-8")
        if len(encoded_id) > 512:
            return

        from task_fence import AttemptTerminal

        digest = hashlib.sha256()
        digest.update(b"task-fence-openai-chat-completion-response-id-v1\0")
        digest.update(encoded_id)
        policy.finish_attempt(
            attempt_id,
            AttemptTerminal.SUCCEEDED,
            "openai:chat_completions:response:v1:sha256:" + digest.hexdigest(),
        )
    except Exception as exc:
        logger.warning(
            "Task Fence shadow model terminal observation failed: %s",
            type(exc).__name__,
        )


@contextmanager
def task_fence_model_stream_handoff(
    *,
    adapter: str,
    request: Mapping[str, Any],
    route: Mapping[str, Any],
    policy: Any | None,
    open_stream: Callable[[], Any],
) -> Iterator[Any]:
    """Audit the lazy stream open, then keep provider lifecycle unprivileged."""

    with task_fence_model_handoff(
        adapter=adapter,
        request=request,
        route=route,
        policy=policy,
    ):
        with open_stream() as stream:
            yield stream
