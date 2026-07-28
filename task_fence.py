"""Typed ingress contracts for the Task Fence control protocol.

This module defines data that may cross the trusted ingress boundary. It does
not authorize dispatch and deliberately contains no prompt, transcript, tool,
or result payload fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


CONTROL_PROTOCOL_VERSION = 1

_MAX_SOURCE_BYTES = 256
_MAX_IDENTIFIER_BYTES = 512
_MAX_OPAQUE_REFERENCE_BYTES = 2_048
_MAX_CORRELATION_IDS = 64
_MAX_EVIDENCE_REFS = 32
_MAX_SQLITE_INTEGER = 2**63 - 1
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


class Origin(str, Enum):
    HUMAN = "human"
    RUNTIME = "runtime"


class IngressClass(str, Enum):
    TASK_INPUT = "task_input"
    CONTROL = "control"
    ADVISORY = "advisory"
    SYNTHETIC = "synthetic"


class IntentEffect(str, Enum):
    KEEP = "keep"
    REPLACE = "replace"


class ExecutionEffect(str, Enum):
    NONE = "none"
    HOLD = "hold"
    RUN = "run"
    TERMINATE = "terminate"


class InputEffect(str, Enum):
    NONE = "none"
    APPEND = "append"
    DISCARD_SELECTED = "discard_selected"


class CorrelationKind(str, Enum):
    NONE = "none"
    OPEN_QUESTION = "open_question"
    PENDING_INPUTS = "pending_inputs"
    INCIDENT_ATTEMPTS = "incident_attempts"


class ResolutionDisposition(str, Enum):
    CONFIRMED_SUCCESS = "confirmed_success"
    CONFIRMED_FAILURE = "confirmed_failure"
    ACCEPTED_UNKNOWN_NO_RETRY = "accepted_unknown_no_retry"


class TerminalReason(str, Enum):
    STOPPED = "stopped"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TaskFenceAction:
    """One member of the closed, versioned ingress union.

    UI action names are intentionally not part of this value. For example,
    initial submit and change-and-run share one authoritative wire shape.
    """

    origin: Origin
    ingress_class: IngressClass
    intent: IntentEffect
    execution: ExecutionEffect
    input_effect: InputEffect
    correlation_kind: CorrelationKind = CorrelationKind.NONE


def action_shape(action: TaskFenceAction) -> tuple[object, ...]:
    return (
        action.ingress_class,
        action.intent,
        action.execution,
        action.input_effect,
        action.correlation_kind,
        action.origin,
    )


_ACTION_DEFINITIONS = {
    "initial_submit": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.TASK_INPUT,
        IntentEffect.REPLACE,
        ExecutionEffect.RUN,
        InputEffect.APPEND,
    ),
    "change_and_hold": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.TASK_INPUT,
        IntentEffect.REPLACE,
        ExecutionEffect.HOLD,
        InputEffect.APPEND,
    ),
    "change_and_run": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.TASK_INPUT,
        IntentEffect.REPLACE,
        ExecutionEffect.RUN,
        InputEffect.APPEND,
    ),
    "comment_hold": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.TASK_INPUT,
        IntentEffect.KEEP,
        ExecutionEffect.HOLD,
        InputEffect.APPEND,
    ),
    "answer_only": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.TASK_INPUT,
        IntentEffect.KEEP,
        ExecutionEffect.HOLD,
        InputEffect.APPEND,
        CorrelationKind.OPEN_QUESTION,
    ),
    "answer_and_resume": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.TASK_INPUT,
        IntentEffect.KEEP,
        ExecutionEffect.RUN,
        InputEffect.APPEND,
        CorrelationKind.OPEN_QUESTION,
    ),
    "pause": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.CONTROL,
        IntentEffect.KEEP,
        ExecutionEffect.HOLD,
        InputEffect.NONE,
    ),
    "resume": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.CONTROL,
        IntentEffect.KEEP,
        ExecutionEffect.RUN,
        InputEffect.NONE,
    ),
    "stop": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.CONTROL,
        IntentEffect.KEEP,
        ExecutionEffect.TERMINATE,
        InputEffect.NONE,
    ),
    "discard_pending": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.CONTROL,
        IntentEffect.KEEP,
        ExecutionEffect.HOLD,
        InputEffect.DISCARD_SELECTED,
        CorrelationKind.PENDING_INPUTS,
    ),
    "resolve_incident": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.CONTROL,
        IntentEffect.KEEP,
        ExecutionEffect.HOLD,
        InputEffect.NONE,
        CorrelationKind.INCIDENT_ATTEMPTS,
    ),
    "explicit_note": TaskFenceAction(
        Origin.HUMAN,
        IngressClass.ADVISORY,
        IntentEffect.KEEP,
        ExecutionEffect.NONE,
        InputEffect.NONE,
    ),
    "synthetic_notice": TaskFenceAction(
        Origin.RUNTIME,
        IngressClass.SYNTHETIC,
        IntentEffect.KEEP,
        ExecutionEffect.NONE,
        InputEffect.NONE,
    ),
}

TASK_FENCE_ACTIONS: Mapping[str, TaskFenceAction] = MappingProxyType(
    _ACTION_DEFINITIONS
)
_ALLOWED_ACTION_SHAPES = frozenset(
    action_shape(action) for action in TASK_FENCE_ACTIONS.values()
)


class TaskFenceProtocolRejected(ValueError):
    """A typed ingress value does not satisfy the closed protocol."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class TaskFenceIngressRejected(TaskFenceProtocolRejected):
    """The durable store rejected an otherwise typed ingress envelope."""

    def __init__(self, reason: str, *, incident_id: str | None = None):
        self.incident_id = incident_id
        super().__init__(reason)


class TaskFenceIngressUnavailable(RuntimeError):
    """Shadow acceptance could not use a compatible writable control store."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def validate_action(action: TaskFenceAction) -> None:
    if not isinstance(action, TaskFenceAction):
        raise TaskFenceProtocolRejected("invalid_action_type")
    if not (
        isinstance(action.origin, Origin)
        and isinstance(action.ingress_class, IngressClass)
        and isinstance(action.intent, IntentEffect)
        and isinstance(action.execution, ExecutionEffect)
        and isinstance(action.input_effect, InputEffect)
        and isinstance(action.correlation_kind, CorrelationKind)
    ):
        raise TaskFenceProtocolRejected("invalid_action_fields")
    if action_shape(action) not in _ALLOWED_ACTION_SHAPES:
        raise TaskFenceProtocolRejected("unsupported_ingress_tuple")


def _bounded_text(
    value: object,
    *,
    field: str,
    max_bytes: int,
    optional: bool = False,
) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TaskFenceProtocolRejected(f"invalid_{field}")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TaskFenceProtocolRejected(f"invalid_{field}") from exc
    if len(encoded) > max_bytes:
        raise TaskFenceProtocolRejected(f"{field}_too_large")


def _canonical_refs(
    values: tuple[str, ...],
    *,
    field: str,
    max_items: int,
) -> tuple[str, ...]:
    if not isinstance(values, tuple) or len(values) > max_items:
        raise TaskFenceProtocolRejected(f"invalid_{field}")
    for value in values:
        _bounded_text(
            value,
            field=field,
            max_bytes=_MAX_IDENTIFIER_BYTES,
        )
    if len(set(values)) != len(values):
        raise TaskFenceProtocolRejected(f"duplicate_{field}")
    return tuple(sorted(values))


@dataclass(frozen=True)
class IngressEnvelope:
    """Immutable, secret-free input to the durable acceptance transaction."""

    source: str
    source_event_id: str
    conversation_id: str
    action: TaskFenceAction
    payload_hash: str | None = None
    opaque_payload_ref: str | None = None
    task_id: str | None = None
    protocol_version: int = CONTROL_PROTOCOL_VERSION
    correlation_ids: tuple[str, ...] = ()
    resolution_disposition: ResolutionDisposition | None = None
    evidence_refs: tuple[str, ...] = ()
    terminal_reason: TerminalReason | None = None
    source_sequence: int | None = None
    causal_parent_generation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "correlation_ids",
            _canonical_refs(
                self.correlation_ids,
                field="correlation_ids",
                max_items=_MAX_CORRELATION_IDS,
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _canonical_refs(
                self.evidence_refs,
                field="evidence_refs",
                max_items=_MAX_EVIDENCE_REFS,
            ),
        )
        validate_ingress_envelope(self)


@dataclass(frozen=True)
class TaskFenceIngressSidecar:
    """Secret-free typed authority attached by a trusted ingress adapter.

    ``active_lane_action`` lets an adapter state both closed protocol actions
    for a plain message without consulting mutable runtime state itself.  The
    durable writer selects between them under the same SQLite write lock that
    accepts the event.  Redelivery reuses the already-recorded action before
    checking the full envelope fingerprint.
    """

    source: str
    source_event_id: str
    action: TaskFenceAction
    payload_hash: str | None = None
    opaque_payload_ref: str | None = None
    active_lane_action: TaskFenceAction | None = None
    protocol_version: int = CONTROL_PROTOCOL_VERSION
    correlation_ids: tuple[str, ...] = ()
    resolution_disposition: ResolutionDisposition | None = None
    evidence_refs: tuple[str, ...] = ()
    terminal_reason: TerminalReason | None = None
    source_sequence: int | None = None
    causal_parent_generation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "correlation_ids",
            _canonical_refs(
                self.correlation_ids,
                field="correlation_ids",
                max_items=_MAX_CORRELATION_IDS,
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _canonical_refs(
                self.evidence_refs,
                field="evidence_refs",
                max_items=_MAX_EVIDENCE_REFS,
            ),
        )
        for candidate in (self.action, self.active_lane_action):
            if candidate is None:
                continue
            self.to_envelope(
                conversation_id="task-fence-sidecar-validation",
                action=candidate,
            )

    def to_envelope(
        self,
        *,
        conversation_id: str,
        action: TaskFenceAction | None = None,
    ) -> IngressEnvelope:
        """Bind adapter authority to one canonical gateway conversation."""

        return IngressEnvelope(
            source=self.source,
            source_event_id=self.source_event_id,
            conversation_id=conversation_id,
            action=self.action if action is None else action,
            payload_hash=self.payload_hash,
            opaque_payload_ref=self.opaque_payload_ref,
            protocol_version=self.protocol_version,
            correlation_ids=self.correlation_ids,
            resolution_disposition=self.resolution_disposition,
            evidence_refs=self.evidence_refs,
            terminal_reason=self.terminal_reason,
            source_sequence=self.source_sequence,
            causal_parent_generation_id=self.causal_parent_generation_id,
        )


@dataclass(frozen=True)
class TaskFenceTaskControl:
    """Immutable scalar task projection captured by one acceptance."""

    task_id: str
    conversation_id: str
    cohort_key: str | None
    store_schema_version: int
    control_protocol_version: int
    intent_epoch: int
    control_revision: int
    status: str
    active_authority_event_id: str | None
    active_execution_run_id: str | None
    current_generation_id: str | None
    current_runtime_epoch: int
    last_accepted_order: int
    last_transition_event_id: str | None
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class IngressAcceptance:
    """Exact historical result returned after the acceptance commit."""

    event_id: str
    accepted_order: int
    task_id: str | None
    task_projection: TaskFenceTaskControl | None
    pending_input_ids: tuple[str, ...]
    replayed: bool
    opened_run_id: str | None
    closed_run_id: str | None
    accepted_at: float


def validate_ingress_envelope(envelope: IngressEnvelope) -> None:
    if not isinstance(envelope, IngressEnvelope):
        raise TaskFenceProtocolRejected("invalid_envelope_type")
    validate_action(envelope.action)
    _bounded_text(
        envelope.source,
        field="source",
        max_bytes=_MAX_SOURCE_BYTES,
    )
    for field in ("source_event_id", "conversation_id"):
        _bounded_text(
            getattr(envelope, field),
            field=field,
            max_bytes=_MAX_IDENTIFIER_BYTES,
        )
    _bounded_text(
        envelope.task_id,
        field="task_id",
        max_bytes=_MAX_IDENTIFIER_BYTES,
        optional=True,
    )
    _bounded_text(
        envelope.opaque_payload_ref,
        field="opaque_payload_ref",
        max_bytes=_MAX_OPAQUE_REFERENCE_BYTES,
        optional=True,
    )
    _bounded_text(
        envelope.causal_parent_generation_id,
        field="causal_parent_generation_id",
        max_bytes=_MAX_IDENTIFIER_BYTES,
        optional=True,
    )
    if type(envelope.protocol_version) is not int or (
        envelope.protocol_version != CONTROL_PROTOCOL_VERSION
    ):
        raise TaskFenceProtocolRejected("unsupported_protocol_version")
    if envelope.payload_hash is not None and (
        not isinstance(envelope.payload_hash, str)
        or _SHA256_RE.fullmatch(envelope.payload_hash) is None
    ):
        raise TaskFenceProtocolRejected("invalid_payload_hash")
    if envelope.payload_hash is None and envelope.opaque_payload_ref is None:
        raise TaskFenceProtocolRejected("missing_payload_identity")
    if envelope.source_sequence is not None and (
        type(envelope.source_sequence) is not int
        or envelope.source_sequence < 0
        or envelope.source_sequence > _MAX_SQLITE_INTEGER
    ):
        raise TaskFenceProtocolRejected("invalid_source_sequence")

    correlation = envelope.action.correlation_kind
    if correlation is CorrelationKind.NONE and envelope.correlation_ids:
        raise TaskFenceProtocolRejected("unexpected_correlation_ids")
    if correlation is not CorrelationKind.NONE and not envelope.correlation_ids:
        raise TaskFenceProtocolRejected("missing_correlation_ids")
    if correlation is CorrelationKind.INCIDENT_ATTEMPTS:
        if not isinstance(
            envelope.resolution_disposition,
            ResolutionDisposition,
        ):
            raise TaskFenceProtocolRejected("missing_resolution_disposition")
        if not envelope.evidence_refs:
            raise TaskFenceProtocolRejected("missing_resolution_evidence")
    elif envelope.resolution_disposition is not None:
        raise TaskFenceProtocolRejected("unexpected_resolution_disposition")

    if envelope.action.execution is ExecutionEffect.TERMINATE:
        if not isinstance(envelope.terminal_reason, TerminalReason):
            raise TaskFenceProtocolRejected("missing_terminal_reason")
    elif envelope.terminal_reason is not None:
        raise TaskFenceProtocolRejected("unexpected_terminal_reason")
    if (
        envelope.action.origin is Origin.HUMAN
        and envelope.causal_parent_generation_id is not None
    ):
        raise TaskFenceProtocolRejected("human_ingress_has_causal_parent")


# Compatibility alias for the Phase 0 executable model terminology.
Correlation = CorrelationKind
