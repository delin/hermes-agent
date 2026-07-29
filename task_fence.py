"""Typed ingress and causal contracts for the Task Fence control protocol.

This module defines bounded, secret-free data that may cross trusted runtime
boundaries. Its policy facade computes audit-only decisions; callers do not
gate legacy dispatch in this increment. It deliberately contains no prompt,
transcript, tool argument, or result payload fields.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterator, Mapping, Protocol


CONTROL_PROTOCOL_VERSION = 1
TASK_FENCE_STORE_SCHEMA_VERSION = 4

_MAX_SOURCE_BYTES = 256
_MAX_IDENTIFIER_BYTES = 512
_MAX_OPAQUE_REFERENCE_BYTES = 2_048
_MAX_CORRELATION_IDS = 64
_MAX_EVIDENCE_REFS = 32
_MAX_CAUSAL_ENVELOPE_BYTES = 16_384
_MAX_SQLITE_INTEGER = 2**63 - 1
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_DECISION_ID_RE = re.compile(r"\Atfd_[0-9a-f]{64}\Z")

TASK_FENCE_POLICY_VERSION = "task-fence-policy-v1"


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


class OperationKind(str, Enum):
    MODEL = "model"
    TOOL = "tool"


class DecisionOutcome(str, Enum):
    WOULD_RESERVE = "would_reserve"
    WOULD_ALLOW = "would_allow"
    WOULD_BLOCK = "would_block"
    HALT_DISPATCH = "halt_dispatch"


class DecisionReason(str, Enum):
    CURRENT_AUTHORITY = "current_authority"
    MISSING_PROVENANCE = "missing_provenance"
    STORE_UNAVAILABLE = "store_unavailable"
    STORE_INCOMPATIBLE = "store_incompatible"
    AUDIT_DEGRADED = "audit_degraded"
    COHORT_MISMATCH = "cohort_mismatch"
    COHORT_HALTED = "cohort_halted"
    UNSUPPORTED_MODE = "unsupported_mode"
    TASK_NOT_FOUND = "task_not_found"
    TASK_NOT_RUNNABLE = "task_not_runnable"
    STALE_AUTHORITY = "stale_authority"
    NEWER_INPUT_PENDING = "newer_input_pending"
    INVOCATION_CONFLICT = "invocation_conflict"
    PERMIT_NOT_FOUND = "permit_not_found"
    PERMIT_ALREADY_CONSUMED = "permit_already_consumed"
    PERMIT_REVOKED = "permit_revoked"
    PERMIT_EXPIRED = "permit_expired"
    PERMIT_OPERATION_MISMATCH = "permit_operation_mismatch"
    PERMIT_GENERATION_MISMATCH = "permit_generation_mismatch"
    PERMIT_RUNTIME_EPOCH_MISMATCH = "permit_runtime_epoch_mismatch"


class AttemptTerminal(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED_DEFINITE = "FAILED_DEFINITE"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


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
    """A typed Task Fence value does not satisfy the closed protocol."""

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


class TaskFenceProvenanceRejected(TaskFenceProtocolRejected):
    """A causal operation does not match its recorded task authority."""


class TaskFenceProvenanceUnavailable(RuntimeError):
    """Shadow provenance could not use a compatible writable control store."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class TaskFencePolicyRejected(TaskFenceProtocolRejected):
    """A dispatch-policy operation conflicts with durable protocol state."""


class TaskFencePolicyUnavailable(RuntimeError):
    """A terminal policy write could not use the durable control store."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class OperationDescriptor:
    """Secret-free identity of one post-default external operation."""

    invocation_id: str
    kind: OperationKind
    adapter: str
    invocation_fingerprint: str

    def __post_init__(self) -> None:
        validate_operation_descriptor(self)

    def to_dict(self) -> dict[str, str]:
        return {
            "invocation_id": self.invocation_id,
            "kind": self.kind.value,
            "adapter": self.adapter,
            "invocation_fingerprint": self.invocation_fingerprint,
        }


@dataclass(frozen=True)
class DispatchDecision:
    """Stable shadow observation returned by the shared policy service."""

    outcome: DecisionOutcome
    reason: DecisionReason
    permit_id: str | None = None
    attempt_id: str | None = None
    decision_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, DecisionOutcome) or not isinstance(
            self.reason,
            DecisionReason,
        ):
            raise TaskFenceProtocolRejected("invalid_dispatch_decision")
        current = self.reason is DecisionReason.CURRENT_AUTHORITY
        allowed = self.outcome in {
            DecisionOutcome.WOULD_RESERVE,
            DecisionOutcome.WOULD_ALLOW,
        }
        if current != allowed:
            raise TaskFenceProtocolRejected("invalid_dispatch_decision_reason")
        for field in ("permit_id", "attempt_id", "decision_id"):
            _bounded_text(
                getattr(self, field),
                field=field,
                max_bytes=_MAX_IDENTIFIER_BYTES,
                optional=True,
            )
        if (
            self.decision_id is not None
            and _DECISION_ID_RE.fullmatch(self.decision_id) is None
        ):
            raise TaskFenceProtocolRejected("invalid_decision_id")
        if self.outcome is DecisionOutcome.WOULD_RESERVE:
            valid_shape = self.permit_id is not None and self.attempt_id is None
        elif self.outcome is DecisionOutcome.WOULD_ALLOW:
            valid_shape = self.permit_id is not None and self.attempt_id is not None
        else:
            valid_shape = self.attempt_id is None
        if not valid_shape:
            raise TaskFenceProtocolRejected("invalid_dispatch_decision_shape")


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


def validate_operation_descriptor(operation: OperationDescriptor) -> None:
    if not isinstance(operation, OperationDescriptor):
        raise TaskFenceProtocolRejected("invalid_operation_descriptor_type")
    if not isinstance(operation.kind, OperationKind):
        raise TaskFenceProtocolRejected("invalid_operation_kind")
    for field in ("invocation_id", "adapter"):
        _bounded_text(
            getattr(operation, field),
            field=field,
            max_bytes=_MAX_IDENTIFIER_BYTES,
        )
    if not (
        isinstance(operation.invocation_fingerprint, str)
        and _SHA256_RE.fullmatch(operation.invocation_fingerprint)
    ):
        raise TaskFenceProtocolRejected("invalid_invocation_fingerprint")


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


_CAUSAL_ENVELOPE_FIELDS = frozenset(
    {
        "task_id",
        "authority_event_id",
        "run_id",
        "generation_id",
        "snapshot_event_id",
        "input_manifest_hash",
        "store_schema_version",
        "control_protocol_version",
        "intent_epoch",
        "control_revision",
        "runtime_epoch",
        "accepted_order",
        "invocation_id",
        "parent_invocation_id",
    }
)


@dataclass(frozen=True)
class CausalEnvelope:
    """Immutable authority copied from one recorded model generation.

    `invocation_id` is absent on the generation envelope. Independently
    dispatchable descendants derive a new envelope and may reference only the
    immediately preceding invocation ID; every authority field is copied.
    """

    task_id: str
    authority_event_id: str
    run_id: str
    generation_id: str
    snapshot_event_id: str
    input_manifest_hash: str
    store_schema_version: int
    control_protocol_version: int
    intent_epoch: int
    control_revision: int
    runtime_epoch: int
    accepted_order: int
    invocation_id: str | None = None
    parent_invocation_id: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "task_id",
            "authority_event_id",
            "run_id",
            "generation_id",
            "snapshot_event_id",
        ):
            _bounded_text(
                getattr(self, field),
                field=field,
                max_bytes=_MAX_IDENTIFIER_BYTES,
            )
        for field in ("invocation_id", "parent_invocation_id"):
            _bounded_text(
                getattr(self, field),
                field=field,
                max_bytes=_MAX_IDENTIFIER_BYTES,
                optional=True,
            )
        if not (
            isinstance(self.input_manifest_hash, str)
            and _SHA256_RE.fullmatch(self.input_manifest_hash)
        ):
            raise TaskFenceProtocolRejected("invalid_input_manifest_hash")
        for field in (
            "store_schema_version",
            "control_protocol_version",
            "intent_epoch",
            "control_revision",
            "runtime_epoch",
            "accepted_order",
        ):
            value = getattr(self, field)
            if type(value) is not int or not 0 <= value <= _MAX_SQLITE_INTEGER:
                raise TaskFenceProtocolRejected(f"invalid_{field}")
        if self.store_schema_version != TASK_FENCE_STORE_SCHEMA_VERSION:
            raise TaskFenceProtocolRejected("invalid_store_schema_version")
        if self.accepted_order < 1:
            raise TaskFenceProtocolRejected("invalid_accepted_order")
        if self.control_protocol_version != CONTROL_PROTOCOL_VERSION:
            raise TaskFenceProtocolRejected("unsupported_protocol_version")
        if self.invocation_id is None and self.parent_invocation_id is not None:
            raise TaskFenceProtocolRejected("orphan_parent_invocation_id")
        if (
            self.invocation_id is not None
            and self.invocation_id == self.parent_invocation_id
        ):
            raise TaskFenceProtocolRejected("cyclic_invocation_id")

    def for_invocation(self, invocation_id: str | None = None) -> "CausalEnvelope":
        """Copy authority into one new server-owned invocation identity."""

        return CausalEnvelope(
            task_id=self.task_id,
            authority_event_id=self.authority_event_id,
            run_id=self.run_id,
            generation_id=self.generation_id,
            snapshot_event_id=self.snapshot_event_id,
            input_manifest_hash=self.input_manifest_hash,
            store_schema_version=self.store_schema_version,
            control_protocol_version=self.control_protocol_version,
            intent_epoch=self.intent_epoch,
            control_revision=self.control_revision,
            runtime_epoch=self.runtime_epoch,
            accepted_order=self.accepted_order,
            invocation_id=(
                f"tfiv_{uuid.uuid4().hex}"
                if invocation_id is None
                else invocation_id
            ),
            parent_invocation_id=self.invocation_id,
        )

    def _authority_tuple(self) -> tuple[object, ...]:
        return (
            self.task_id,
            self.authority_event_id,
            self.run_id,
            self.generation_id,
            self.snapshot_event_id,
            self.input_manifest_hash,
            self.store_schema_version,
            self.control_protocol_version,
            self.intent_epoch,
            self.control_revision,
            self.runtime_epoch,
            self.accepted_order,
        )

    def to_dict(self) -> dict[str, str | int | None]:
        """Return the exact closed serialization shape."""

        return {
            "task_id": self.task_id,
            "authority_event_id": self.authority_event_id,
            "run_id": self.run_id,
            "generation_id": self.generation_id,
            "snapshot_event_id": self.snapshot_event_id,
            "input_manifest_hash": self.input_manifest_hash,
            "store_schema_version": self.store_schema_version,
            "control_protocol_version": self.control_protocol_version,
            "intent_epoch": self.intent_epoch,
            "control_revision": self.control_revision,
            "runtime_epoch": self.runtime_epoch,
            "accepted_order": self.accepted_order,
            "invocation_id": self.invocation_id,
            "parent_invocation_id": self.parent_invocation_id,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: object) -> "CausalEnvelope":
        if not isinstance(payload, dict) or set(payload) != _CAUSAL_ENVELOPE_FIELDS:
            raise TaskFenceProtocolRejected("invalid_causal_envelope_fields")
        try:
            return cls(**payload)
        except TypeError as exc:
            raise TaskFenceProtocolRejected(
                "invalid_causal_envelope_fields"
            ) from exc

    @classmethod
    def from_json(cls, payload: object) -> "CausalEnvelope":
        if not isinstance(payload, str):
            raise TaskFenceProtocolRejected("invalid_causal_envelope_json")
        try:
            if len(payload.encode("utf-8")) > _MAX_CAUSAL_ENVELOPE_BYTES:
                raise TaskFenceProtocolRejected(
                    "causal_envelope_json_too_large"
                )
            decoded = json.loads(payload)
        except TaskFenceProtocolRejected:
            raise
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise TaskFenceProtocolRejected(
                "invalid_causal_envelope_json"
            ) from exc
        envelope = cls.from_dict(decoded)
        if envelope.to_json() != payload:
            raise TaskFenceProtocolRejected("noncanonical_causal_envelope_json")
        return envelope

    @classmethod
    def invocation_from_dict(
        cls,
        payload: object,
        *,
        parent: "CausalEnvelope",
    ) -> "CausalEnvelope":
        """Decode a child only when it preserves exact parent authority."""

        if not isinstance(parent, CausalEnvelope):
            raise TaskFenceProtocolRejected("invalid_causal_parent_type")
        child = cls.from_dict(payload)
        if (
            child.invocation_id is None
            or child.parent_invocation_id != parent.invocation_id
            or child._authority_tuple() != parent._authority_tuple()
        ):
            raise TaskFenceProtocolRejected("mixed_causal_parentage")
        return child


def operation_binding_fingerprint(
    envelope: CausalEnvelope,
    operation: OperationDescriptor,
) -> str:
    """Commit to complete causal authority and post-default operation identity."""

    if not isinstance(envelope, CausalEnvelope):
        raise TaskFenceProtocolRejected("invalid_causal_envelope_type")
    validate_operation_descriptor(operation)
    payload = json.dumps(
        {
            "causal_envelope": envelope.to_dict(),
            "operation": operation.to_dict(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _TaskFencePolicyStore(Protocol):
    def _admit_task_fence_operation(
        self,
        envelope: CausalEnvelope | None,
        operation: OperationDescriptor,
    ) -> DispatchDecision: ...

    def _authorize_and_start_task_fence_operation(
        self,
        envelope: CausalEnvelope | None,
        operation: OperationDescriptor,
        permit_id: str,
    ) -> DispatchDecision: ...

    def _finish_task_fence_attempt(
        self,
        attempt_id: str,
        terminal: AttemptTerminal,
        evidence_reference: str | None,
    ) -> None: ...


class TaskFencePolicy:
    """Shared audit-only policy facade; decisions do not gate legacy dispatch."""

    def __init__(self, store: _TaskFencePolicyStore):
        self._store = store

    def admit_operation(
        self,
        envelope: CausalEnvelope | None,
        operation: OperationDescriptor,
    ) -> DispatchDecision:
        validate_operation_descriptor(operation)
        if envelope is not None and not isinstance(envelope, CausalEnvelope):
            raise TaskFenceProtocolRejected("invalid_causal_envelope_type")
        return self._store._admit_task_fence_operation(envelope, operation)

    def authorize_and_start(
        self,
        envelope: CausalEnvelope | None,
        operation: OperationDescriptor,
        permit_id: str,
    ) -> DispatchDecision:
        validate_operation_descriptor(operation)
        _bounded_text(
            permit_id,
            field="permit_id",
            max_bytes=_MAX_IDENTIFIER_BYTES,
        )
        if envelope is not None and not isinstance(envelope, CausalEnvelope):
            raise TaskFenceProtocolRejected("invalid_causal_envelope_type")
        return self._store._authorize_and_start_task_fence_operation(
            envelope,
            operation,
            permit_id,
        )

    def finish_attempt(
        self,
        attempt_id: str,
        terminal: AttemptTerminal,
        evidence_reference: str | None,
    ) -> None:
        _bounded_text(
            attempt_id,
            field="attempt_id",
            max_bytes=_MAX_IDENTIFIER_BYTES,
        )
        if not isinstance(terminal, AttemptTerminal):
            raise TaskFenceProtocolRejected("invalid_attempt_terminal")
        if (
            terminal
            in {AttemptTerminal.SUCCEEDED, AttemptTerminal.FAILED_DEFINITE}
            and evidence_reference is None
        ):
            raise TaskFencePolicyRejected("missing_terminal_evidence")
        _bounded_text(
            evidence_reference,
            field="evidence_reference",
            max_bytes=_MAX_OPAQUE_REFERENCE_BYTES,
            optional=True,
        )
        self._store._finish_task_fence_attempt(
            attempt_id,
            terminal,
            evidence_reference,
        )


_CURRENT_CAUSAL_ENVELOPE: ContextVar[CausalEnvelope | None] = ContextVar(
    "task_fence_causal_envelope",
    default=None,
)


def current_causal_envelope() -> CausalEnvelope | None:
    """Return process-local immutable provenance without synthesizing it."""

    return _CURRENT_CAUSAL_ENVELOPE.get()


@contextmanager
def bind_causal_envelope(
    envelope: CausalEnvelope | None,
) -> Iterator[CausalEnvelope | None]:
    """Transport an already-created envelope inside one in-process scope."""

    if envelope is not None and not isinstance(envelope, CausalEnvelope):
        raise TaskFenceProtocolRejected("invalid_causal_envelope_type")
    token = _CURRENT_CAUSAL_ENVELOPE.set(envelope)
    try:
        yield envelope
    finally:
        _CURRENT_CAUSAL_ENVELOPE.reset(token)


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
