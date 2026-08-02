"""Executable protocol specification for Task Fence.

This file deliberately models the intended control protocol without changing
runtime behaviour.  The final test records the sanitized stale-dispatch gap in
the current runtime; enforcement work must replace that assertion with a real
membrane conformance test.
"""

from dataclasses import dataclass, replace
from enum import Enum

import pytest

from task_fence import (
    TASK_FENCE_ACTIONS as ACTIONS,
    CorrelationKind as Correlation,
    DecisionOutcome,
    DecisionReason,
    DispatchDecision,
    ExecutionEffect,
    IngressClass,
    InputEffect,
    IntentEffect,
    Origin,
    ResolutionDisposition,
    TaskFenceAction as Action,
    TaskFenceModeRecord as ModeRecord,
    TaskFenceProtocolRejected as ProtocolRejected,
    TaskFenceRuntimeMode as RuntimeMode,
    action_shape as wire_shape,
    effective_task_fence_startup_mode as effective_startup_mode,
    validate_action,
)


class TaskStatus(str, Enum):
    EMPTY = "empty"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    PAUSED = "paused"
    INCIDENT = "incident"
    STOPPED = "stopped"
    DONE = "done"


@dataclass(frozen=True)
class TaskControlState:
    task_id: str | None = None
    intent_epoch: int = 0
    control_revision: int = 0
    runtime_epoch: int = 1
    status: TaskStatus = TaskStatus.EMPTY
    active_run_id: str | None = None
    generation_id: str | None = None
    bound_input_ids: tuple[str, ...] = ()
    pending_input_ids: tuple[str, ...] = ()
    open_question_ids: tuple[str, ...] = ()
    incident_attempt_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DispatchEnvelope:
    task_id: str | None
    intent_epoch: int
    control_revision: int
    runtime_epoch: int
    run_id: str | None
    generation_id: str | None
    invocation_id: str


class DispatchBlocked(RuntimeError):
    pass


def accept_source_event(
    state: TaskControlState,
    ledger: dict[tuple[str, str], tuple[object, ...]],
    action: Action,
    *,
    source: str,
    source_event_id: str,
    payload_hash: str,
    correlation_ids: tuple[str, ...] = (),
    resolution_disposition: ResolutionDisposition | None = None,
) -> TaskControlState:
    """Model source-scoped idempotence in the same acceptance transaction."""
    if not source or not source_event_id or not payload_hash:
        raise ProtocolRejected("source event identity must be complete")
    key = (source, source_event_id)
    fingerprint = (
        wire_shape(action),
        payload_hash,
        correlation_ids,
        resolution_disposition,
    )
    existing = ledger.get(key)
    if existing is not None:
        if existing != fingerprint:
            raise ProtocolRejected("source event id collision")
        return state

    accepted_event_id = f"accepted-{len(ledger) + 1}"
    next_state = accept_ingress(
        state,
        action,
        accepted_event_id,
        correlation_ids=correlation_ids,
        resolution_disposition=resolution_disposition,
    )
    ledger[key] = fingerprint
    return next_state


def accept_ingress(
    state: TaskControlState,
    action: Action,
    event_id: str,
    *,
    correlation_ids: tuple[str, ...] = (),
    resolution_disposition: ResolutionDisposition | None = None,
) -> TaskControlState:
    """Commit one accepted ingress and close the superseded run atomically."""
    validate_action(action)
    authority_affecting = action.ingress_class in {
        IngressClass.TASK_INPUT,
        IngressClass.CONTROL,
    }
    if not authority_affecting:
        return state
    if state.status in {TaskStatus.DONE, TaskStatus.STOPPED}:
        raise ProtocolRejected("terminal task cannot accept authority transition")
    has_task = state.task_id is not None
    if not has_task:
        if wire_shape(action) != wire_shape(ACTIONS["initial_submit"]):
            raise ProtocolRejected("only replace/run/append may create a task")
    if action.correlation_kind is not Correlation.NONE and not correlation_ids:
        raise ProtocolRejected(
            f"action requires exact {action.correlation_kind.value} ids"
        )
    if action.correlation_kind is Correlation.OPEN_QUESTION:
        if state.status is not TaskStatus.WAITING_USER:
            raise ProtocolRejected("question answer requires waiting_user state")
        if set(correlation_ids) != set(state.open_question_ids) or len(
            correlation_ids
        ) != len(state.open_question_ids):
            raise ProtocolRejected("question answer requires exact open question ids")
    if action.correlation_kind is Correlation.INCIDENT_ATTEMPTS:
        if state.status is not TaskStatus.INCIDENT:
            raise ProtocolRejected("incident resolution requires incident state")
        if set(correlation_ids) != set(state.incident_attempt_ids) or len(
            correlation_ids
        ) != len(state.incident_attempt_ids):
            raise ProtocolRejected("incident resolution requires exact attempt ids")
        if not isinstance(resolution_disposition, ResolutionDisposition):
            raise ProtocolRejected("incident resolution requires a disposition")
    elif resolution_disposition is not None:
        raise ProtocolRejected("resolution disposition requires incident attempts")
    if (
        state.status is TaskStatus.INCIDENT
        and action.correlation_kind is not Correlation.INCIDENT_ATTEMPTS
        and action.execution is not ExecutionEffect.TERMINATE
    ):
        raise ProtocolRejected("incident requires exact resolution or stop")
    is_plain_resume = (
        action.ingress_class is IngressClass.CONTROL
        and action.execution is ExecutionEffect.RUN
        and action.input_effect is InputEffect.NONE
        and action.correlation_kind is Correlation.NONE
    )
    if is_plain_resume and (
        state.status is not TaskStatus.PAUSED or state.pending_input_ids
    ):
        raise ProtocolRejected("resume requires paused task without pending input")

    revision = state.control_revision + 1
    intent_epoch = state.intent_epoch + (action.intent is IntentEffect.REPLACE)
    task_id = state.task_id or "task-1"
    pending = list(state.pending_input_ids)
    if action.input_effect is InputEffect.APPEND:
        pending.append(event_id)
    elif action.input_effect is InputEffect.DISCARD_SELECTED:
        if not set(correlation_ids).issubset(pending):
            raise ProtocolRejected("discard_pending requires exact pending input ids")
        pending = [item for item in pending if item not in correlation_ids]

    next_state = replace(
        state,
        task_id=task_id,
        intent_epoch=intent_epoch,
        control_revision=revision,
        active_run_id=None,
        generation_id=None,
        bound_input_ids=(),
        pending_input_ids=tuple(pending),
        open_question_ids=(),
        incident_attempt_ids=(
            ()
            if action.correlation_kind is Correlation.INCIDENT_ATTEMPTS
            or action.execution is ExecutionEffect.TERMINATE
            else state.incident_attempt_ids
        ),
    )

    if action.execution is ExecutionEffect.RUN:
        return replace(
            next_state,
            status=TaskStatus.RUNNING,
            active_run_id=f"{task_id}:run:{revision}",
            bound_input_ids=next_state.pending_input_ids,
            pending_input_ids=(),
        )
    if action.execution is ExecutionEffect.TERMINATE:
        return replace(next_state, status=TaskStatus.STOPPED)
    return replace(next_state, status=TaskStatus.PAUSED)


def wait_for_question(state: TaskControlState, question_id: str) -> TaskControlState:
    """Model a generation closing its run while awaiting one exact human answer."""
    if state.status is not TaskStatus.RUNNING or not question_id:
        raise ProtocolRejected("open question requires a running task and exact id")
    return replace(
        state,
        status=TaskStatus.WAITING_USER,
        active_run_id=None,
        generation_id=None,
        bound_input_ids=(),
        open_question_ids=(question_id,),
    )


def record_incident(state: TaskControlState, attempt_id: str) -> TaskControlState:
    """Model an unknown may-effect attempt closing dispatch pending resolution."""
    if state.task_id is None or state.status in {TaskStatus.DONE, TaskStatus.STOPPED}:
        raise ProtocolRejected("incident requires a non-terminal task")
    if not attempt_id:
        raise ProtocolRejected("incident requires an exact attempt id")
    return replace(
        state,
        status=TaskStatus.INCIDENT,
        active_run_id=None,
        generation_id=None,
        bound_input_ids=(),
        open_question_ids=(),
        incident_attempt_ids=(attempt_id,),
    )


def open_generation(state: TaskControlState) -> TaskControlState:
    if state.status is not TaskStatus.RUNNING or state.active_run_id is None:
        raise ProtocolRejected("generation requires an active run")
    return replace(
        state,
        generation_id=f"{state.active_run_id}:generation:{state.control_revision}",
    )


def envelope_for(state: TaskControlState, invocation_id: str) -> DispatchEnvelope:
    return DispatchEnvelope(
        task_id=state.task_id,
        intent_epoch=state.intent_epoch,
        control_revision=state.control_revision,
        runtime_epoch=state.runtime_epoch,
        run_id=state.active_run_id,
        generation_id=state.generation_id,
        invocation_id=invocation_id,
    )


def authorize_and_start(
    state: TaskControlState,
    envelope: DispatchEnvelope,
    consumed_invocations: set[str],
) -> str:
    """Model the single transactional permit-consumption linearization point."""
    if state.status is not TaskStatus.RUNNING:
        raise DispatchBlocked("task_not_running")
    if state.pending_input_ids:
        raise DispatchBlocked("newer_input_pending")
    expected = (
        state.task_id,
        state.intent_epoch,
        state.control_revision,
        state.runtime_epoch,
        state.active_run_id,
        state.generation_id,
    )
    actual = (
        envelope.task_id,
        envelope.intent_epoch,
        envelope.control_revision,
        envelope.runtime_epoch,
        envelope.run_id,
        envelope.generation_id,
    )
    if None in actual or actual != expected:
        raise DispatchBlocked("stale_or_missing_provenance")
    if envelope.invocation_id in consumed_invocations:
        raise DispatchBlocked("permit_already_consumed")
    consumed_invocations.add(envelope.invocation_id)
    return "started"


def transition_mode(
    record: ModeRecord,
    target: RuntimeMode,
    *,
    expected_generation: int,
    offline: bool,
) -> ModeRecord:
    """Model a reviewed mode transition; emergency halt is always reachable."""
    if expected_generation != record.mode_generation:
        raise ProtocolRejected("stale mode generation")
    if target is not RuntimeMode.HALT_DISPATCH and not offline:
        raise ProtocolRejected("mode activation requires an offline transition")
    if record.ever_enforced and target is RuntimeMode.AUDIT:
        raise ProtocolRejected("enforced cohort cannot return to audit")
    if target is RuntimeMode.ENFORCE and record.audit_degraded:
        raise ProtocolRejected("degraded audit cannot promote to enforce")
    return ModeRecord(
        mode=target,
        mode_generation=record.mode_generation + 1,
        ever_enforced=record.ever_enforced or target is RuntimeMode.ENFORCE,
        audit_degraded=record.audit_degraded,
    )


def clear_audit_degraded(
    record: ModeRecord,
    *,
    expected_generation: int,
    offline: bool,
    conformance_passed: bool,
) -> ModeRecord:
    if expected_generation != record.mode_generation:
        raise ProtocolRejected("stale mode generation")
    if not offline or not conformance_passed:
        raise ProtocolRejected("degraded audit requires offline conformance")
    return replace(
        record,
        mode_generation=record.mode_generation + 1,
        audit_degraded=False,
    )


def recover_after_restart(
    state: TaskControlState, *, new_runtime_epoch: int
) -> TaskControlState:
    """Pause a durable task and invalidate every pre-restart descendant."""
    if new_runtime_epoch <= state.runtime_epoch:
        raise ProtocolRejected("runtime epoch must increase")
    if state.task_id is None:
        return replace(state, runtime_epoch=new_runtime_epoch)
    if state.status in {TaskStatus.DONE, TaskStatus.STOPPED}:
        return replace(state, runtime_epoch=new_runtime_epoch)
    if state.status is TaskStatus.INCIDENT:
        return replace(
            state,
            runtime_epoch=new_runtime_epoch,
            active_run_id=None,
            generation_id=None,
            bound_input_ids=(),
        )
    if state.status is TaskStatus.WAITING_USER:
        return replace(
            state,
            runtime_epoch=new_runtime_epoch,
            active_run_id=None,
            generation_id=None,
            bound_input_ids=(),
        )
    pending = state.pending_input_ids
    if state.status is TaskStatus.RUNNING:
        pending = tuple(dict.fromkeys(pending + state.bound_input_ids))
    return replace(
        state,
        runtime_epoch=new_runtime_epoch,
        status=TaskStatus.PAUSED,
        active_run_id=None,
        generation_id=None,
        bound_input_ids=(),
        pending_input_ids=pending,
    )


def test_closed_ingress_matrix_accepts_only_documented_shapes() -> None:
    for action in ACTIONS.values():
        validate_action(action)

    invalid = Action(
        Origin.RUNTIME,
        IngressClass.SYNTHETIC,
        IntentEffect.KEEP,
        ExecutionEffect.RUN,
        InputEffect.NONE,
    )
    with pytest.raises(ProtocolRejected, match="unsupported_ingress_tuple"):
        validate_action(invalid)


def test_dispatch_decision_id_is_closed_bounded_and_optional() -> None:
    decision_id = f"tfd_{'a' * 64}"
    persisted = DispatchDecision(
        DecisionOutcome.WOULD_BLOCK,
        DecisionReason.STALE_AUTHORITY,
        decision_id=decision_id,
    )
    unavailable = DispatchDecision(
        DecisionOutcome.HALT_DISPATCH,
        DecisionReason.STORE_UNAVAILABLE,
    )

    assert persisted.decision_id == decision_id
    assert unavailable.decision_id is None

    for invalid in (
        "",
        "decision-1",
        f"tfd_{'g' * 64}",
        f"tfd_{'a' * 63}",
        f"tfd_{'a' * 65}",
        1,
    ):
        with pytest.raises(ProtocolRejected, match="decision_id"):
            DispatchDecision(
                DecisionOutcome.WOULD_BLOCK,
                DecisionReason.STALE_AUTHORITY,
                decision_id=invalid,  # type: ignore[arg-type]
            )


def test_source_event_dedupe_is_idempotent_and_collision_safe() -> None:
    ledger: dict[tuple[str, str], tuple[object, ...]] = {}
    accepted = accept_source_event(
        TaskControlState(),
        ledger,
        ACTIONS["initial_submit"],
        source="adapter-a",
        source_event_id="message-1",
        payload_hash="payload-hash-a",
    )
    replay = accept_source_event(
        accepted,
        ledger,
        ACTIONS["initial_submit"],
        source="adapter-a",
        source_event_id="message-1",
        payload_hash="payload-hash-a",
    )
    assert replay == accepted
    assert replay.control_revision == 1

    with pytest.raises(ProtocolRejected, match="source event id collision"):
        accept_source_event(
            accepted,
            ledger,
            ACTIONS["initial_submit"],
            source="adapter-a",
            source_event_id="message-1",
            payload_hash="payload-hash-different",
        )

    other_source = accept_source_event(
        accepted,
        ledger,
        ACTIONS["comment_hold"],
        source="adapter-b",
        source_event_id="message-1",
        payload_hash="payload-hash-b",
    )
    assert other_source.control_revision == 2
    assert len(ledger) == 2


def test_only_explicit_run_transition_opens_a_run() -> None:
    initial = accept_ingress(TaskControlState(), ACTIONS["initial_submit"], "input-1")
    assert initial.active_run_id is not None

    for name in (
        "change_and_hold",
        "comment_hold",
        "pause",
        "stop",
        "explicit_note",
        "synthetic_notice",
    ):
        next_state = accept_ingress(initial, ACTIONS[name], f"event-{name}")
        if ACTIONS[name].ingress_class in {
            IngressClass.ADVISORY,
            IngressClass.SYNTHETIC,
        }:
            assert next_state == initial
        else:
            assert next_state.active_run_id is None
            assert next_state.generation_id is None


def test_empty_lane_and_correlations_fail_closed() -> None:
    with pytest.raises(ProtocolRejected, match="only replace/run/append"):
        accept_ingress(TaskControlState(), ACTIONS["resume"], "resume-1")

    running = open_generation(
        accept_ingress(TaskControlState(), ACTIONS["initial_submit"], "human-1")
    )
    with pytest.raises(ProtocolRejected, match="exact open_question ids"):
        accept_ingress(running, ACTIONS["answer_only"], "answer-1")
    with pytest.raises(ProtocolRejected, match="waiting_user"):
        accept_ingress(
            running,
            ACTIONS["answer_only"],
            "answer-1",
            correlation_ids=("question-1",),
        )

    waiting = wait_for_question(running, "question-1")
    with pytest.raises(ProtocolRejected, match="exact open question ids"):
        accept_ingress(
            waiting,
            ACTIONS["answer_only"],
            "answer-wrong",
            correlation_ids=("question-other",),
        )
    answered = accept_ingress(
        waiting,
        ACTIONS["answer_only"],
        "answer-1",
        correlation_ids=("question-1",),
    )
    assert answered.status is TaskStatus.PAUSED
    assert answered.pending_input_ids == ("answer-1",)
    assert answered.open_question_ids == ()

    waiting = wait_for_question(running, "question-2")
    resumed = accept_ingress(
        waiting,
        ACTIONS["answer_and_resume"],
        "answer-2",
        correlation_ids=("question-2",),
    )
    assert resumed.status is TaskStatus.RUNNING
    assert resumed.bound_input_ids == ("answer-2",)
    assert resumed.open_question_ids == ()

    incident = record_incident(running, "attempt-1")
    with pytest.raises(ProtocolRejected, match="requires a disposition"):
        accept_ingress(
            incident,
            ACTIONS["resolve_incident"],
            "resolution-1",
            correlation_ids=("attempt-1",),
        )
    with pytest.raises(ProtocolRejected, match="exact attempt ids"):
        accept_ingress(
            incident,
            ACTIONS["resolve_incident"],
            "resolution-wrong",
            correlation_ids=("attempt-other",),
            resolution_disposition=ResolutionDisposition.CONFIRMED_FAILURE,
        )
    resolved = accept_ingress(
        incident,
        ACTIONS["resolve_incident"],
        "resolution-1",
        correlation_ids=("attempt-1",),
        resolution_disposition=ResolutionDisposition.ACCEPTED_UNKNOWN_NO_RETRY,
    )
    assert resolved.status is TaskStatus.PAUSED
    assert resolved.incident_attempt_ids == ()


@pytest.mark.parametrize("terminal_status", [TaskStatus.STOPPED, TaskStatus.DONE])
def test_terminal_states_never_reopen(terminal_status: TaskStatus) -> None:
    running = open_generation(
        accept_ingress(TaskControlState(), ACTIONS["initial_submit"], "human-1")
    )
    terminal = replace(
        running,
        status=terminal_status,
        active_run_id=None,
        generation_id=None,
        bound_input_ids=(),
    )
    for name, action in ACTIONS.items():
        if action.ingress_class in {IngressClass.ADVISORY, IngressClass.SYNTHETIC}:
            assert accept_ingress(terminal, action, f"event-{name}") == terminal
            continue
        with pytest.raises(ProtocolRejected, match="terminal task"):
            accept_ingress(terminal, action, f"event-{name}")


def test_advisory_and_synthetic_events_never_gain_authority() -> None:
    state = open_generation(
        accept_ingress(TaskControlState(), ACTIONS["initial_submit"], "human-1")
    )
    for name in ("explicit_note", "synthetic_notice"):
        assert accept_ingress(state, ACTIONS[name], f"runtime-{name}") == state


def test_dispatch_permit_is_exact_and_single_use() -> None:
    state = open_generation(
        accept_ingress(TaskControlState(), ACTIONS["initial_submit"], "human-1")
    )
    envelope = envelope_for(state, "invoke-1")
    consumed: set[str] = set()
    assert authorize_and_start(state, envelope, consumed) == "started"
    with pytest.raises(DispatchBlocked, match="permit_already_consumed"):
        authorize_and_start(state, envelope, consumed)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"mode": "audit"}, "invalid_task_fence_runtime_mode"),
        ({"mode_generation": True}, "invalid_mode_generation"),
        ({"mode_generation": -1}, "invalid_mode_generation"),
        ({"mode_generation": 2**63}, "invalid_mode_generation"),
        ({"ever_enforced": 1}, "invalid_ever_enforced"),
        ({"audit_degraded": 0}, "invalid_audit_degraded"),
    ],
)
def test_mode_record_rejects_untyped_or_unbounded_state(
    kwargs: dict[str, object],
    reason: str,
) -> None:
    with pytest.raises(ProtocolRejected, match=reason):
        ModeRecord(**kwargs)


def test_mode_activation_and_restart_fail_safe() -> None:
    initial = ModeRecord()
    assert effective_startup_mode(initial, None) is RuntimeMode.AUDIT

    for corrupt_enforced in (
        ModeRecord(
            mode=RuntimeMode.ENFORCE,
            mode_generation=1,
            ever_enforced=False,
        ),
        ModeRecord(
            mode=RuntimeMode.ENFORCE,
            mode_generation=0,
            ever_enforced=True,
        ),
    ):
        assert (
            effective_startup_mode(corrupt_enforced, "enforce")
            is RuntimeMode.HALT_DISPATCH
        )

    with pytest.raises(ProtocolRejected, match="offline transition"):
        transition_mode(
            initial,
            RuntimeMode.ENFORCE,
            expected_generation=0,
            offline=False,
        )

    enforced = transition_mode(
        initial,
        RuntimeMode.ENFORCE,
        expected_generation=0,
        offline=True,
    )
    assert effective_startup_mode(enforced, "enforce") is RuntimeMode.ENFORCE
    with pytest.raises(ProtocolRejected, match="cannot return to audit"):
        transition_mode(
            enforced,
            RuntimeMode.AUDIT,
            expected_generation=enforced.mode_generation,
            offline=True,
        )
    corrupt_audit = replace(enforced, mode=RuntimeMode.AUDIT)
    assert effective_startup_mode(corrupt_audit, "audit") is RuntimeMode.HALT_DISPATCH
    assert effective_startup_mode(enforced, None) is RuntimeMode.HALT_DISPATCH
    assert effective_startup_mode(enforced, "audit") is RuntimeMode.HALT_DISPATCH
    assert effective_startup_mode(enforced, "unknown") is RuntimeMode.HALT_DISPATCH
    assert (
        effective_startup_mode(enforced, "enforce", store_healthy=False)
        is RuntimeMode.HALT_DISPATCH
    )
    assert (
        effective_startup_mode(enforced, "enforce", compatible=False)
        is RuntimeMode.HALT_DISPATCH
    )

    emergency_halt = transition_mode(
        enforced,
        RuntimeMode.HALT_DISPATCH,
        expected_generation=enforced.mode_generation,
        offline=False,
    )
    assert emergency_halt.mode is RuntimeMode.HALT_DISPATCH
    with pytest.raises(ProtocolRejected, match="stale mode generation"):
        transition_mode(
            emergency_halt,
            RuntimeMode.AUDIT,
            expected_generation=enforced.mode_generation,
            offline=True,
        )
    with pytest.raises(ProtocolRejected, match="cannot return to audit"):
        transition_mode(
            emergency_halt,
            RuntimeMode.AUDIT,
            expected_generation=emergency_halt.mode_generation,
            offline=True,
        )

    degraded = replace(initial, audit_degraded=True)
    with pytest.raises(ProtocolRejected, match="degraded audit"):
        transition_mode(
            degraded,
            RuntimeMode.ENFORCE,
            expected_generation=0,
            offline=True,
        )
    with pytest.raises(ProtocolRejected, match="offline conformance"):
        clear_audit_degraded(
            degraded,
            expected_generation=0,
            offline=True,
            conformance_passed=False,
        )
    cleared = clear_audit_degraded(
        degraded,
        expected_generation=0,
        offline=True,
        conformance_passed=True,
    )
    assert not cleared.audit_degraded
    assert cleared.mode_generation == 1


def test_restart_pauses_task_and_invalidates_old_descendants() -> None:
    running = open_generation(
        accept_ingress(TaskControlState(), ACTIONS["initial_submit"], "human-1")
    )
    old_envelope = envelope_for(running, "invoke-before-restart")
    recovered = recover_after_restart(running, new_runtime_epoch=2)

    assert recovered.status is TaskStatus.PAUSED
    assert recovered.runtime_epoch == 2
    assert recovered.active_run_id is None
    assert recovered.generation_id is None
    assert recovered.bound_input_ids == ()
    assert recovered.pending_input_ids == ("human-1",)
    with pytest.raises(DispatchBlocked, match="task_not_running"):
        authorize_and_start(recovered, old_envelope, set())

    with pytest.raises(ProtocolRejected, match="without pending input"):
        accept_ingress(recovered, ACTIONS["resume"], "resume-with-pending")
    recovered = accept_ingress(
        recovered,
        ACTIONS["discard_pending"],
        "discard-after-restart",
        correlation_ids=("human-1",),
    )
    resumed = open_generation(
        accept_ingress(recovered, ACTIONS["resume"], "resume-after-restart")
    )
    with pytest.raises(DispatchBlocked, match="stale_or_missing_provenance"):
        authorize_and_start(resumed, old_envelope, set())

    paused = accept_ingress(running, ACTIONS["comment_hold"], "human-2")
    assert paused.bound_input_ids == ()
    recovered_paused = recover_after_restart(paused, new_runtime_epoch=2)
    assert recovered_paused.pending_input_ids == ("human-2",)

    stopped = accept_ingress(running, ACTIONS["stop"], "stop-1")
    recovered_stopped = recover_after_restart(stopped, new_runtime_epoch=2)
    assert recovered_stopped.status is TaskStatus.STOPPED
    done = replace(stopped, status=TaskStatus.DONE)
    recovered_done = recover_after_restart(done, new_runtime_epoch=2)
    assert recovered_done.status is TaskStatus.DONE


def test_ingress_dispatch_race_has_only_two_legal_orderings() -> None:
    running = open_generation(
        accept_ingress(TaskControlState(), ACTIONS["initial_submit"], "human-1")
    )
    old_envelope = envelope_for(running, "invoke-1")

    dispatched: set[str] = set()
    assert authorize_and_start(running, old_envelope, dispatched) == "started"
    after_dispatch = accept_ingress(running, ACTIONS["comment_hold"], "human-2")
    assert after_dispatch.status is TaskStatus.PAUSED

    ingress_first = accept_ingress(running, ACTIONS["comment_hold"], "human-2")
    with pytest.raises(DispatchBlocked, match="task_not_running"):
        authorize_and_start(ingress_first, old_envelope, set())


def test_sanitized_stale_todo_replay_records_current_runtime_gap(monkeypatch) -> None:
    """Expected model blocks; today's shared tool wrapper still dispatches.

    Replay: an old run carries a stale todo snapshot, compaction preserves that
    synthetic snapshot, a newer human input is durably accepted as comment/hold,
    and the old generation then attempts a tool dispatch.
    """
    running = open_generation(
        accept_ingress(TaskControlState(), ACTIONS["initial_submit"], "human-1")
    )
    old_envelope = envelope_for(running, "old-generation-write")

    after_todo_injection = accept_ingress(
        running, ACTIONS["synthetic_notice"], "stale-todo-snapshot"
    )
    after_compaction = accept_ingress(
        after_todo_injection,
        ACTIONS["synthetic_notice"],
        "compacted-stale-todo-snapshot",
    )
    assert after_compaction == running

    held = accept_ingress(after_compaction, ACTIONS["comment_hold"], "human-2")
    with pytest.raises(DispatchBlocked, match="task_not_running"):
        authorize_and_start(held, old_envelope, set())

    from hermes_cli import middleware

    monkeypatch.setattr(middleware, "_get_middleware_callbacks", lambda _kind: [])
    dispatched: list[dict] = []

    def current_dispatch(args: dict) -> str:
        dispatched.append(args)
        return "dispatched"

    result = middleware.run_tool_execution_middleware(
        "write_file",
        {"path": "artifact.txt", "content": "sanitized"},
        current_dispatch,
        task_id="task-1",
        session_id="session-1",
        turn_id="old-turn",
        api_request_id="old-generation",
    )
    assert result == "dispatched"
    assert dispatched == [{"path": "artifact.txt", "content": "sanitized"}]
