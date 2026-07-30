from dataclasses import replace
import hashlib
import sqlite3
import threading

import pytest

import hermes_state
from hermes_state import SessionDB
from task_fence import (
    AttemptTerminal,
    DecisionOutcome,
    DecisionReason,
    IngressEnvelope,
    OperationDescriptor,
    OperationKind,
    ResolutionDisposition,
    TASK_FENCE_ACTIONS,
    TaskFenceIngressRejected,
    TaskFenceIngressUnavailable,
    TaskFencePolicy,
    TaskFenceRecoveryUnavailable,
    TerminalReason,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ingress(
    action: str,
    source_event_id: str,
    *,
    task_id: str | None = None,
    correlation_ids: tuple[str, ...] = (),
    conversation_id: str = "recovery-conversation",
    resolution_disposition: ResolutionDisposition | None = None,
    evidence_refs: tuple[str, ...] = (),
) -> IngressEnvelope:
    return IngressEnvelope(
        source="gateway:test:recovery",
        source_event_id=source_event_id,
        conversation_id=conversation_id,
        action=TASK_FENCE_ACTIONS[action],
        payload_hash=_hash(source_event_id),
        task_id=task_id,
        correlation_ids=correlation_ids,
        resolution_disposition=resolution_disposition,
        evidence_refs=evidence_refs,
        terminal_reason=(TerminalReason.STOPPED if action == "stop" else None),
    )


def _operation(
    invocation_id: str,
    *,
    kind: OperationKind = OperationKind.TOOL,
) -> OperationDescriptor:
    return OperationDescriptor(
        invocation_id=invocation_id,
        kind=kind,
        adapter=(
            "provider:test-model"
            if kind is OperationKind.MODEL
            else "registry:write_file"
        ),
        invocation_fingerprint=_hash(invocation_id),
    )


def _recover(db: SessionDB):
    inspection = db.inspect_task_fence_store()
    assert inspection.runtime_epoch is not None
    assert inspection.mode_generation is not None
    return db.recover_task_fence_state(
        expected_runtime_epoch=inspection.runtime_epoch,
        expected_mode_generation=inspection.mode_generation,
    )


def _start_effect(
    db: SessionDB,
    *,
    marker: str,
    conversation_id: str = "recovery-conversation",
):
    accepted = db.accept_task_fence_ingress(
        _ingress(
            "initial_submit",
            f"{marker}-initial",
            conversation_id=conversation_id,
        )
    )
    generation = db.reserve_task_fence_generation(accepted)
    assert db.finish_task_fence_generation(generation, state="committed")
    invocation = generation.for_invocation(f"tfiv-{marker}")
    operation = _operation(invocation.invocation_id)
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(invocation, operation)
    started = policy.authorize_and_start(
        invocation,
        operation,
        admitted.permit_id,
    )
    return accepted, started


def _terminal_unknown_incident(
    db: SessionDB,
    *,
    marker: str,
    conversation_id: str = "recovery-conversation",
):
    accepted, started = _start_effect(
        db,
        marker=marker,
        conversation_id=conversation_id,
    )
    stopped = db.accept_task_fence_ingress(
        _ingress(
            "stop",
            f"{marker}-stop",
            task_id=accepted.task_id,
            conversation_id=conversation_id,
        )
    )
    assert stopped.task_projection is not None
    assert stopped.task_projection.status == "stopped"
    _recover(db)
    incident = db._conn.execute(
        "SELECT incident_id FROM task_fence_incidents "
        "WHERE task_id = ? AND state = 'open'",
        (accepted.task_id,),
    ).fetchone()
    assert incident is not None
    return accepted, started, incident[0]


def _task_fence_state(db: SessionDB) -> tuple[tuple[str, tuple[tuple, ...]], ...]:
    table_names = tuple(
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name GLOB 'task_fence_*' ORDER BY name"
        )
    )
    return tuple(
        (
            table,
            tuple(
                tuple(row)
                for row in db._conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
            ),
        )
        for table in table_names
    )


def _rewrite_append_only_row(
    db: SessionDB,
    *,
    trigger_name: str,
    statement: str,
    params: tuple,
) -> None:
    trigger = db._conn.execute(
        "SELECT sql FROM main.sqlite_master "
        "WHERE type = 'trigger' AND name = ?",
        (trigger_name,),
    ).fetchone()
    assert trigger is not None and isinstance(trigger[0], str)
    db._conn.execute("BEGIN IMMEDIATE")
    try:
        db._conn.execute(f'DROP TRIGGER main."{trigger_name}"')
        db._conn.execute(statement, params)
        db._conn.execute(trigger[0])
        db._conn.commit()
    except BaseException:
        db._conn.rollback()
        raise
    assert db.inspect_task_fence_store().compatible is True


def test_recovery_requeues_unpresented_input_without_rewriting_history(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    original_envelope = _ingress("initial_submit", "message-initial")
    db = SessionDB(path)
    accepted = db.accept_task_fence_ingress(original_envelope)
    generation = db.reserve_task_fence_generation(accepted)
    invocation = generation.for_invocation("tfiv-recovery-reserved")
    reserved_operation = _operation(
        invocation.invocation_id,
        kind=OperationKind.MODEL,
    )
    reserved = TaskFencePolicy(db).admit_operation(
        invocation,
        reserved_operation,
    )
    db.close()

    reopened = SessionDB(path)
    recovered = _recover(reopened)
    try:
        assert recovered.previous_runtime_epoch == 0
        assert recovered.runtime_epoch == 1
        task = reopened.inspect_task_fence_task(accepted.task_id).task
        assert task is not None
        assert task.status == "paused"
        assert task.current_runtime_epoch == 1
        assert task.active_execution_run_id is None
        assert task.current_generation_id is None
        assert tuple(
            reopened._conn.execute(
                "SELECT state, bound_run_id FROM task_fence_task_inputs "
                "WHERE event_id = ?",
                (accepted.event_id,),
            ).fetchone()
        ) == ("pending", None)
        assert tuple(
            reopened._conn.execute(
                "SELECT state, close_reason FROM task_fence_execution_runs "
                "WHERE run_id = ?",
                (accepted.opened_run_id,),
            ).fetchone()
        ) == ("closed", "runtime_recovery_unpresented")
        assert (
            reopened._conn.execute(
                "SELECT state FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (generation.generation_id,),
            ).fetchone()[0]
            == "cancelled"
        )
        assert (
            reopened._conn.execute(
                "SELECT state FROM task_fence_dispatch_permits WHERE permit_id = ?",
                (reserved.permit_id,),
            ).fetchone()[0]
            == "revoked"
        )
        recovery_row = tuple(
            reopened._conn.execute(
                "SELECT runtime_epoch, task_id, input_event_id, source_run_id, "
                "after_accepted_order FROM task_fence_recovery_requeues"
            ).fetchone()
        )
        assert recovery_row == (
            1,
            accepted.task_id,
            accepted.event_id,
            accepted.opened_run_id,
            accepted.accepted_order,
        )
        for statement in (
            "UPDATE task_fence_recovery_requeues SET requeued_at = 0.0",
            "DELETE FROM task_fence_recovery_requeues",
            "INSERT INTO task_fence_recovery_requeues "
            "SELECT * FROM task_fence_recovery_requeues",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                reopened._conn.execute(statement)

        replay = reopened.accept_task_fence_ingress(original_envelope)
        assert replay == replace(accepted, replayed=True)
        held = reopened.accept_task_fence_ingress(
            _ingress(
                "comment_hold",
                "message-after-recovery",
                task_id=accepted.task_id,
            )
        )
        assert held.pending_input_ids == (
            accepted.event_id,
            held.event_id,
        )
    finally:
        reopened.close()

    durable = SessionDB(path)
    try:
        replayed_held = durable.accept_task_fence_ingress(
            _ingress(
                "comment_hold",
                "message-after-recovery",
                task_id=accepted.task_id,
            )
        )
        assert replayed_held.replayed is True
        assert replayed_held.pending_input_ids == (
            accepted.event_id,
            replayed_held.event_id,
        )
        discarded = durable.accept_task_fence_ingress(
            _ingress(
                "discard_pending",
                "message-discard-recovered",
                task_id=accepted.task_id,
                correlation_ids=(accepted.event_id,),
            )
        )
        assert discarded.pending_input_ids == (replayed_held.event_id,)
        resumed = durable.accept_task_fence_ingress(
            _ingress(
                "change_and_run",
                "message-run-recovered",
                task_id=accepted.task_id,
            )
        )
        assert resumed.opened_run_id is not None
        assert resumed.pending_input_ids == ()
        permit_count = durable._conn.execute(
            "SELECT COUNT(*) FROM task_fence_dispatch_permits"
        ).fetchone()[0]
        attempt_count = durable._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0]
        stale = TaskFencePolicy(durable).admit_operation(
            invocation,
            reserved_operation,
        )
        assert stale.outcome is DecisionOutcome.WOULD_BLOCK
        assert stale.reason is DecisionReason.STALE_AUTHORITY
        revoked = TaskFencePolicy(durable).authorize_and_start(
            invocation,
            reserved_operation,
            reserved.permit_id,
        )
        assert revoked.outcome is DecisionOutcome.WOULD_BLOCK
        assert revoked.reason is DecisionReason.PERMIT_REVOKED
        assert (
            durable._conn.execute(
                "SELECT COUNT(*) FROM task_fence_dispatch_permits"
            ).fetchone()[0]
            == permit_count
        )
        assert (
            durable._conn.execute(
                "SELECT COUNT(*) FROM task_fence_attempts"
            ).fetchone()[0]
            == attempt_count
        )
    finally:
        durable.close()


def test_recovery_preserves_accepted_input_before_generation_exists(tmp_path) -> None:
    path = tmp_path / "state.db"
    envelope = _ingress("initial_submit", "message-before-generation")
    db = SessionDB(path)
    accepted = db.accept_task_fence_ingress(envelope)
    db.close()

    reopened = SessionDB(path)
    recovered = _recover(reopened)
    try:
        assert (recovered.previous_runtime_epoch, recovered.runtime_epoch) == (0, 1)
        task = reopened.inspect_task_fence_task(accepted.task_id).task
        assert task is not None
        assert task.status == "paused"
        assert tuple(
            reopened._conn.execute(
                "SELECT state, bound_run_id FROM task_fence_task_inputs "
                "WHERE event_id = ?",
                (accepted.event_id,),
            ).fetchone()
        ) == ("pending", None)
        assert tuple(
            reopened._conn.execute(
                "SELECT state, close_reason FROM task_fence_execution_runs "
                "WHERE run_id = ?",
                (accepted.opened_run_id,),
            ).fetchone()
        ) == ("closed", "runtime_recovery_unpresented")
        assert (
            reopened._conn.execute(
                "SELECT COUNT(*) FROM task_fence_model_generations"
            ).fetchone()[0]
            == 0
        )
        assert reopened.accept_task_fence_ingress(envelope) == replace(
            accepted,
            replayed=True,
        )
    finally:
        reopened.close()


def test_recovery_rejects_latent_skipped_epoch_ledger_without_mutation(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    accepted = db.accept_task_fence_ingress(
        _ingress("initial_submit", "message-ledger-corruption")
    )
    _recover(db)
    _recover(db)
    resumed = db.accept_task_fence_ingress(
        _ingress(
            "change_and_run",
            "message-after-ledger-boundary",
            task_id=accepted.task_id,
        )
    )
    assert resumed.opened_run_id is not None

    _rewrite_append_only_row(
        db,
        trigger_name="task_fence_recovery_requeues_no_update",
        statement=(
            "UPDATE task_fence_recovery_requeues SET runtime_epoch = 2 "
            "WHERE input_event_id = ?"
        ),
        params=(accepted.event_id,),
    )

    before = _task_fence_state(db)
    with pytest.raises(
        TaskFenceRecoveryUnavailable,
        match="incompatible_recovery_projection",
    ):
        _recover(db)
    assert _task_fence_state(db) == before
    db.close()


def test_recovery_does_not_resurrect_uncommitted_ingress(tmp_path) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_before_ingress_commit "
        "BEFORE INSERT ON task_fence_acceptance_snapshots BEGIN "
        "SELECT RAISE(ABORT, 'injected pre-commit fault'); END"
    )
    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="acceptance_database_error",
    ):
        db.accept_task_fence_ingress(
            _ingress("initial_submit", "message-never-committed")
        )
    db.close()

    reopened = SessionDB(path)
    try:
        for table in (
            "task_fence_tasks",
            "task_fence_ingress",
            "task_fence_task_inputs",
            "task_fence_execution_runs",
            "task_fence_acceptance_snapshots",
        ):
            assert (
                reopened._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                == 0
            )
        recovered = _recover(reopened)
        assert (recovered.previous_runtime_epoch, recovered.runtime_epoch) == (0, 1)
        assert (
            reopened._conn.execute("SELECT COUNT(*) FROM task_fence_tasks").fetchone()[
                0
            ]
            == 0
        )
    finally:
        reopened.close()


def test_recovery_turns_started_effect_into_one_durable_incident(tmp_path) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    accepted = db.accept_task_fence_ingress(
        _ingress("initial_submit", "message-started")
    )
    generation = db.reserve_task_fence_generation(accepted)
    assert db.finish_task_fence_generation(generation, state="committed")
    invocation = generation.for_invocation("tfiv-recovery-started")
    operation = _operation(invocation.invocation_id)
    policy = TaskFencePolicy(db)
    reserved = policy.admit_operation(invocation, operation)
    started = policy.authorize_and_start(
        invocation,
        operation,
        reserved.permit_id,
    )
    db.close()

    reopened = SessionDB(path)
    first = _recover(reopened)
    after_first = _task_fence_state(reopened)
    with pytest.raises(
        TaskFenceRecoveryUnavailable,
        match="runtime_epoch_changed",
    ):
        reopened.recover_task_fence_state(
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )
    assert _task_fence_state(reopened) == after_first
    second = _recover(reopened)
    try:
        assert (first.previous_runtime_epoch, first.runtime_epoch) == (0, 1)
        assert (second.previous_runtime_epoch, second.runtime_epoch) == (1, 2)
        task = reopened.inspect_task_fence_task(accepted.task_id).task
        assert task is not None
        assert task.status == "incident"
        assert task.current_runtime_epoch == 2
        frontier = reopened.inspect_task_fence_conversation(
            "recovery-conversation"
        )
        assert frontier.compatible is True
        assert frontier.reason == "compatible"
        assert frontier.task is not None
        assert frontier.task.task_id == accepted.task_id
        assert frontier.active_run is None
        assert frontier.pending_input_ids == ()
        assert frontier.pending_inputs_truncated is False
        assert frontier.started_attempts == ()
        assert frontier.started_attempts_truncated is False
        assert frontier.open_incident is not None
        assert frontier.open_incident.source_run_id == accepted.opened_run_id
        assert frontier.open_incident.reason_code == "outcome_unknown"
        assert frontier.open_incident.attempt_ids == (started.attempt_id,)
        assert (
            reopened._conn.execute(
                "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
                (started.attempt_id,),
            ).fetchone()[0]
            == "OUTCOME_UNKNOWN"
        )
        assert tuple(
            tuple(row)
            for row in reopened._conn.execute(
                "SELECT from_state, to_state, evidence_ref "
                "FROM task_fence_attempt_transitions WHERE attempt_id = ? "
                "ORDER BY transition_order",
                (started.attempt_id,),
            )
        ) == (
            (
                None,
                "STARTED",
                hermes_state.operation_binding_fingerprint(
                    invocation,
                    operation,
                ),
            ),
            ("STARTED", "OUTCOME_UNKNOWN", None),
        )
        incidents = reopened._conn.execute(
            "SELECT incident_id, state, reason_code FROM task_fence_incidents "
            "WHERE task_id = ?",
            (accepted.task_id,),
        ).fetchall()
        assert len(incidents) == 1
        assert tuple(incidents[0][1:]) == ("open", "outcome_unknown")
        assert (
            reopened._conn.execute(
                "SELECT attempt_id FROM task_fence_incident_attempts "
                "WHERE incident_id = ?",
                (incidents[0][0],),
            ).fetchone()[0]
            == started.attempt_id
        )
        assert (
            reopened._conn.execute(
                "SELECT COUNT(*) FROM task_fence_recovery_requeues WHERE task_id = ?",
                (accepted.task_id,),
            ).fetchone()[0]
            == 0
        )

        blocked = TaskFencePolicy(reopened).admit_operation(
            invocation.for_invocation("tfiv-stale-after-recovery"),
            _operation("tfiv-stale-after-recovery"),
        )
        assert blocked.outcome is DecisionOutcome.WOULD_BLOCK

        reopened._conn.execute(
            "UPDATE task_fence_attempts "
            "SET recovery_classification = 'known_read' "
            "WHERE attempt_id = ?",
            (started.attempt_id,),
        )
        reopened._conn.commit()
        assert reopened.inspect_task_fence_store().compatible is True
        corrupt = reopened.inspect_task_fence_conversation(
            "recovery-conversation"
        )
        assert corrupt.compatible is False
        assert corrupt.reason == "incompatible_open_incident_projection"
        assert corrupt.task is None
        assert corrupt.open_incident is None
    finally:
        reopened.close()


@pytest.mark.parametrize("attempt_count", (64, 65))
def test_conversation_inspection_requires_exact_bounded_incident_links(
    tmp_path,
    monkeypatch,
    attempt_count,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    accepted = db.accept_task_fence_ingress(
        _ingress("initial_submit", f"incident-bound-{attempt_count}")
    )
    generation = db.reserve_task_fence_generation(accepted)
    assert db.finish_task_fence_generation(generation, state="committed")
    policy = TaskFencePolicy(db)
    started_ids = []
    for index in range(attempt_count):
        invocation_id = f"tfiv-incident-bound-{attempt_count}-{index}"
        invocation = generation.for_invocation(invocation_id)
        operation = _operation(invocation_id)
        admitted = policy.admit_operation(invocation, operation)
        started = policy.authorize_and_start(
            invocation,
            operation,
            admitted.permit_id,
        )
        assert started.attempt_id is not None
        started_ids.append(started.attempt_id)
    db.close()

    reopened = SessionDB(path)
    inspection_limit = hermes_state._TASK_FENCE_MAX_RECOVERY_INCIDENT_ATTEMPTS
    monkeypatch.setattr(
        hermes_state,
        "_TASK_FENCE_MAX_RECOVERY_INCIDENT_ATTEMPTS",
        attempt_count,
    )
    _recover(reopened)
    monkeypatch.setattr(
        hermes_state,
        "_TASK_FENCE_MAX_RECOVERY_INCIDENT_ATTEMPTS",
        inspection_limit,
    )
    try:
        inspected = reopened.inspect_task_fence_conversation(
            "recovery-conversation"
        )
    finally:
        reopened.close()

    if attempt_count == inspection_limit:
        assert inspected.compatible is True
        assert inspected.open_incident is not None
        assert inspected.open_incident.attempt_ids == tuple(sorted(started_ids))
        assert inspected.started_attempts == ()
    else:
        assert inspected.compatible is False
        assert inspected.reason == "open_incident_attempt_limit_exceeded"
        assert inspected.task is None
        assert inspected.open_incident is None


@pytest.mark.parametrize(
    "resolution_fault",
    (None, "action", "correlation", "evidence", "cardinality"),
)
def test_recovery_adopts_preexisting_unknowns_into_one_durable_incident(
    tmp_path,
    monkeypatch,
    resolution_fault,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    accepted = db.accept_task_fence_ingress(
        _ingress("initial_submit", "message-preexisting-unknown")
    )
    generation = db.reserve_task_fence_generation(accepted)
    assert db.finish_task_fence_generation(generation, state="committed")
    policy = TaskFencePolicy(db)
    attempt_ids = []
    for suffix in ("first", "second"):
        invocation = generation.for_invocation(f"tfiv-preexisting-unknown-{suffix}")
        operation = _operation(invocation.invocation_id)
        reserved = policy.admit_operation(invocation, operation)
        started = policy.authorize_and_start(
            invocation,
            operation,
            reserved.permit_id,
        )
        policy.finish_attempt(
            started.attempt_id,
            AttemptTerminal.OUTCOME_UNKNOWN,
            None,
        )
        attempt_ids.append(started.attempt_id)

    def transitions(attempt_id: str) -> tuple[tuple, ...]:
        return tuple(
            tuple(row)
            for row in db._conn.execute(
                "SELECT from_state, to_state, disposition, evidence_ref "
                "FROM task_fence_attempt_transitions WHERE attempt_id = ? "
                "ORDER BY transition_order",
                (attempt_id,),
            )
        )

    transitions_before = {
        attempt_id: transitions(attempt_id) for attempt_id in attempt_ids
    }
    assert all(len(items) == 2 for items in transitions_before.values())
    assert db._conn.execute(
        "SELECT COUNT(*) FROM task_fence_incident_attempts "
        "WHERE attempt_id IN (?, ?)",
        tuple(attempt_ids),
    ).fetchone()[0] == 0
    db.close()

    reopened = SessionDB(path)

    def reopened_transitions(attempt_id: str) -> tuple[tuple, ...]:
        return tuple(
            tuple(row)
            for row in reopened._conn.execute(
                "SELECT from_state, to_state, disposition, evidence_ref "
                "FROM task_fence_attempt_transitions WHERE attempt_id = ? "
                "ORDER BY transition_order",
                (attempt_id,),
            )
        )

    first = _recover(reopened)
    try:
        assert (first.previous_runtime_epoch, first.runtime_epoch) == (0, 1)
        task = reopened.inspect_task_fence_task(accepted.task_id).task
        assert task is not None
        assert task.status == "incident"
        incident = reopened._conn.execute(
            "SELECT incident_id, state, reason_code FROM task_fence_incidents "
            "WHERE task_id = ?",
            (accepted.task_id,),
        ).fetchone()
        assert incident is not None
        assert tuple(incident[1:]) == ("open", "outcome_unknown")
        assert tuple(
            row[0]
            for row in reopened._conn.execute(
                "SELECT attempt_id FROM task_fence_incident_attempts "
                "WHERE incident_id = ? ORDER BY attempt_id",
                (incident[0],),
            )
        ) == tuple(sorted(attempt_ids))
        assert all(
            reopened_transitions(attempt_id) == transitions_before[attempt_id]
            for attempt_id in attempt_ids
        )
        if resolution_fault == "cardinality":
            monkeypatch.setattr(
                hermes_state,
                "_TASK_FENCE_MAX_RECOVERY_INCIDENT_ATTEMPTS",
                1,
            )
            before = _task_fence_state(reopened)
            with pytest.raises(
                TaskFenceRecoveryUnavailable,
                match="incompatible_recovery_projection",
            ):
                _recover(reopened)
            assert _task_fence_state(reopened) == before
            return
        with pytest.raises(TaskFenceIngressRejected):
            reopened.accept_task_fence_ingress(
                _ingress(
                    "resume",
                    "message-resume-unresolved-unknown",
                    task_id=accepted.task_id,
                )
            )

        second = _recover(reopened)
        assert (second.previous_runtime_epoch, second.runtime_epoch) == (1, 2)
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM task_fence_incidents WHERE task_id = ?",
            (accepted.task_id,),
        ).fetchone()[0] == 1
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM task_fence_incident_attempts "
            "WHERE attempt_id IN (?, ?)",
            tuple(attempt_ids),
        ).fetchone()[0] == 2

        resolved = reopened.accept_task_fence_ingress(
            _ingress(
                "resolve_incident",
                "message-resolve-preexisting-unknowns",
                task_id=accepted.task_id,
                correlation_ids=tuple(sorted(attempt_ids)),
                resolution_disposition=(
                    ResolutionDisposition.ACCEPTED_UNKNOWN_NO_RETRY
                ),
                evidence_refs=("operator:recovery-review",),
            )
        )
        resolved_task = reopened.inspect_task_fence_task(accepted.task_id).task
        assert resolved_task is not None
        assert resolved_task.status == "paused"
        resolved_frontier = reopened.inspect_task_fence_conversation(
            "recovery-conversation"
        )
        assert resolved_frontier.compatible is True
        assert resolved_frontier.task is not None
        assert resolved_frontier.open_incident is None

        resolution = reopened._conn.execute(
            "SELECT resolution_id FROM task_fence_resolutions "
            "WHERE incident_id = ?",
            (incident[0],),
        ).fetchone()
        assert resolution is not None
        if resolution_fault is not None:
            trigger_name, statement, params = {
                "action": (
                    "task_fence_ingress_no_update",
                    "UPDATE task_fence_ingress SET origin = 'runtime' "
                    "WHERE event_id = ?",
                    (resolved.event_id,),
                ),
                "correlation": (
                    "task_fence_ingress_correlations_no_update",
                    "UPDATE task_fence_ingress_correlations "
                    "SET correlation_id = 'tfat_forged' "
                    "WHERE event_id = ? AND correlation_id = ?",
                    (resolved.event_id, attempt_ids[0]),
                ),
                "evidence": (
                    "task_fence_resolution_evidence_no_update",
                    "UPDATE task_fence_resolution_evidence "
                    "SET evidence_ref = 'operator:forged' "
                    "WHERE resolution_id = ?",
                    (resolution[0],),
                ),
            }[resolution_fault]
            _rewrite_append_only_row(
                reopened,
                trigger_name=trigger_name,
                statement=statement,
                params=params,
            )
            before = _task_fence_state(reopened)
            with pytest.raises(
                TaskFenceRecoveryUnavailable,
                match="incompatible_recovery_projection",
            ):
                _recover(reopened)
            assert _task_fence_state(reopened) == before
            return

        third = _recover(reopened)
        assert (third.previous_runtime_epoch, third.runtime_epoch) == (2, 3)
        assert tuple(
            reopened._conn.execute(
                "SELECT state, reason_code FROM task_fence_incidents "
                "WHERE incident_id = ?",
                (incident[0],),
            ).fetchone()
        ) == ("resolved", "outcome_unknown")
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM task_fence_incidents WHERE task_id = ?",
            (accepted.task_id,),
        ).fetchone()[0] == 1
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM task_fence_incidents "
            "WHERE task_id = ? AND state = 'open'",
            (accepted.task_id,),
        ).fetchone()[0] == 0
        assert tuple(
            row[0]
            for row in reopened._conn.execute(
                "SELECT attempt_id FROM task_fence_incident_attempts "
                "WHERE incident_id = ? ORDER BY attempt_id",
                (incident[0],),
            )
        ) == tuple(sorted(attempt_ids))
        assert all(
            reopened_transitions(attempt_id) == transitions_before[attempt_id]
            for attempt_id in attempt_ids
        )
        recovered_task = reopened.inspect_task_fence_task(accepted.task_id).task
        assert recovered_task is not None
        assert recovered_task.status == "paused"
    finally:
        reopened.close()


def test_recovery_rejects_empty_preexisting_incident_instead_of_repairing_it(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    accepted = db.accept_task_fence_ingress(
        _ingress("initial_submit", "message-empty-incident")
    )
    generation = db.reserve_task_fence_generation(accepted)
    assert db.finish_task_fence_generation(generation, state="committed")
    invocation = generation.for_invocation("tfiv-empty-incident")
    operation = _operation(invocation.invocation_id)
    policy = TaskFencePolicy(db)
    reserved = policy.admit_operation(invocation, operation)
    started = policy.authorize_and_start(
        invocation,
        operation,
        reserved.permit_id,
    )
    policy.finish_attempt(
        started.attempt_id,
        AttemptTerminal.OUTCOME_UNKNOWN,
        None,
    )
    stopped = db.accept_task_fence_ingress(
        _ingress("stop", "message-empty-incident-stop", task_id=accepted.task_id)
    )
    assert stopped.task_projection is not None
    assert stopped.task_projection.status == "stopped"
    terminal_at = db._conn.execute(
        "SELECT terminal_at FROM task_fence_attempts WHERE attempt_id = ?",
        (started.attempt_id,),
    ).fetchone()[0]
    db._conn.execute(
        "INSERT INTO task_fence_incidents ("
        "incident_id, task_id, source_run_id, reason_code, state, opened_at"
        ") VALUES ('tfin_empty_preexisting', ?, ?, "
        "'outcome_unknown', 'open', ?)",
        (accepted.task_id, accepted.opened_run_id, terminal_at),
    )

    before = _task_fence_state(db)
    with pytest.raises(
        TaskFenceRecoveryUnavailable,
        match="incompatible_recovery_projection",
    ):
        _recover(db)
    assert _task_fence_state(db) == before
    assert db._conn.execute(
        "SELECT COUNT(*) FROM task_fence_incident_attempts"
    ).fetchone()[0] == 0
    db.close()


def test_recovery_deep_validates_only_mutable_or_unresolved_permits(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    accepted = db.accept_task_fence_ingress(
        _ingress("initial_submit", "message-bounded-frontier")
    )
    policy = TaskFencePolicy(db)
    historical_reserved_ids = []
    for ordinal in range(16):
        historical_generation = db.reserve_task_fence_generation(accepted)
        assert db.finish_task_fence_generation(
            historical_generation,
            state="committed",
        )
        historical_invocation = historical_generation.for_invocation(
            f"tfiv-historical-reserved-{ordinal}"
        )
        historical_permit = policy.admit_operation(
            historical_invocation,
            _operation(historical_invocation.invocation_id),
        )
        assert historical_permit.permit_id is not None
        historical_reserved_ids.append(historical_permit.permit_id)

    generation = db.reserve_task_fence_generation(accepted)
    assert db.finish_task_fence_generation(generation, state="committed")

    terminal_permit_ids = []
    for ordinal in range(32):
        invocation = generation.for_invocation(f"tfiv-terminal-history-{ordinal}")
        operation = _operation(invocation.invocation_id)
        permit = policy.admit_operation(invocation, operation)
        started = policy.authorize_and_start(
            invocation,
            operation,
            permit.permit_id,
        )
        policy.finish_attempt(
            started.attempt_id,
            (
                AttemptTerminal.SUCCEEDED
                if ordinal % 2 == 0
                else AttemptTerminal.FAILED_DEFINITE
            ),
            f"ack:terminal-history-{ordinal}",
        )
        terminal_permit_ids.append(permit.permit_id)

    reserved_invocation = generation.for_invocation("tfiv-frontier-reserved")
    reserved_operation = _operation(reserved_invocation.invocation_id)
    reserved = policy.admit_operation(reserved_invocation, reserved_operation)
    started_invocation = generation.for_invocation("tfiv-frontier-started")
    started_operation = _operation(started_invocation.invocation_id)
    started_permit = policy.admit_operation(started_invocation, started_operation)
    started = policy.authorize_and_start(
        started_invocation,
        started_operation,
        started_permit.permit_id,
    )

    terminal_rows_before = tuple(
        tuple(row)
        for row in db._conn.execute(
            "SELECT permit.*, attempt.* "
            "FROM task_fence_dispatch_permits AS permit "
            "JOIN task_fence_attempts AS attempt "
            "ON attempt.permit_id = permit.permit_id "
            "WHERE attempt.state IN ('SUCCEEDED', 'FAILED_DEFINITE') "
            "ORDER BY permit.permit_id"
        )
    )
    terminal_transitions_before = tuple(
        tuple(row)
        for row in db._conn.execute(
            "SELECT transition.* "
            "FROM task_fence_attempt_transitions AS transition "
            "JOIN task_fence_attempts AS attempt "
            "ON attempt.attempt_id = transition.attempt_id "
            "WHERE attempt.state IN ('SUCCEEDED', 'FAILED_DEFINITE') "
            "ORDER BY transition.transition_order"
        )
    )
    assert len(terminal_rows_before) == len(terminal_permit_ids) == 32
    assert len(historical_reserved_ids) == 16
    db.close()

    checked_permit_ids = []
    original_validator = (
        SessionDB._task_fence_policy_permit_storage_compatible_unlocked
    )

    def track_validator(self, conn, permit):
        checked_permit_ids.append(permit["permit_id"])
        return original_validator(self, conn, permit)

    monkeypatch.setattr(
        SessionDB,
        "_task_fence_policy_permit_storage_compatible_unlocked",
        track_validator,
    )
    reopened = SessionDB(path)
    try:
        _recover(reopened)
        assert set(checked_permit_ids) == {
            *historical_reserved_ids,
            reserved.permit_id,
            started_permit.permit_id,
        }
        assert set(checked_permit_ids).isdisjoint(terminal_permit_ids)
        assert reopened._conn.execute(
            "SELECT state FROM task_fence_dispatch_permits WHERE permit_id = ?",
            (reserved.permit_id,),
        ).fetchone()[0] == "revoked"
        assert reopened._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0] == "OUTCOME_UNKNOWN"
        assert tuple(
            tuple(row)
            for row in reopened._conn.execute(
                "SELECT permit.*, attempt.* "
                "FROM task_fence_dispatch_permits AS permit "
                "JOIN task_fence_attempts AS attempt "
                "ON attempt.permit_id = permit.permit_id "
                "WHERE attempt.state IN ('SUCCEEDED', 'FAILED_DEFINITE') "
                "ORDER BY permit.permit_id"
            )
        ) == terminal_rows_before
        assert tuple(
            tuple(row)
            for row in reopened._conn.execute(
                "SELECT transition.* "
                "FROM task_fence_attempt_transitions AS transition "
                "JOIN task_fence_attempts AS attempt "
                "ON attempt.attempt_id = transition.attempt_id "
                "WHERE attempt.state IN ('SUCCEEDED', 'FAILED_DEFINITE') "
                "ORDER BY transition.transition_order"
            )
        ) == terminal_transitions_before
        assert {
            row[0]
            for row in reopened._conn.execute(
                "SELECT state FROM task_fence_dispatch_permits "
                "WHERE permit_id IN ("
                + ",".join("?" for _permit_id in historical_reserved_ids)
                + ")",
                historical_reserved_ids,
            )
        } == {"revoked"}
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "terminal",
    (AttemptTerminal.SUCCEEDED, AttemptTerminal.FAILED_DEFINITE),
)
def test_recovery_rejects_terminal_projection_without_terminal_proof(
    tmp_path,
    terminal,
) -> None:
    path = tmp_path / f"state-{terminal.value}.db"
    db = SessionDB(path)
    accepted = db.accept_task_fence_ingress(
        _ingress("initial_submit", f"message-forged-{terminal.value}")
    )
    generation = db.reserve_task_fence_generation(accepted)
    assert db.finish_task_fence_generation(generation, state="committed")
    invocation = generation.for_invocation(f"tfiv-forged-{terminal.value}")
    operation = _operation(invocation.invocation_id)
    policy = TaskFencePolicy(db)
    permit = policy.admit_operation(invocation, operation)
    started = policy.authorize_and_start(
        invocation,
        operation,
        permit.permit_id,
    )
    started_at = db._conn.execute(
        "SELECT started_at FROM task_fence_attempts WHERE attempt_id = ?",
        (started.attempt_id,),
    ).fetchone()[0]
    db._conn.execute(
        "UPDATE task_fence_attempts SET state = ?, disposition = ?, "
        "acknowledgement_ref = ?, terminal_at = ? WHERE attempt_id = ?",
        (
            terminal.value,
            terminal.value,
            f"ack:forged-{terminal.value}",
            started_at,
            started.attempt_id,
        ),
    )
    assert db._conn.execute(
        "SELECT COUNT(*) FROM task_fence_attempt_transitions "
        "WHERE attempt_id = ?",
        (started.attempt_id,),
    ).fetchone()[0] == 1
    db.close()

    reopened = SessionDB(path)
    try:
        before = _task_fence_state(reopened)
        with pytest.raises(
            TaskFenceRecoveryUnavailable,
            match="incompatible_recovery_projection",
        ):
            _recover(reopened)
        assert _task_fence_state(reopened) == before
    finally:
        reopened.close()


def test_recovery_frontier_limit_rejects_without_mutation(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    accepted = db.accept_task_fence_ingress(
        _ingress("initial_submit", "message-frontier-limit")
    )
    generation = db.reserve_task_fence_generation(accepted)
    assert db.finish_task_fence_generation(generation, state="committed")
    policy = TaskFencePolicy(db)
    for ordinal in range(2):
        invocation = generation.for_invocation(f"tfiv-frontier-limit-{ordinal}")
        policy.admit_operation(invocation, _operation(invocation.invocation_id))
    started_invocation = generation.for_invocation("tfiv-frontier-limit-started")
    started_operation = _operation(started_invocation.invocation_id)
    started_permit = policy.admit_operation(started_invocation, started_operation)
    policy.authorize_and_start(
        started_invocation,
        started_operation,
        started_permit.permit_id,
    )
    db.close()

    monkeypatch.setattr(hermes_state, "_TASK_FENCE_MAX_RECOVERY_AUTHORITIES", 2)
    reopened = SessionDB(path)
    try:
        before = _task_fence_state(reopened)
        with pytest.raises(
            TaskFenceRecoveryUnavailable,
            match="recovery_authority_limit_exceeded",
        ):
            _recover(reopened)
        assert _task_fence_state(reopened) == before
    finally:
        reopened.close()


@pytest.mark.parametrize("fault_target", ("task", "control"))
def test_recovery_fault_rolls_back_every_authority_write(
    tmp_path,
    fault_target,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    accepted = db.accept_task_fence_ingress(
        _ingress("initial_submit", "message-rollback")
    )
    db.reserve_task_fence_generation(accepted)
    effect = db.accept_task_fence_ingress(
        _ingress(
            "initial_submit",
            "message-rollback-effect",
            conversation_id="recovery-rollback-effect",
        )
    )
    effect_generation = db.reserve_task_fence_generation(effect)
    assert db.finish_task_fence_generation(effect_generation, state="committed")
    effect_invocation = effect_generation.for_invocation("tfiv-rollback-effect")
    effect_operation = _operation(effect_invocation.invocation_id)
    effect_policy = TaskFencePolicy(db)
    effect_permit = effect_policy.admit_operation(
        effect_invocation,
        effect_operation,
    )
    effect_policy.authorize_and_start(
        effect_invocation,
        effect_operation,
        effect_permit.permit_id,
    )
    before = _task_fence_state(db)
    table, condition = {
        "task": (
            "task_fence_tasks",
            "NEW.current_runtime_epoch != OLD.current_runtime_epoch",
        ),
        "control": (
            "task_fence_control",
            "NEW.runtime_epoch != OLD.runtime_epoch",
        ),
    }[fault_target]
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_task_fence_recovery "
        f"BEFORE UPDATE ON {table} "
        f"WHEN {condition} BEGIN "
        "SELECT RAISE(ABORT, 'private recovery fault'); END"
    )

    with pytest.raises(
        TaskFenceRecoveryUnavailable,
        match="recovery_database_error",
    ) as raised:
        _recover(db)

    assert raised.value.__cause__ is None
    assert "private" not in str(raised.value)
    assert _task_fence_state(db) == before
    db.close()


def test_recovery_preserves_terminal_task_and_records_unknown_started_effect(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    accepted = db.accept_task_fence_ingress(
        _ingress("initial_submit", "message-terminal-started")
    )
    generation = db.reserve_task_fence_generation(accepted)
    assert db.finish_task_fence_generation(generation, state="committed")
    invocation = generation.for_invocation("tfiv-terminal-started")
    operation = _operation(invocation.invocation_id)
    policy = TaskFencePolicy(db)
    reserved = policy.admit_operation(invocation, operation)
    started = policy.authorize_and_start(
        invocation,
        operation,
        reserved.permit_id,
    )
    stopped = db.accept_task_fence_ingress(
        _ingress("stop", "message-terminal-stop", task_id=accepted.task_id)
    )
    assert stopped.task_projection is not None
    assert stopped.task_projection.status == "stopped"
    first = _recover(db)
    assert (first.previous_runtime_epoch, first.runtime_epoch) == (0, 1)
    task = db.inspect_task_fence_task(accepted.task_id).task
    assert task is not None
    assert task.status == "stopped"
    assert task.current_runtime_epoch == 1
    assert (
        db._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0]
        == "OUTCOME_UNKNOWN"
    )
    incident = db._conn.execute(
        "SELECT incident_id, state, reason_code FROM task_fence_incidents "
        "WHERE task_id = ?",
        (accepted.task_id,),
    ).fetchone()
    assert incident is not None
    assert tuple(incident[1:]) == ("open", "outcome_unknown")
    assert (
        db._conn.execute(
            "SELECT attempt_id FROM task_fence_incident_attempts WHERE incident_id = ?",
            (incident[0],),
        ).fetchone()[0]
        == started.attempt_id
    )
    terminal_frontier = db.inspect_task_fence_conversation(
        "recovery-conversation"
    )
    assert terminal_frontier.compatible is True
    assert terminal_frontier.reason == "no_active_task"
    assert terminal_frontier.task is None
    assert terminal_frontier.open_incident is None
    assert len(terminal_frontier.terminal_open_incidents) == 1
    terminal_incident = terminal_frontier.terminal_open_incidents[0]
    assert terminal_incident.task_id == accepted.task_id
    assert terminal_incident.task_status == "stopped"
    assert terminal_incident.incident.incident_id == incident[0]
    assert terminal_incident.incident.source_run_id == accepted.opened_run_id
    assert terminal_incident.incident.reason_code == "outcome_unknown"
    assert terminal_incident.incident.attempt_ids == (started.attempt_id,)

    second = _recover(db)
    assert (second.previous_runtime_epoch, second.runtime_epoch) == (1, 2)
    task = db.inspect_task_fence_task(accepted.task_id).task
    assert task is not None
    assert task.status == "stopped"
    assert (
        db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_incidents WHERE task_id = ?",
            (accepted.task_id,),
        ).fetchone()[0]
        == 1
    )
    terminal_replay = db.inspect_task_fence_conversation("recovery-conversation")
    assert (
        terminal_replay.terminal_open_incidents
        == terminal_frontier.terminal_open_incidents
    )
    db.close()


def test_terminal_open_incident_coexists_with_new_active_task_and_is_exact(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    conversation_id = "terminal-exact:raw-secret-lane"
    terminal, started, incident_id = _terminal_unknown_incident(
        db,
        marker="terminal-exact-target",
        conversation_id=conversation_id,
    )
    foreign, _, foreign_incident_id = _terminal_unknown_incident(
        db,
        marker="terminal-exact-foreign",
        conversation_id="terminal-exact:foreign-secret-lane",
    )
    active = db.accept_task_fence_ingress(
        _ingress(
            "initial_submit",
            "terminal-exact-active",
            conversation_id=conversation_id,
        )
    )

    inspected = db.inspect_task_fence_conversation(conversation_id)
    db.close()

    assert inspected.compatible is True
    assert inspected.task is not None
    assert inspected.task.task_id == active.task_id
    assert inspected.active_run is not None
    assert inspected.active_run.run_id == active.opened_run_id
    assert len(inspected.terminal_open_incidents) == 1
    projection = inspected.terminal_open_incidents[0]
    assert projection.task_id == terminal.task_id
    assert projection.task_status == "stopped"
    assert projection.incident.incident_id == incident_id
    assert projection.incident.source_run_id == terminal.opened_run_id
    assert projection.incident.attempt_ids == (started.attempt_id,)
    rendered = repr(inspected)
    assert foreign.task_id not in rendered
    assert foreign_incident_id not in rendered
    assert conversation_id not in rendered
    assert "foreign-secret-lane" not in rendered


def test_terminal_open_incident_set_is_deterministic_and_exact_bounded(
    tmp_path,
    monkeypatch,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    conversation_id = "terminal-set-bound"
    first, _, _ = _terminal_unknown_incident(
        db,
        marker="terminal-set-first",
        conversation_id=conversation_id,
    )
    second, _, _ = _terminal_unknown_incident(
        db,
        marker="terminal-set-second",
        conversation_id=conversation_id,
    )
    db._conn.execute(
        "UPDATE task_fence_tasks SET status = 'done' WHERE task_id = ?",
        (first.task_id,),
    )
    db._conn.commit()
    active = db.accept_task_fence_ingress(
        _ingress(
            "initial_submit",
            "terminal-set-active",
            conversation_id=conversation_id,
        )
    )

    monkeypatch.setattr(
        hermes_state,
        "_TASK_FENCE_MAX_TERMINAL_OPEN_INCIDENTS",
        2,
    )
    complete = db.inspect_task_fence_conversation(conversation_id)
    replay = db.inspect_task_fence_conversation(conversation_id)
    assert complete.compatible is True
    assert complete.task is not None
    assert complete.task.task_id == active.task_id
    assert complete.terminal_open_incidents == replay.terminal_open_incidents
    assert tuple(item.task_id for item in complete.terminal_open_incidents) == tuple(
        sorted((first.task_id, second.task_id))
    )
    assert {
        item.task_id: item.task_status
        for item in complete.terminal_open_incidents
    } == {first.task_id: "done", second.task_id: "stopped"}

    monkeypatch.setattr(
        hermes_state,
        "_TASK_FENCE_MAX_TERMINAL_OPEN_INCIDENTS",
        1,
    )
    overflow = db.inspect_task_fence_conversation(conversation_id)
    db.close()

    assert overflow.compatible is False
    assert overflow.reason == "terminal_open_incident_limit_exceeded"
    assert overflow.task is None
    assert overflow.active_run is None
    assert overflow.open_incident is None
    assert overflow.terminal_open_incidents == ()


def test_terminal_open_incident_scan_uses_partial_index_and_global_work_bound(
    tmp_path,
    monkeypatch,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    _terminal_unknown_incident(
        db,
        marker="terminal-work-first",
        conversation_id="terminal-work-foreign-a",
    )
    _terminal_unknown_incident(
        db,
        marker="terminal-work-second",
        conversation_id="terminal-work-foreign-b",
    )
    plan = tuple(
        row[-1]
        for row in db._conn.execute(
            "EXPLAIN QUERY PLAN "
            + hermes_state._TASK_FENCE_TERMINAL_OPEN_INCIDENT_SCAN_SQL,
            (hermes_state._TASK_FENCE_MAX_INSPECTION_WORK + 1,),
        )
    )
    assert any("idx_task_fence_incidents_one_open" in detail for detail in plan)
    assert all("USE TEMP B-TREE" not in detail.upper() for detail in plan)

    monkeypatch.setattr(hermes_state, "_TASK_FENCE_MAX_INSPECTION_WORK", 2)
    complete = db.inspect_task_fence_conversation("terminal-work-empty-target")
    assert complete.compatible is True
    assert complete.reason == "no_active_task"
    assert complete.terminal_open_incidents == ()

    monkeypatch.setattr(hermes_state, "_TASK_FENCE_MAX_INSPECTION_WORK", 1)
    overflow = db.inspect_task_fence_conversation("terminal-work-empty-target")
    db.close()

    assert overflow.compatible is False
    assert overflow.reason == "terminal_open_incident_work_limit_exceeded"
    assert overflow.task is None
    assert overflow.terminal_open_incidents == ()


def test_terminal_open_incident_projection_reuses_exact_link_validation(
    tmp_path,
    monkeypatch,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    conversation_id = "terminal-exact-links"
    _, started, _ = _terminal_unknown_incident(
        db,
        marker="terminal-exact-links",
        conversation_id=conversation_id,
    )
    incident_attempt_limit = hermes_state._TASK_FENCE_MAX_RECOVERY_INCIDENT_ATTEMPTS
    monkeypatch.setattr(
        hermes_state,
        "_TASK_FENCE_MAX_RECOVERY_INCIDENT_ATTEMPTS",
        0,
    )
    over_limit = db.inspect_task_fence_conversation(conversation_id)
    assert over_limit.compatible is False
    assert over_limit.reason == "open_incident_attempt_limit_exceeded"
    assert over_limit.terminal_open_incidents == ()

    monkeypatch.setattr(
        hermes_state,
        "_TASK_FENCE_MAX_RECOVERY_INCIDENT_ATTEMPTS",
        incident_attempt_limit,
    )
    db._conn.execute(
        "UPDATE task_fence_attempts SET recovery_classification = 'known_read' "
        "WHERE attempt_id = ?",
        (started.attempt_id,),
    )
    db._conn.commit()
    malformed = db.inspect_task_fence_conversation(conversation_id)
    db.close()

    assert malformed.compatible is False
    assert malformed.reason == "incompatible_open_incident_projection"
    assert malformed.task is None
    assert malformed.terminal_open_incidents == ()


def test_resolved_and_incident_free_terminal_tasks_are_not_projected(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    conversation_id = "terminal-resolved-history"
    resolved_task, started = _start_effect(
        db,
        marker="terminal-resolved",
        conversation_id=conversation_id,
    )
    _recover(db)
    db.accept_task_fence_ingress(
        _ingress(
            "resolve_incident",
            "terminal-resolved-decision",
            task_id=resolved_task.task_id,
            correlation_ids=(started.attempt_id,),
            conversation_id=conversation_id,
            resolution_disposition=(ResolutionDisposition.ACCEPTED_UNKNOWN_NO_RETRY),
            evidence_refs=("operator:terminal-review",),
        )
    )
    db.accept_task_fence_ingress(
        _ingress(
            "stop",
            "terminal-resolved-stop",
            task_id=resolved_task.task_id,
            conversation_id=conversation_id,
        )
    )
    incident_free = db.accept_task_fence_ingress(
        _ingress(
            "initial_submit",
            "terminal-incident-free-initial",
            conversation_id=conversation_id,
        )
    )
    db.accept_task_fence_ingress(
        _ingress(
            "stop",
            "terminal-incident-free-stop",
            task_id=incident_free.task_id,
            conversation_id=conversation_id,
        )
    )

    inspected = db.inspect_task_fence_conversation(conversation_id)
    db.close()

    assert inspected.compatible is True
    assert inspected.reason == "no_active_task"
    assert inspected.task is None
    assert inspected.terminal_open_incidents == ()


def test_terminal_open_incidents_share_the_active_frontier_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.db"
    conversation_id = "terminal-snapshot"
    db = SessionDB(path)
    accepted, started = _start_effect(
        db,
        marker="terminal-snapshot",
        conversation_id=conversation_id,
    )
    _recover(db)
    active = db.inspect_task_fence_conversation(conversation_id)
    assert active.open_incident is not None
    incident_id = active.open_incident.incident_id
    db.close()

    reader = SessionDB(path, read_only=True)
    store_read = threading.Event()
    writer_done = threading.Event()
    writer_errors = []
    original_inspect = reader._inspect_task_fence_store_unlocked

    def pause_after_store(*, include_counts):
        inspected = original_inspect(include_counts=include_counts)
        store_read.set()
        if not writer_done.wait(timeout=5):
            raise RuntimeError("terminal snapshot update timed out")
        return inspected

    def stop_task():
        try:
            if not store_read.wait(timeout=5):
                raise RuntimeError("inspection did not establish its snapshot")
            writer = SessionDB(path)
            try:
                writer.accept_task_fence_ingress(
                    _ingress(
                        "stop",
                        "terminal-snapshot-stop",
                        task_id=accepted.task_id,
                        conversation_id=conversation_id,
                    )
                )
            finally:
                writer.close()
        except Exception as exc:
            writer_errors.append(exc)
        finally:
            writer_done.set()

    monkeypatch.setattr(
        reader,
        "_inspect_task_fence_store_unlocked",
        pause_after_store,
    )
    writer_thread = threading.Thread(target=stop_task)
    writer_thread.start()
    try:
        before = reader.inspect_task_fence_conversation(conversation_id)
        writer_thread.join(timeout=5)
        after = reader.inspect_task_fence_conversation(conversation_id)
    finally:
        reader.close()

    assert writer_thread.is_alive() is False
    assert writer_errors == []
    assert before.compatible is True
    assert before.task is not None
    assert before.task.task_id == accepted.task_id
    assert before.open_incident is not None
    assert before.open_incident.incident_id == incident_id
    assert before.terminal_open_incidents == ()
    assert after.compatible is True
    assert after.task is None
    assert after.open_incident is None
    assert len(after.terminal_open_incidents) == 1
    terminal = after.terminal_open_incidents[0]
    assert terminal.task_id == accepted.task_id
    assert terminal.incident.incident_id == incident_id
    assert terminal.incident.attempt_ids == (started.attempt_id,)


@pytest.mark.parametrize(
    ("control_column", "control_value"),
    (("tested_artifact_commit", "future"), ("ever_enforced", 1)),
)
def test_recovery_is_explicit_shadow_only_and_preserves_legacy_rows(
    tmp_path,
    control_column,
    control_value,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    session_id = "legacy-session"
    db.create_session(session_id, source="cli")
    initial = db.accept_task_fence_ingress(
        _ingress("initial_submit", "message-terminal")
    )
    stopped = db.accept_task_fence_ingress(
        _ingress("stop", "message-stop", task_id=initial.task_id)
    )
    db.close()

    reopened = SessionDB(path)
    try:
        assert reopened.inspect_task_fence_store().runtime_epoch == 0
        result = _recover(reopened)
        assert result.runtime_epoch == 1
        task = reopened.inspect_task_fence_task(stopped.task_id).task
        assert task is not None
        assert task.status == "stopped"
        assert task.current_runtime_epoch == 1
        assert reopened.get_session(session_id) is not None

        reopened._conn.execute(
            f"UPDATE task_fence_control SET {control_column} = ? WHERE singleton = 1",
            (control_value,),
        )
        before = _task_fence_state(reopened)
        with pytest.raises(
            TaskFenceRecoveryUnavailable,
            match="shadow_recovery_precondition",
        ):
            _recover(reopened)
        assert _task_fence_state(reopened) == before
    finally:
        reopened.close()
