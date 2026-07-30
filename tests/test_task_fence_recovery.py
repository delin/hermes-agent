from dataclasses import replace
import hashlib
import sqlite3

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
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "resolution_fault",
    (None, "action", "correlation", "evidence"),
)
def test_recovery_adopts_preexisting_unknowns_into_one_durable_incident(
    tmp_path,
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
    db.close()


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
