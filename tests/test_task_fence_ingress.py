from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import hashlib
import sqlite3
import threading

import pytest

from hermes_state import SessionDB
from task_fence import (
    TASK_FENCE_ACTIONS,
    CorrelationKind,
    ExecutionEffect,
    IngressClass,
    IngressEnvelope,
    InputEffect,
    IntentEffect,
    Origin,
    ResolutionDisposition,
    TaskFenceAction,
    TaskFenceIngressRejected,
    TaskFenceIngressUnavailable,
    TaskFenceProtocolRejected,
    TerminalReason,
    validate_action,
)


def _hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _envelope(
    action_name: str,
    source_event_id: str,
    *,
    payload: str | None = None,
    task_id: str | None = None,
    correlation_ids: tuple[str, ...] = (),
    resolution_disposition: ResolutionDisposition | None = None,
    evidence_refs: tuple[str, ...] = (),
) -> IngressEnvelope:
    return IngressEnvelope(
        source="gateway:test:conversation-1",
        source_event_id=source_event_id,
        conversation_id="conversation-1",
        task_id=task_id,
        action=TASK_FENCE_ACTIONS[action_name],
        payload_hash=_hash(payload or source_event_id),
        correlation_ids=correlation_ids,
        resolution_disposition=resolution_disposition,
        evidence_refs=evidence_refs,
        terminal_reason=(
            TerminalReason.STOPPED if action_name == "stop" else None
        ),
    )


def _scalar(conn: sqlite3.Connection, query: str, params=()):
    return conn.execute(query, params).fetchone()[0]


def _connection(db: SessionDB) -> sqlite3.Connection:
    conn = db._conn
    assert conn is not None
    return conn


def _seed_started_generation_and_permit(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    run_id: str,
    event_id: str,
) -> None:
    conn.execute(
        "INSERT INTO main.task_fence_model_generations ("
        "generation_id, task_id, run_id, intent_epoch, control_revision, "
        "runtime_epoch, input_manifest_hash, snapshot_event_id, state, opened_at"
        ") VALUES ('generation-1', ?, ?, 1, 1, 0, ?, ?, 'started', 1.0)",
        (task_id, run_id, _hash("manifest"), event_id),
    )
    conn.execute(
        "UPDATE main.task_fence_tasks SET current_generation_id = 'generation-1' "
        "WHERE task_id = ?",
        (task_id,),
    )
    conn.execute(
        "INSERT INTO main.task_fence_dispatch_permits ("
        "permit_id, task_id, authority_event_id, run_id, generation_id, "
        "intent_epoch, control_revision, runtime_epoch, "
        "invocation_envelope_id, invocation_fingerprint, executor, audience, "
        "expires_at, state, reserved_at"
        ") VALUES ('permit-1', ?, ?, ?, 'generation-1', 1, 1, 0, "
        "'invocation-1', ?, 'model', 'provider', 100.0, 'reserved', 1.0)",
        (task_id, event_id, run_id, _hash("call")),
    )


def test_production_ingress_union_is_closed_and_envelopes_are_bounded() -> None:
    for action in TASK_FENCE_ACTIONS.values():
        validate_action(action)

    invalid = TaskFenceAction(
        Origin.RUNTIME,
        IngressClass.SYNTHETIC,
        IntentEffect.KEEP,
        ExecutionEffect.RUN,
        InputEffect.NONE,
        CorrelationKind.NONE,
    )
    with pytest.raises(
        TaskFenceProtocolRejected,
        match="unsupported_ingress_tuple",
    ):
        validate_action(invalid)
    raw_strings = TaskFenceAction(
        "human",  # type: ignore[arg-type]
        "task_input",  # type: ignore[arg-type]
        "replace",  # type: ignore[arg-type]
        "run",  # type: ignore[arg-type]
        "append",  # type: ignore[arg-type]
        "none",  # type: ignore[arg-type]
    )
    with pytest.raises(TaskFenceProtocolRejected, match="invalid_action_fields"):
        validate_action(raw_strings)
    with pytest.raises(
        TaskFenceProtocolRejected,
        match="missing_payload_identity",
    ):
        IngressEnvelope(
            source="adapter",
            source_event_id="event",
            conversation_id="conversation",
            action=TASK_FENCE_ACTIONS["initial_submit"],
        )
    with pytest.raises(
        TaskFenceProtocolRejected,
        match="source_event_id_too_large",
    ):
        IngressEnvelope(
            source="adapter",
            source_event_id="x" * 513,
            conversation_id="conversation",
            action=TASK_FENCE_ACTIONS["initial_submit"],
            payload_hash=_hash("payload"),
        )
    with pytest.raises(
        TaskFenceProtocolRejected,
        match="invalid_source_sequence",
    ):
        IngressEnvelope(
            source="adapter",
            source_event_id="event",
            conversation_id="conversation",
            action=TASK_FENCE_ACTIONS["initial_submit"],
            payload_hash=_hash("payload"),
            source_sequence=2**63,
        )
    envelope = _envelope("initial_submit", "message-1")
    with pytest.raises(FrozenInstanceError):
        envelope.source_event_id = "changed"  # type: ignore[misc]


def test_initial_submit_commits_task_input_and_run_before_return(tmp_path) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    acceptance = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1", payload="private prompt")
    )
    db.close()

    assert acceptance.replayed is False
    assert acceptance.task_id
    assert acceptance.opened_run_id
    assert acceptance.closed_run_id is None

    reopened = SessionDB(path)
    try:
        assert acceptance.task_id is not None
        task = reopened.inspect_task_fence_task(acceptance.task_id)
        assert task.compatible is True
        assert task.task is not None
        assert task.task.intent_epoch == 1
        assert task.task.control_revision == 1
        assert task.task.status == "running"
        assert task.task.active_authority_event_id == acceptance.event_id
        assert task.task.active_execution_run_id == acceptance.opened_run_id
        assert task.task.last_accepted_order == acceptance.accepted_order

        conn = _connection(reopened)
        assert conn.execute("PRAGMA main.foreign_key_check").fetchall() == []
        assert (
            _scalar(
                conn,
                "SELECT state FROM main.task_fence_task_inputs "
                "WHERE event_id = ?",
                (acceptance.event_id,),
            )
            == "bound"
        )
        assert (
            _scalar(
                conn,
                "SELECT state FROM main.task_fence_execution_runs "
                "WHERE run_id = ?",
                (acceptance.opened_run_id,),
            )
            == "open"
        )
    finally:
        reopened.close()


def test_comment_hold_closes_generation_run_and_reserved_permit_atomically(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    assert initial.opened_run_id is not None
    conn = _connection(db)
    _seed_started_generation_and_permit(
        conn,
        task_id=initial.task_id,
        run_id=initial.opened_run_id,
        event_id=initial.event_id,
    )

    held = db.accept_task_fence_ingress(
        _envelope(
            "comment_hold",
            "message-2",
            payload="new private direction",
            task_id=initial.task_id,
        )
    )
    task = db.inspect_task_fence_task(initial.task_id).task

    assert held.closed_run_id == initial.opened_run_id
    assert held.opened_run_id is None
    assert task is not None
    assert task.intent_epoch == 1
    assert task.control_revision == 2
    assert task.status == "paused"
    assert task.active_execution_run_id is None
    assert task.current_generation_id is None
    assert (
        _scalar(
            conn,
            "SELECT state FROM main.task_fence_model_generations "
            "WHERE generation_id = 'generation-1'",
        )
        == "cancelled"
    )
    assert (
        _scalar(
            conn,
            "SELECT state FROM main.task_fence_dispatch_permits "
            "WHERE permit_id = 'permit-1'",
        )
        == "revoked"
    )
    assert (
        _scalar(
            conn,
            "SELECT state FROM main.task_fence_task_inputs WHERE event_id = ?",
            (held.event_id,),
        )
        == "pending"
    )
    db.close()


def test_replace_run_binds_pending_inputs_and_opens_one_run(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    conn = _connection(db)
    held = db.accept_task_fence_ingress(
        _envelope("comment_hold", "message-2", task_id=initial.task_id)
    )
    resumed = db.accept_task_fence_ingress(
        _envelope("change_and_run", "message-3", task_id=initial.task_id)
    )

    task = db.inspect_task_fence_task(initial.task_id).task
    assert task is not None
    assert task.intent_epoch == 2
    assert task.control_revision == 3
    assert task.status == "running"
    assert task.active_execution_run_id == resumed.opened_run_id
    assert resumed.closed_run_id is None
    assert (
        _scalar(
            conn,
            "SELECT COUNT(*) FROM main.task_fence_execution_runs "
            "WHERE task_id = ? AND state = 'open'",
            (initial.task_id,),
        )
        == 1
    )
    inputs = conn.execute(
        "SELECT event_id, state, bound_run_id "
        "FROM main.task_fence_task_inputs WHERE task_id = ? "
        "ORDER BY event_id",
        (initial.task_id,),
    ).fetchall()
    by_event = {row[0]: (row[1], row[2]) for row in inputs}
    assert by_event[held.event_id] == ("bound", resumed.opened_run_id)
    assert by_event[resumed.event_id] == ("bound", resumed.opened_run_id)
    db.close()


def test_exact_replay_after_later_acceptance_returns_original_identity(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    original_envelope = _envelope("initial_submit", "message-1")
    initial = db.accept_task_fence_ingress(original_envelope)
    assert initial.task_id is not None
    db.accept_task_fence_ingress(
        _envelope("comment_hold", "message-2", task_id=initial.task_id)
    )
    before = db.inspect_task_fence_task(initial.task_id).task
    assert before is not None
    revision_before = before.control_revision

    replay = db.accept_task_fence_ingress(original_envelope)

    assert replay.replayed is True
    assert replay.event_id == initial.event_id
    assert replay.accepted_order == initial.accepted_order
    assert replay.task_id == initial.task_id
    assert replay.opened_run_id == initial.opened_run_id
    assert replay.accepted_at == initial.accepted_at
    after = db.inspect_task_fence_task(initial.task_id).task
    assert after is not None
    assert after.control_revision == revision_before
    assert (
        _scalar(
            _connection(db),
            "SELECT COUNT(*) FROM main.task_fence_ingress",
        )
        == 2
    )
    db.close()


def test_mismatched_source_event_collision_rejects_without_authority_mutation(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    accepted = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1", payload="first")
    )
    assert accepted.task_id is not None

    with pytest.raises(
        TaskFenceIngressRejected,
        match="source_event_id_collision",
    ) as caught:
        db.accept_task_fence_ingress(
            _envelope("initial_submit", "message-1", payload="different")
        )

    assert caught.value.incident_id is None
    task = db.inspect_task_fence_task(accepted.task_id).task
    assert task is not None
    assert task.control_revision == 1
    assert (
        _scalar(_connection(db), "SELECT COUNT(*) FROM main.task_fence_ingress")
        == 1
    )
    assert _scalar(
        _connection(db),
        "SELECT COUNT(*) FROM main.task_fence_incidents",
    ) == 0
    db.close()


def test_duplicate_acceptance_serializes_across_two_connections(tmp_path) -> None:
    path = tmp_path / "state.db"
    SessionDB(path).close()
    first = SessionDB(path)
    second = SessionDB(path)
    barrier = threading.Barrier(2)
    envelope = _envelope("initial_submit", "message-1")

    def accept(db: SessionDB):
        barrier.wait(timeout=5)
        return db.accept_task_fence_ingress(envelope)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(accept, (first, second)))
    finally:
        first.close()
        second.close()

    assert {result.replayed for result in results} == {False, True}
    assert len({result.event_id for result in results}) == 1
    verify = sqlite3.connect(path)
    try:
        assert _scalar(
            verify,
            "SELECT COUNT(*) FROM task_fence_ingress",
        ) == 1
        assert _scalar(
            verify,
            "SELECT control_revision FROM task_fence_tasks",
        ) == 1
    finally:
        verify.close()


def test_distinct_replace_run_acceptances_have_one_serial_order(tmp_path) -> None:
    path = tmp_path / "state.db"
    SessionDB(path).close()
    first = SessionDB(path)
    second = SessionDB(path)
    barrier = threading.Barrier(2)
    envelopes = (
        _envelope("initial_submit", "message-a"),
        _envelope("initial_submit", "message-b"),
    )

    def accept(pair):
        db, envelope = pair
        barrier.wait(timeout=5)
        return db.accept_task_fence_ingress(envelope)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(accept, zip((first, second), envelopes)))
    finally:
        first.close()
        second.close()

    assert all(result.replayed is False for result in results)
    assert sorted(result.accepted_order for result in results) == [1, 2]
    earlier, later = sorted(results, key=lambda result: result.accepted_order)
    assert later.closed_run_id == earlier.opened_run_id
    verify = sqlite3.connect(path)
    try:
        assert _scalar(
            verify,
            "SELECT control_revision FROM task_fence_tasks",
        ) == 2
        assert _scalar(
            verify,
            "SELECT intent_epoch FROM task_fence_tasks",
        ) == 2
        assert _scalar(
            verify,
            "SELECT COUNT(*) FROM task_fence_execution_runs "
            "WHERE state = 'open'",
        ) == 1
        assert _scalar(
            verify,
            "SELECT COUNT(*) FROM task_fence_execution_runs "
            "WHERE state = 'closed'",
        ) == 1
        current = verify.execute(
            "SELECT active_authority_event_id, active_execution_run_id "
            "FROM task_fence_tasks",
        ).fetchone()
        assert tuple(current) == (later.event_id, later.opened_run_id)
    finally:
        verify.close()


@pytest.mark.parametrize("failure_stage", ["run_insert", "task_update"])
def test_mid_transaction_sqlite_abort_rolls_back_every_authority_write(
    tmp_path,
    failure_stage: str,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    conn = _connection(db)
    operation, table = {
        "run_insert": ("INSERT", "task_fence_execution_runs"),
        "task_update": ("UPDATE", "task_fence_tasks"),
    }[failure_stage]
    conn.execute(
        "CREATE TEMP TRIGGER fail_task_fence_acceptance "
        f"BEFORE {operation} ON main.{table} BEGIN "
        "SELECT RAISE(ABORT, 'injected task fence failure'); END"
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="acceptance_database_error",
    ):
        db.accept_task_fence_ingress(
            _envelope("initial_submit", "message-1")
        )

    for table in (
        "task_fence_tasks",
        "task_fence_ingress",
        "task_fence_task_inputs",
        "task_fence_execution_runs",
    ):
        assert _scalar(
            conn,
            f"SELECT COUNT(*) FROM main.{table}",
        ) == 0
    conn.execute("DROP TRIGGER temp.fail_task_fence_acceptance")
    assert db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    ).opened_run_id
    db.close()


def test_existing_authority_rolls_back_on_late_projection_failure(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    assert initial.opened_run_id is not None
    conn = _connection(db)
    _seed_started_generation_and_permit(
        conn,
        task_id=initial.task_id,
        run_id=initial.opened_run_id,
        event_id=initial.event_id,
    )
    conn.execute(
        "CREATE TEMP TRIGGER fail_task_fence_projection "
        "BEFORE UPDATE ON main.task_fence_tasks BEGIN "
        "SELECT RAISE(ABORT, 'injected late task fence failure'); END"
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="acceptance_database_error",
    ):
        db.accept_task_fence_ingress(
            _envelope("comment_hold", "message-2", task_id=initial.task_id)
        )

    task = db.inspect_task_fence_task(initial.task_id).task
    assert task is not None
    assert task.control_revision == 1
    assert task.status == "running"
    assert task.active_execution_run_id == initial.opened_run_id
    assert task.current_generation_id == "generation-1"
    assert _scalar(
        conn,
        "SELECT state FROM main.task_fence_execution_runs WHERE run_id = ?",
        (initial.opened_run_id,),
    ) == "open"
    assert _scalar(
        conn,
        "SELECT state FROM main.task_fence_model_generations "
        "WHERE generation_id = 'generation-1'",
    ) == "started"
    assert _scalar(
        conn,
        "SELECT state FROM main.task_fence_dispatch_permits "
        "WHERE permit_id = 'permit-1'",
    ) == "reserved"
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    ) == 1
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_task_inputs",
    ) == 1
    db.close()


def test_invalid_state_correlations_reject_without_partial_rows(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    conn = _connection(db)
    with pytest.raises(
        TaskFenceIngressRejected,
        match="empty_lane_requires_initial_submit",
    ):
        db.accept_task_fence_ingress(
            _envelope("comment_hold", "orphan-comment")
        )
    assert _scalar(conn, "SELECT COUNT(*) FROM main.task_fence_tasks") == 0
    assert _scalar(conn, "SELECT COUNT(*) FROM main.task_fence_ingress") == 0

    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    held = db.accept_task_fence_ingress(
        _envelope("comment_hold", "message-2", task_id=initial.task_id)
    )
    count_before = _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    )
    with pytest.raises(
        TaskFenceIngressRejected,
        match="pending_input_correlation_mismatch",
    ):
        db.accept_task_fence_ingress(
            _envelope(
                "discard_pending",
                "message-3",
                task_id=initial.task_id,
                correlation_ids=("not-a-pending-input",),
            )
        )
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    ) == count_before
    assert _scalar(
        conn,
        "SELECT state FROM main.task_fence_task_inputs WHERE event_id = ?",
        (held.event_id,),
    ) == "pending"
    db.close()


def test_stale_question_correlation_rejects_without_partial_writes(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    assert initial.opened_run_id is not None
    conn = _connection(db)
    conn.execute(
        "INSERT INTO main.task_fence_questions ("
        "question_id, task_id, source_run_id, state, opened_at"
        ") VALUES ('question-current', ?, ?, 'open', 1.0)",
        (initial.task_id, initial.opened_run_id),
    )
    conn.execute(
        "UPDATE main.task_fence_execution_runs SET state = 'closed', "
        "close_reason = 'awaiting_user', closed_at = 1.0 "
        "WHERE run_id = ?",
        (initial.opened_run_id,),
    )
    conn.execute(
        "UPDATE main.task_fence_tasks SET status = 'waiting_user', "
        "active_execution_run_id = NULL "
        "WHERE task_id = ?",
        (initial.task_id,),
    )
    count_before = _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    )

    with pytest.raises(
        TaskFenceIngressRejected,
        match="open_question_correlation_mismatch",
    ):
        db.accept_task_fence_ingress(
            _envelope(
                "answer_and_resume",
                "message-2",
                task_id=initial.task_id,
                correlation_ids=("question-stale",),
            )
        )

    task = db.inspect_task_fence_task(initial.task_id).task
    assert task is not None
    assert task.status == "waiting_user"
    assert task.control_revision == 1
    assert task.active_execution_run_id is None
    assert _scalar(
        conn,
        "SELECT state FROM main.task_fence_questions "
        "WHERE question_id = 'question-current'",
    ) == "open"
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    ) == count_before

    answered = db.accept_task_fence_ingress(
        _envelope(
            "answer_and_resume",
            "message-3",
            task_id=initial.task_id,
            correlation_ids=("question-current",),
        )
    )
    task = db.inspect_task_fence_task(initial.task_id).task
    assert task is not None
    assert task.status == "running"
    assert task.control_revision == 2
    assert task.active_execution_run_id == answered.opened_run_id
    question = conn.execute(
        "SELECT state, answer_event_id FROM main.task_fence_questions "
        "WHERE question_id = 'question-current'",
    ).fetchone()
    assert tuple(question) == ("answered", answered.event_id)
    replay = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert replay.replayed is True
    assert replay.event_id == initial.event_id
    assert replay.opened_run_id == initial.opened_run_id
    assert replay.closed_run_id is None
    db.close()


def test_incident_resolution_requires_exact_attempts_and_evidence(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    assert initial.opened_run_id is not None
    conn = _connection(db)
    conn.execute(
        "INSERT INTO main.task_fence_model_generations ("
        "generation_id, task_id, run_id, intent_epoch, control_revision, "
        "runtime_epoch, input_manifest_hash, snapshot_event_id, state, "
        "opened_at, closed_at"
        ") VALUES ('generation-1', ?, ?, 1, 1, 0, ?, ?, 'failed', 1.0, 2.0)",
        (initial.task_id, initial.opened_run_id, _hash("manifest"), initial.event_id),
    )
    conn.execute(
        "INSERT INTO main.task_fence_dispatch_permits ("
        "permit_id, task_id, authority_event_id, run_id, generation_id, "
        "intent_epoch, control_revision, runtime_epoch, invocation_envelope_id, "
        "invocation_fingerprint, executor, audience, expires_at, state, "
        "reserved_at, consumed_at"
        ") VALUES ('permit-1', ?, ?, ?, 'generation-1', 1, 1, 0, "
        "'invocation-1', ?, 'tool', 'external', 100.0, 'consumed', 1.0, 2.0)",
        (initial.task_id, initial.event_id, initial.opened_run_id, _hash("call")),
    )
    conn.execute(
        "INSERT INTO main.task_fence_attempts ("
        "attempt_id, permit_id, recovery_classification, state, prepared_at, "
        "started_at"
        ") VALUES ('attempt-1', 'permit-1', 'may_effect', "
        "'OUTCOME_UNKNOWN', 1.0, 2.0)"
    )
    conn.execute(
        "INSERT INTO main.task_fence_incidents ("
        "incident_id, task_id, source_run_id, reason_code, state, opened_at"
        ") VALUES ('incident-1', ?, ?, 'outcome_unknown', 'open', 3.0)",
        (initial.task_id, initial.opened_run_id),
    )
    conn.execute(
        "INSERT INTO main.task_fence_incident_attempts (incident_id, attempt_id) "
        "VALUES ('incident-1', 'attempt-1')"
    )
    conn.execute(
        "UPDATE main.task_fence_execution_runs SET state = 'closed', "
        "close_reason = 'incident', closed_at = 3.0 "
        "WHERE run_id = ?",
        (initial.opened_run_id,),
    )
    conn.execute(
        "UPDATE main.task_fence_tasks SET status = 'incident', "
        "active_execution_run_id = NULL WHERE task_id = ?",
        (initial.task_id,),
    )

    with pytest.raises(
        TaskFenceIngressRejected,
        match="incident_correlation_mismatch",
    ):
        db.accept_task_fence_ingress(
            _envelope(
                "resolve_incident",
                "message-stale",
                task_id=initial.task_id,
                correlation_ids=("attempt-stale",),
                resolution_disposition=(
                    ResolutionDisposition.ACCEPTED_UNKNOWN_NO_RETRY
                ),
                evidence_refs=("operator:ticket-stale",),
            )
        )
    task_before = db.inspect_task_fence_task(initial.task_id).task
    assert task_before is not None
    assert task_before.status == "incident"
    assert task_before.control_revision == 1
    assert _scalar(
        conn,
        "SELECT state FROM main.task_fence_incidents "
        "WHERE incident_id = 'incident-1'",
    ) == "open"
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    ) == 1

    resolved = db.accept_task_fence_ingress(
        _envelope(
            "resolve_incident",
            "message-2",
            task_id=initial.task_id,
            correlation_ids=("attempt-1",),
            resolution_disposition=(
                ResolutionDisposition.ACCEPTED_UNKNOWN_NO_RETRY
            ),
            evidence_refs=("operator:ticket-1",),
        )
    )

    task = db.inspect_task_fence_task(initial.task_id).task
    assert task is not None
    assert task.status == "paused"
    assert task.control_revision == 2
    assert resolved.closed_run_id is None
    assert _scalar(
        conn,
        "SELECT state FROM main.task_fence_incidents "
        "WHERE incident_id = 'incident-1'",
    ) == "resolved"
    resolution = conn.execute(
        "SELECT disposition, resolution_event_id "
        "FROM main.task_fence_resolutions WHERE incident_id = 'incident-1'",
    ).fetchone()
    assert tuple(resolution) == (
        "accepted_unknown_no_retry",
        resolved.event_id,
    )
    assert _scalar(
        conn,
        "SELECT evidence_ref FROM main.task_fence_resolution_evidence",
    ) == "operator:ticket-1"
    db.close()


def test_terminal_task_never_reopens(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    stopped = db.accept_task_fence_ingress(
        _envelope("stop", "message-2", task_id=initial.task_id)
    )
    assert stopped.closed_run_id == initial.opened_run_id
    with pytest.raises(TaskFenceIngressRejected, match="terminal_task"):
        db.accept_task_fence_ingress(
            _envelope("resume", "message-3", task_id=initial.task_id)
        )
    task = db.inspect_task_fence_task(initial.task_id).task
    assert task is not None
    assert task.status == "stopped"
    assert task.control_revision == 2
    db.close()


def test_advisory_and_synthetic_ingress_never_change_authority(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    before = db.inspect_task_fence_task(initial.task_id).task
    note = db.accept_task_fence_ingress(
        _envelope("explicit_note", "message-2", task_id=initial.task_id)
    )
    synthetic = db.accept_task_fence_ingress(
        IngressEnvelope(
            source="runtime:test",
            source_event_id="runtime-1",
            conversation_id="conversation-1",
            task_id=initial.task_id,
            action=TASK_FENCE_ACTIONS["synthetic_notice"],
            payload_hash=_hash("synthetic"),
        )
    )
    after = db.inspect_task_fence_task(initial.task_id).task

    assert before is not None and after is not None
    assert after.control_revision == before.control_revision
    assert after.intent_epoch == before.intent_epoch
    assert after.status == before.status
    assert after.active_execution_run_id == before.active_execution_run_id
    assert note.opened_run_id is None
    assert synthetic.opened_run_id is None
    assert _scalar(
        _connection(db),
        "SELECT COUNT(*) FROM main.task_fence_task_inputs",
    ) == 1
    db.close()


def test_non_authority_ingress_rejects_an_explicit_unknown_task(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    envelope = _envelope(
        "explicit_note",
        "message-1",
        task_id="unknown-task",
    )

    for _ in range(2):
        with pytest.raises(TaskFenceIngressRejected, match="unknown_task_id"):
            db.accept_task_fence_ingress(envelope)

    assert _scalar(
        _connection(db),
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    ) == 0
    db.close()


def test_future_task_projection_rejects_without_mutation(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    conn = _connection(db)
    conn.execute(
        "UPDATE main.task_fence_tasks SET store_schema_version = 2 "
        "WHERE task_id = ?",
        (initial.task_id,),
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="incompatible_task_projection",
    ):
        db.accept_task_fence_ingress(
            _envelope("comment_hold", "message-2", task_id=initial.task_id)
        )

    row = conn.execute(
        "SELECT store_schema_version, control_revision, status "
        "FROM main.task_fence_tasks WHERE task_id = ?",
        (initial.task_id,),
    ).fetchone()
    assert tuple(row) == (2, 1, "running")
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    ) == 1
    db.close()


def test_unpointed_active_generation_rejects_without_mutation(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    assert initial.opened_run_id is not None
    conn = _connection(db)
    conn.execute(
        "INSERT INTO main.task_fence_model_generations ("
        "generation_id, task_id, run_id, intent_epoch, control_revision, "
        "runtime_epoch, input_manifest_hash, snapshot_event_id, state, opened_at"
        ") VALUES ('orphan-active', ?, ?, 1, 1, 0, ?, ?, 'started', 1.0)",
        (initial.task_id, initial.opened_run_id, _hash("manifest"), initial.event_id),
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="incompatible_current_generation",
    ):
        db.accept_task_fence_ingress(
            _envelope("comment_hold", "message-2", task_id=initial.task_id)
        )

    task = db.inspect_task_fence_task(initial.task_id).task
    assert task is not None
    assert task.control_revision == 1
    assert task.active_execution_run_id == initial.opened_run_id
    assert task.current_generation_id is None
    assert _scalar(
        conn,
        "SELECT state FROM main.task_fence_model_generations "
        "WHERE generation_id = 'orphan-active'",
    ) == "started"
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    ) == 1
    db.close()


def test_runtime_epoch_mismatch_rejects_until_recovery(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    conn = _connection(db)
    conn.execute(
        "UPDATE main.task_fence_control SET runtime_epoch = 1 WHERE singleton = 1"
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="runtime_epoch_mismatch",
    ):
        db.accept_task_fence_ingress(
            _envelope("comment_hold", "message-2", task_id=initial.task_id)
        )

    task = db.inspect_task_fence_task(initial.task_id).task
    assert task is not None
    assert task.current_runtime_epoch == 0
    assert task.control_revision == 1
    assert task.status == "running"
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    ) == 1
    db.close()


def test_incompatible_read_only_and_closed_stores_never_accept(tmp_path) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()
    envelope = _envelope("initial_submit", "message-1")

    with pytest.raises(TaskFenceIngressUnavailable, match="closed_store"):
        db.accept_task_fence_ingress(envelope)

    read_only = SessionDB(path, read_only=True)
    try:
        with pytest.raises(
            TaskFenceIngressUnavailable,
            match="read_only_store",
        ):
            read_only.accept_task_fence_ingress(envelope)
    finally:
        read_only.close()

    conn = sqlite3.connect(path)
    conn.execute("DROP INDEX idx_task_fence_runs_one_open")
    conn.commit()
    conn.close()
    incompatible = SessionDB(path)
    try:
        with pytest.raises(
            TaskFenceIngressUnavailable,
            match="malformed_schema",
        ):
            incompatible.accept_task_fence_ingress(envelope)
        assert _scalar(
            _connection(incompatible),
            "SELECT COUNT(*) FROM main.task_fence_ingress",
        ) == 0
    finally:
        incompatible.close()


def test_raw_payload_is_never_persisted_in_task_fence_namespace(tmp_path) -> None:
    secret = "do-not-store-this-private-prompt"
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1", payload=secret)
    )
    db.close()

    conn = sqlite3.connect(path)
    try:
        dump = "\n".join(
            line
            for line in conn.iterdump()
            if "task_fence_" in line
        )
    finally:
        conn.close()
    assert secret not in dump
    assert _hash(secret) in dump
