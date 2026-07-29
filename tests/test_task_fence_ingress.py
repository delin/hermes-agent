from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
import sqlite3
import threading

import pytest

import hermes_state
from hermes_state import TASK_FENCE_STORE_SCHEMA_VERSION, SessionDB
from task_fence import (
    TASK_FENCE_ACTIONS,
    CorrelationKind,
    ExecutionEffect,
    IngressAcceptance,
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


_TASK_FENCE_V2_DDL_SHA256 = (
    "283bc43a083c97421e712fd29dfc802677558a35c75a890836831e1e3fdac847"
)
_TASK_FENCE_V3_DDL_SHA256 = (
    "f6e4c0a1bd7b586a7793ced62cd2f9eb6f54ac2acebb50d780e509d00fe10d72"
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


def _task_fence_table_state(
    conn: sqlite3.Connection,
) -> tuple[tuple[str, tuple[tuple, ...]], ...]:
    table_names = tuple(
        row[0]
        for row in conn.execute(
            "SELECT name FROM main.sqlite_master "
            "WHERE type = 'table' AND name GLOB 'task_fence_*' "
            "ORDER BY name"
        )
    )
    return tuple(
        (
            table_name,
            tuple(
                tuple(row)
                for row in conn.execute(
                    f'SELECT * FROM main."{table_name}" ORDER BY rowid'
                )
            ),
        )
        for table_name in table_names
    )


def _rewrite_append_only_row(
    db: SessionDB,
    *,
    trigger_name: str,
    statement: str,
    params: tuple,
) -> None:
    conn = _connection(db)
    trigger = conn.execute(
        "SELECT sql FROM main.sqlite_master "
        "WHERE type = 'trigger' AND name = ?",
        (trigger_name,),
    ).fetchone()
    assert trigger is not None
    assert isinstance(trigger[0], str)
    quoted_name = trigger_name.replace('"', '""')
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(f'DROP TRIGGER main."{quoted_name}"')
        conn.execute(statement, params)
        conn.execute(trigger[0])
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    assert db.inspect_task_fence_store().compatible is True


def _downgrade_current_store_to_exact_v2(
    path,
    acceptances: tuple[IngressAcceptance, ...],
) -> None:
    reference = sqlite3.connect(":memory:")
    reference.executescript(hermes_state.TASK_FENCE_SCHEMA_V2_SQL)

    def _reference_sql(name: str) -> str:
        row = reference.execute(
            "SELECT sql FROM main.sqlite_master WHERE name = ?",
            (name,),
        ).fetchone()
        assert row is not None and isinstance(row[0], str)
        return row[0]

    pending_table_sql = _reference_sql(
        "task_fence_acceptance_pending_inputs"
    )
    pending_trigger_sql = tuple(
        _reference_sql(name)
        for name in (
            "task_fence_acceptance_pending_inputs_no_replace",
            "task_fence_acceptance_pending_inputs_no_update",
            "task_fence_acceptance_pending_inputs_no_delete",
        )
    )
    snapshot_update_trigger_sql = _reference_sql(
        "task_fence_acceptance_snapshots_no_update"
    )
    reference.close()

    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DROP TABLE main.task_fence_policy_decisions")
        conn.execute(
            "DROP INDEX main.idx_task_fence_ingress_task_run_order"
        )
        conn.execute(
            "DROP INDEX main.idx_task_fence_ingress_task_input_history"
        )
        conn.execute(pending_table_sql)
        conn.execute(
            "DROP TRIGGER main.task_fence_acceptance_snapshots_no_update"
        )
        conn.execute(
            "UPDATE main.task_fence_acceptance_snapshots "
            "SET task_store_schema_version = 2 "
            "WHERE task_store_schema_version = ?",
            (TASK_FENCE_STORE_SCHEMA_VERSION,),
        )
        conn.execute(
            "UPDATE main.task_fence_tasks SET store_schema_version = 2 "
            "WHERE store_schema_version = ?",
            (TASK_FENCE_STORE_SCHEMA_VERSION,),
        )
        conn.execute(
            "UPDATE main.task_fence_control SET store_schema_version = 2 "
            "WHERE singleton = 1 AND store_schema_version = ?",
            (TASK_FENCE_STORE_SCHEMA_VERSION,),
        )
        conn.executemany(
            "INSERT INTO main.task_fence_acceptance_pending_inputs "
            "(event_id, ordinal, input_event_id) VALUES (?, ?, ?)",
            (
                (acceptance.event_id, ordinal, input_event_id)
                for acceptance in acceptances
                for ordinal, input_event_id in enumerate(
                    acceptance.pending_input_ids
                )
            ),
        )
        for trigger_sql in pending_trigger_sql:
            conn.execute(trigger_sql)
        conn.execute(snapshot_update_trigger_sql)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    check = sqlite3.connect(path)
    try:
        assert hermes_state._read_task_fence_schema_objects(
            check
        ) == hermes_state._expected_task_fence_v2_schema_objects()
    finally:
        check.close()


def _rewrite_exact_v2_row(
    path,
    *,
    trigger_name: str,
    statement: str,
    params: tuple,
) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    trigger = conn.execute(
        "SELECT sql FROM main.sqlite_master "
        "WHERE type = 'trigger' AND name = ?",
        (trigger_name,),
    ).fetchone()
    assert trigger is not None and isinstance(trigger[0], str)
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(f'DROP TRIGGER main."{trigger_name}"')
        conn.execute(statement, params)
        conn.execute(trigger[0])
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _downgrade_current_store_to_exact_v3(path) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    snapshot_trigger = conn.execute(
        "SELECT sql FROM main.sqlite_master "
        "WHERE type = 'trigger' "
        "AND name = 'task_fence_acceptance_snapshots_no_update'"
    ).fetchone()
    assert snapshot_trigger is not None
    assert isinstance(snapshot_trigger[0], str)
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DROP TABLE main.task_fence_policy_decisions")
        conn.execute(
            "DROP TRIGGER main.task_fence_acceptance_snapshots_no_update"
        )
        conn.execute(
            "UPDATE main.task_fence_acceptance_snapshots "
            "SET task_store_schema_version = 3 "
            "WHERE task_store_schema_version = ?",
            (TASK_FENCE_STORE_SCHEMA_VERSION,),
        )
        conn.execute(
            "UPDATE main.task_fence_tasks SET store_schema_version = 3 "
            "WHERE store_schema_version = ?",
            (TASK_FENCE_STORE_SCHEMA_VERSION,),
        )
        conn.execute(
            "UPDATE main.task_fence_control SET store_schema_version = 3 "
            "WHERE singleton = 1 AND store_schema_version = ?",
            (TASK_FENCE_STORE_SCHEMA_VERSION,),
        )
        conn.execute(snapshot_trigger[0])
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()

    check = sqlite3.connect(path)
    try:
        assert hermes_state._read_task_fence_schema_objects(
            check
        ) == hermes_state._expected_task_fence_v3_schema_objects()
        assert _scalar(
            check,
            "SELECT store_schema_version FROM task_fence_control",
        ) == 3
        assert check.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        check.close()


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


def test_archived_v2_migration_source_is_frozen() -> None:
    assert (
        hashlib.sha256(
            hermes_state.TASK_FENCE_SCHEMA_V2_SQL.encode("utf-8")
        ).hexdigest()
        == _TASK_FENCE_V2_DDL_SHA256
    )


def test_archived_v3_migration_source_is_frozen() -> None:
    assert (
        hashlib.sha256(
            hermes_state.TASK_FENCE_SCHEMA_V3_SQL.encode("utf-8")
        ).hexdigest()
        == _TASK_FENCE_V3_DDL_SHA256
    )


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
    assert acceptance.pending_input_ids == ()
    assert acceptance.task_projection is not None
    assert acceptance.task_projection.task_id == acceptance.task_id
    assert acceptance.task_projection.status == "running"
    assert acceptance.task_projection.intent_epoch == 1
    assert acceptance.task_projection.control_revision == 1
    assert (
        acceptance.task_projection.active_authority_event_id
        == acceptance.event_id
    )
    assert (
        acceptance.task_projection.active_execution_run_id
        == acceptance.opened_run_id
    )

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
        assert _scalar(
            conn,
            "SELECT COUNT(*) FROM main.task_fence_acceptance_snapshots",
        ) == 1
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


def test_acceptance_snapshot_rejects_update_delete_and_replace(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    envelope = _envelope("initial_submit", "message-1")
    accepted = db.accept_task_fence_ingress(envelope)
    conn = _connection(db)
    statements = (
        "UPDATE main.task_fence_acceptance_snapshots "
        f"SET task_status = 'paused' WHERE event_id = '{accepted.event_id}'",
        "DELETE FROM main.task_fence_acceptance_snapshots "
        f"WHERE event_id = '{accepted.event_id}'",
        "INSERT OR REPLACE INTO main.task_fence_acceptance_snapshots "
        "SELECT * FROM main.task_fence_acceptance_snapshots "
        f"WHERE event_id = '{accepted.event_id}'",
        "REPLACE INTO main.task_fence_acceptance_snapshots "
        "SELECT * FROM main.task_fence_acceptance_snapshots "
        f"WHERE event_id = '{accepted.event_id}'",
    )
    for statement in statements:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(statement)

    assert db.accept_task_fence_ingress(envelope) == replace(
        accepted,
        replayed=True,
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


def test_exact_replay_after_later_acceptance_returns_original_projection(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    original_envelope = _envelope("initial_submit", "message-1")
    initial = db.accept_task_fence_ingress(original_envelope)
    assert initial.task_id is not None
    db.accept_task_fence_ingress(
        _envelope("comment_hold", "message-2", task_id=initial.task_id)
    )
    before = db.inspect_task_fence_task(initial.task_id).task
    assert before is not None
    revision_before = before.control_revision
    db.close()
    db = SessionDB(path)

    replay = db.accept_task_fence_ingress(original_envelope)

    assert replay == replace(initial, replayed=True)
    assert replay.task_projection is not None
    assert replay.task_projection.control_revision == 1
    assert replay.task_projection.status == "running"
    assert replay.pending_input_ids == ()
    assert db.accept_task_fence_ingress(
        replace(original_envelope, task_id=initial.task_id)
    ) == replace(initial, replayed=True)
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


def test_wall_clock_regression_preserves_typed_acceptance(
    tmp_path,
    monkeypatch,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    timestamps = iter((100.0, 90.0))
    monkeypatch.setattr(hermes_state.time, "time", lambda: next(timestamps))
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-clock-initial")
    )
    assert initial.task_id is not None
    note_envelope = _envelope(
        "explicit_note",
        "message-clock-note",
        task_id=initial.task_id,
    )

    note = db.accept_task_fence_ingress(note_envelope)

    assert note.task_projection is not None
    assert note.task_projection.created_at == 100.0
    assert note.task_projection.updated_at == 90.0
    assert db.accept_task_fence_ingress(note_envelope) == replace(
        note,
        replayed=True,
    )
    db.close()


def test_replay_preserves_fifo_pending_projection_after_inputs_are_bound(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-initial")
    )
    assert initial.task_id is not None
    first_pending = db.accept_task_fence_ingress(
        _envelope("comment_hold", "message-z", task_id=initial.task_id)
    )
    second_envelope = _envelope(
        "comment_hold",
        "message-a",
        task_id=initial.task_id,
    )
    second_pending = db.accept_task_fence_ingress(second_envelope)

    assert first_pending.pending_input_ids == (
        initial.event_id,
        first_pending.event_id,
    )
    assert second_pending.pending_input_ids == (
        initial.event_id,
        first_pending.event_id,
        second_pending.event_id,
    )
    db.accept_task_fence_ingress(
        _envelope("change_and_run", "message-run", task_id=initial.task_id)
    )
    conn = _connection(db)
    assert conn.execute(
        "SELECT 1 FROM main.sqlite_master "
        "WHERE type = 'table' "
        "AND name = 'task_fence_acceptance_pending_inputs'"
    ).fetchone() is None
    replay = db.accept_task_fence_ingress(second_envelope)

    assert replay == replace(second_pending, replayed=True)
    assert replay.pending_input_ids == (
        initial.event_id,
        first_pending.event_id,
        second_pending.event_id,
    )
    db.close()


def test_pending_history_replay_is_exact_across_append_discard_bind_and_reopen(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    history: list[tuple[IngressEnvelope, object]] = []

    initial_envelope = _envelope("initial_submit", "history-initial")
    initial = db.accept_task_fence_ingress(initial_envelope)
    assert initial.task_id is not None
    history.append((initial_envelope, initial))

    first_envelope = _envelope(
        "comment_hold",
        "history-a",
        task_id=initial.task_id,
    )
    first = db.accept_task_fence_ingress(first_envelope)
    history.append((first_envelope, first))

    second_envelope = _envelope(
        "comment_hold",
        "history-b",
        task_id=initial.task_id,
    )
    second = db.accept_task_fence_ingress(second_envelope)
    history.append((second_envelope, second))

    note_envelope = _envelope(
        "explicit_note",
        "history-note",
        task_id=initial.task_id,
    )
    note = db.accept_task_fence_ingress(note_envelope)
    history.append((note_envelope, note))

    discard_envelope = _envelope(
        "discard_pending",
        "history-discard-a",
        task_id=initial.task_id,
        correlation_ids=(first.event_id,),
    )
    discarded = db.accept_task_fence_ingress(discard_envelope)
    history.append((discard_envelope, discarded))

    third_envelope = _envelope(
        "comment_hold",
        "history-c",
        task_id=initial.task_id,
    )
    third = db.accept_task_fence_ingress(third_envelope)
    history.append((third_envelope, third))

    run_envelope = _envelope(
        "change_and_run",
        "history-run",
        task_id=initial.task_id,
    )
    resumed = db.accept_task_fence_ingress(run_envelope)
    history.append((run_envelope, resumed))

    fourth_envelope = _envelope(
        "comment_hold",
        "history-d",
        task_id=initial.task_id,
    )
    fourth = db.accept_task_fence_ingress(fourth_envelope)
    history.append((fourth_envelope, fourth))

    assert initial.pending_input_ids == ()
    assert first.pending_input_ids == (initial.event_id, first.event_id)
    assert second.pending_input_ids == (
        initial.event_id,
        first.event_id,
        second.event_id,
    )
    assert note.pending_input_ids == second.pending_input_ids
    assert discarded.pending_input_ids == (initial.event_id, second.event_id)
    assert third.pending_input_ids == (
        initial.event_id,
        second.event_id,
        third.event_id,
    )
    assert resumed.pending_input_ids == ()
    assert fourth.pending_input_ids == (
        initial.event_id,
        second.event_id,
        third.event_id,
        resumed.event_id,
        fourth.event_id,
    )
    db.close()

    reopened = SessionDB(path)
    try:
        for envelope, accepted in history:
            assert reopened.accept_task_fence_ingress(envelope) == replace(
                accepted,
                replayed=True,
            )
    finally:
        reopened.close()


def test_pending_history_rows_grow_linearly(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "linear-initial")
    )
    assert initial.task_id is not None
    latest = initial
    event_count = 100
    baseline_rows = sum(
        len(rows) for _table_name, rows in _task_fence_table_state(_connection(db))
    )
    halfway_rows = 0
    for index in range(event_count):
        latest = db.accept_task_fence_ingress(
            _envelope(
                "comment_hold",
                f"linear-{index:03d}",
                task_id=initial.task_id,
            )
        )
        if index + 1 == event_count // 2:
            halfway_rows = sum(
                len(rows)
                for _table_name, rows in _task_fence_table_state(_connection(db))
            )

    conn = _connection(db)
    final_rows = sum(
        len(rows) for _table_name, rows in _task_fence_table_state(conn)
    )
    assert halfway_rows - baseline_rows == 3 * (event_count // 2)
    assert final_rows - halfway_rows == 3 * (event_count // 2)
    assert len(latest.pending_input_ids) == event_count + 1
    assert _scalar(conn, "SELECT COUNT(*) FROM task_fence_ingress") == (
        event_count + 1
    )
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM task_fence_acceptance_snapshots",
    ) == (event_count + 1)
    assert _scalar(conn, "SELECT COUNT(*) FROM task_fence_task_inputs") == (
        event_count + 1
    )
    assert conn.execute(
        "SELECT 1 FROM main.sqlite_master "
        "WHERE type = 'table' "
        "AND name = 'task_fence_acceptance_pending_inputs'"
    ).fetchone() is None
    db.close()


def test_pending_history_reconstruction_is_bounded_before_collision(
    tmp_path,
    monkeypatch,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "bounded-initial")
    )
    assert initial.task_id is not None
    events = []
    acceptances = []
    for index in range(4):
        envelope = _envelope(
            "comment_hold",
            f"bounded-{index}",
            task_id=initial.task_id,
        )
        events.append(envelope)
        acceptances.append(db.accept_task_fence_ingress(envelope))

    monkeypatch.setattr(
        hermes_state,
        "_TASK_FENCE_MAX_PENDING_HISTORY_WORK",
        4,
    )
    traced_statements: list[str] = []
    _connection(db).set_trace_callback(traced_statements.append)
    try:
        assert db.accept_task_fence_ingress(events[2]) == replace(
            acceptances[2],
            replayed=True,
        )
        for envelope in (
            events[3],
            replace(events[3], payload_hash=_hash("changed")),
        ):
            with pytest.raises(
                TaskFenceIngressUnavailable,
                match="pending_history_limit_exceeded",
            ):
                db.accept_task_fence_ingress(envelope)
    finally:
        _connection(db).set_trace_callback(None)
    history_queries = tuple(
        " ".join(statement.split())
        for statement in traced_statements
        if "FROM main.task_fence_ingress" in statement
        and "input_effect IN ('append', 'discard_selected')" in statement
    )
    assert history_queries
    assert all("LIMIT 5" in statement for statement in history_queries)
    assert _scalar(
        _connection(db),
        "SELECT COUNT(*) FROM task_fence_ingress_collisions",
    ) == 0
    db.close()


@pytest.mark.parametrize("corruption", ("missing_pending", "stale_pending"))
def test_run_rejects_mutable_pending_divergence_before_history_reset(
    tmp_path,
    corruption,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", f"{corruption}-initial")
    )
    assert initial.task_id is not None
    conn = _connection(db)

    if corruption == "missing_pending":
        held = db.accept_task_fence_ingress(
            _envelope(
                "comment_hold",
                "missing-pending-held",
                task_id=initial.task_id,
            )
        )
        conn.execute(
            "UPDATE main.task_fence_task_inputs "
            "SET state = 'bound' WHERE event_id = ?",
            (held.event_id,),
        )
        next_envelope = _envelope(
            "resume",
            "missing-pending-resume",
            task_id=initial.task_id,
        )
    else:
        conn.execute(
            "UPDATE main.task_fence_task_inputs "
            "SET state = 'pending', bound_run_id = NULL WHERE event_id = ?",
            (initial.event_id,),
        )
        next_envelope = _envelope(
            "change_and_run",
            "stale-pending-run",
            task_id=initial.task_id,
        )

    task_before = db.inspect_task_fence_task(initial.task_id).task
    ingress_count = _scalar(conn, "SELECT COUNT(*) FROM task_fence_ingress")
    snapshot_count = _scalar(
        conn,
        "SELECT COUNT(*) FROM task_fence_acceptance_snapshots",
    )
    run_count = _scalar(conn, "SELECT COUNT(*) FROM task_fence_execution_runs")
    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="incompatible_acceptance_projection",
    ):
        db.accept_task_fence_ingress(next_envelope)

    assert db.inspect_task_fence_task(initial.task_id).task == task_before
    assert _scalar(conn, "SELECT COUNT(*) FROM task_fence_ingress") == ingress_count
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM task_fence_acceptance_snapshots",
    ) == snapshot_count
    assert _scalar(conn, "SELECT COUNT(*) FROM task_fence_execution_runs") == run_count
    db.close()


def test_post_mutation_pending_failure_rolls_back_exact_state(
    tmp_path,
    monkeypatch,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "rollback-initial")
    )
    assert initial.task_id is not None
    db.accept_task_fence_ingress(
        _envelope(
            "comment_hold",
            "rollback-held",
            task_id=initial.task_id,
        )
    )
    conn = _connection(db)
    state_before = _task_fence_table_state(conn)
    original = SessionDB._task_fence_current_pending_input_ids_unlocked
    call_count = 0

    def fail_post_mutation(connection, *, task_id):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return hermes_state._TaskFenceIngressFailure(
                "injected_post_mutation_failure",
                unavailable=True,
            )
        return original(connection, task_id=task_id)

    monkeypatch.setattr(
        SessionDB,
        "_task_fence_current_pending_input_ids_unlocked",
        staticmethod(fail_post_mutation),
    )
    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="acceptance_database_error",
    ):
        db.accept_task_fence_ingress(
            _envelope(
                "change_and_run",
                "rollback-change-and-run",
                task_id=initial.task_id,
            )
        )

    assert call_count == 2
    assert _task_fence_table_state(conn) == state_before
    db.close()


def test_populated_v2_migrates_with_byte_equivalent_historical_replay(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    history: list[tuple[IngressEnvelope, IngressAcceptance]] = []

    initial_envelope = _envelope("initial_submit", "v2-initial")
    initial = db.accept_task_fence_ingress(initial_envelope)
    assert initial.task_id is not None
    history.append((initial_envelope, initial))

    first_envelope = _envelope(
        "comment_hold",
        "v2-a",
        task_id=initial.task_id,
    )
    first = db.accept_task_fence_ingress(first_envelope)
    history.append((first_envelope, first))

    second_envelope = _envelope(
        "comment_hold",
        "v2-b",
        task_id=initial.task_id,
    )
    second = db.accept_task_fence_ingress(second_envelope)
    history.append((second_envelope, second))

    discard_envelope = _envelope(
        "discard_pending",
        "v2-discard-a",
        task_id=initial.task_id,
        correlation_ids=(first.event_id,),
    )
    discarded = db.accept_task_fence_ingress(discard_envelope)
    history.append((discard_envelope, discarded))

    run_envelope = _envelope(
        "change_and_run",
        "v2-run",
        task_id=initial.task_id,
    )
    resumed = db.accept_task_fence_ingress(run_envelope)
    history.append((run_envelope, resumed))
    db.close()

    _downgrade_current_store_to_exact_v2(
        path,
        tuple(acceptance for _envelope_value, acceptance in history),
    )
    migrated = SessionDB(path)
    try:
        inspection = migrated.inspect_task_fence_store()
        assert inspection.compatible is True
        assert (
            inspection.observed_store_schema_version
            == TASK_FENCE_STORE_SCHEMA_VERSION
        )
        assert hermes_state._read_task_fence_schema_objects(
            _connection(migrated)
        ) == hermes_state._expected_task_fence_schema_objects()
        assert _connection(migrated).execute(
            "SELECT 1 FROM main.sqlite_master "
            "WHERE name = 'task_fence_acceptance_pending_inputs'"
        ).fetchone() is None

        for envelope, acceptance in history:
            historical_projection = acceptance.task_projection
            if historical_projection is not None:
                historical_projection = replace(
                    historical_projection,
                    store_schema_version=2,
                )
            expected = replace(
                acceptance,
                task_projection=historical_projection,
                replayed=True,
            )
            assert migrated.accept_task_fence_ingress(envelope) == expected

        before = migrated.inspect_task_fence_task(initial.task_id).task
        assert before is not None
        with pytest.raises(
            TaskFenceIngressRejected,
            match="source_event_id_collision",
        ):
            migrated.accept_task_fence_ingress(
                replace(initial_envelope, payload_hash=_hash("changed"))
            )
        after = migrated.inspect_task_fence_task(initial.task_id).task
        assert after == before
        assert _scalar(
            _connection(migrated),
            "SELECT COUNT(*) FROM task_fence_ingress_collisions",
        ) == 1
    finally:
        migrated.close()


@pytest.mark.parametrize(
    "failure_stage",
    ("reduction", "task_version", "control_version"),
)
def test_populated_v2_migration_fault_rolls_back_exact_state(
    tmp_path,
    monkeypatch,
    failure_stage: str,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "fault-v2-initial")
    )
    assert initial.task_id is not None
    held = db.accept_task_fence_ingress(
        _envelope(
            "comment_hold",
            "fault-v2-held",
            task_id=initial.task_id,
        )
    )
    db.close()
    _downgrade_current_store_to_exact_v2(path, (initial, held))
    objects_before = hermes_state._expected_task_fence_v2_schema_objects()
    state_connection = sqlite3.connect(path)
    try:
        state_before = _task_fence_table_state(state_connection)
    finally:
        state_connection.close()

    if failure_stage == "reduction":
        monkeypatch.setattr(
            hermes_state,
            "TASK_FENCE_SCHEMA_V3_REDUCTION_SQL",
            hermes_state.TASK_FENCE_SCHEMA_V3_REDUCTION_SQL
            + "INVALID TASK FENCE V3 REDUCTION;",
        )
        failed = SessionDB(path)
        try:
            assert failed._task_fence_schema_init_failed is True
        finally:
            failed.close()
    else:
        target = {
            "task_version": "task_fence_tasks",
            "control_version": "task_fence_control",
        }[failure_stage]
        probe = SessionDB.__new__(SessionDB)
        probe._conn = sqlite3.connect(path, isolation_level=None)
        probe._conn.row_factory = sqlite3.Row
        probe._conn.execute("PRAGMA foreign_keys=ON")
        probe._task_fence_schema_init_failed = False
        probe._conn.execute(
            "CREATE TEMP TRIGGER fail_task_fence_v2_migration "
            f"BEFORE UPDATE ON main.{target} BEGIN "
            "SELECT RAISE(ABORT, 'injected v2 migration failure'); END"
        )
        try:
            probe._init_task_fence_schema()
            assert probe._task_fence_schema_init_failed is True
        finally:
            probe._conn.close()

    check = sqlite3.connect(path)
    try:
        assert hermes_state._read_task_fence_schema_objects(
            check
        ) == objects_before
        assert _scalar(
            check,
            "SELECT store_schema_version FROM task_fence_control",
        ) == 2
        assert check.execute(
            "SELECT DISTINCT store_schema_version FROM task_fence_tasks"
        ).fetchall() == [(2,)]
        assert _scalar(
            check,
            "SELECT COUNT(*) FROM task_fence_acceptance_pending_inputs",
        ) == 2
        assert _task_fence_table_state(check) == state_before
        assert check.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        check.close()


def test_inconsistent_populated_v2_is_left_exactly_untouched(tmp_path) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "invalid-v2-initial")
    )
    assert initial.task_id is not None
    first = db.accept_task_fence_ingress(
        _envelope("comment_hold", "invalid-v2-a", task_id=initial.task_id)
    )
    second = db.accept_task_fence_ingress(
        _envelope("comment_hold", "invalid-v2-b", task_id=initial.task_id)
    )
    db.close()
    _downgrade_current_store_to_exact_v2(path, (initial, first, second))
    _rewrite_exact_v2_row(
        path,
        trigger_name="task_fence_acceptance_pending_inputs_no_delete",
        statement=(
            "DELETE FROM main.task_fence_acceptance_pending_inputs "
            "WHERE event_id = ? AND ordinal = 0"
        ),
        params=(second.event_id,),
    )
    objects_before = hermes_state._expected_task_fence_v2_schema_objects()

    rejected = SessionDB(path)
    try:
        inspection = rejected.inspect_task_fence_store()
        assert inspection.compatible is False
        assert inspection.reason == "unsupported_store_schema"
    finally:
        rejected.close()

    check = sqlite3.connect(path)
    try:
        assert hermes_state._read_task_fence_schema_objects(
            check
        ) == objects_before
        assert _scalar(
            check,
            "SELECT store_schema_version FROM task_fence_control",
        ) == 2
        assert _scalar(
            check,
            "SELECT COUNT(*) FROM task_fence_acceptance_pending_inputs",
        ) == 4
    finally:
        check.close()


def test_semantically_stale_populated_v2_is_left_exactly_untouched(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "stale-v2-initial")
    )
    assert initial.task_id is not None
    held = db.accept_task_fence_ingress(
        _envelope("comment_hold", "stale-v2-held", task_id=initial.task_id)
    )
    latest = db.accept_task_fence_ingress(
        _envelope("change_and_run", "stale-v2-run", task_id=initial.task_id)
    )
    assert held.task_projection is not None
    db.close()
    _downgrade_current_store_to_exact_v2(path, (initial, held, latest))

    stale_task = replace(held.task_projection, store_schema_version=2)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "UPDATE main.task_fence_tasks SET conversation_id = ?, "
            "cohort_key = ?, store_schema_version = ?, "
            "control_protocol_version = ?, intent_epoch = ?, "
            "control_revision = ?, status = ?, active_authority_event_id = ?, "
            "active_execution_run_id = ?, current_generation_id = ?, "
            "current_runtime_epoch = ?, last_accepted_order = ?, "
            "last_transition_event_id = ?, created_at = ?, updated_at = ? "
            "WHERE task_id = ?",
            (
                stale_task.conversation_id,
                stale_task.cohort_key,
                stale_task.store_schema_version,
                stale_task.control_protocol_version,
                stale_task.intent_epoch,
                stale_task.control_revision,
                stale_task.status,
                stale_task.active_authority_event_id,
                stale_task.active_execution_run_id,
                stale_task.current_generation_id,
                stale_task.current_runtime_epoch,
                stale_task.last_accepted_order,
                stale_task.last_transition_event_id,
                stale_task.created_at,
                stale_task.updated_at,
                stale_task.task_id,
            ),
        )
        conn.execute(
            "UPDATE main.task_fence_task_inputs "
            "SET state = 'pending', bound_run_id = NULL WHERE event_id = ?",
            (held.event_id,),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    state_before = _task_fence_table_state(conn)
    conn.close()

    rejected = SessionDB(path)
    try:
        inspection = rejected.inspect_task_fence_store()
        assert inspection.compatible is False
        assert inspection.reason == "unsupported_store_schema"
    finally:
        rejected.close()

    check = sqlite3.connect(path)
    try:
        assert hermes_state._read_task_fence_schema_objects(
            check
        ) == hermes_state._expected_task_fence_v2_schema_objects()
        assert _task_fence_table_state(check) == state_before
    finally:
        check.close()


def test_oversized_v2_migration_is_bounded_and_untouched(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "oversized-v2-initial")
    )
    assert initial.task_id is not None
    held = db.accept_task_fence_ingress(
        _envelope(
            "comment_hold",
            "oversized-v2-held",
            task_id=initial.task_id,
        )
    )
    db.close()
    _downgrade_current_store_to_exact_v2(path, (initial, held))
    monkeypatch.setattr(
        hermes_state,
        "_TASK_FENCE_MAX_V2_MIGRATION_ACCEPTANCES",
        1,
    )

    rejected = SessionDB(path)
    try:
        assert rejected.inspect_task_fence_store().compatible is False
    finally:
        rejected.close()

    check = sqlite3.connect(path)
    try:
        assert hermes_state._read_task_fence_schema_objects(
            check
        ) == hermes_state._expected_task_fence_v2_schema_objects()
        assert _scalar(
            check,
            "SELECT store_schema_version FROM task_fence_control",
        ) == 2
    finally:
        check.close()


def test_concurrent_populated_v2_migrators_publish_only_complete_current_schema(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    initial_envelope = _envelope("initial_submit", "race-v2-initial")
    initial = db.accept_task_fence_ingress(initial_envelope)
    assert initial.task_id is not None
    held_envelope = _envelope(
        "comment_hold",
        "race-v2-held",
        task_id=initial.task_id,
    )
    held = db.accept_task_fence_ingress(held_envelope)
    db.close()
    _downgrade_current_store_to_exact_v2(path, (initial, held))

    expected_v2 = hermes_state._expected_task_fence_v2_schema_objects()
    expected_current = hermes_state._expected_task_fence_schema_objects()
    original_read = hermes_state._read_task_fence_schema_objects
    barrier = threading.Barrier(2)
    first_reads: set[int] = set()
    reads_lock = threading.Lock()

    def synchronized_read(conn):
        observed = original_read(conn)
        connection_id = id(conn)
        should_wait = False
        with reads_lock:
            if (
                observed == expected_v2
                and not conn.in_transaction
                and connection_id not in first_reads
            ):
                first_reads.add(connection_id)
                should_wait = True
        if should_wait:
            barrier.wait(timeout=5)
        return observed

    monkeypatch.setattr(
        hermes_state,
        "_read_task_fence_schema_objects",
        synchronized_read,
    )

    def migrate() -> bool:
        candidate = SessionDB(path)
        try:
            return candidate.inspect_task_fence_store().compatible
        finally:
            candidate.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _index: migrate(), range(2)))

    assert results == (True, True)
    assert len(first_reads) == 2
    check = sqlite3.connect(path)
    try:
        assert original_read(check) == expected_current
    finally:
        check.close()
    verify = SessionDB(path)
    try:
        historical_projection = held.task_projection
        assert historical_projection is not None
        assert verify.accept_task_fence_ingress(held_envelope) == replace(
            held,
            task_projection=replace(
                historical_projection,
                store_schema_version=2,
            ),
            replayed=True,
        )
    finally:
        verify.close()


def test_populated_exact_v3_migrates_to_v4_without_decision_backfill(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    history: list[tuple[IngressEnvelope, IngressAcceptance]] = []

    initial_envelope = _envelope("initial_submit", "v3-initial")
    initial = db.accept_task_fence_ingress(initial_envelope)
    assert initial.task_id is not None
    history.append((initial_envelope, initial))

    held_envelope = _envelope(
        "comment_hold",
        "v3-held",
        task_id=initial.task_id,
    )
    held = db.accept_task_fence_ingress(held_envelope)
    history.append((held_envelope, held))

    resumed_envelope = _envelope(
        "change_and_run",
        "v3-resumed",
        task_id=initial.task_id,
    )
    resumed = db.accept_task_fence_ingress(resumed_envelope)
    history.append((resumed_envelope, resumed))
    db.close()

    _downgrade_current_store_to_exact_v3(path)
    migrated = SessionDB(path)
    try:
        inspection = migrated.inspect_task_fence_store(include_counts=True)
        assert inspection.compatible is True
        assert (
            inspection.observed_store_schema_version
            == TASK_FENCE_STORE_SCHEMA_VERSION
        )
        assert hermes_state._read_task_fence_schema_objects(
            _connection(migrated)
        ) == hermes_state._expected_task_fence_schema_objects()
        assert _scalar(
            _connection(migrated),
            "SELECT COUNT(*) FROM task_fence_policy_decisions",
        ) == 0
        assert _scalar(
            _connection(migrated),
            "SELECT MIN(store_schema_version) FROM task_fence_tasks",
        ) == TASK_FENCE_STORE_SCHEMA_VERSION
        assert _scalar(
            _connection(migrated),
            "SELECT MAX(store_schema_version) FROM task_fence_tasks",
        ) == TASK_FENCE_STORE_SCHEMA_VERSION

        for envelope, acceptance in history:
            historical_projection = acceptance.task_projection
            assert historical_projection is not None
            expected = replace(
                acceptance,
                task_projection=replace(
                    historical_projection,
                    store_schema_version=3,
                ),
                replayed=True,
            )
            assert migrated.accept_task_fence_ingress(envelope) == expected
    finally:
        migrated.close()


def test_exact_v3_with_pre_journal_policy_lifecycle_is_left_untouched(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "v3-policy-lifecycle")
    )
    assert initial.task_id is not None
    assert initial.opened_run_id is not None
    _seed_started_generation_and_permit(
        _connection(db),
        task_id=initial.task_id,
        run_id=initial.opened_run_id,
        event_id=initial.event_id,
    )
    db.close()
    _downgrade_current_store_to_exact_v3(path)

    before = sqlite3.connect(path)
    try:
        state_before = _task_fence_table_state(before)
    finally:
        before.close()

    rejected = SessionDB(path)
    try:
        inspection = rejected.inspect_task_fence_store()
        assert inspection.compatible is False
        assert inspection.reason == "unsupported_store_schema"
        assert inspection.observed_store_schema_version == 3
    finally:
        rejected.close()

    check = sqlite3.connect(path)
    try:
        assert hermes_state._read_task_fence_schema_objects(
            check
        ) == hermes_state._expected_task_fence_v3_schema_objects()
        assert _task_fence_table_state(check) == state_before
        assert _scalar(
            check,
            "SELECT COUNT(*) FROM task_fence_dispatch_permits",
        ) == 1
    finally:
        check.close()


def test_exact_v3_with_stale_cohort_generation_is_left_untouched(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db._conn.execute(
        "INSERT INTO task_fence_cohorts ("
        "cohort_key, mode, mode_generation, activation_state, "
        "audit_degraded, created_at, updated_at"
        ") VALUES ('stale-v3-cohort', 'audit', 1, 'inactive', 0, 1.0, 1.0)"
    )
    db.close()
    _downgrade_current_store_to_exact_v3(path)

    rejected = SessionDB(path)
    try:
        inspection = rejected.inspect_task_fence_store()
        assert inspection.compatible is False
        assert inspection.reason == "unsupported_store_schema"
        assert inspection.observed_store_schema_version == 3
    finally:
        rejected.close()

    check = sqlite3.connect(path)
    try:
        assert hermes_state._read_task_fence_schema_objects(
            check
        ) == hermes_state._expected_task_fence_v3_schema_objects()
        assert _scalar(
            check,
            "SELECT store_schema_version FROM task_fence_control",
        ) == 3
        assert _scalar(
            check,
            "SELECT mode_generation FROM task_fence_cohorts "
            "WHERE cohort_key = 'stale-v3-cohort'",
        ) == 1
    finally:
        check.close()


def test_oversized_v3_migration_is_bounded_and_untouched(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.accept_task_fence_ingress(
        _envelope("initial_submit", "oversized-v3-initial")
    )
    db.close()
    _downgrade_current_store_to_exact_v3(path)
    monkeypatch.setattr(hermes_state, "_TASK_FENCE_MAX_V3_MIGRATION_ROWS", 1)

    before = sqlite3.connect(path)
    try:
        state_before = _task_fence_table_state(before)
    finally:
        before.close()

    rejected = SessionDB(path)
    try:
        inspection = rejected.inspect_task_fence_store()
        assert inspection.compatible is False
        assert inspection.observed_store_schema_version == 3
    finally:
        rejected.close()

    check = sqlite3.connect(path)
    try:
        assert hermes_state._read_task_fence_schema_objects(
            check
        ) == hermes_state._expected_task_fence_v3_schema_objects()
        assert _task_fence_table_state(check) == state_before
    finally:
        check.close()


@pytest.mark.parametrize(
    "failure_stage",
    ("extension", "task_version", "control_version"),
)
def test_v3_migration_fault_rolls_back_exact_state(
    tmp_path,
    monkeypatch,
    failure_stage: str,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", f"fault-v3-{failure_stage}")
    )
    assert initial.task_id is not None
    db.close()
    _downgrade_current_store_to_exact_v3(path)
    objects_before = hermes_state._expected_task_fence_v3_schema_objects()
    state_connection = sqlite3.connect(path)
    try:
        state_before = _task_fence_table_state(state_connection)
    finally:
        state_connection.close()

    if failure_stage == "extension":
        monkeypatch.setattr(
            hermes_state,
            "TASK_FENCE_SCHEMA_V4_EXTENSION_SQL",
            hermes_state.TASK_FENCE_SCHEMA_V4_EXTENSION_SQL
            + "CREATE TABLE task_fence_v4_partial (value INTEGER);"
            + "INVALID TASK FENCE V4;",
        )
        failed = SessionDB(path)
        try:
            assert failed._task_fence_schema_init_failed is True
        finally:
            failed.close()
    else:
        target = {
            "task_version": "task_fence_tasks",
            "control_version": "task_fence_control",
        }[failure_stage]
        probe = SessionDB.__new__(SessionDB)
        probe._conn = sqlite3.connect(path, isolation_level=None)
        probe._conn.row_factory = sqlite3.Row
        probe._conn.execute("PRAGMA foreign_keys=ON")
        probe._task_fence_schema_init_failed = False
        probe._conn.execute(
            "CREATE TEMP TRIGGER fail_task_fence_v3_migration "
            f"BEFORE UPDATE ON main.{target} BEGIN "
            "SELECT RAISE(ABORT, 'injected v3 migration failure'); END"
        )
        try:
            probe._init_task_fence_schema()
            assert probe._task_fence_schema_init_failed is True
        finally:
            probe._conn.close()

    check = sqlite3.connect(path)
    try:
        assert hermes_state._read_task_fence_schema_objects(check) == objects_before
        assert _task_fence_table_state(check) == state_before
        assert _scalar(
            check,
            "SELECT store_schema_version FROM task_fence_control",
        ) == 3
        assert check.execute(
            "SELECT DISTINCT store_schema_version FROM task_fence_tasks"
        ).fetchall() == [(3,)]
        assert check.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        check.close()


def test_concurrent_populated_v3_migrators_publish_only_complete_v4(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "race-v3-initial")
    )
    assert initial.task_id is not None
    db.accept_task_fence_ingress(
        _envelope("comment_hold", "race-v3-held", task_id=initial.task_id)
    )
    db.close()
    _downgrade_current_store_to_exact_v3(path)

    expected_v3 = hermes_state._expected_task_fence_v3_schema_objects()
    expected_v4 = hermes_state._expected_task_fence_schema_objects()
    original_read = hermes_state._read_task_fence_schema_objects
    barrier = threading.Barrier(2)
    first_reads: set[int] = set()
    reads_lock = threading.Lock()

    def synchronized_read(conn):
        observed = original_read(conn)
        connection_id = id(conn)
        should_wait = False
        with reads_lock:
            if (
                observed == expected_v3
                and not conn.in_transaction
                and connection_id not in first_reads
            ):
                first_reads.add(connection_id)
                should_wait = True
        if should_wait:
            barrier.wait(timeout=5)
        return observed

    monkeypatch.setattr(
        hermes_state,
        "_read_task_fence_schema_objects",
        synchronized_read,
    )

    def migrate() -> bool:
        candidate = SessionDB(path)
        try:
            return candidate.inspect_task_fence_store().compatible
        finally:
            candidate.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _index: migrate(), range(2)))

    assert results == (True, True)
    assert len(first_reads) == 2
    check = sqlite3.connect(path)
    try:
        assert original_read(check) == expected_v4
        assert _scalar(
            check,
            "SELECT COUNT(*) FROM task_fence_policy_decisions",
        ) == 0
        assert check.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        check.close()


def test_task_bound_and_taskless_advisory_snapshots_are_historical(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    taskless_envelope = _envelope("explicit_note", "note-taskless")
    taskless = db.accept_task_fence_ingress(taskless_envelope)
    assert taskless.task_id is None
    assert taskless.task_projection is None
    assert taskless.pending_input_ids == ()

    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-initial")
    )
    assert initial.task_id is not None
    bound_note = db.accept_task_fence_ingress(
        _envelope("explicit_note", "note-bound", task_id=initial.task_id)
    )
    assert bound_note.task_projection is not None
    assert bound_note.task_projection.task_id == initial.task_id
    assert bound_note.task_projection.status == "running"
    assert (
        bound_note.task_projection.active_execution_run_id
        == initial.opened_run_id
    )
    assert (
        bound_note.task_projection.last_accepted_order
        == bound_note.accepted_order
    )
    assert db.accept_task_fence_ingress(taskless_envelope) == replace(
        taskless,
        replayed=True,
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

    assert caught.value.incident_id is not None
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
    collision = _connection(db).execute(
        "SELECT collision_id, accepted_event_id, incoming_fingerprint "
        "FROM main.task_fence_ingress_collisions"
    ).fetchone()
    assert collision[0] == caught.value.incident_id
    assert collision[1] == accepted.event_id
    assert len(collision[2]) == 64
    db.close()


def test_collision_retries_are_idempotent_and_distinct_payloads_append(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    accepted = db.accept_task_fence_ingress(
        _envelope("explicit_note", "note-1", payload="accepted")
    )
    assert accepted.task_id is None

    def collide(payload: str) -> str:
        with pytest.raises(
            TaskFenceIngressRejected,
            match="source_event_id_collision",
        ) as caught:
            db.accept_task_fence_ingress(
                _envelope("explicit_note", "note-1", payload=payload)
            )
        assert caught.value.incident_id is not None
        return caught.value.incident_id

    first_id = collide("different-a")
    assert collide("different-a") == first_id
    second_id = collide("different-b")

    assert second_id != first_id
    rows = _connection(db).execute(
        "SELECT collision_order, collision_id, accepted_event_id, "
        "incoming_fingerprint "
        "FROM main.task_fence_ingress_collisions ORDER BY collision_order"
    ).fetchall()
    assert [row[0] for row in rows] == [1, 2]
    assert [row[1] for row in rows] == [first_id, second_id]
    conn = _connection(db)
    for statement, params in (
        (
            "UPDATE main.task_fence_ingress_collisions "
            "SET collision_id = 'changed' WHERE collision_order = 1",
            (),
        ),
        (
            "DELETE FROM main.task_fence_ingress_collisions "
            "WHERE collision_order = 1",
            (),
        ),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(statement, params)
    for verb in ("INSERT OR REPLACE", "REPLACE"):
        replacements = (
            (
                f"{verb} INTO main.task_fence_ingress_collisions ("
                "collision_order, collision_id, accepted_event_id, "
                "incoming_fingerprint, first_detected_at"
                ") VALUES (1, 'replacement-order', ?, ?, 2.0)",
                (accepted.event_id, "a" * 64),
            ),
            (
                f"{verb} INTO main.task_fence_ingress_collisions ("
                "collision_id, accepted_event_id, incoming_fingerprint, "
                "first_detected_at) VALUES (?, ?, ?, 2.0)",
                (first_id, accepted.event_id, "b" * 64),
            ),
            (
                f"{verb} INTO main.task_fence_ingress_collisions ("
                "collision_id, accepted_event_id, incoming_fingerprint, "
                "first_detected_at) VALUES ('replacement-pair', ?, ?, 2.0)",
                (accepted.event_id, rows[0][3]),
            ),
        )
        for statement, params in replacements:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement, params)
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_incidents",
    ) == 0
    db.close()


def test_concurrent_identical_collisions_share_one_durable_incident(tmp_path) -> None:
    path = tmp_path / "state.db"
    seed = SessionDB(path)
    seed.accept_task_fence_ingress(
        _envelope("explicit_note", "note-1", payload="accepted")
    )
    seed.close()
    first = SessionDB(path)
    second = SessionDB(path)
    barrier = threading.Barrier(2)
    collision = _envelope("explicit_note", "note-1", payload="different")

    def collide(db: SessionDB) -> str:
        barrier.wait(timeout=5)
        with pytest.raises(TaskFenceIngressRejected) as caught:
            db.accept_task_fence_ingress(collision)
        assert caught.value.incident_id is not None
        return caught.value.incident_id

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            collision_ids = tuple(pool.map(collide, (first, second)))
    finally:
        first.close()
        second.close()

    assert len(set(collision_ids)) == 1
    verify = sqlite3.connect(path)
    try:
        assert _scalar(
            verify,
            "SELECT COUNT(*) FROM task_fence_ingress_collisions",
        ) == 1
    finally:
        verify.close()


def test_collision_journal_failure_is_unavailable_without_authority_change(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    accepted = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1", payload="accepted")
    )
    assert accepted.task_id is not None
    conn = _connection(db)
    conn.execute(
        "CREATE TEMP TRIGGER fail_task_fence_collision "
        "BEFORE INSERT ON main.task_fence_ingress_collisions BEGIN "
        "SELECT RAISE(ABORT, 'injected collision failure'); END"
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="acceptance_database_error",
    ):
        db.accept_task_fence_ingress(
            _envelope("initial_submit", "message-1", payload="different")
        )

    task = db.inspect_task_fence_task(accepted.task_id).task
    assert task is not None
    assert task.control_revision == 1
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress_collisions",
    ) == 0
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    ) == 1
    db.close()


def test_missing_historical_snapshot_fails_closed_without_live_fallback(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    conn = _connection(db)
    envelope = _envelope("explicit_note", "note-1")
    conn.execute(
        "INSERT INTO main.task_fence_ingress ("
        "event_id, source, source_event_id, conversation_id, protocol_version, "
        "origin, ingress_class, intent, execution, input_effect, "
        "correlation_kind, payload_hash, accepted_at"
        ") VALUES ('manual-event', ?, ?, ?, 1, 'human', 'advisory', 'keep', "
        "'none', 'none', 'none', ?, 1.0)",
        (
            envelope.source,
            envelope.source_event_id,
            envelope.conversation_id,
            envelope.payload_hash,
        ),
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="incompatible_acceptance_projection",
    ):
        db.accept_task_fence_ingress(envelope)

    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress_collisions",
    ) == 0
    db.close()


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("task_conversation_id", sqlite3.Binary(b"not-text")),
        ("task_conversation_id", "other-conversation"),
        ("task_cohort_key", "invalid\x00cohort"),
        ("task_cohort_key", "x" * 513),
        ("task_updated_at", sqlite3.Binary(b"1.0")),
        ("task_updated_at", float("inf")),
    ),
)
def test_malformed_historical_projection_is_unavailable_before_collision(
    tmp_path,
    column,
    value,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    envelope = _envelope("initial_submit", "message-1", payload="accepted")
    accepted = db.accept_task_fence_ingress(envelope)
    _rewrite_append_only_row(
        db,
        trigger_name="task_fence_acceptance_snapshots_no_update",
        statement=(
            f"UPDATE main.task_fence_acceptance_snapshots SET {column} = ? "
            "WHERE event_id = ?"
        ),
        params=(value, accepted.event_id),
    )

    for replay_envelope in (
        envelope,
        _envelope("initial_submit", "message-1", payload="different"),
    ):
        with pytest.raises(
            TaskFenceIngressUnavailable,
            match="incompatible_acceptance_projection",
        ):
            db.accept_task_fence_ingress(replay_envelope)

    assert _scalar(
        _connection(db),
        "SELECT COUNT(*) FROM main.task_fence_ingress_collisions",
    ) == 0
    db.close()


def test_historical_acceptance_rejects_blob_timestamp(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    envelope = _envelope("explicit_note", "note-blob-timestamp")
    accepted = db.accept_task_fence_ingress(envelope)
    _rewrite_append_only_row(
        db,
        trigger_name="task_fence_ingress_no_update",
        statement=(
            "UPDATE main.task_fence_ingress SET accepted_at = ? "
            "WHERE event_id = ?"
        ),
        params=(sqlite3.Binary(b"1.0"), accepted.event_id),
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="incompatible_acceptance_projection",
    ):
        db.accept_task_fence_ingress(envelope)
    db.close()


def test_historical_pending_projection_rejects_cross_task_discard(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    task_a_initial_envelope = replace(
        _envelope("initial_submit", "a-initial"),
        source="gateway:test:conversation-a",
        conversation_id="conversation-a",
    )
    task_a = db.accept_task_fence_ingress(task_a_initial_envelope)
    assert task_a.task_id is not None
    task_a_pending = db.accept_task_fence_ingress(
        replace(
            _envelope("comment_hold", "a-pending", task_id=task_a.task_id),
            source="gateway:test:conversation-a",
            conversation_id="conversation-a",
        )
    )
    task_b_initial_envelope = replace(
        _envelope("initial_submit", "b-initial"),
        source="gateway:test:conversation-b",
        conversation_id="conversation-b",
    )
    task_b = db.accept_task_fence_ingress(task_b_initial_envelope)
    assert task_b.task_id is not None
    task_b_pending_envelope = replace(
        _envelope("comment_hold", "b-pending", task_id=task_b.task_id),
        source="gateway:test:conversation-b",
        conversation_id="conversation-b",
    )
    task_b_pending = db.accept_task_fence_ingress(task_b_pending_envelope)
    task_b_discard_envelope = replace(
        _envelope(
            "discard_pending",
            "b-discard",
            task_id=task_b.task_id,
            correlation_ids=(task_b_pending.event_id,),
        ),
        source="gateway:test:conversation-b",
        conversation_id="conversation-b",
    )
    task_b_discard = db.accept_task_fence_ingress(task_b_discard_envelope)
    _rewrite_append_only_row(
        db,
        trigger_name="task_fence_ingress_correlations_no_update",
        statement=(
            "UPDATE main.task_fence_ingress_correlations "
            "SET correlation_id = ? WHERE event_id = ?"
        ),
        params=(task_a_pending.event_id, task_b_discard.event_id),
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="incompatible_acceptance_projection",
    ):
        db.accept_task_fence_ingress(task_b_discard_envelope)
    db.close()


@pytest.mark.parametrize(
    ("snapshot_field", "foreign_id_kind"),
    (
        ("task_active_authority_event_id", "event"),
        ("task_last_transition_event_id", "event"),
        ("task_active_execution_run_id", "run"),
        ("opened_run_id", "run"),
        ("closed_run_id", "run"),
    ),
)
def test_historical_projection_rejects_cross_task_authority_links(
    tmp_path,
    snapshot_field,
    foreign_id_kind,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    task_a = db.accept_task_fence_ingress(
        replace(
            _envelope("initial_submit", "a-initial"),
            source="gateway:test:conversation-a",
            conversation_id="conversation-a",
        )
    )
    task_b_envelope = replace(
        _envelope("initial_submit", "b-initial"),
        source="gateway:test:conversation-b",
        conversation_id="conversation-b",
    )
    task_b = db.accept_task_fence_ingress(task_b_envelope)
    assert task_a.opened_run_id is not None
    foreign_id = (
        task_a.event_id
        if foreign_id_kind == "event"
        else task_a.opened_run_id
    )
    _rewrite_append_only_row(
        db,
        trigger_name="task_fence_acceptance_snapshots_no_update",
        statement=(
            "UPDATE main.task_fence_acceptance_snapshots "
            f"SET {snapshot_field} = ? WHERE event_id = ?"
        ),
        params=(foreign_id, task_b.event_id),
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="incompatible_acceptance_projection",
    ):
        db.accept_task_fence_ingress(task_b_envelope)
    db.close()


@pytest.mark.parametrize("corruption", ("paused_active", "older_active_run"))
def test_historical_projection_rejects_impossible_run_shape(
    tmp_path,
    corruption,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "shape-initial")
    )
    assert initial.task_id is not None
    assert initial.opened_run_id is not None
    if corruption == "paused_active":
        replay_envelope = _envelope("initial_submit", "shape-initial")
        acceptance = initial
        assignment = "task_status = 'paused'"
        params = (acceptance.event_id,)
    else:
        replay_envelope = _envelope(
            "change_and_run",
            "shape-replacement",
            task_id=initial.task_id,
        )
        acceptance = db.accept_task_fence_ingress(replay_envelope)
        assignment = "task_active_execution_run_id = ?"
        params = (initial.opened_run_id, acceptance.event_id)
    _rewrite_append_only_row(
        db,
        trigger_name="task_fence_acceptance_snapshots_no_update",
        statement=(
            "UPDATE main.task_fence_acceptance_snapshots "
            f"SET {assignment} WHERE event_id = ?"
        ),
        params=params,
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="incompatible_acceptance_projection",
    ):
        db.accept_task_fence_ingress(replay_envelope)
    db.close()


def test_historical_generation_must_match_active_run_and_counters(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "generation-initial")
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
    note_envelope = _envelope(
        "explicit_note",
        "generation-note",
        task_id=initial.task_id,
    )
    note = db.accept_task_fence_ingress(note_envelope)
    assert note.task_projection is not None
    assert note.task_projection.current_generation_id == "generation-1"
    conn.execute(
        "INSERT INTO main.task_fence_model_generations ("
        "generation_id, task_id, run_id, intent_epoch, control_revision, "
        "runtime_epoch, input_manifest_hash, snapshot_event_id, state, "
        "opened_at, closed_at"
        ") VALUES ('generation-wrong-counters', ?, ?, 0, 0, 0, ?, ?, "
        "'failed', 1.0, 2.0)",
        (
            initial.task_id,
            initial.opened_run_id,
            _hash("wrong-generation"),
            initial.event_id,
        ),
    )
    _rewrite_append_only_row(
        db,
        trigger_name="task_fence_acceptance_snapshots_no_update",
        statement=(
            "UPDATE main.task_fence_acceptance_snapshots "
            "SET task_current_generation_id = 'generation-wrong-counters' "
            "WHERE event_id = ?"
        ),
        params=(note.event_id,),
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="incompatible_acceptance_projection",
    ):
        db.accept_task_fence_ingress(note_envelope)
    db.close()


def test_historical_closed_run_must_close_on_acceptance_event(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "close-initial")
    )
    assert initial.task_id is not None
    second = db.accept_task_fence_ingress(
        _envelope("change_and_run", "close-second", task_id=initial.task_id)
    )
    third_envelope = _envelope(
        "change_and_run",
        "close-third",
        task_id=initial.task_id,
    )
    third = db.accept_task_fence_ingress(third_envelope)
    assert second.opened_run_id == third.closed_run_id
    _rewrite_append_only_row(
        db,
        trigger_name="task_fence_acceptance_snapshots_no_update",
        statement=(
            "UPDATE main.task_fence_acceptance_snapshots "
            "SET closed_run_id = ? WHERE event_id = ?"
        ),
        params=(initial.opened_run_id, third.event_id),
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="incompatible_acceptance_projection",
    ):
        db.accept_task_fence_ingress(third_envelope)
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
    accepted_result = next(result for result in results if not result.replayed)
    replayed_result = next(result for result in results if result.replayed)
    assert replayed_result == replace(accepted_result, replayed=True)
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


def test_pending_history_serializes_across_two_connections(tmp_path) -> None:
    path = tmp_path / "state.db"
    initial_db = SessionDB(path)
    initial = initial_db.accept_task_fence_ingress(
        _envelope("initial_submit", "pending-race-initial")
    )
    assert initial.task_id is not None
    initial_db.close()
    first = SessionDB(path)
    second = SessionDB(path)
    barrier = threading.Barrier(2)
    envelopes = (
        _envelope("comment_hold", "pending-race-a", task_id=initial.task_id),
        _envelope("comment_hold", "pending-race-b", task_id=initial.task_id),
    )

    def accept(pair):
        candidate, envelope = pair
        barrier.wait(timeout=5)
        return candidate.accept_task_fence_ingress(envelope)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(accept, zip((first, second), envelopes)))
    finally:
        first.close()
        second.close()

    earlier, later = sorted(results, key=lambda item: item.accepted_order)
    assert earlier.pending_input_ids == (initial.event_id, earlier.event_id)
    assert later.pending_input_ids == (
        initial.event_id,
        earlier.event_id,
        later.event_id,
    )
    envelope_by_source = {item.source_event_id: item for item in envelopes}
    reopened = SessionDB(path)
    try:
        for accepted in (earlier, later):
            envelope = envelope_by_source[
                _scalar(
                    _connection(reopened),
                    "SELECT source_event_id FROM task_fence_ingress "
                    "WHERE event_id = ?",
                    (accepted.event_id,),
                )
            ]
            assert reopened.accept_task_fence_ingress(envelope) == replace(
                accepted,
                replayed=True,
            )
    finally:
        reopened.close()


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


@pytest.mark.parametrize(
    "failure_stage",
    ["run_insert", "task_update", "snapshot_insert"],
)
def test_mid_transaction_sqlite_abort_rolls_back_every_authority_write(
    tmp_path,
    failure_stage: str,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    conn = _connection(db)
    operation, table = {
        "run_insert": ("INSERT", "task_fence_execution_runs"),
        "task_update": ("UPDATE", "task_fence_tasks"),
        "snapshot_insert": ("INSERT", "task_fence_acceptance_snapshots"),
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
        "task_fence_acceptance_snapshots",
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


def test_pre_mutation_pending_failure_preserves_existing_authority(
    tmp_path,
    monkeypatch,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    initial = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1")
    )
    assert initial.task_id is not None
    assert initial.opened_run_id is not None
    conn = _connection(db)
    monkeypatch.setattr(
        SessionDB,
        "_task_fence_pending_input_ids_at_unlocked",
        staticmethod(
            lambda *_args, **_kwargs: hermes_state._TaskFenceIngressFailure(
                "pending_history_unavailable",
                unavailable=True,
            )
        ),
    )

    with pytest.raises(
        TaskFenceIngressUnavailable,
        match="pending_history_unavailable",
    ):
        db.accept_task_fence_ingress(
            _envelope("comment_hold", "message-2", task_id=initial.task_id)
        )

    task = db.inspect_task_fence_task(initial.task_id).task
    assert task is not None
    assert task.control_revision == 1
    assert task.status == "running"
    assert task.active_execution_run_id == initial.opened_run_id
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_ingress",
    ) == 1
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_acceptance_snapshots",
    ) == 1
    assert _scalar(
        conn,
        "SELECT COUNT(*) FROM main.task_fence_execution_runs "
        "WHERE state = 'open'",
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

    resolution_envelope = _envelope(
        "resolve_incident",
        "message-2",
        task_id=initial.task_id,
        correlation_ids=("attempt-1",),
        resolution_disposition=(
            ResolutionDisposition.ACCEPTED_UNKNOWN_NO_RETRY
        ),
        evidence_refs=("operator:ticket-1",),
    )
    resolved = db.accept_task_fence_ingress(resolution_envelope)

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
    guarded_statements = (
        "INSERT INTO main.task_fence_ingress_correlations "
        f"VALUES ('{resolved.event_id}', 'attempt-late')",
        "UPDATE main.task_fence_ingress_correlations "
        "SET correlation_id = 'attempt-other' "
        f"WHERE event_id = '{resolved.event_id}'",
        "DELETE FROM main.task_fence_ingress_correlations "
        f"WHERE event_id = '{resolved.event_id}'",
        "INSERT OR REPLACE INTO main.task_fence_ingress_evidence "
        f"VALUES ('{resolved.event_id}', 'operator:replacement')",
        "UPDATE main.task_fence_resolution_evidence "
        "SET evidence_ref = 'operator:replacement'",
        "DELETE FROM main.task_fence_resolution_evidence",
        "INSERT INTO main.task_fence_incident_evidence "
        "VALUES ('incident-1', 'operator:late')",
        "INSERT OR REPLACE INTO main.task_fence_incident_attempts "
        "VALUES ('incident-1', 'attempt-1')",
    )
    for statement in guarded_statements:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(statement)
    assert db.accept_task_fence_ingress(resolution_envelope) == replace(
        resolved,
        replayed=True,
    )
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
        "UPDATE main.task_fence_tasks SET store_schema_version = ? "
        "WHERE task_id = ?",
        (TASK_FENCE_STORE_SCHEMA_VERSION + 1, initial.task_id),
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
    assert tuple(row) == (
        TASK_FENCE_STORE_SCHEMA_VERSION + 1,
        1,
        "running",
    )
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
    rejected_reference = "do-not-store-this-rejected-private-reference"
    path = tmp_path / "state.db"
    db = SessionDB(path)
    accepted = db.accept_task_fence_ingress(
        _envelope("initial_submit", "message-1", payload=secret)
    )
    with pytest.raises(TaskFenceIngressRejected):
        db.accept_task_fence_ingress(
            IngressEnvelope(
                source="gateway:test:conversation-1",
                source_event_id="message-1",
                conversation_id="conversation-1",
                task_id=accepted.task_id,
                action=TASK_FENCE_ACTIONS["initial_submit"],
                opaque_payload_ref=rejected_reference,
            )
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
    assert rejected_reference not in dump
    assert _hash(secret) in dump
