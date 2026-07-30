from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import hashlib
import sqlite3
import threading

import pytest

import hermes_state
from hermes_state import (
    SCHEMA_VERSION,
    TASK_FENCE_CONTROL_PROTOCOL_VERSION,
    TASK_FENCE_STORE_SCHEMA_VERSION,
    SessionDB,
)


EXPECTED_TASK_FENCE_TABLES = {
    "task_fence_acceptance_snapshots",
    "task_fence_attempt_transitions",
    "task_fence_attempts",
    "task_fence_cohort_capabilities",
    "task_fence_cohorts",
    "task_fence_control",
    "task_fence_dispatch_permits",
    "task_fence_execution_runs",
    "task_fence_incident_attempts",
    "task_fence_incident_evidence",
    "task_fence_incidents",
    "task_fence_ingress",
    "task_fence_ingress_collisions",
    "task_fence_ingress_correlations",
    "task_fence_ingress_evidence",
    "task_fence_model_generations",
    "task_fence_policy_decisions",
    "task_fence_questions",
    "task_fence_recovery_requeues",
    "task_fence_resolution_evidence",
    "task_fence_resolutions",
    "task_fence_task_inputs",
    "task_fence_tasks",
}


def _task_fence_schema_objects(path):
    conn = sqlite3.connect(path)
    try:
        return hermes_state._read_task_fence_schema_objects(conn)
    finally:
        conn.close()


def _schema_cookie(path):
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("PRAGMA schema_version").fetchone()[0])
    finally:
        conn.close()


def _drop_task_fence_schema(path):
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name GLOB 'task_fence_*'"
        ).fetchall()
        for (table,) in tables:
            safe_table = table.replace('"', '""')
            conn.execute(f'DROP TABLE "{safe_table}"')
        conn.commit()
    finally:
        conn.close()


def _install_task_fence_v1_schema(path, **control_overrides):
    _drop_task_fence_schema(path)
    control = {
        "runtime_epoch": 0,
        "mode_generation": 0,
        "ever_enforced": 0,
        "tested_artifact_commit": None,
        "tested_artifact_checksum": None,
        "dependency_lock_fingerprint": None,
    }
    control.update(control_overrides)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(hermes_state.TASK_FENCE_SCHEMA_V1_SQL)
        conn.execute(
            "INSERT INTO task_fence_control ("
            "singleton, store_schema_version, control_protocol_version, "
            "runtime_epoch, mode_generation, ever_enforced, "
            "tested_artifact_commit, tested_artifact_checksum, "
            "dependency_lock_fingerprint, created_at, updated_at"
            ") VALUES (1, 1, ?, ?, ?, ?, ?, ?, ?, 1.0, 1.0)",
            (
                TASK_FENCE_CONTROL_PROTOCOL_VERSION,
                control["runtime_epoch"],
                control["mode_generation"],
                control["ever_enforced"],
                control["tested_artifact_commit"],
                control["tested_artifact_checksum"],
                control["dependency_lock_fingerprint"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _install_task_fence_v4_schema(path, **control_overrides):
    _drop_task_fence_schema(path)
    control = {
        "runtime_epoch": 0,
        "mode_generation": 0,
        "ever_enforced": 0,
        "tested_artifact_commit": None,
        "tested_artifact_checksum": None,
        "dependency_lock_fingerprint": None,
    }
    control.update(control_overrides)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(hermes_state.TASK_FENCE_SCHEMA_V4_SQL)
        conn.execute(
            "INSERT INTO task_fence_control ("
            "singleton, store_schema_version, control_protocol_version, "
            "runtime_epoch, mode_generation, ever_enforced, "
            "tested_artifact_commit, tested_artifact_checksum, "
            "dependency_lock_fingerprint, created_at, updated_at"
            ") VALUES (1, 4, ?, ?, ?, ?, ?, ?, ?, 1.0, 1.0)",
            (
                TASK_FENCE_CONTROL_PROTOCOL_VERSION,
                control["runtime_epoch"],
                control["mode_generation"],
                control["ever_enforced"],
                control["tested_artifact_commit"],
                control["tested_artifact_checksum"],
                control["dependency_lock_fingerprint"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _install_task_fence_v5_schema(path, **control_overrides):
    _drop_task_fence_schema(path)
    control = {
        "runtime_epoch": 0,
        "mode_generation": 0,
        "ever_enforced": 0,
        "tested_artifact_commit": None,
        "tested_artifact_checksum": None,
        "dependency_lock_fingerprint": None,
    }
    control.update(control_overrides)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(hermes_state.TASK_FENCE_SCHEMA_V5_SQL)
        conn.execute(
            "INSERT INTO task_fence_control ("
            "singleton, store_schema_version, control_protocol_version, "
            "runtime_epoch, mode_generation, ever_enforced, "
            "tested_artifact_commit, tested_artifact_checksum, "
            "dependency_lock_fingerprint, created_at, updated_at"
            ") VALUES (1, 5, ?, ?, ?, ?, ?, ?, ?, 1.0, 1.0)",
            (
                TASK_FENCE_CONTROL_PROTOCOL_VERSION,
                control["runtime_epoch"],
                control["mode_generation"],
                control["ever_enforced"],
                control["tested_artifact_commit"],
                control["tested_artifact_checksum"],
                control["dependency_lock_fingerprint"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_v4_policy_decision(
    conn,
    *,
    decision_order=7,
    invocation_id="tfiv-v4",
):
    invocation_fingerprint = "b" * 64
    decision_id = hermes_state._task_fence_policy_decision_identifier(
        {
            "policy_version": "task-fence-policy-v1",
            "decision_point": "admission",
            "outcome": "would_block",
            "reason_code": "missing_provenance",
            "candidate_task_id": None,
            "candidate_authority_event_id": None,
            "candidate_run_id": None,
            "candidate_generation_id": None,
            "candidate_intent_epoch": None,
            "candidate_control_revision": None,
            "candidate_runtime_epoch": None,
            "cohort_key": None,
            "mode_generation": 0,
            "operation": {
                "invocation_id": invocation_id,
                "kind": "tool",
                "adapter": "registry:v4",
                "invocation_fingerprint": invocation_fingerprint,
            },
            "envelope_invocation_id": None,
            "causal_binding_fingerprint": None,
            "permit_id": None,
            "attempt_id": None,
        }
    )
    order_column = "decision_order, " if decision_order is not None else ""
    order_value = "?, " if decision_order is not None else ""
    params = (
        (decision_order, decision_id, invocation_id, invocation_fingerprint)
        if decision_order is not None
        else (decision_id, invocation_id, invocation_fingerprint)
    )
    conn.execute(
        "INSERT INTO task_fence_policy_decisions ("
        f"{order_column}decision_id, decision_point, outcome, reason_code, "
        "policy_version, mode_generation, operation_invocation_id, "
        "operation_kind, adapter, invocation_fingerprint, decided_at"
        f") VALUES ({order_value}?, 'admission', 'would_block', "
        "'missing_provenance', 'task-fence-policy-v1', 0, ?, 'tool', "
        "'registry:v4', ?, 1.0)",
        params,
    )
    return decision_id


def _insert_v4_dispatch_permit(conn, *, audience):
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO task_fence_tasks ("
            "task_id, conversation_id, store_schema_version, "
            "control_protocol_version, intent_epoch, control_revision, status, "
            "active_authority_event_id, active_execution_run_id, "
            "current_generation_id, current_runtime_epoch, last_accepted_order, "
            "last_transition_event_id, created_at, updated_at"
            ") VALUES ("
            "'task-v4-permit', 'conversation-v4-permit', 4, ?, 0, 0, "
            "'running', 'event-v4-permit', 'run-v4-permit', "
            "'generation-v4-permit', 0, 1, 'event-v4-permit', 1.0, 1.0"
            ")",
            (TASK_FENCE_CONTROL_PROTOCOL_VERSION,),
        )
        conn.execute(
            "INSERT INTO task_fence_ingress ("
            "accepted_order, event_id, source, source_event_id, conversation_id, "
            "task_id, protocol_version, origin, ingress_class, intent, execution, "
            "input_effect, correlation_kind, payload_hash, accepted_at"
            ") VALUES ("
            "1, 'event-v4-permit', 'gateway:test', 'source-v4-permit', "
            "'conversation-v4-permit', 'task-v4-permit', 1, 'human', "
            "'task_input', 'replace', 'run', 'append', 'none', ?, 1.0"
            ")",
            ("a" * 64,),
        )
        conn.execute(
            "INSERT INTO task_fence_execution_runs ("
            "run_id, task_id, authority_event_id, intent_epoch, control_revision, "
            "runtime_epoch, bound_input_hash, state, open_event_id, opened_at"
            ") VALUES ("
            "'run-v4-permit', 'task-v4-permit', 'event-v4-permit', 0, 0, 0, "
            "?, 'open', 'event-v4-permit', 1.0"
            ")",
            ("b" * 64,),
        )
        conn.execute(
            "INSERT INTO task_fence_model_generations ("
            "generation_id, task_id, run_id, intent_epoch, control_revision, "
            "runtime_epoch, input_manifest_hash, snapshot_event_id, state, opened_at"
            ") VALUES ("
            "'generation-v4-permit', 'task-v4-permit', 'run-v4-permit', "
            "0, 0, 0, ?, 'event-v4-permit', 'committed', 1.0"
            ")",
            ("b" * 64,),
        )
        conn.execute(
            "INSERT INTO task_fence_task_inputs ("
            "event_id, task_id, state, bound_run_id, state_changed_at"
            ") VALUES ("
            "'event-v4-permit', 'task-v4-permit', 'presented', "
            "'run-v4-permit', 1.0"
            ")"
        )
        conn.execute(
            "INSERT INTO task_fence_acceptance_snapshots ("
            "event_id, envelope_fingerprint, task_id, task_conversation_id, "
            "task_store_schema_version, task_control_protocol_version, "
            "task_intent_epoch, task_control_revision, task_status, "
            "task_active_authority_event_id, task_active_execution_run_id, "
            "task_current_generation_id, task_current_runtime_epoch, "
            "task_last_accepted_order, task_last_transition_event_id, "
            "task_created_at, task_updated_at, opened_run_id, pending_input_count"
            ") VALUES ("
            "'event-v4-permit', ?, 'task-v4-permit', "
            "'conversation-v4-permit', 4, ?, 0, 0, 'running', "
            "'event-v4-permit', 'run-v4-permit', 'generation-v4-permit', "
            "0, 1, 'event-v4-permit', 1.0, 1.0, 'run-v4-permit', 1"
            ")",
            ("c" * 64, TASK_FENCE_CONTROL_PROTOCOL_VERSION),
        )
        conn.execute(
            "INSERT INTO task_fence_dispatch_permits ("
            "permit_id, task_id, authority_event_id, run_id, generation_id, "
            "intent_epoch, control_revision, runtime_epoch, "
            "invocation_envelope_id, invocation_fingerprint, executor, audience, "
            "policy_decision, policy_version, expires_at, state, reserved_at"
            ") VALUES ("
            "'permit-v4', 'task-v4-permit', 'event-v4-permit', "
            "'run-v4-permit', 'generation-v4-permit', 0, 0, 0, "
            "'tfiv-v4-permit', ?, 'gateway:test', ?, "
            "'would_reserve:current_authority', 'task-fence-policy-v1', "
            "2.0, 'reserved', 1.0"
            ")",
            ("d" * 64, audience),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _insert_task(
    conn,
    *,
    task_id,
    conversation_id,
    store_schema_version=TASK_FENCE_STORE_SCHEMA_VERSION,
    status="planning",
    intent_epoch=0,
    control_revision=0,
    runtime_epoch=0,
):
    conn.execute(
        "INSERT INTO task_fence_tasks ("
        "task_id, conversation_id, store_schema_version, "
        "control_protocol_version, intent_epoch, control_revision, status, "
        "current_runtime_epoch, created_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            task_id,
            conversation_id,
            store_schema_version,
            TASK_FENCE_CONTROL_PROTOCOL_VERSION,
            intent_epoch,
            control_revision,
            status,
            runtime_epoch,
            1.0,
            2.0,
        ),
    )


def _insert_cohort(
    conn,
    *,
    cohort_key,
    mode="audit",
    mode_generation=0,
    activation_state="inactive",
    audit_degraded=0,
):
    conn.execute(
        "INSERT INTO task_fence_cohorts ("
        "cohort_key, mode, mode_generation, activation_state, "
        "audit_degraded, created_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, 1.0, 2.0)",
        (
            cohort_key,
            mode,
            mode_generation,
            activation_state,
            audit_degraded,
        ),
    )


def test_fresh_store_initializes_exact_current_task_fence_schema(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        inspection = db.inspect_task_fence_store(include_counts=True)
    finally:
        db.close()

    assert inspection.initialized is True
    assert inspection.compatible is True
    assert inspection.reason == "compatible"
    assert inspection.expected_store_schema_version == TASK_FENCE_STORE_SCHEMA_VERSION
    assert inspection.observed_store_schema_version == TASK_FENCE_STORE_SCHEMA_VERSION
    assert (
        inspection.expected_control_protocol_version
        == TASK_FENCE_CONTROL_PROTOCOL_VERSION
    )
    assert (
        inspection.observed_control_protocol_version
        == TASK_FENCE_CONTROL_PROTOCOL_VERSION
    )
    assert inspection.runtime_epoch == 0
    assert inspection.mode_generation == 0
    assert inspection.ever_enforced is False
    assert {
        item.table for item in inspection.table_counts
    } == EXPECTED_TASK_FENCE_TABLES
    assert (
        next(
            item
            for item in inspection.table_counts
            if item.table == "task_fence_control"
        ).rows
        == 1
    )
    assert (
        next(
            item
            for item in inspection.table_counts
            if item.table == "task_fence_policy_decisions"
        ).rows
        == 0
    )

    conn = sqlite3.connect(path)
    try:
        main_version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    finally:
        conn.close()
    assert main_version == SCHEMA_VERSION


def test_current_task_fence_schema_open_is_idempotent(tmp_path):
    path = tmp_path / "state.db"
    first = SessionDB(path)
    first.close()
    initial_objects = _task_fence_schema_objects(path)

    second = SessionDB(path)
    try:
        inspection = second.inspect_task_fence_store()
    finally:
        second.close()

    assert inspection.compatible is True
    assert _task_fence_schema_objects(path) == initial_objects

    conn = sqlite3.connect(path)
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM task_fence_control").fetchone()[0] == 1
        )
    finally:
        conn.close()


def test_writable_open_migrates_store_without_task_fence_tables(tmp_path):
    path = tmp_path / "state.db"
    legacy = SessionDB(path)
    legacy.close()
    _drop_task_fence_schema(path)
    assert _task_fence_schema_objects(path) == ()

    migrated = SessionDB(path)
    try:
        inspection = migrated.inspect_task_fence_store()
    finally:
        migrated.close()

    assert inspection.compatible is True
    assert {
        row[1] for row in _task_fence_schema_objects(path) if row[0] == "table"
    } == EXPECTED_TASK_FENCE_TABLES


def test_exact_pristine_v1_store_migrates_atomically_to_current(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db._conn.execute("CREATE TABLE unrelated_v1_guard (value TEXT)")
    db._conn.execute("INSERT INTO unrelated_v1_guard VALUES ('preserved')")
    db.close()
    _install_task_fence_v1_schema(path)
    assert (
        _task_fence_schema_objects(path)
        == hermes_state._expected_task_fence_v1_schema_objects()
    )

    migrated = SessionDB(path)
    try:
        inspection = migrated.inspect_task_fence_store(include_counts=True)
        unrelated = migrated._conn.execute(
            "SELECT value FROM unrelated_v1_guard"
        ).fetchone()[0]
    finally:
        migrated.close()

    assert inspection.compatible is True
    assert (
        inspection.observed_store_schema_version
        == TASK_FENCE_STORE_SCHEMA_VERSION
    )
    assert unrelated == "preserved"
    assert _task_fence_schema_objects(path) == (
        hermes_state._expected_task_fence_schema_objects()
    )


def test_exact_populated_v4_store_migrates_atomically_to_current(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db._conn.execute("CREATE TABLE unrelated_v4_guard (value TEXT)")
    db._conn.execute("INSERT INTO unrelated_v4_guard VALUES ('preserved')")
    db.close()
    _install_task_fence_v4_schema(path)
    conn = sqlite3.connect(path)
    try:
        _insert_task(
            conn,
            task_id="task-v4",
            conversation_id="conversation-v4",
            store_schema_version=4,
        )
        decision_id = _insert_v4_policy_decision(conn)
        conn.execute(
            "UPDATE sqlite_sequence SET seq = 99 "
            "WHERE name = 'task_fence_policy_decisions'"
        )
        conn.commit()
    finally:
        conn.close()
    assert _task_fence_schema_objects(path) == (
        hermes_state._expected_task_fence_v4_schema_objects()
    )

    migrated = SessionDB(path)
    try:
        inspection = migrated.inspect_task_fence_store()
        task = migrated.inspect_task_fence_task("task-v4")
        decision = migrated.inspect_task_fence_policy_decision(decision_id)
        unrelated = migrated._conn.execute(
            "SELECT value FROM unrelated_v4_guard"
        ).fetchone()[0]

        assert inspection.compatible is True
        assert (
            inspection.observed_store_schema_version
            == TASK_FENCE_STORE_SCHEMA_VERSION
        )
        assert task.compatible is True
        assert task.task is not None
        assert task.task.store_schema_version == TASK_FENCE_STORE_SCHEMA_VERSION
        assert decision.compatible is True
        assert decision.decision is not None
        assert decision.decision.decision_order == 7
        assert decision.decision.operation.kind.value == "tool"
        assert unrelated == "preserved"
        assert migrated._conn.execute(
            "SELECT seq FROM sqlite_sequence "
            "WHERE name = 'task_fence_policy_decisions'"
        ).fetchone()[0] == 99
        next_decision_id = _insert_v4_policy_decision(
            migrated._conn,
            decision_order=None,
            invocation_id="tfiv-v5-next",
        )
        assert migrated._conn.execute(
            "SELECT decision_order FROM task_fence_policy_decisions "
            "WHERE decision_id = ?",
            (next_decision_id,),
        ).fetchone()[0] == 100
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            migrated._conn.execute(
                "UPDATE task_fence_policy_decisions SET adapter = 'changed' "
                "WHERE decision_id = ?",
                (decision_id,),
            )
    finally:
        migrated.close()

    assert _task_fence_schema_objects(path) == (
        hermes_state._expected_task_fence_schema_objects()
    )


def test_read_only_open_does_not_migrate_exact_v4(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v4_schema(path)
    objects_before = _task_fence_schema_objects(path)

    read_only = SessionDB(path, read_only=True)
    try:
        inspection = read_only.inspect_task_fence_store()
    finally:
        read_only.close()

    assert inspection.compatible is False
    assert inspection.observed_store_schema_version == 4
    assert _task_fence_schema_objects(path) == objects_before


def test_exact_populated_v5_store_migrates_atomically_to_current(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v5_schema(path)
    conn = sqlite3.connect(path)
    try:
        _insert_task(
            conn,
            task_id="task-v5",
            conversation_id="conversation-v5",
            store_schema_version=5,
        )
        decision_id = _insert_v4_policy_decision(
            conn,
            invocation_id="tfiv-v5-preserved",
        )
        conn.commit()
    finally:
        conn.close()
    assert _task_fence_schema_objects(path) == (
        hermes_state._expected_task_fence_v5_schema_objects()
    )

    migrated = SessionDB(path)
    try:
        inspection = migrated.inspect_task_fence_store()
        task = migrated.inspect_task_fence_task("task-v5")
        decision = migrated.inspect_task_fence_policy_decision(decision_id)

        assert inspection.compatible is True
        assert (
            inspection.observed_store_schema_version
            == TASK_FENCE_STORE_SCHEMA_VERSION
        )
        assert task.compatible is True
        assert task.task is not None
        assert task.task.store_schema_version == TASK_FENCE_STORE_SCHEMA_VERSION
        assert decision.compatible is True
        assert decision.decision is not None
        assert migrated._conn.execute(
            "SELECT COUNT(*) FROM task_fence_recovery_requeues"
        ).fetchone()[0] == 0
        assert migrated._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        migrated.close()

    assert _task_fence_schema_objects(path) == (
        hermes_state._expected_task_fence_schema_objects()
    )


def test_read_only_open_does_not_migrate_exact_v5(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v5_schema(path)
    objects_before = _task_fence_schema_objects(path)

    read_only = SessionDB(path, read_only=True)
    try:
        inspection = read_only.inspect_task_fence_store()
    finally:
        read_only.close()

    assert inspection.compatible is False
    assert inspection.observed_store_schema_version == 5
    assert _task_fence_schema_objects(path) == objects_before


@pytest.mark.parametrize(
    "control_overrides",
    (
        {"runtime_epoch": 1},
        {"ever_enforced": 1},
        {"tested_artifact_commit": "artifact"},
    ),
)
def test_nonpristine_v5_control_metadata_refuses_automatic_migration(
    tmp_path,
    control_overrides,
):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v5_schema(path, **control_overrides)
    objects_before = _task_fence_schema_objects(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.observed_store_schema_version == 5
    assert _task_fence_schema_objects(path) == objects_before


@pytest.mark.parametrize("failure_stage", ("ddl", "task_version", "control_version"))
def test_v5_migration_failure_rolls_back_exact_state(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v5_schema(path)
    conn = sqlite3.connect(path)
    try:
        _insert_task(
            conn,
            task_id="task-v5-fault",
            conversation_id="conversation-v5-fault",
            store_schema_version=5,
        )
        decision_id = _insert_v4_policy_decision(
            conn,
            invocation_id="tfiv-v5-fault",
        )
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)

    if failure_stage == "ddl":
        monkeypatch.setattr(
            hermes_state,
            "TASK_FENCE_SCHEMA_V6_EXTENSION_SQL",
            hermes_state.TASK_FENCE_SCHEMA_V6_EXTENSION_SQL
            + "CREATE TABLE task_fence_v6_partial (value INTEGER);"
            + "INVALID TASK FENCE V6;",
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
            "CREATE TEMP TRIGGER fail_task_fence_v5_migration "
            f"BEFORE UPDATE ON main.{target} BEGIN "
            "SELECT RAISE(ABORT, 'injected v5 migration failure'); END"
        )
        try:
            probe._init_task_fence_schema()
            assert probe._task_fence_schema_init_failed is True
        finally:
            probe._conn.close()

    assert _task_fence_schema_objects(path) == objects_before
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT store_schema_version FROM task_fence_control"
        ).fetchone()[0] == 5
        assert conn.execute(
            "SELECT store_schema_version FROM task_fence_tasks "
            "WHERE task_id = 'task-v5-fault'"
        ).fetchone()[0] == 5
        assert conn.execute(
            "SELECT decision_order FROM task_fence_policy_decisions "
            "WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()[0] == 7
    finally:
        conn.close()


def test_v4_delivery_permit_is_not_reclassified_by_migration(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v4_schema(path)
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        _insert_v4_dispatch_permit(conn, audience="delivery")
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        permit_before = conn.execute(
            "SELECT audience, state FROM task_fence_dispatch_permits "
            "WHERE permit_id = 'permit-v4'"
        ).fetchone()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)
    cookie_before = _schema_cookie(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.observed_store_schema_version == 4
    assert _task_fence_schema_objects(path) == objects_before
    assert _schema_cookie(path) == cookie_before
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT store_schema_version FROM task_fence_control"
        ).fetchone()[0] == 4
        assert conn.execute(
            "SELECT audience, state FROM task_fence_dispatch_permits "
            "WHERE permit_id = 'permit-v4'"
        ).fetchone() == permit_before
    finally:
        conn.close()


@pytest.mark.parametrize(
    "control_overrides",
    (
        {"runtime_epoch": 1},
        {"ever_enforced": 1},
        {"tested_artifact_commit": "artifact"},
    ),
)
def test_nonpristine_v4_control_metadata_refuses_automatic_migration(
    tmp_path,
    control_overrides,
):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v4_schema(path, **control_overrides)
    objects_before = _task_fence_schema_objects(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.observed_store_schema_version == 4
    assert _task_fence_schema_objects(path) == objects_before


@pytest.mark.parametrize("failure_stage", ("ddl", "control_version"))
def test_v4_migration_failure_rolls_back_exact_state(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v4_schema(path)
    conn = sqlite3.connect(path)
    try:
        _insert_task(
            conn,
            task_id="task-v4-fault",
            conversation_id="conversation-v4-fault",
            store_schema_version=4,
        )
        decision_id = _insert_v4_policy_decision(conn)
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)

    if failure_stage == "ddl":
        monkeypatch.setattr(
            hermes_state,
            "TASK_FENCE_SCHEMA_V5_EXTENSION_SQL",
            hermes_state.TASK_FENCE_SCHEMA_V5_EXTENSION_SQL
            + "INVALID TASK FENCE V5;",
        )
        failed = SessionDB(path)
        try:
            assert failed._task_fence_schema_init_failed is True
        finally:
            failed.close()
    else:
        probe = SessionDB.__new__(SessionDB)
        probe._conn = sqlite3.connect(path, isolation_level=None)
        probe._conn.row_factory = sqlite3.Row
        probe._conn.execute("PRAGMA foreign_keys=ON")
        probe._task_fence_schema_init_failed = False
        probe._conn.execute(
            "CREATE TEMP TRIGGER fail_task_fence_v4_control_update "
            "BEFORE UPDATE ON main.task_fence_control BEGIN "
            "SELECT RAISE(ABORT, 'injected v4 migration failure'); END"
        )
        try:
            probe._init_task_fence_schema()
            assert probe._task_fence_schema_init_failed is True
        finally:
            probe._conn.close()

    assert _task_fence_schema_objects(path) == objects_before
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT store_schema_version FROM task_fence_control"
        ).fetchone()[0] == 4
        assert conn.execute(
            "SELECT decision_order FROM task_fence_policy_decisions "
            "WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()[0] == 7
    finally:
        conn.close()


def test_partial_v4_schema_is_left_exactly_untouched(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v4_schema(path)
    conn = sqlite3.connect(path)
    try:
        decision_id = _insert_v4_policy_decision(conn)
        conn.execute("DROP INDEX idx_task_fence_policy_decisions_invocation")
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)
    cookie_before = _schema_cookie(path)

    rejected = SessionDB(path)
    try:
        inspection = rejected.inspect_task_fence_store()
    finally:
        rejected.close()

    assert inspection.compatible is False
    assert _task_fence_schema_objects(path) == objects_before
    assert _schema_cookie(path) == cookie_before
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT decision_order FROM task_fence_policy_decisions "
            "WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()[0] == 7
    finally:
        conn.close()


def test_oversized_v4_migration_is_bounded_and_untouched(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v4_schema(path)
    conn = sqlite3.connect(path)
    try:
        _insert_task(
            conn,
            task_id="task-v4-bounded",
            conversation_id="conversation-v4-bounded",
            store_schema_version=4,
        )
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)
    monkeypatch.setattr(hermes_state, "_TASK_FENCE_MAX_V4_MIGRATION_ROWS", 1)

    rejected = SessionDB(path)
    try:
        inspection = rejected.inspect_task_fence_store()
    finally:
        rejected.close()

    assert inspection.compatible is False
    assert inspection.observed_store_schema_version == 4
    assert _task_fence_schema_objects(path) == objects_before


def test_concurrent_populated_v4_migrators_publish_only_complete_current_schema(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v4_schema(path)
    conn = sqlite3.connect(path)
    try:
        decision_id = _insert_v4_policy_decision(conn)
        conn.commit()
    finally:
        conn.close()

    expected_v4 = hermes_state._expected_task_fence_v4_schema_objects()
    expected_current = hermes_state._expected_task_fence_schema_objects()
    original_read = hermes_state._read_task_fence_schema_objects
    barrier = threading.Barrier(2)
    first_reads = set()
    reads_lock = threading.Lock()

    def synchronized_read(conn):
        observed = original_read(conn)
        connection_id = id(conn)
        should_wait = False
        with reads_lock:
            if (
                observed == expected_v4
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

    def migrate():
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
    verified = SessionDB(path)
    try:
        decision = verified.inspect_task_fence_policy_decision(decision_id)
        assert decision.compatible is True
    finally:
        verified.close()


def test_nonempty_exact_v1_store_is_not_partially_backfilled(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v1_schema(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO task_fence_ingress ("
            "event_id, source, source_event_id, conversation_id, "
            "protocol_version, origin, ingress_class, intent, execution, "
            "input_effect, correlation_kind, accepted_at"
            ") VALUES ('event-v1', 'gateway', 'message-v1', 'conversation-v1', "
            "1, 'human', 'advisory', 'keep', 'none', 'none', 'none', 1.0)"
        )
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
        ingress_count = reopened._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress"
        ).fetchone()[0]
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "unsupported_store_schema"
    assert inspection.observed_store_schema_version == 1
    assert ingress_count == 1
    assert _task_fence_schema_objects(path) == objects_before


def test_orphan_v1_child_row_also_blocks_automatic_migration(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v1_schema(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO task_fence_ingress_evidence "
            "VALUES ('orphan-event', 'operator:orphan')"
        )
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
        child_count = reopened._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress_evidence"
        ).fetchone()[0]
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "unsupported_store_schema"
    assert child_count == 1
    assert _task_fence_schema_objects(path) == objects_before


@pytest.mark.parametrize(
    "control_overrides",
    (
        {"runtime_epoch": 1},
        {"mode_generation": 1},
        {"ever_enforced": 1},
        {"tested_artifact_commit": "artifact"},
    ),
)
def test_nonpristine_v1_control_metadata_refuses_automatic_migration(
    tmp_path,
    control_overrides,
):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v1_schema(path, **control_overrides)
    objects_before = _task_fence_schema_objects(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "unsupported_store_schema"
    assert inspection.observed_store_schema_version == 1
    assert _task_fence_schema_objects(path) == objects_before


def test_read_only_open_does_not_migrate_exact_v1(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v1_schema(path)
    objects_before = _task_fence_schema_objects(path)

    read_only = SessionDB(path, read_only=True)
    try:
        inspection = read_only.inspect_task_fence_store()
    finally:
        read_only.close()

    assert inspection.compatible is False
    assert inspection.reason == "unsupported_store_schema"
    assert inspection.observed_store_schema_version == 1
    assert _task_fence_schema_objects(path) == objects_before


def test_concurrent_v1_migration_rechecks_under_write_lock(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v1_schema(path)
    expected_v1 = hermes_state._expected_task_fence_v1_schema_objects()
    expected_v2 = hermes_state._expected_task_fence_schema_objects()
    original_read = hermes_state._read_task_fence_schema_objects
    barrier = threading.Barrier(2)
    reads_by_connection = {}
    reads_lock = threading.Lock()

    def synchronized_read(conn):
        observed = original_read(conn)
        connection_id = id(conn)
        with reads_lock:
            observations = reads_by_connection.setdefault(connection_id, [])
            read_count = len(observations)
            observations.append((observed, conn.in_transaction))
        if read_count == 0:
            assert observed == expected_v1
            barrier.wait(timeout=5)
        return observed

    monkeypatch.setattr(
        hermes_state,
        "_read_task_fence_schema_objects",
        synchronized_read,
    )

    def migrate():
        db = SessionDB(path)
        try:
            return db.inspect_task_fence_store().compatible
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _: migrate(), range(2)))

    monkeypatch.setattr(
        hermes_state,
        "_read_task_fence_schema_objects",
        original_read,
    )
    assert results == (True, True)
    assert len(reads_by_connection) == 2
    first_locked_reads = tuple(
        observations[1] for observations in reads_by_connection.values()
    )
    assert all(in_transaction for _observed, in_transaction in first_locked_reads)
    assert {
        observed for observed, _in_transaction in first_locked_reads
    } == {expected_v1, expected_v2}
    assert _task_fence_schema_objects(path) == (
        hermes_state._expected_task_fence_schema_objects()
    )


def test_v1_migration_ddl_failure_rolls_back_to_exact_v1(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v1_schema(path)
    objects_before = _task_fence_schema_objects(path)
    monkeypatch.setattr(
        hermes_state,
        "TASK_FENCE_SCHEMA_POST_V1_SQL",
        hermes_state.TASK_FENCE_SCHEMA_POST_V1_SQL
        + "CREATE TABLE task_fence_v3_partial (value INTEGER);"
        + "INVALID TASK FENCE V3;",
    )

    failed = SessionDB(path)
    try:
        inspection = failed.inspect_task_fence_store()
    finally:
        failed.close()

    assert inspection.compatible is False
    assert inspection.reason == "schema_initialization_failed"
    assert _task_fence_schema_objects(path) == objects_before
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT store_schema_version FROM task_fence_control"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_v1_migration_control_update_failure_rolls_back_extension(tmp_path):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    _install_task_fence_v1_schema(path)
    objects_before = _task_fence_schema_objects(path)
    probe = SessionDB.__new__(SessionDB)
    probe._conn = sqlite3.connect(path, isolation_level=None)
    probe._task_fence_schema_init_failed = False
    try:
        probe._conn.execute(
            "CREATE TEMP TRIGGER fail_task_fence_v1_update "
            "BEFORE UPDATE ON main.task_fence_control BEGIN "
            "SELECT RAISE(ABORT, 'injected migration metadata failure'); END"
        )
        probe._init_task_fence_schema()
        assert probe._task_fence_schema_init_failed is True
    finally:
        probe._conn.close()

    assert _task_fence_schema_objects(path) == objects_before
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT store_schema_version FROM task_fence_control"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_schema_statement_executor_handles_semicolon_inside_sql_literal(monkeypatch):
    monkeypatch.setattr(
        hermes_state,
        "TASK_FENCE_SCHEMA_SQL",
        "CREATE TABLE task_fence_literal_guard ("
        "value TEXT CHECK (value NOT LIKE '%;%'));",
    )
    conn = sqlite3.connect(":memory:")
    try:
        SessionDB._execute_task_fence_schema_sql(conn.cursor())
        conn.execute("INSERT INTO task_fence_literal_guard VALUES ('safe')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO task_fence_literal_guard VALUES ('unsafe;value')")
    finally:
        conn.close()


def test_initializer_rejects_non_autocommit_connection_contract(tmp_path):
    path = tmp_path / "state.db"
    probe = SessionDB.__new__(SessionDB)
    probe._conn = sqlite3.connect(path)
    probe._task_fence_schema_init_failed = False
    try:
        assert probe._conn.isolation_level is not None

        probe._init_task_fence_schema()

        assert probe._task_fence_schema_init_failed is True
        assert not probe._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name GLOB 'task_fence_*' LIMIT 1"
        ).fetchall()
    finally:
        probe._conn.close()


def test_initializer_refuses_to_commit_an_unrelated_transaction(tmp_path):
    path = tmp_path / "state.db"
    probe = SessionDB.__new__(SessionDB)
    probe._conn = sqlite3.connect(path, isolation_level=None)
    probe._task_fence_schema_init_failed = False
    try:
        probe._conn.execute("BEGIN IMMEDIATE")
        probe._conn.execute("CREATE TABLE unrelated_uncommitted (value INTEGER)")

        probe._init_task_fence_schema()

        assert probe._task_fence_schema_init_failed is True
        assert probe._conn.in_transaction is True
        assert not probe._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name GLOB 'task_fence_*' LIMIT 1"
        ).fetchall()
        probe._conn.rollback()
    finally:
        probe._conn.close()

    conn = sqlite3.connect(path)
    try:
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'unrelated_uncommitted'"
        ).fetchall()
    finally:
        conn.close()


def test_concurrent_first_open_rechecks_absence_under_write_lock(tmp_path):
    path = tmp_path / "state.db"
    barrier = threading.Barrier(2)

    class BufferedRows:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class FirstNamespaceReadBarrier:
        def __init__(self, conn):
            self._conn = conn
            self._first_namespace_read = True
            self.barrier_engaged = False

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def execute(self, sql, parameters=()):
            if self._first_namespace_read and sql.startswith(
                "SELECT type, name, tbl_name, sql FROM main.sqlite_master"
            ):
                self._first_namespace_read = False
                self.barrier_engaged = True
                rows = self._conn.execute(sql, parameters).fetchall()
                barrier.wait(timeout=5)
                return BufferedRows(rows)
            return self._conn.execute(sql, parameters)

    def initialize():
        probe = SessionDB.__new__(SessionDB)
        raw = sqlite3.connect(path, timeout=5, isolation_level=None)
        wrapped = FirstNamespaceReadBarrier(raw)
        probe._conn = wrapped
        probe._task_fence_schema_init_failed = False
        try:
            probe._init_task_fence_schema()
            return probe._task_fence_schema_init_failed, wrapped.barrier_engaged
        finally:
            raw.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _: initialize(), range(2)))

    assert results == ((False, True), (False, True))
    db = SessionDB(path)
    try:
        assert db.inspect_task_fence_store().compatible is True
    finally:
        db.close()


def test_mid_ddl_failure_rolls_back_and_reports_shadow_failure(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    monkeypatch.setattr(
        hermes_state,
        "TASK_FENCE_SCHEMA_SQL",
        "CREATE TABLE task_fence_partial (value INTEGER); INVALID TASK FENCE;",
    )

    db = SessionDB(path)
    try:
        inspection = db.inspect_task_fence_store()
        assert inspection.compatible is False
        assert inspection.reason == "schema_initialization_failed"
        assert _task_fence_schema_objects(path) == ()
    finally:
        db.close()


def test_recorded_initialization_failure_overrides_exact_layout(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        assert db.inspect_task_fence_store().compatible is True

        db._task_fence_schema_init_failed = True
        inspection = db.inspect_task_fence_store()

        assert inspection.initialized is True
        assert inspection.compatible is False
        assert inspection.reason == "schema_initialization_failed"
    finally:
        db.close()


def test_read_only_inspection_does_not_initialize_legacy_store(tmp_path):
    path = tmp_path / "state.db"
    legacy = SessionDB(path)
    legacy.close()
    _drop_task_fence_schema(path)

    conn = sqlite3.connect(path)
    try:
        schema_cookie_before = conn.execute("PRAGMA schema_version").fetchone()[0]
        objects_before = tuple(
            conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "ORDER BY type, name"
            ).fetchall()
        )
    finally:
        conn.close()

    read_only = SessionDB(path, read_only=True)
    try:
        inspection = read_only.inspect_task_fence_store(include_counts=True)
    finally:
        read_only.close()

    assert inspection.initialized is False
    assert inspection.compatible is False
    assert inspection.reason == "not_initialized"

    conn = sqlite3.connect(path)
    try:
        schema_cookie_after = conn.execute("PRAGMA schema_version").fetchone()[0]
        objects_after = tuple(
            conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "ORDER BY type, name"
            ).fetchall()
        )
    finally:
        conn.close()
    assert schema_cookie_after == schema_cookie_before
    assert objects_after == objects_before


def test_temp_objects_cannot_shadow_read_only_inspection(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        _insert_task(db._conn, task_id="task-1", conversation_id="main-conversation")
    finally:
        db.close()

    read_only = SessionDB(path, read_only=True)
    try:
        read_only._conn.execute(
            "CREATE TEMP TABLE task_fence_control AS "
            "SELECT * FROM main.task_fence_control"
        )
        read_only._conn.execute(
            "UPDATE temp.task_fence_control SET store_schema_version = 999"
        )
        read_only._conn.execute(
            "CREATE TEMP TABLE task_fence_tasks AS SELECT * FROM main.task_fence_tasks"
        )
        read_only._conn.execute(
            "UPDATE temp.task_fence_tasks SET conversation_id = 'temp-conversation'"
        )
        read_only._conn.execute(
            "INSERT INTO temp.task_fence_tasks SELECT * FROM temp.task_fence_tasks"
        )

        store = read_only.inspect_task_fence_store(include_counts=True)
        task = read_only.inspect_task_fence_task("task-1")
    finally:
        read_only.close()

    task_count = next(
        item for item in store.table_counts if item.table == "task_fence_tasks"
    )
    assert store.compatible is True
    assert task.compatible is True
    assert task.task is not None
    assert task.task.conversation_id == "main-conversation"
    assert task_count.rows == 1


def test_temp_control_table_cannot_capture_initializer_metadata(tmp_path):
    path = tmp_path / "state.db"
    probe = SessionDB.__new__(SessionDB)
    probe._conn = sqlite3.connect(path, isolation_level=None)
    probe._task_fence_schema_init_failed = False
    try:
        probe._conn.execute("CREATE TEMP TABLE task_fence_control (marker TEXT)")
        probe._conn.execute("INSERT INTO temp.task_fence_control VALUES ('temp')")
        probe._conn.execute("CREATE TEMP TABLE task_fence_tasks (marker TEXT)")
        probe._conn.execute("INSERT INTO temp.task_fence_tasks VALUES ('temp-task')")

        probe._init_task_fence_schema()

        durable = probe._conn.execute(
            "SELECT store_schema_version, control_protocol_version "
            "FROM main.task_fence_control"
        ).fetchall()
        temporary = probe._conn.execute(
            "SELECT marker FROM temp.task_fence_control"
        ).fetchall()
        temporary_tasks = probe._conn.execute(
            "SELECT marker FROM temp.task_fence_tasks"
        ).fetchall()
        durable_tasks = probe._conn.execute(
            "SELECT COUNT(*) FROM main.task_fence_tasks"
        ).fetchone()[0]
    finally:
        probe._conn.close()

    assert probe._task_fence_schema_init_failed is False
    assert durable == [
        (TASK_FENCE_STORE_SCHEMA_VERSION, TASK_FENCE_CONTROL_PROTOCOL_VERSION)
    ]
    assert temporary == [("temp",)]
    assert temporary_tasks == [("temp-task",)]
    assert durable_tasks == 0


def test_orphan_schema_object_is_malformed_not_clean_absence(tmp_path):
    path = tmp_path / "state.db"
    legacy = SessionDB(path)
    legacy.close()
    _drop_task_fence_schema(path)

    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE VIEW task_fence_orphan_view AS SELECT 1 AS value")
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)
    schema_cookie_before = _schema_cookie(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.initialized is True
    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    assert _task_fence_schema_objects(path) == objects_before
    assert _schema_cookie(path) == schema_cookie_before


def test_virtual_control_table_is_never_read_during_incompatible_inspection(tmp_path):
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE VIRTUAL TABLE task_fence_control USING fts5(content)")
        conn.commit()
    finally:
        conn.close()

    db = SessionDB(path)
    traced = []
    try:
        db._conn.set_trace_callback(traced.append)
        inspection = db.inspect_task_fence_store(include_counts=True)
    finally:
        db.close()

    statements = "\n".join(traced).lower()
    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    assert "from main.task_fence_control" not in statements
    assert "pragma table_info" not in statements
    assert "select count(*)" not in statements


def test_incompatible_virtual_table_is_not_counted(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE VIRTUAL TABLE task_fence_untrusted USING fts5(content)")
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(path)
    traced = []
    try:
        reopened._conn.set_trace_callback(traced.append)
        inspection = reopened.inspect_task_fence_store(include_counts=True)
    finally:
        reopened.close()

    statements = "\n".join(traced).lower()
    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    assert 'from "task_fence_untrusted"' not in statements
    assert "select count(*)" not in statements
    assert all(item.rows is None for item in inspection.table_counts)


def test_inspection_uses_one_snapshot_across_schema_and_row_reads(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    schema_read = threading.Event()
    replacement_done = threading.Event()
    writer_errors = []
    original_expected = hermes_state._expected_task_fence_schema_objects

    def pause_after_schema_read():
        expected = original_expected()
        schema_read.set()
        if not replacement_done.wait(timeout=5):
            raise RuntimeError("concurrent schema replacement timed out")
        return expected

    def replace_control_with_virtual_table():
        try:
            if not schema_read.wait(timeout=5):
                raise RuntimeError("inspection did not read the schema")
            conn = sqlite3.connect(path, timeout=5, isolation_level=None)
            try:
                conn.execute("PRAGMA foreign_keys=OFF")
                conn.execute("DROP TABLE task_fence_control")
                conn.execute(
                    "CREATE VIRTUAL TABLE task_fence_control USING fts5("
                    "singleton UNINDEXED, store_schema_version UNINDEXED, "
                    "control_protocol_version UNINDEXED, runtime_epoch UNINDEXED, "
                    "mode_generation UNINDEXED, ever_enforced UNINDEXED, "
                    "tested_artifact_commit UNINDEXED, "
                    "tested_artifact_checksum UNINDEXED, "
                    "dependency_lock_fingerprint UNINDEXED)"
                )
            finally:
                conn.close()
        except Exception as exc:
            writer_errors.append(exc)
        finally:
            replacement_done.set()

    monkeypatch.setattr(
        hermes_state,
        "_expected_task_fence_schema_objects",
        pause_after_schema_read,
    )
    writer = threading.Thread(target=replace_control_with_virtual_table)
    writer.start()
    try:
        before_replacement = db.inspect_task_fence_store(include_counts=True)
        writer.join(timeout=5)
        after_replacement = db.inspect_task_fence_store(include_counts=True)
    finally:
        db.close()

    assert writer.is_alive() is False
    assert writer_errors == []
    assert before_replacement.compatible is True
    assert after_replacement.compatible is False
    assert after_replacement.reason == "malformed_schema"


def test_inspection_does_not_end_a_borrowed_transaction(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        db._conn.execute("BEGIN")
        inspection = db.inspect_task_fence_store()
        assert inspection.compatible is True
        assert db._conn.in_transaction is True
        db._conn.rollback()
    finally:
        db.close()


def test_inspection_failure_reason_is_stable_and_secret_free(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    store = db.inspect_task_fence_store()
    task = db.inspect_task_fence_task("task-1")

    assert store.compatible is False
    assert store.reason == "inspection_closed"
    assert task.compatible is False
    assert task.reason == "inspection_closed"
    assert task.store.reason == "inspection_closed"


def test_future_store_schema_is_reported_and_not_rewritten(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE task_fence_control SET store_schema_version = ?, "
            "runtime_epoch = 7 WHERE singleton = 1",
            (TASK_FENCE_STORE_SCHEMA_VERSION + 1,),
        )
        conn.execute("CREATE TABLE task_fence_future_record (id TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    future_objects = _task_fence_schema_objects(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.initialized is True
    assert inspection.compatible is False
    assert inspection.reason == "unsupported_store_schema"
    assert (
        inspection.observed_store_schema_version
        == TASK_FENCE_STORE_SCHEMA_VERSION + 1
    )
    assert inspection.runtime_epoch == 7
    assert _task_fence_schema_objects(path) == future_objects


def test_unknown_protocol_is_reported_and_not_downgraded(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE task_fence_control SET control_protocol_version = 2 "
            "WHERE singleton = 1"
        )
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "unsupported_control_protocol"
    assert inspection.observed_control_protocol_version == 2

    conn = sqlite3.connect(path)
    try:
        observed = conn.execute(
            "SELECT control_protocol_version FROM task_fence_control "
            "WHERE singleton = 1"
        ).fetchone()[0]
    finally:
        conn.close()
    assert observed == 2


@pytest.mark.parametrize(
    "field",
    (
        "tested_artifact_commit",
        "tested_artifact_checksum",
        "dependency_lock_fingerprint",
    ),
)
def test_malformed_optional_control_text_is_not_reported_compatible(tmp_path, field):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            f'UPDATE task_fence_control SET "{field}" = ? WHERE singleton = 1',
            (sqlite3.Binary(b"not-text"),),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "malformed_control_metadata"


@pytest.mark.parametrize(
    "field",
    (
        "store_schema_version",
        "control_protocol_version",
        "runtime_epoch",
        "mode_generation",
        "ever_enforced",
    ),
)
@pytest.mark.parametrize(
    "value",
    (pytest.param(1.5, id="real"), pytest.param(sqlite3.Binary(b"1"), id="blob")),
)
def test_noninteger_control_metadata_is_not_coerced_into_compatibility(
    tmp_path, field, value
):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA ignore_check_constraints=ON")
        conn.execute(
            f'UPDATE main.task_fence_control SET "{field}" = ? WHERE singleton = 1',
            (value,),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "malformed_control_metadata"


def test_malformed_current_schema_is_not_reconciled_into_compatibility(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("ALTER TABLE task_fence_tasks ADD COLUMN unexpected TEXT")
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    conn = sqlite3.connect(path)
    try:
        columns = {
            row[1] for row in conn.execute('PRAGMA table_info("task_fence_tasks")')
        }
    finally:
        conn.close()
    assert "unexpected" in columns


def test_partial_current_schema_is_not_auto_healed(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DROP TABLE task_fence_questions")
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    assert _task_fence_schema_objects(path) == objects_before
    assert not any(row[1] == "task_fence_questions" for row in objects_before)


def test_missing_current_index_is_not_auto_healed(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP INDEX idx_task_fence_runs_one_open")
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)
    schema_cookie_before = _schema_cookie(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    assert _task_fence_schema_objects(path) == objects_before
    assert _schema_cookie(path) == schema_cookie_before
    assert not any(row[1] == "idx_task_fence_runs_one_open" for row in objects_before)


def test_malformed_control_table_does_not_break_legacy_startup(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DROP TABLE task_fence_control")
        conn.execute(
            "CREATE TABLE task_fence_control ("
            "store_schema_version INTEGER, control_protocol_version INTEGER)"
        )
        conn.execute("INSERT INTO task_fence_control VALUES (1, 1)")
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.initialized is True
    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"


def test_unexpected_trigger_cannot_hide_inside_compatible_schema(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TRIGGER rogue_control_mutator "
            "AFTER INSERT ON task_fence_tasks BEGIN "
            "UPDATE task_fence_control SET mode_generation = mode_generation + 1 "
            "WHERE singleton = 1; END"
        )
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    assert _task_fence_schema_objects(path) == objects_before


def test_external_trigger_referencing_task_fence_is_incompatible(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TRIGGER rogue_external AFTER INSERT ON sessions BEGIN "
            "UPDATE task_fence_control SET runtime_epoch = runtime_epoch + 1 "
            "WHERE singleton = 1; END"
        )
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.initialized is True
    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    assert any(item[1] == "rogue_external" for item in _task_fence_schema_objects(path))


def test_external_reference_prevents_absent_namespace_initialization(tmp_path):
    path = tmp_path / "state.db"
    legacy = SessionDB(path)
    legacy.close()
    _drop_task_fence_schema(path)

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE VIEW legacy_projection AS SELECT task_id FROM task_fence_tasks"
        )
        conn.commit()
    finally:
        conn.close()
    schema_cookie_before = _schema_cookie(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    conn = sqlite3.connect(path)
    try:
        durable_control = conn.execute(
            "SELECT 1 FROM main.sqlite_master "
            "WHERE type = 'table' AND name = 'task_fence_control'"
        ).fetchall()
    finally:
        conn.close()

    assert inspection.initialized is True
    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    assert durable_control == []
    assert _schema_cookie(path) == schema_cookie_before
    assert any(
        item[1] == "legacy_projection" for item in _task_fence_schema_objects(path)
    )


@pytest.mark.parametrize(
    "statement",
    (
        "CREATE VIEW task_fence_unexpected_view AS "
        "SELECT task_id FROM task_fence_tasks",
        "CREATE TRIGGER task_fence_unexpected_trigger "
        "AFTER INSERT ON task_fence_tasks BEGIN SELECT 1; END",
    ),
    ids=("view", "trigger"),
)
def test_unexpected_prefixed_schema_object_is_not_hidden_or_rewritten(
    tmp_path, statement
):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute(statement)
        conn.commit()
    finally:
        conn.close()
    objects_before = _task_fence_schema_objects(path)
    schema_cookie_before = _schema_cookie(path)

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    assert _task_fence_schema_objects(path) == objects_before
    assert _schema_cookie(path) == schema_cookie_before


def test_task_projection_is_scalar_immutable_and_read_only(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        _insert_task(
            conn,
            task_id="task-1",
            conversation_id="conversation-1",
            status="waiting_user",
            intent_epoch=3,
            control_revision=5,
            runtime_epoch=7,
        )
        conn.commit()
    finally:
        conn.close()

    before = _task_fence_schema_objects(path)
    read_only = SessionDB(path, read_only=True)
    try:
        inspection = read_only.inspect_task_fence_task("task-1")
        missing = read_only.inspect_task_fence_task("missing")
    finally:
        read_only.close()

    assert inspection.compatible is True
    assert inspection.reason == "compatible"
    assert inspection.task is not None
    assert inspection.task.task_id == "task-1"
    assert inspection.task.intent_epoch == 3
    assert inspection.task.control_revision == 5
    assert inspection.task.current_runtime_epoch == 7
    assert inspection.task.last_accepted_order == 0
    assert inspection.task.status == "waiting_user"
    with pytest.raises(FrozenInstanceError):
        setattr(inspection.task, "status", "running")
    assert missing.compatible is True
    assert missing.reason == "not_found"
    assert missing.task is None
    assert _task_fence_schema_objects(path) == before


def test_conversation_inspection_is_exact_secret_free_and_active_only(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    selected_conversation = "workspace:secret-selected-conversation"
    other_conversation = "workspace:secret-other-conversation"
    terminal_conversation = "workspace:secret-terminal-conversation"
    implicit_conversation = "workspace:secret-implicit-conversation"
    reserved_conversation = "workspace:secret-reserved-conversation"
    conn = sqlite3.connect(path)
    try:
        _insert_cohort(
            conn,
            cohort_key="cohort-selected",
            audit_degraded=1,
        )
        _insert_cohort(conn, cohort_key="cohort-other")
        _insert_task(
            conn,
            task_id="task-terminal-old",
            conversation_id=selected_conversation,
            status="done",
        )
        _insert_task(
            conn,
            task_id="task-selected",
            conversation_id=selected_conversation,
            status="paused",
            intent_epoch=3,
            control_revision=5,
        )
        conn.execute(
            "UPDATE task_fence_tasks SET cohort_key = 'cohort-selected' "
            "WHERE task_id = 'task-selected'"
        )
        _insert_task(
            conn,
            task_id="task-other",
            conversation_id=other_conversation,
            status="waiting_user",
        )
        conn.execute(
            "UPDATE task_fence_tasks SET cohort_key = 'cohort-other' "
            "WHERE task_id = 'task-other'"
        )
        _insert_task(
            conn,
            task_id="task-terminal-only",
            conversation_id=terminal_conversation,
            status="stopped",
        )
        _insert_task(
            conn,
            task_id="task-implicit",
            conversation_id=implicit_conversation,
            status="paused",
        )
        _insert_task(
            conn,
            task_id="task-explicit-reserved",
            conversation_id=reserved_conversation,
            status="paused",
        )
        conn.execute(
            "UPDATE task_fence_tasks SET cohort_key = ? "
            "WHERE task_id = 'task-explicit-reserved'",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        )
        conn.commit()
    finally:
        conn.close()

    read_only = SessionDB(path, read_only=True)
    try:
        selected = read_only.inspect_task_fence_conversation(
            selected_conversation
        )
        selected_replay = read_only.inspect_task_fence_conversation(
            selected_conversation
        )
        other = read_only.inspect_task_fence_conversation(other_conversation)
        terminal_only = read_only.inspect_task_fence_conversation(
            terminal_conversation
        )
        implicit = read_only.inspect_task_fence_conversation(
            implicit_conversation
        )
        explicit_reserved = read_only.inspect_task_fence_conversation(
            reserved_conversation
        )
        invalid = read_only.inspect_task_fence_conversation("")
        assert read_only._conn.total_changes == 0
    finally:
        read_only.close()

    assert selected.compatible is True
    assert selected.reason == "compatible"
    assert selected.task is not None
    assert selected.task.task_id == "task-selected"
    assert selected.task.status == "paused"
    assert selected.task.intent_epoch == 3
    assert selected.task.control_revision == 5
    assert selected.conversation_fingerprint is not None
    assert selected.conversation_fingerprint == hashlib.sha256(
        b"hermes.task_fence.conversation.v1\0"
        + selected_conversation.encode("utf-8")
    ).hexdigest()
    assert (
        selected_replay.conversation_fingerprint
        == selected.conversation_fingerprint
    )
    assert other.conversation_fingerprint != selected.conversation_fingerprint
    assert selected.cohort is not None
    assert selected.cohort.binding == "explicit"
    assert selected.cohort.cohort_fingerprint == hashlib.sha256(
        b"hermes.task_fence.cohort.v1\0cohort-selected"
    ).hexdigest()
    assert selected.cohort.mode == "audit"
    assert selected.cohort.activation_state == "inactive"
    assert selected.cohort.audit_degraded is True
    rendered = repr(selected)
    assert selected_conversation not in rendered
    assert other_conversation not in rendered
    assert terminal_conversation not in rendered
    assert implicit_conversation not in rendered
    assert reserved_conversation not in rendered
    assert "cohort-selected" not in rendered
    with pytest.raises(FrozenInstanceError):
        setattr(selected.task, "status", "running")

    assert terminal_only.compatible is True
    assert terminal_only.reason == "no_active_task"
    assert terminal_only.task is None
    assert terminal_only.cohort is None
    assert implicit.compatible is True
    assert implicit.reason == "cohort_not_materialized"
    assert implicit.task is not None
    assert implicit.task.task_id == "task-implicit"
    assert implicit.cohort is None
    assert explicit_reserved.compatible is False
    assert explicit_reserved.reason == "cohort_not_found"
    assert explicit_reserved.task is None
    assert explicit_reserved.cohort is None
    assert invalid.compatible is False
    assert invalid.reason == "invalid_conversation_id"
    assert invalid.conversation_fingerprint is None

    conn = sqlite3.connect(path)
    try:
        implicit_rows = conn.execute(
            "SELECT COUNT(*) FROM task_fence_cohorts "
            "WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert implicit_rows == 0


def test_conversation_inspection_rejects_non_audit_implicit_cohort(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        _insert_cohort(
            conn,
            cohort_key=hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,
        )
        _insert_task(
            conn,
            task_id="task-implicit",
            conversation_id="conversation-implicit",
            status="paused",
        )
        conn.commit()
        conn.execute(
            "UPDATE task_fence_cohorts "
            "SET mode = 'enforce', activation_state = 'active' "
            "WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        )
        conn.commit()
    finally:
        conn.close()

    read_only = SessionDB(path, read_only=True)
    try:
        enforced = read_only.inspect_task_fence_conversation(
            "conversation-implicit"
        )
    finally:
        read_only.close()
    assert enforced.compatible is False
    assert enforced.reason == "implicit_cohort_mismatch"
    assert enforced.task is None
    assert enforced.cohort is None

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE task_fence_cohorts "
            "SET mode = 'audit', activation_state = 'halted' "
            "WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        )
        conn.commit()
    finally:
        conn.close()

    read_only = SessionDB(path, read_only=True)
    try:
        halted = read_only.inspect_task_fence_conversation(
            "conversation-implicit"
        )
    finally:
        read_only.close()
    assert halted.compatible is False
    assert halted.reason == "implicit_cohort_mismatch"
    assert halted.task is None
    assert halted.cohort is None


def test_conversation_inspection_uses_one_store_task_cohort_snapshot(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        _insert_cohort(conn, cohort_key="cohort-snapshot")
        _insert_task(
            conn,
            task_id="task-snapshot",
            conversation_id="conversation-snapshot",
            status="paused",
            control_revision=1,
        )
        conn.execute(
            "UPDATE task_fence_tasks SET cohort_key = 'cohort-snapshot' "
            "WHERE task_id = 'task-snapshot'"
        )
        conn.commit()
    finally:
        conn.close()

    reader = SessionDB(path, read_only=True)
    store_read = threading.Event()
    writer_done = threading.Event()
    writer_errors = []
    original_inspect = reader._inspect_task_fence_store_unlocked

    def pause_after_store(*, include_counts):
        inspected = original_inspect(include_counts=include_counts)
        store_read.set()
        if not writer_done.wait(timeout=5):
            raise RuntimeError("concurrent projection update timed out")
        return inspected

    def update_projection():
        try:
            if not store_read.wait(timeout=5):
                raise RuntimeError("inspection did not read the store")
            writer = sqlite3.connect(path, timeout=5, isolation_level=None)
            try:
                writer.execute("BEGIN IMMEDIATE")
                writer.execute(
                    "UPDATE task_fence_tasks SET control_revision = 2 "
                    "WHERE task_id = 'task-snapshot'"
                )
                writer.execute(
                    "UPDATE task_fence_cohorts SET audit_degraded = 1 "
                    "WHERE cohort_key = 'cohort-snapshot'"
                )
                writer.commit()
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
    writer_thread = threading.Thread(target=update_projection)
    writer_thread.start()
    try:
        before = reader.inspect_task_fence_conversation(
            "conversation-snapshot"
        )
        writer_thread.join(timeout=5)
        after = reader.inspect_task_fence_conversation(
            "conversation-snapshot"
        )
    finally:
        reader.close()

    assert writer_thread.is_alive() is False
    assert writer_errors == []
    assert before.compatible is True
    assert before.task is not None
    assert before.cohort is not None
    assert before.task.control_revision == 1
    assert before.cohort.audit_degraded is False
    assert after.compatible is True
    assert after.task is not None
    assert after.cohort is not None
    assert after.task.control_revision == 2
    assert after.cohort.audit_degraded is True


@pytest.mark.parametrize(
    "field",
    (
        "cohort_key",
        "active_authority_event_id",
        "active_execution_run_id",
        "current_generation_id",
        "last_transition_event_id",
    ),
)
def test_malformed_optional_task_identifier_is_not_returned_as_typed(tmp_path, field):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        _insert_task(conn, task_id="task-1", conversation_id="conversation-1")
        conn.execute(
            f'UPDATE task_fence_tasks SET "{field}" = ? WHERE task_id = ?',
            (sqlite3.Binary(b"not-text"), "task-1"),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_task("task-1")
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "malformed_task_projection"
    assert inspection.task is None


@pytest.mark.parametrize(
    "field",
    (
        "store_schema_version",
        "control_protocol_version",
        "intent_epoch",
        "control_revision",
        "current_runtime_epoch",
        "last_accepted_order",
    ),
)
@pytest.mark.parametrize(
    "value",
    (pytest.param(1.5, id="real"), pytest.param(sqlite3.Binary(b"1"), id="blob")),
)
def test_noninteger_task_state_is_not_coerced_into_typed_projection(
    tmp_path, field, value
):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("PRAGMA ignore_check_constraints=ON")
        _insert_task(conn, task_id="task-1", conversation_id="conversation-1")
        conn.execute(
            f'UPDATE main.task_fence_tasks SET "{field}" = ? WHERE task_id = ?',
            (value, "task-1"),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_task("task-1")
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "malformed_task_projection"
    assert inspection.task is None


def test_only_one_nonterminal_task_can_own_a_conversation_lane(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        _insert_task(conn, task_id="task-1", conversation_id="conversation-1")
        with pytest.raises(sqlite3.IntegrityError):
            _insert_task(conn, task_id="task-2", conversation_id="conversation-1")
        conn.rollback()

        _insert_task(
            conn,
            task_id="task-terminal",
            conversation_id="conversation-2",
            status="done",
        )
        _insert_task(
            conn,
            task_id="task-active",
            conversation_id="conversation-2",
            status="planning",
        )
        conn.commit()
    finally:
        conn.close()


def test_authority_journals_reject_update_and_delete(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO task_fence_ingress ("
            "event_id, source, source_event_id, conversation_id, "
            "protocol_version, origin, ingress_class, intent, execution, "
            "input_effect, correlation_kind, accepted_at"
            ") VALUES ("
            "'event-1', 'gateway', 'source-1', 'conversation-1', "
            "1, 'human', 'advisory', 'keep', 'none', 'none', 'none', 1.0)"
        )
        conn.execute(
            "INSERT INTO task_fence_attempt_transitions ("
            "attempt_id, from_state, to_state, transitioned_at"
            ") VALUES ('attempt-1', NULL, 'PREPARED', 1.0)"
        )
        conn.execute(
            "INSERT INTO task_fence_resolutions ("
            "resolution_id, incident_id, resolution_event_id, disposition, resolved_at"
            ") VALUES ("
            "'resolution-1', 'incident-1', 'event-1', "
            "'accepted_unknown_no_retry', 1.0)"
        )
        decision_id = f"tfd_{'a' * 64}"
        conn.execute(
            "INSERT INTO task_fence_policy_decisions ("
            "decision_id, decision_point, outcome, reason_code, policy_version, "
            "mode_generation, operation_invocation_id, operation_kind, adapter, "
            "invocation_fingerprint, decided_at"
            ") VALUES (?, 'admission', 'would_block', 'missing_provenance', "
            "'task-fence-policy-v1', 0, 'invocation-1', 'tool', 'adapter-1', ?, 1.0)",
            (decision_id, "b" * 64),
        )
        conn.commit()

        statements = (
            "UPDATE task_fence_ingress SET source = 'other' WHERE event_id = 'event-1'",
            "DELETE FROM task_fence_ingress WHERE event_id = 'event-1'",
            "UPDATE task_fence_attempt_transitions SET to_state = 'STARTED' "
            "WHERE attempt_id = 'attempt-1'",
            "DELETE FROM task_fence_attempt_transitions WHERE attempt_id = 'attempt-1'",
            "UPDATE task_fence_resolutions SET disposition = 'confirmed_failure' "
            "WHERE resolution_id = 'resolution-1'",
            "DELETE FROM task_fence_resolutions WHERE resolution_id = 'resolution-1'",
            "UPDATE task_fence_policy_decisions SET reason_code = "
            f"'stale_authority' WHERE decision_id = '{decision_id}'",
            "DELETE FROM task_fence_policy_decisions "
            f"WHERE decision_id = '{decision_id}'",
        )
        for statement in statements:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement)
    finally:
        conn.close()


def test_authority_journals_reject_all_replace_conflicts(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        assert conn.execute("PRAGMA recursive_triggers").fetchone()[0] == 0
        conn.execute(
            "INSERT INTO task_fence_ingress ("
            "accepted_order, event_id, source, source_event_id, conversation_id, "
            "protocol_version, origin, ingress_class, intent, execution, "
            "input_effect, correlation_kind, accepted_at"
            ") VALUES ("
            "7, 'event-1', 'gateway', 'source-1', 'conversation-1', "
            "1, 'human', 'advisory', 'keep', 'none', 'none', 'none', 1.0)"
        )
        conn.execute(
            "INSERT INTO task_fence_attempt_transitions ("
            "transition_order, attempt_id, from_state, to_state, transitioned_at"
            ") VALUES (11, 'attempt-1', NULL, 'PREPARED', 1.0)"
        )
        conn.execute(
            "INSERT INTO task_fence_resolutions ("
            "resolution_id, incident_id, resolution_event_id, disposition, resolved_at"
            ") VALUES ("
            "'resolution-1', 'incident-1', 'event-1', "
            "'accepted_unknown_no_retry', 1.0)"
        )
        decision_id = f"tfd_{'a' * 64}"
        conn.execute(
            "INSERT INTO task_fence_policy_decisions ("
            "decision_order, decision_id, decision_point, outcome, reason_code, "
            "policy_version, mode_generation, operation_invocation_id, "
            "operation_kind, adapter, "
            "invocation_fingerprint, decided_at"
            ") VALUES (13, ?, 'admission', 'would_block', 'missing_provenance', "
            "'task-fence-policy-v1', 0, 'invocation-1', 'tool', 'adapter-1', ?, 1.0)",
            (decision_id, "b" * 64),
        )
        conn.commit()

        for verb in ("INSERT OR REPLACE", "REPLACE"):
            statements = (
                f"{verb} INTO task_fence_ingress ("
                "accepted_order, event_id, source, source_event_id, conversation_id, "
                "protocol_version, origin, ingress_class, intent, execution, "
                "input_effect, correlation_kind, accepted_at) VALUES ("
                "7, 'event-order', 'gateway-order', 'source-order', "
                "'conversation-1', 1, 'human', 'advisory', 'keep', 'none', "
                "'none', 'none', 2.0)",
                f"{verb} INTO task_fence_ingress ("
                "accepted_order, event_id, source, source_event_id, conversation_id, "
                "protocol_version, origin, ingress_class, intent, execution, "
                "input_effect, correlation_kind, accepted_at) VALUES ("
                "8, 'event-1', 'gateway-event', 'source-event', "
                "'conversation-1', 1, 'human', 'advisory', 'keep', 'none', "
                "'none', 'none', 2.0)",
                f"{verb} INTO task_fence_ingress ("
                "accepted_order, event_id, source, source_event_id, conversation_id, "
                "protocol_version, origin, ingress_class, intent, execution, "
                "input_effect, correlation_kind, accepted_at) VALUES ("
                "9, 'event-source', 'gateway', 'source-1', 'conversation-1', "
                "1, 'human', 'advisory', 'keep', 'none', 'none', 'none', 2.0)",
                f"{verb} INTO task_fence_attempt_transitions ("
                "transition_order, attempt_id, from_state, to_state, transitioned_at"
                ") VALUES (11, 'attempt-replacement', 'PREPARED', 'STARTED', 2.0)",
                f"{verb} INTO task_fence_resolutions ("
                "resolution_id, incident_id, resolution_event_id, disposition, resolved_at"
                ") VALUES ('resolution-1', 'incident-resolution', 'event-1', "
                "'confirmed_failure', 2.0)",
                f"{verb} INTO task_fence_resolutions ("
                "resolution_id, incident_id, resolution_event_id, disposition, resolved_at"
                ") VALUES ('resolution-incident', 'incident-1', 'event-1', "
                "'confirmed_failure', 2.0)",
                f"{verb} INTO task_fence_policy_decisions ("
                "decision_order, decision_id, decision_point, outcome, reason_code, "
                "policy_version, mode_generation, operation_invocation_id, "
                "operation_kind, adapter, "
                "invocation_fingerprint, decided_at) VALUES ("
                f"13, 'tfd_{'c' * 64}', 'admission', 'would_block', "
                "'missing_provenance', 'task-fence-policy-v1', 0, "
                "'invocation-order', 'tool', 'adapter-1', "
                f"'{'d' * 64}', 2.0)",
                f"{verb} INTO task_fence_policy_decisions ("
                "decision_order, decision_id, decision_point, outcome, reason_code, "
                "policy_version, mode_generation, operation_invocation_id, "
                "operation_kind, adapter, "
                "invocation_fingerprint, decided_at) VALUES ("
                f"14, '{decision_id}', 'admission', 'would_block', "
                "'missing_provenance', 'task-fence-policy-v1', 0, "
                "'invocation-id', 'tool', 'adapter-1', "
                f"'{'e' * 64}', 2.0)",
            )
            for statement in statements:
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    conn.execute(statement)

        ingress = conn.execute(
            "SELECT accepted_order, event_id, source, source_event_id "
            "FROM task_fence_ingress"
        ).fetchall()
        transitions = conn.execute(
            "SELECT transition_order, attempt_id, to_state "
            "FROM task_fence_attempt_transitions"
        ).fetchall()
        resolutions = conn.execute(
            "SELECT resolution_id, incident_id, disposition FROM task_fence_resolutions"
        ).fetchall()
        decisions = conn.execute(
            "SELECT decision_order, decision_id, reason_code "
            "FROM task_fence_policy_decisions"
        ).fetchall()
    finally:
        conn.close()

    assert ingress == [(7, "event-1", "gateway", "source-1")]
    assert transitions == [(11, "attempt-1", "PREPARED")]
    assert resolutions == [("resolution-1", "incident-1", "accepted_unknown_no_retry")]
    assert decisions == [(13, decision_id, "missing_provenance")]


def test_inspection_counts_are_capped(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        db._conn.executemany(
            "INSERT INTO task_fence_tasks ("
            "task_id, conversation_id, store_schema_version, "
            "control_protocol_version, status, current_runtime_epoch, "
            "created_at, updated_at"
            ") VALUES (?, ?, ?, ?, 'done', 0, 1.0, 1.0)",
            (
                (
                    f"task-{index}",
                    f"conversation-{index}",
                    TASK_FENCE_STORE_SCHEMA_VERSION,
                    TASK_FENCE_CONTROL_PROTOCOL_VERSION,
                )
                for index in range(1001)
            ),
        )
        inspection = db.inspect_task_fence_store(include_counts=True)
    finally:
        db.close()

    task_count = next(
        item for item in inspection.table_counts if item.table == "task_fence_tasks"
    )
    assert task_count.rows == 1000
    assert task_count.truncated is True


def test_count_inspection_never_interpolates_unexpected_identifier(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        conn.execute('CREATE TABLE "task_fence_bad""name" (value INTEGER)')
        conn.commit()
    finally:
        conn.close()

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store(include_counts=True)
    finally:
        reopened.close()

    assert inspection.compatible is False
    assert inspection.reason == "malformed_schema"
    unexpected = next(
        item for item in inspection.table_counts if item.table == 'task_fence_bad"name'
    )
    assert unexpected.rows is None
    assert unexpected.truncated is False


def test_policy_decision_journal_schema_is_closed_and_secret_free(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        column_info = {
            row[1]: row
            for row in conn.execute(
                'PRAGMA table_info("task_fence_policy_decisions")'
            )
        }
        columns = set(column_info)
        foreign_keys = conn.execute(
            'PRAGMA foreign_key_list("task_fence_policy_decisions")'
        ).fetchall()
        indexes = {
            row[1]
            for row in conn.execute('PRAGMA index_list("task_fence_policy_decisions")')
        }
    finally:
        conn.close()

    assert columns == {
        "decision_order",
        "decision_id",
        "decision_point",
        "outcome",
        "reason_code",
        "policy_version",
        "candidate_task_id",
        "candidate_authority_event_id",
        "candidate_run_id",
        "candidate_generation_id",
        "candidate_intent_epoch",
        "candidate_control_revision",
        "candidate_runtime_epoch",
        "cohort_key",
        "mode_generation",
        "operation_invocation_id",
        "envelope_invocation_id",
        "operation_kind",
        "adapter",
        "invocation_fingerprint",
        "causal_binding_fingerprint",
        "permit_id",
        "attempt_id",
        "decided_at",
    }
    assert foreign_keys == []
    assert column_info["mode_generation"][3] == 1
    assert "idx_task_fence_policy_decisions_invocation" in indexes
    assert all(
        forbidden not in column.lower()
        for column in columns
        for forbidden in (
            "prompt",
            "message",
            "argument",
            "result",
            "payload",
            "secret",
            "credential",
            "transcript",
            "envelope_json",
        )
    )


def test_policy_decision_exact_inspection_is_bounded_and_rejects_corruption(
    tmp_path,
):
    db = SessionDB(tmp_path / "state.db")
    missing_id = f"tfd_{'f' * 64}"
    corrupt_id = f"tfd_{'a' * 64}"
    try:
        invalid = db.inspect_task_fence_policy_decision("not-a-decision")
        missing = db.inspect_task_fence_policy_decision(missing_id)
        db._conn.execute(
            "INSERT INTO task_fence_policy_decisions ("
            "decision_id, decision_point, outcome, reason_code, policy_version, "
            "mode_generation, operation_invocation_id, operation_kind, adapter, "
            "invocation_fingerprint, decided_at"
            ") VALUES (?, 'admission', 'would_block', 'missing_provenance', "
            "'task-fence-policy-v1', 0, 'invocation-corrupt', 'tool', "
            "'adapter-corrupt', ?, 1.0)",
            (corrupt_id, "b" * 64),
        )
        corrupt = db.inspect_task_fence_policy_decision(corrupt_id)

        assert invalid.compatible is False
        assert invalid.reason == "invalid_decision_id"
        assert invalid.decision is None
        assert missing.compatible is True
        assert missing.reason == "not_found"
        assert missing.decision is None
        assert corrupt.compatible is False
        assert corrupt.reason == "malformed_policy_decision"
        assert corrupt.decision is None
    finally:
        db.close()

    closed = db.inspect_task_fence_policy_decision(missing_id)
    assert closed.compatible is False
    assert closed.reason == "inspection_closed"
    assert closed.decision is None


def test_policy_decision_schema_and_reader_reject_writer_impossible_success(
    tmp_path,
):
    db = SessionDB(tmp_path / "state.db")
    operation = {
        "invocation_id": "invocation-corrupt",
        "kind": "tool",
        "adapter": "adapter-corrupt",
        "invocation_fingerprint": "b" * 64,
    }
    semantic = {
        "policy_version": hermes_state.TASK_FENCE_POLICY_VERSION,
        "decision_point": "admission",
        "outcome": "would_reserve",
        "reason_code": "current_authority",
        "candidate_task_id": "task-corrupt",
        "candidate_authority_event_id": "event-corrupt",
        "candidate_run_id": "run-corrupt",
        "candidate_generation_id": "generation-corrupt",
        "candidate_intent_epoch": 0,
        "candidate_control_revision": 0,
        "candidate_runtime_epoch": 0,
        "cohort_key": None,
        "mode_generation": 0,
        "operation": operation,
        "envelope_invocation_id": operation["invocation_id"],
        "causal_binding_fingerprint": "c" * 64,
        "permit_id": "tfp_corrupt",
        "attempt_id": None,
    }
    decision_id = hermes_state._task_fence_policy_decision_identifier(semantic)
    values = (
        decision_id,
        semantic["policy_version"],
        semantic["candidate_task_id"],
        semantic["candidate_authority_event_id"],
        semantic["candidate_run_id"],
        semantic["candidate_generation_id"],
        semantic["candidate_intent_epoch"],
        semantic["candidate_control_revision"],
        semantic["candidate_runtime_epoch"],
        semantic["mode_generation"],
        operation["invocation_id"],
        semantic["envelope_invocation_id"],
        operation["adapter"],
        operation["invocation_fingerprint"],
        semantic["causal_binding_fingerprint"],
        semantic["permit_id"],
    )
    statement = (
        "INSERT INTO task_fence_policy_decisions ("
        "decision_id, decision_point, outcome, reason_code, policy_version, "
        "candidate_task_id, candidate_authority_event_id, candidate_run_id, "
        "candidate_generation_id, candidate_intent_epoch, "
        "candidate_control_revision, candidate_runtime_epoch, cohort_key, "
        "mode_generation, operation_invocation_id, envelope_invocation_id, "
        "operation_kind, adapter, invocation_fingerprint, "
        "causal_binding_fingerprint, permit_id, attempt_id, decided_at"
        ") VALUES (?, 'admission', 'would_reserve', 'current_authority', ?, "
        "?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, 'tool', ?, ?, ?, ?, NULL, 1.0)"
    )
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db._conn.execute(statement, values)

        db._conn.execute("PRAGMA ignore_check_constraints=ON")
        db._conn.execute(statement, values)
        db._conn.execute("PRAGMA ignore_check_constraints=OFF")
        corrupt = db.inspect_task_fence_policy_decision(decision_id)

        assert corrupt.compatible is False
        assert corrupt.reason == "malformed_policy_decision"
        assert corrupt.decision is None
    finally:
        db.close()


def test_task_fence_schema_has_no_raw_prompt_or_tool_payload_columns(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()

    conn = sqlite3.connect(path)
    try:
        columns = set()
        for table in EXPECTED_TASK_FENCE_TABLES:
            columns.update(
                row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')
            )
    finally:
        conn.close()

    assert {
        "prompt",
        "message_content",
        "tool_arguments",
        "tool_result",
        "worker_result",
        "secret",
    }.isdisjoint(columns)
