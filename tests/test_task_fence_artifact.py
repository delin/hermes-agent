import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state import SessionDB
from task_fence import (
    CausalEnvelope,
    IngressAcceptance,
    IngressEnvelope,
    TASK_FENCE_ACTIONS,
    TaskFenceArtifactIdentity,
    TaskFenceArtifactUnavailable,
    TaskFenceProtocolRejected,
    TaskFenceRecoveryUnavailable,
)


def _identity(marker: str = "a") -> TaskFenceArtifactIdentity:
    values = {"a": ("a", "b", "c"), "d": ("d", "e", "f")}
    commit, artifact, lock = values[marker]
    return TaskFenceArtifactIdentity(
        tested_artifact_commit=commit * 40,
        tested_artifact_checksum="sha256:" + artifact * 64,
        dependency_lock_fingerprint="sha256:" + lock * 64,
    )


def _control_row(db: SessionDB) -> tuple:
    return tuple(
        db._conn.execute(
            "SELECT runtime_epoch, mode_generation, ever_enforced, "
            "tested_artifact_commit, tested_artifact_checksum, "
            "dependency_lock_fingerprint, updated_at "
            "FROM main.task_fence_control WHERE singleton = 1"
        ).fetchone()
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


def _prepare_recovery_subject(
    db: SessionDB,
    marker: str,
) -> tuple[IngressAcceptance, CausalEnvelope]:
    source_event_id = f"artifact-startup-{marker}"
    accepted = db.accept_task_fence_ingress(
        IngressEnvelope(
            source="gateway:test:artifact-startup",
            source_event_id=source_event_id,
            conversation_id=f"artifact-startup-{marker}",
            action=TASK_FENCE_ACTIONS["initial_submit"],
            payload_hash=hashlib.sha256(source_event_id.encode()).hexdigest(),
        )
    )
    assert accepted.task_id is not None
    assert accepted.opened_run_id is not None
    generation = db.reserve_task_fence_generation(accepted)
    return accepted, generation


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("tested_artifact_commit", "a" * 39, "invalid_tested_artifact_commit"),
        ("tested_artifact_commit", "A" * 40, "invalid_tested_artifact_commit"),
        ("tested_artifact_commit", "0" * 40, "invalid_tested_artifact_commit"),
        (
            "tested_artifact_commit",
            "refs/tags/latest",
            "invalid_tested_artifact_commit",
        ),
        (
            "tested_artifact_checksum",
            "b" * 64,
            "invalid_tested_artifact_checksum",
        ),
        (
            "tested_artifact_checksum",
            "sha256:" + "B" * 64,
            "invalid_tested_artifact_checksum",
        ),
        (
            "tested_artifact_checksum",
            "sha256:" + "0" * 64,
            "invalid_tested_artifact_checksum",
        ),
        (
            "dependency_lock_fingerprint",
            "sha256:" + "c" * 63,
            "invalid_dependency_lock_fingerprint",
        ),
        (
            "dependency_lock_fingerprint",
            "sha512:" + "c" * 64,
            "invalid_dependency_lock_fingerprint",
        ),
        (
            "dependency_lock_fingerprint",
            "sha256:" + "0" * 64,
            "invalid_dependency_lock_fingerprint",
        ),
    ),
)
def test_artifact_identity_is_closed_and_canonical(field, value, reason) -> None:
    values = {
        "tested_artifact_commit": "a" * 40,
        "tested_artifact_checksum": "sha256:" + "b" * 64,
        "dependency_lock_fingerprint": "sha256:" + "c" * 64,
    }
    values[field] = value

    with pytest.raises(TaskFenceProtocolRejected, match=reason) as exc:
        TaskFenceArtifactIdentity(**values)

    assert exc.value.reason == reason


def test_unset_artifact_verification_is_read_only(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    before = _control_row(db)

    result = db.verify_task_fence_tested_artifact(_identity())

    assert result.store.compatible is True
    assert result.verified is False
    assert result.reason == "artifact_unset"
    assert result.tested_identity is None
    assert _control_row(db) == before
    db.close()


def test_artifact_pin_is_atomic_durable_and_exactly_verifiable(tmp_path) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    identity = _identity()

    pinned = db.pin_task_fence_tested_artifact(
        identity,
        expected_runtime_epoch=0,
        expected_mode_generation=0,
    )

    assert pinned.created is True
    assert pinned.identity == identity
    db.close()

    reopened = SessionDB(path)
    result = reopened.verify_task_fence_tested_artifact(identity)
    store = reopened.inspect_task_fence_store()
    assert result.verified is True
    assert result.reason == "verified"
    assert result.tested_identity == identity
    assert (
        store.tested_artifact_commit,
        store.tested_artifact_checksum,
        store.dependency_lock_fingerprint,
    ) == (
        identity.tested_artifact_commit,
        identity.tested_artifact_checksum,
        identity.dependency_lock_fingerprint,
    )
    reopened.close()


def test_exact_artifact_replay_preserves_control_state(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    identity = _identity()
    db.pin_task_fence_tested_artifact(
        identity,
        expected_runtime_epoch=0,
        expected_mode_generation=0,
    )
    before = _control_row(db)

    replayed = db.pin_task_fence_tested_artifact(
        identity,
        expected_runtime_epoch=0,
        expected_mode_generation=0,
    )

    assert replayed.created is False
    assert replayed.identity == identity
    assert _control_row(db) == before
    db.close()


def test_artifact_mismatch_is_read_only_and_reports_tested_identity(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    tested = _identity("a")
    db.pin_task_fence_tested_artifact(
        tested,
        expected_runtime_epoch=0,
        expected_mode_generation=0,
    )
    before = _control_row(db)

    result = db.verify_task_fence_tested_artifact(_identity("d"))

    assert result.verified is False
    assert result.reason == "artifact_mismatch"
    assert result.tested_identity == tested
    assert _control_row(db) == before
    db.close()


def test_artifact_pin_is_set_once_and_never_replaced(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    first = _identity("a")
    db.pin_task_fence_tested_artifact(
        first,
        expected_runtime_epoch=0,
        expected_mode_generation=0,
    )
    before = _control_row(db)

    with pytest.raises(
        TaskFenceArtifactUnavailable, match="artifact_identity_conflict"
    ) as exc:
        db.pin_task_fence_tested_artifact(
            _identity("d"),
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )

    assert exc.value.reason == "artifact_identity_conflict"
    assert _control_row(db) == before
    db.close()


@pytest.mark.parametrize(
    "stored_values",
    (
        ("a" * 40, None, None),
        ("not-a-commit", "sha256:" + "b" * 64, "sha256:" + "c" * 64),
    ),
)
def test_partial_or_malformed_artifact_state_never_becomes_authority(
    tmp_path, stored_values
) -> None:
    db = SessionDB(tmp_path / "state.db")
    db._conn.execute(
        "UPDATE main.task_fence_control SET tested_artifact_commit = ?, "
        "tested_artifact_checksum = ?, dependency_lock_fingerprint = ? "
        "WHERE singleton = 1",
        stored_values,
    )
    before = _control_row(db)

    result = db.verify_task_fence_tested_artifact(_identity())
    assert result.verified is False
    assert result.reason == "malformed_artifact_identity"
    assert result.tested_identity is None

    with pytest.raises(
        TaskFenceArtifactUnavailable, match="malformed_artifact_identity"
    ):
        db.pin_task_fence_tested_artifact(
            _identity(),
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )

    assert _control_row(db) == before
    db.close()


def test_concurrent_competing_artifact_pins_never_mix_fields(tmp_path) -> None:
    path = tmp_path / "state.db"
    SessionDB(path).close()
    barrier = threading.Barrier(2)

    def pin(identity: TaskFenceArtifactIdentity):
        db = SessionDB(path)
        try:
            barrier.wait(timeout=5)
            result = db.pin_task_fence_tested_artifact(
                identity,
                expected_runtime_epoch=0,
                expected_mode_generation=0,
            )
            return ("created" if result.created else "replayed", identity)
        except TaskFenceArtifactUnavailable as exc:
            return (exc.reason, identity)
        finally:
            db.close()

    identities = (_identity("a"), _identity("d"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(pin, identities))

    assert sorted(result[0] for result in results) == [
        "artifact_identity_conflict",
        "created",
    ]
    db = SessionDB(path)
    stored = db.inspect_task_fence_store()
    triples = {
        (
            identity.tested_artifact_commit,
            identity.tested_artifact_checksum,
            identity.dependency_lock_fingerprint,
        )
        for identity in identities
    }
    assert (
        stored.tested_artifact_commit,
        stored.tested_artifact_checksum,
        stored.dependency_lock_fingerprint,
    ) in triples
    db.close()


def test_concurrent_identical_artifact_pins_create_once_then_replay(tmp_path) -> None:
    path = tmp_path / "state.db"
    SessionDB(path).close()
    barrier = threading.Barrier(2)
    identity = _identity()

    def pin() -> str:
        db = SessionDB(path)
        try:
            barrier.wait(timeout=5)
            result = db.pin_task_fence_tested_artifact(
                identity,
                expected_runtime_epoch=0,
                expected_mode_generation=0,
            )
            return "created" if result.created else "replayed"
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _: pin(), range(2)))

    assert sorted(results) == ["created", "replayed"]
    db = SessionDB(path)
    assert db.verify_task_fence_tested_artifact(identity).verified is True
    db.close()


def test_artifact_pin_database_fault_rolls_back_the_complete_triple(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_artifact_pin "
        "BEFORE UPDATE OF tested_artifact_commit ON main.task_fence_control "
        "BEGIN SELECT RAISE(ABORT, 'injected artifact pin failure'); END"
    )

    with pytest.raises(
        TaskFenceArtifactUnavailable, match="artifact_database_error"
    ) as exc:
        db.pin_task_fence_tested_artifact(
            _identity(),
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )

    assert exc.value.reason == "artifact_database_error"
    assert _control_row(db)[3:6] == (None, None, None)
    db.close()


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    (
        ("runtime_epoch", 1, "runtime_epoch_changed"),
        ("mode_generation", 1, "mode_generation_changed"),
        ("ever_enforced", 1, "artifact_pin_precondition"),
    ),
)
def test_artifact_pin_checks_exact_control_generation(
    tmp_path, column, value, reason
) -> None:
    db = SessionDB(tmp_path / "state.db")
    db._conn.execute(
        f"UPDATE main.task_fence_control SET {column} = ? WHERE singleton = 1",
        (value,),
    )
    before = _control_row(db)

    with pytest.raises(TaskFenceArtifactUnavailable, match=reason) as exc:
        db.pin_task_fence_tested_artifact(
            _identity(),
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )

    assert exc.value.reason == reason
    assert _control_row(db) == before
    db.close()


def test_artifact_pin_refuses_read_only_closed_and_incompatible_stores(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    SessionDB(path).close()

    read_only = SessionDB(path, read_only=True)
    with pytest.raises(TaskFenceArtifactUnavailable, match="store_unavailable"):
        read_only.pin_task_fence_tested_artifact(
            _identity(),
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )
    read_only.close()

    closed = SessionDB(path)
    closed.close()
    with pytest.raises(TaskFenceArtifactUnavailable, match="store_unavailable"):
        closed.pin_task_fence_tested_artifact(
            _identity(),
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )
    assert closed.verify_task_fence_tested_artifact(_identity()).reason == (
        "inspection_closed"
    )

    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE task_fence_control SET control_protocol_version = "
        "control_protocol_version + 1"
    )
    conn.commit()
    conn.close()
    incompatible = SessionDB(path)
    before = _control_row(incompatible)
    with pytest.raises(
        TaskFenceArtifactUnavailable, match="unsupported_control_protocol"
    ):
        incompatible.pin_task_fence_tested_artifact(
            _identity(),
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )
    assert _control_row(incompatible) == before
    incompatible.close()


def test_artifact_pin_and_shadow_recovery_have_only_two_serial_orders(tmp_path) -> None:
    pinned_path = tmp_path / "pin-first.db"
    pinned = SessionDB(pinned_path)
    pinned.pin_task_fence_tested_artifact(
        _identity(),
        expected_runtime_epoch=0,
        expected_mode_generation=0,
    )
    pinned_before_recovery = _control_row(pinned)
    with pytest.raises(
        TaskFenceRecoveryUnavailable, match="shadow_recovery_precondition"
    ) as exc:
        pinned.recover_task_fence_state(
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )
    assert exc.value.reason == "shadow_recovery_precondition"
    assert _control_row(pinned) == pinned_before_recovery
    pinned.close()

    recovered_path = tmp_path / "recovery-first.db"
    recovered = SessionDB(recovered_path)
    recovery = recovered.recover_task_fence_state(
        expected_runtime_epoch=0,
        expected_mode_generation=0,
    )
    assert recovery.runtime_epoch == 1
    recovered_before_stale_pin = _control_row(recovered)
    with pytest.raises(TaskFenceArtifactUnavailable, match="runtime_epoch_changed"):
        recovered.pin_task_fence_tested_artifact(
            _identity(),
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )
    assert _control_row(recovered) == recovered_before_stale_pin
    pinned_after_recovery = recovered.pin_task_fence_tested_artifact(
        _identity(),
        expected_runtime_epoch=1,
        expected_mode_generation=0,
    )
    assert pinned_after_recovery.created is True
    assert recovered.inspect_task_fence_store().runtime_epoch == 1
    recovered.close()


@pytest.mark.parametrize(
    ("shadow_session_key", "with_artifact", "reason"),
    (
        ("", True, "invalid_shadow_session_key"),
        ("lane\x00key", True, "invalid_shadow_session_key"),
        ("x" * 513, True, "invalid_shadow_session_key"),
        ("lane-key", False, "shadow_session_key_requires_artifact"),
    ),
)
def test_shadow_session_recovery_scope_is_closed(
    tmp_path,
    shadow_session_key,
    with_artifact,
    reason,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    before = _task_fence_state(db)

    with pytest.raises(TaskFenceProtocolRejected, match=reason) as exc:
        db.recover_task_fence_state(
            expected_runtime_epoch=0,
            expected_mode_generation=0,
            tested_artifact_identity=_identity() if with_artifact else None,
            shadow_session_key=shadow_session_key,
        )

    assert exc.value.reason == reason
    assert _task_fence_state(db) == before
    db.close()


def test_startup_artifact_first_pin_recovers_durable_authority(tmp_path) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    identity = _identity()
    accepted, generation = _prepare_recovery_subject(db, "first-pin")

    recovery = db.recover_task_fence_state(
        expected_runtime_epoch=0,
        expected_mode_generation=0,
        tested_artifact_identity=identity,
    )

    assert (recovery.previous_runtime_epoch, recovery.runtime_epoch) == (0, 1)
    db.close()

    reopened = SessionDB(path)
    store = reopened.inspect_task_fence_store()
    assert store.runtime_epoch == 1
    assert (
        store.tested_artifact_commit,
        store.tested_artifact_checksum,
        store.dependency_lock_fingerprint,
    ) == (
        identity.tested_artifact_commit,
        identity.tested_artifact_checksum,
        identity.dependency_lock_fingerprint,
    )
    assert reopened.verify_task_fence_tested_artifact(identity).verified is True
    task = reopened.inspect_task_fence_task(accepted.task_id).task
    assert task is not None
    assert task.status == "paused"
    assert task.current_runtime_epoch == 1
    assert task.active_execution_run_id is None
    assert task.current_generation_id is None
    assert tuple(
        reopened._conn.execute(
            "SELECT state, bound_run_id FROM task_fence_task_inputs WHERE event_id = ?",
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
            "SELECT state FROM task_fence_model_generations WHERE generation_id = ?",
            (generation.generation_id,),
        ).fetchone()[0]
        == "cancelled"
    )
    assert tuple(
        reopened._conn.execute(
            "SELECT runtime_epoch, task_id, input_event_id, source_run_id "
            "FROM task_fence_recovery_requeues"
        ).fetchone()
    ) == (
        1,
        accepted.task_id,
        accepted.event_id,
        accepted.opened_run_id,
    )
    reopened.close()


def test_startup_artifact_exact_replay_advances_epoch_without_replacement(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    identity = _identity()
    db = SessionDB(path)
    first = db.recover_task_fence_state(
        expected_runtime_epoch=0,
        expected_mode_generation=0,
        tested_artifact_identity=identity,
    )
    first_identity = _control_row(db)[3:6]

    replay = db.recover_task_fence_state(
        expected_runtime_epoch=1,
        expected_mode_generation=0,
        tested_artifact_identity=identity,
    )

    assert (first.previous_runtime_epoch, first.runtime_epoch) == (0, 1)
    assert (replay.previous_runtime_epoch, replay.runtime_epoch) == (1, 2)
    assert _control_row(db)[3:6] == first_identity
    db.close()

    reopened = SessionDB(path)
    assert reopened.inspect_task_fence_store().runtime_epoch == 2
    assert reopened.verify_task_fence_tested_artifact(identity).verified is True
    reopened.close()


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("conflict", "artifact_identity_conflict"),
        ("partial", "malformed_artifact_identity"),
    ),
)
def test_startup_artifact_conflict_or_partial_state_is_wholly_read_only(
    tmp_path,
    case,
    reason,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    _prepare_recovery_subject(db, case)
    identity = _identity()
    observed = identity
    if case == "conflict":
        db.pin_task_fence_tested_artifact(
            identity,
            expected_runtime_epoch=0,
            expected_mode_generation=0,
        )
        observed = _identity("d")
    else:
        db._conn.execute(
            "UPDATE main.task_fence_control SET tested_artifact_commit = ? "
            "WHERE singleton = 1",
            (identity.tested_artifact_commit,),
        )
    before = _task_fence_state(db)

    with pytest.raises(TaskFenceRecoveryUnavailable, match=reason) as exc:
        db.recover_task_fence_state(
            expected_runtime_epoch=0,
            expected_mode_generation=0,
            tested_artifact_identity=observed,
        )

    assert exc.value.reason == reason
    assert _task_fence_state(db) == before
    db.close()

    reopened = SessionDB(path)
    assert _task_fence_state(reopened) == before
    reopened.close()


@pytest.mark.parametrize("fault_target", ("task_epoch", "artifact_pin"))
def test_startup_artifact_fault_rolls_back_pin_and_recovery_together(
    tmp_path,
    fault_target,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    _prepare_recovery_subject(db, fault_target)
    trigger = {
        "task_epoch": (
            "BEFORE UPDATE OF current_runtime_epoch ON main.task_fence_tasks "
            "WHEN NEW.current_runtime_epoch != OLD.current_runtime_epoch"
        ),
        "artifact_pin": (
            "BEFORE UPDATE OF tested_artifact_commit ON main.task_fence_control "
            "WHEN NEW.tested_artifact_commit IS NOT OLD.tested_artifact_commit"
        ),
    }[fault_target]
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_task_fence_artifact_startup "
        f"{trigger} BEGIN "
        "SELECT RAISE(ABORT, 'private startup transaction fault'); END"
    )
    before = _task_fence_state(db)

    with pytest.raises(
        TaskFenceRecoveryUnavailable,
        match="recovery_database_error",
    ) as exc:
        db.recover_task_fence_state(
            expected_runtime_epoch=0,
            expected_mode_generation=0,
            tested_artifact_identity=_identity(),
        )

    assert exc.value.reason == "recovery_database_error"
    assert exc.value.__cause__ is None
    assert "private" not in str(exc.value)
    assert _task_fence_state(db) == before
    db.close()

    reopened = SessionDB(path)
    assert _task_fence_state(reopened) == before
    assert _control_row(reopened)[:6] == (0, 0, 0, None, None, None)
    reopened.close()


def test_artifact_verification_preserves_caller_read_transaction(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    identity = _identity()
    db.pin_task_fence_tested_artifact(
        identity,
        expected_runtime_epoch=0,
        expected_mode_generation=0,
    )
    db._conn.execute("BEGIN")

    result = db.verify_task_fence_tested_artifact(identity)

    assert result.verified is True
    assert db._conn.in_transaction is True
    db._conn.rollback()
    db.close()
