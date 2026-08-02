from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import threading

import pytest

import hermes_state
from hermes_state import SessionDB
from task_fence import (
    TASK_FENCE_CAPABILITY_VERSION,
    TASK_FENCE_SELECTED_COHORT_CAPABILITIES,
    TaskFenceCapabilityDeclaration,
    TaskFenceCapabilityKind,
    TaskFenceCapabilityState,
    TaskFenceCapabilityUnavailable,
    TaskFenceProtocolRejected,
)


_COHORT_KEY = "__task_fence_shadow_v1__"


def _materialize(db: SessionDB):
    store = db.inspect_task_fence_store(include_counts=False)
    assert store.compatible is True
    return db.materialize_task_fence_selected_cohort_capabilities(
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )


def _capability_rows(db: SessionDB):
    return tuple(
        tuple(row)
        for row in db._conn.execute(
            "SELECT capability_kind, capability_id, capability_version, "
            "declaration_state, declared_at "
            "FROM main.task_fence_cohort_capabilities "
            "WHERE cohort_key = ? "
            "ORDER BY capability_kind, capability_id, capability_version",
            (_COHORT_KEY,),
        ).fetchall()
    )


def _selected_storage_key():
    declaration = TASK_FENCE_SELECTED_COHORT_CAPABILITIES[0]
    return (
        declaration.kind.value,
        declaration.capability_id,
        declaration.capability_version,
    )


def test_selected_capability_vocabulary_is_closed_canonical_and_secret_free() -> None:
    declarations = TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    keys = tuple(
        (item.kind.value, item.capability_id, item.capability_version)
        for item in declarations
    )

    assert isinstance(declarations, tuple)
    assert declarations
    assert len(declarations) <= 64
    assert keys == tuple(sorted(keys))
    assert len(keys) == len(set(keys))
    assert len({item.capability_id for item in declarations}) == len(declarations)
    assert {item.kind for item in declarations} == {
        TaskFenceCapabilityKind.ADAPTER,
        TaskFenceCapabilityKind.RUNTIME,
    }
    assert {item.state for item in declarations} == {
        TaskFenceCapabilityState.SUPPORTED,
        TaskFenceCapabilityState.UNSUPPORTED,
    }
    assert {
        item.capability_version for item in declarations
    } == {TASK_FENCE_CAPABILITY_VERSION}

    by_id = {item.capability_id: item.state for item in declarations}
    assert (
        by_id["provider:openai.chat.completions.create"]
        is TaskFenceCapabilityState.SUPPORTED
    )
    assert (
        by_id["gateway:slack:chat_post_message"]
        is TaskFenceCapabilityState.SUPPORTED
    )
    assert (
        by_id["runtime:execute-code-rpc-descendant"]
        is TaskFenceCapabilityState.SUPPORTED
    )
    assert (
        by_id["gateway:slack:other_ingress"]
        is TaskFenceCapabilityState.UNSUPPORTED
    )
    assert (
        by_id["gateway:telegram:typed_ingress"]
        is TaskFenceCapabilityState.SUPPORTED
    )
    assert (
        by_id["gateway:telegram:other_ingress"]
        is TaskFenceCapabilityState.UNSUPPORTED
    )
    assert (
        by_id["runtime:internal-retries"]
        is TaskFenceCapabilityState.UNSUPPORTED
    )

    with pytest.raises(FrozenInstanceError):
        declarations[0].capability_id = "changed"
    with pytest.raises(TaskFenceProtocolRejected, match="invalid_capability_kind"):
        TaskFenceCapabilityDeclaration(
            kind="adapter",
            capability_id="route",
            capability_version=TASK_FENCE_CAPABILITY_VERSION,
            state=TaskFenceCapabilityState.SUPPORTED,
        )
    with pytest.raises(TaskFenceProtocolRejected, match="invalid_capability_id"):
        TaskFenceCapabilityDeclaration(
            kind=TaskFenceCapabilityKind.ADAPTER,
            capability_id="",
            capability_version=TASK_FENCE_CAPABILITY_VERSION,
            state=TaskFenceCapabilityState.SUPPORTED,
        )
    with pytest.raises(TaskFenceProtocolRejected, match="capability_id_too_large"):
        TaskFenceCapabilityDeclaration(
            kind=TaskFenceCapabilityKind.ADAPTER,
            capability_id="x" * 513,
            capability_version=TASK_FENCE_CAPABILITY_VERSION,
            state=TaskFenceCapabilityState.SUPPORTED,
        )


def test_materialization_creates_exact_set_and_reopens_read_only(tmp_path) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(path)
    schema_before = hermes_state._read_task_fence_schema_objects(db._conn)
    control_before = tuple(
        db._conn.execute(
            "SELECT * FROM main.task_fence_control WHERE singleton = 1"
        ).fetchone()
    )

    result = _materialize(db)

    assert result.created is True
    assert result.declarations == TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    assert hermes_state._read_task_fence_schema_objects(db._conn) == schema_before
    assert (
        tuple(
            db._conn.execute(
                "SELECT * FROM main.task_fence_control WHERE singleton = 1"
            ).fetchone()
        )
        == control_before
    )
    cohort = tuple(
        db._conn.execute(
            "SELECT mode, mode_generation, activation_state, audit_degraded "
            "FROM main.task_fence_cohorts WHERE cohort_key = ?",
            (_COHORT_KEY,),
        ).fetchone()
    )
    assert cohort == ("audit", 0, "inactive", 0)
    rows = _capability_rows(db)
    assert tuple(row[:4] for row in rows) == tuple(
        (
            declaration.kind.value,
            declaration.capability_id,
            declaration.capability_version,
            declaration.state.value,
        )
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    )
    assert len({row[4] for row in rows}) == 1
    assert db.inspect_task_fence_selected_cohort_capabilities().verified is True
    db.close()

    reopened = SessionDB(path)
    try:
        replayed = _materialize(reopened)
        assert replayed.created is False
        inspection = reopened.inspect_task_fence_selected_cohort_capabilities()
        assert inspection.verified is True
        assert inspection.reason == "verified"
        assert inspection.declarations == TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    finally:
        reopened.close()

    read_only = SessionDB(path, read_only=True)
    try:
        inspection = read_only.inspect_task_fence_selected_cohort_capabilities()
        assert inspection.declarations == TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        assert read_only._conn.total_changes == 0
    finally:
        read_only.close()


def test_exact_materialization_replay_preserves_rows_without_dml(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    _materialize(db)
    rows_before = _capability_rows(db)
    total_changes_before = db._conn.total_changes
    traced = []
    db._conn.set_trace_callback(traced.append)

    replayed = _materialize(db)

    db._conn.set_trace_callback(None)
    assert replayed.created is False
    assert replayed.declarations == TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    assert _capability_rows(db) == rows_before
    assert db._conn.total_changes == total_changes_before
    assert not any(
        statement.lstrip().upper().startswith(
            ("INSERT ", "UPDATE ", "DELETE ", "REPLACE ")
        )
        for statement in traced
    )
    db.close()


def test_v1_selected_set_is_rejected_without_merge_or_dml(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    _materialize(db)
    db._conn.execute(
        "DELETE FROM main.task_fence_cohort_capabilities "
        "WHERE cohort_key = ? AND capability_id IN (?, ?)",
        (
            _COHORT_KEY,
            "gateway:telegram:typed_ingress",
            "gateway:telegram:other_ingress",
        ),
    )
    db._conn.execute(
        "UPDATE main.task_fence_cohort_capabilities "
        "SET capability_version = 'task-fence-capability-v1' "
        "WHERE cohort_key = ?",
        (_COHORT_KEY,),
    )
    db._conn.commit()
    rows_before = _capability_rows(db)
    assert len(rows_before) == 34
    assert {row[2] for row in rows_before} == {
        "task-fence-capability-v1"
    }
    total_changes_before = db._conn.total_changes
    traced = []
    db._conn.set_trace_callback(traced.append)

    with pytest.raises(
        TaskFenceCapabilityUnavailable,
        match="capability_declaration_conflict",
    ):
        _materialize(db)

    db._conn.set_trace_callback(None)
    assert _capability_rows(db) == rows_before
    assert db._conn.total_changes == total_changes_before
    assert not any(
        statement.lstrip().upper().startswith(
            ("INSERT ", "UPDATE ", "DELETE ", "REPLACE ")
        )
        for statement in traced
    )
    db.close()


@pytest.mark.parametrize(
    ("corruption", "inspection_reason"),
    [
        ("missing", "capability_declaration_conflict"),
        ("extra", "capability_declaration_conflict"),
        ("state", "capability_declaration_conflict"),
        ("version", "capability_declaration_conflict"),
        ("mixed_timestamp", "malformed_capability_projection"),
        ("non_finite_timestamp", "malformed_capability_projection"),
        ("oversized_id", "capability_declaration_conflict"),
    ],
)
def test_conflicting_or_malformed_set_fails_without_partial_projection(
    tmp_path,
    corruption,
    inspection_reason,
) -> None:
    db = SessionDB(tmp_path / f"{corruption}.db")
    _materialize(db)
    kind, capability_id, version = _selected_storage_key()
    timestamp = _capability_rows(db)[0][4]

    if corruption == "missing":
        db._conn.execute(
            "DELETE FROM main.task_fence_cohort_capabilities "
            "WHERE cohort_key = ? AND capability_kind = ? "
            "AND capability_id = ? AND capability_version = ?",
            (_COHORT_KEY, kind, capability_id, version),
        )
    elif corruption == "extra":
        db._conn.execute(
            "INSERT INTO main.task_fence_cohort_capabilities "
            "(cohort_key, capability_kind, capability_id, capability_version, "
            "declaration_state, declared_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                _COHORT_KEY,
                "runtime",
                "runtime:injected-extra",
                TASK_FENCE_CAPABILITY_VERSION,
                "unsupported",
                timestamp,
            ),
        )
    elif corruption == "state":
        db._conn.execute(
            "UPDATE main.task_fence_cohort_capabilities "
            "SET declaration_state = CASE declaration_state "
            "WHEN 'supported' THEN 'unsupported' ELSE 'supported' END "
            "WHERE cohort_key = ? AND capability_kind = ? "
            "AND capability_id = ? AND capability_version = ?",
            (_COHORT_KEY, kind, capability_id, version),
        )
    elif corruption == "version":
        db._conn.execute(
            "UPDATE main.task_fence_cohort_capabilities "
            "SET capability_version = 'task-fence-capability-v999' "
            "WHERE cohort_key = ? AND capability_kind = ? "
            "AND capability_id = ? AND capability_version = ?",
            (_COHORT_KEY, kind, capability_id, version),
        )
    elif corruption == "mixed_timestamp":
        db._conn.execute(
            "UPDATE main.task_fence_cohort_capabilities "
            "SET declared_at = declared_at + 1 "
            "WHERE cohort_key = ? AND capability_kind = ? "
            "AND capability_id = ? AND capability_version = ?",
            (_COHORT_KEY, kind, capability_id, version),
        )
    elif corruption == "non_finite_timestamp":
        db._conn.execute(
            "UPDATE main.task_fence_cohort_capabilities "
            "SET declared_at = ? "
            "WHERE cohort_key = ? AND capability_kind = ? "
            "AND capability_id = ? AND capability_version = ?",
            (float("inf"), _COHORT_KEY, kind, capability_id, version),
        )
    else:
        db._conn.execute(
            "UPDATE main.task_fence_cohort_capabilities "
            "SET capability_id = ? "
            "WHERE cohort_key = ? AND capability_kind = ? "
            "AND capability_id = ? AND capability_version = ?",
            (
                "TOP_SECRET_" + "x" * 513,
                _COHORT_KEY,
                kind,
                capability_id,
                version,
            ),
        )
    db._conn.commit()
    rows_before = _capability_rows(db)

    inspection = db.inspect_task_fence_selected_cohort_capabilities()
    assert inspection.verified is False
    assert inspection.reason == inspection_reason
    assert inspection.declarations == ()
    assert "TOP_SECRET" not in repr(inspection)

    with pytest.raises(
        TaskFenceCapabilityUnavailable,
        match="capability_declaration_conflict",
    ) as exc:
        _materialize(db)
    assert exc.value.reason == "capability_declaration_conflict"
    assert _capability_rows(db) == rows_before
    db.close()


def test_capability_limit_fails_reason_only_and_uses_existing_pk_index(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    _materialize(db)
    timestamp = _capability_rows(db)[0][4]
    extras = 65 - len(TASK_FENCE_SELECTED_COHORT_CAPABILITIES)
    assert extras > 0
    db._conn.executemany(
        "INSERT INTO main.task_fence_cohort_capabilities "
        "(cohort_key, capability_kind, capability_id, capability_version, "
        "declaration_state, declared_at) VALUES (?, 'runtime', ?, ?, "
        "'unsupported', ?)",
        (
            (
                _COHORT_KEY,
                f"runtime:injected-limit:{index:02d}",
                TASK_FENCE_CAPABILITY_VERSION,
                timestamp,
            )
            for index in range(extras)
        ),
    )
    db._conn.commit()

    inspection = db.inspect_task_fence_selected_cohort_capabilities()
    assert inspection.verified is False
    assert inspection.reason == "capability_declaration_limit_exceeded"
    assert inspection.declarations == ()

    declaration = TASK_FENCE_SELECTED_COHORT_CAPABILITIES[0]
    plans = (
        db._conn.execute(
            "EXPLAIN QUERY PLAN SELECT 1 "
            "FROM main.task_fence_cohort_capabilities "
            "WHERE cohort_key = ? LIMIT 65",
            (_COHORT_KEY,),
        ).fetchall(),
        db._conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT CASE WHEN declaration_state = ? THEN 1 ELSE 0 END, "
            "CASE WHEN typeof(declared_at) IN ('integer', 'real') "
            "THEN declared_at ELSE NULL END "
            "FROM main.task_fence_cohort_capabilities "
            "WHERE cohort_key = ? AND capability_kind = ? "
            "AND capability_id = ? AND capability_version = ? LIMIT 2",
            (
                declaration.state.value,
                _COHORT_KEY,
                declaration.kind.value,
                declaration.capability_id,
                declaration.capability_version,
            ),
        ).fetchall(),
    )
    for query_plan in plans:
        plan = " ".join(str(row[3]) for row in query_plan)
        assert "sqlite_autoindex_task_fence_cohort_capabilities_1" in plan
        assert "TEMP B-TREE" not in plan
    db.close()


@pytest.mark.parametrize(
    ("trigger_body", "reason"),
    [
        (
            "SELECT RAISE(ABORT, 'private capability insert fault');",
            "capability_database_error",
        ),
        ("SELECT RAISE(IGNORE);", "capability_declaration_conflict"),
    ],
)
def test_mid_insert_fault_rolls_back_cohort_and_complete_set(
    tmp_path,
    trigger_body,
    reason,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    target = TASK_FENCE_SELECTED_COHORT_CAPABILITIES[-1].capability_id
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_capability_materialization "
        "BEFORE INSERT ON main.task_fence_cohort_capabilities "
        f"WHEN NEW.capability_id = '{target}' "
        f"BEGIN {trigger_body} END"
    )

    with pytest.raises(TaskFenceCapabilityUnavailable, match=reason) as exc:
        _materialize(db)

    assert exc.value.reason == reason
    assert (
        db._conn.execute(
            "SELECT COUNT(*) FROM main.task_fence_cohorts WHERE cohort_key = ?",
            (_COHORT_KEY,),
        ).fetchone()[0]
        == 0
    )
    assert _capability_rows(db) == ()
    inspection = db.inspect_task_fence_selected_cohort_capabilities()
    assert inspection.reason == "capabilities_not_materialized"
    assert inspection.declarations == ()
    db.close()


def test_concurrent_identical_materialization_creates_once_then_replays(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    SessionDB(path).close()
    barrier = threading.Barrier(2)

    def materialize() -> bool:
        db = SessionDB(path)
        try:
            barrier.wait(timeout=10)
            return _materialize(db).created
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        created = tuple(pool.map(lambda _: materialize(), range(2)))

    assert sorted(created) == [False, True]
    reopened = SessionDB(path, read_only=True)
    try:
        assert (
            reopened.inspect_task_fence_selected_cohort_capabilities().declarations
            == TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        )
    finally:
        reopened.close()


def test_reader_observes_absent_or_complete_set_during_materialization(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    SessionDB(path).close()
    writer = SessionDB(path)
    reader = SessionDB(path, read_only=True)
    insert_paused = threading.Event()
    release_insert = threading.Event()
    target = TASK_FENCE_SELECTED_COHORT_CAPABILITIES[1].capability_id

    def pause_insert() -> int:
        insert_paused.set()
        if not release_insert.wait(timeout=10):
            raise RuntimeError("timed out waiting to release capability insert")
        return 0

    writer._conn.create_function("pause_capability_insert", 0, pause_insert)
    writer._conn.execute(
        "CREATE TEMP TRIGGER pause_capability_materialization "
        "BEFORE INSERT ON main.task_fence_cohort_capabilities "
        f"WHEN NEW.capability_id = '{target}' "
        "BEGIN SELECT pause_capability_insert(); END"
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_materialize, writer)
        try:
            assert insert_paused.wait(timeout=10)
            during = reader.inspect_task_fence_selected_cohort_capabilities()
            assert during.reason == "capabilities_not_materialized"
            assert during.declarations == ()
        finally:
            release_insert.set()
        assert future.result(timeout=10).created is True

    after = reader.inspect_task_fence_selected_cohort_capabilities()
    assert after.verified is True
    assert after.declarations == TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    reader.close()
    writer.close()


def test_read_only_inspection_ignores_foreign_cohort_and_never_materializes(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    writer = SessionDB(path)
    writer._conn.execute(
        "INSERT INTO main.task_fence_cohorts "
        "(cohort_key, mode, mode_generation, activation_state, audit_degraded, "
        "created_at, updated_at) VALUES "
        "('foreign', 'audit', 0, 'inactive', 0, 1.0, 1.0)"
    )
    writer._conn.execute(
        "INSERT INTO main.task_fence_cohort_capabilities "
        "(cohort_key, capability_kind, capability_id, capability_version, "
        "declaration_state, declared_at) VALUES "
        "('foreign', 'runtime', 'foreign:route', 'foreign-v1', 'supported', 1.0)"
    )
    writer._conn.commit()
    writer.close()

    read_only = SessionDB(path, read_only=True)
    try:
        changes_before = read_only._conn.total_changes
        inspection = read_only.inspect_task_fence_selected_cohort_capabilities()
        assert inspection.verified is False
        assert inspection.reason == "capabilities_not_materialized"
        assert inspection.declarations == ()
        assert read_only._conn.total_changes == changes_before
        assert (
            read_only._conn.execute(
                "SELECT COUNT(*) FROM main.task_fence_cohorts "
                "WHERE cohort_key = ?",
                (_COHORT_KEY,),
            ).fetchone()[0]
            == 0
        )
        with pytest.raises(
            TaskFenceCapabilityUnavailable,
            match="store_unavailable",
        ):
            _materialize(read_only)
    finally:
        read_only.close()

    closed = read_only.inspect_task_fence_selected_cohort_capabilities()
    assert closed.reason == "inspection_closed"
    assert closed.declarations == ()


def test_orphan_selected_rows_fail_closed_without_partial_projection(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    declaration = TASK_FENCE_SELECTED_COHORT_CAPABILITIES[0]
    db._conn.execute("PRAGMA foreign_keys=OFF")
    db._conn.execute(
        "INSERT INTO main.task_fence_cohort_capabilities "
        "(cohort_key, capability_kind, capability_id, capability_version, "
        "declaration_state, declared_at) VALUES (?, ?, ?, ?, ?, 1.0)",
        (
            _COHORT_KEY,
            declaration.kind.value,
            declaration.capability_id,
            declaration.capability_version,
            declaration.state.value,
        ),
    )
    db._conn.commit()
    db._conn.execute("PRAGMA foreign_keys=ON")
    rows_before = _capability_rows(db)

    inspection = db.inspect_task_fence_selected_cohort_capabilities()
    assert inspection.verified is False
    assert inspection.reason == "capability_declaration_conflict"
    assert inspection.declarations == ()
    with pytest.raises(
        TaskFenceCapabilityUnavailable,
        match="capability_declaration_conflict",
    ):
        _materialize(db)
    assert _capability_rows(db) == rows_before
    assert (
        db._conn.execute(
            "SELECT COUNT(*) FROM main.task_fence_cohorts WHERE cohort_key = ?",
            (_COHORT_KEY,),
        ).fetchone()[0]
        == 0
    )
    db.close()


def test_inspection_preserves_borrowed_read_transaction(tmp_path) -> None:
    path = tmp_path / "state.db"
    writer = SessionDB(path)
    _materialize(writer)
    writer.close()

    reader = SessionDB(path, read_only=True)
    try:
        reader._conn.execute("BEGIN")
        inspection = reader.inspect_task_fence_selected_cohort_capabilities()
        assert inspection.verified is True
        assert reader._conn.in_transaction is True
        reader._conn.rollback()
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("column", "value", "inspection_reason"),
    [
        ("mode", "enforce", "implicit_cohort_mismatch"),
        ("mode_generation", 1, "cohort_generation_mismatch"),
    ],
)
def test_selected_cohort_mismatch_fails_without_declarations(
    tmp_path,
    column,
    value,
    inspection_reason,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    _materialize(db)
    db._conn.execute(
        f"UPDATE main.task_fence_cohorts SET {column} = ? WHERE cohort_key = ?",
        (value, _COHORT_KEY),
    )
    db._conn.commit()
    rows_before = _capability_rows(db)

    inspection = db.inspect_task_fence_selected_cohort_capabilities()
    assert inspection.reason == inspection_reason
    assert inspection.declarations == ()
    with pytest.raises(
        TaskFenceCapabilityUnavailable,
        match="capability_materialization_precondition",
    ):
        _materialize(db)
    assert _capability_rows(db) == rows_before
    db.close()


def test_materialization_rejects_stale_control_expectations_before_writes(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")

    with pytest.raises(
        TaskFenceCapabilityUnavailable,
        match="runtime_epoch_changed",
    ):
        db.materialize_task_fence_selected_cohort_capabilities(
            expected_runtime_epoch=1,
            expected_mode_generation=0,
        )
    with pytest.raises(
        TaskFenceCapabilityUnavailable,
        match="mode_generation_changed",
    ):
        db.materialize_task_fence_selected_cohort_capabilities(
            expected_runtime_epoch=0,
            expected_mode_generation=1,
        )

    assert _capability_rows(db) == ()
    assert (
        db._conn.execute(
            "SELECT COUNT(*) FROM main.task_fence_cohorts WHERE cohort_key = ?",
            (_COHORT_KEY,),
        ).fetchone()[0]
        == 0
    )
    db.close()
