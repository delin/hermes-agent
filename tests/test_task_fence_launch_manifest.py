from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
import sqlite3

import pytest

import hermes_state
from hermes_state import SessionDB
from task_fence import (
    TASK_FENCE_ACTIONS,
    TASK_FENCE_SELECTED_COHORT_CAPABILITIES,
    TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
    IngressEnvelope,
    TaskFenceArtifactIdentity,
    TaskFenceCapabilityState,
    TaskFenceLaunchBindingUnavailable,
    TaskFenceLaunchManifest,
    TaskFenceLaunchRoute,
    TaskFenceProtocolRejected,
    task_fence_launch_manifest_fingerprint,
)


_SELECTED_COHORT_KEY = "__task_fence_selected_activation_v1__"
_MIGRATION_CONVERSATION = "agent:main:slack:dm:migration:lane"
_MIGRATION_IDENTITY = TaskFenceArtifactIdentity(
    tested_artifact_commit="a" * 40,
    tested_artifact_checksum="sha256:" + "b" * 64,
    dependency_lock_fingerprint="sha256:" + "c" * 64,
)


def _ready_store(database: SessionDB):
    store = database.inspect_task_fence_store(include_counts=False)
    database.materialize_task_fence_selected_cohort_capabilities(
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )
    return store


def _seed_wp11_1a_store(path):
    database = SessionDB(path)
    envelope = IngressEnvelope(
        source="gateway:test:v6-migration",
        source_event_id="v6-initial",
        conversation_id=_MIGRATION_CONVERSATION,
        action=TASK_FENCE_ACTIONS["initial_submit"],
        payload_hash=hashlib.sha256(b"v6-migration").hexdigest(),
    )
    acceptance = database.accept_task_fence_ingress(envelope)
    store = _ready_store(database)
    recovery = database.recover_task_fence_state(
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
        tested_artifact_identity=_MIGRATION_IDENTITY,
        shadow_session_key=_MIGRATION_CONVERSATION,
    )
    assert recovery.runtime_epoch == 1
    database.close()
    return envelope, acceptance


def _downgrade_current_store_to_exact_v6(path) -> None:
    connection = sqlite3.connect(path, isolation_level=None)
    connection.execute("PRAGMA foreign_keys=ON")
    trigger = connection.execute(
        "SELECT sql FROM main.sqlite_master WHERE type = 'trigger' "
        "AND name = 'task_fence_acceptance_snapshots_no_update'"
    ).fetchone()
    assert trigger is not None and isinstance(trigger[0], str)
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("DROP TABLE main.task_fence_cohort_launch_bindings")
        connection.execute(
            "DROP TRIGGER main.task_fence_acceptance_snapshots_no_update"
        )
        connection.execute(
            "UPDATE main.task_fence_acceptance_snapshots "
            "SET task_store_schema_version = 6 "
            "WHERE task_store_schema_version = 7"
        )
        connection.execute(
            "UPDATE main.task_fence_tasks SET store_schema_version = 6 "
            "WHERE store_schema_version = 7"
        )
        connection.execute(
            "UPDATE main.task_fence_control SET store_schema_version = 6 "
            "WHERE singleton = 1 AND store_schema_version = 7"
        )
        connection.execute(trigger[0])
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()

    check = sqlite3.connect(path)
    try:
        assert (
            hermes_state._read_task_fence_schema_objects(check)
            == hermes_state._expected_task_fence_v6_schema_objects()
        )
    finally:
        check.close()


def test_selected_launch_manifest_is_an_explicit_supported_slack_cohort() -> None:
    declarations = {
        (
            declaration.kind,
            declaration.capability_id,
            declaration.capability_version,
        ): declaration.state
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    }
    routes = {
        (route.kind, route.route_id, route.capability_version)
        for route in TASK_FENCE_SELECTED_LAUNCH_MANIFEST.routes
    }

    assert routes
    assert all(
        declarations[route] is TaskFenceCapabilityState.SUPPORTED for route in routes
    )
    assert {
        "gateway:slack:typed_ingress",
        "gateway:slack:chat_post_message",
    } <= {route.route_id for route in TASK_FENCE_SELECTED_LAUNCH_MANIFEST.routes}
    assert not any(
        route.route_id.startswith(("gateway:telegram:", "cli:"))
        for route in TASK_FENCE_SELECTED_LAUNCH_MANIFEST.routes
    )
    assert any(
        declaration.state is TaskFenceCapabilityState.SUPPORTED
        and declaration.capability_id not in {route[1] for route in routes}
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    )


def test_launch_manifest_fingerprint_commits_version_and_canonical_routes() -> None:
    selected = TASK_FENCE_SELECTED_LAUNCH_MANIFEST
    replay = TaskFenceLaunchManifest(
        manifest_version=selected.manifest_version,
        routes=tuple(reversed(selected.routes)),
    )
    changed = TaskFenceLaunchManifest(
        manifest_version=selected.manifest_version + ".changed",
        routes=selected.routes,
    )

    assert replay.routes == selected.routes
    assert task_fence_launch_manifest_fingerprint(replay) == (
        task_fence_launch_manifest_fingerprint(selected)
    )
    assert task_fence_launch_manifest_fingerprint(changed) != (
        task_fence_launch_manifest_fingerprint(selected)
    )


def test_launch_catalog_is_select_only_immutable_and_classifies_real_witnesses(
    tmp_path,
) -> None:
    conversation_key = "agent:main:slack:dm:workspace:channel:user"
    path = tmp_path / "state.db"
    seed = SessionDB(path)
    store = _ready_store(seed)
    binding = seed.materialize_task_fence_selected_launch_binding(
        shadow_conversation_key=conversation_key,
        manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )
    changes_before_writable_load = seed._conn.total_changes
    with pytest.raises(TaskFenceLaunchBindingUnavailable) as exc_info:
        seed.load_task_fence_selected_launch_catalog(
            shadow_conversation_key=conversation_key,
            manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
            expected_runtime_epoch=store.runtime_epoch,
            expected_mode_generation=store.mode_generation,
        )
    assert exc_info.value.reason == "store_unavailable"
    assert seed._conn.total_changes == changes_before_writable_load
    seed.close()

    database = SessionDB(path, read_only=True)
    try:
        changes_before = database._conn.total_changes
        catalog = database.load_task_fence_selected_launch_catalog(
            shadow_conversation_key=conversation_key,
            manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
            expected_runtime_epoch=store.runtime_epoch,
            expected_mode_generation=store.mode_generation,
        )

        assert database._conn.total_changes == changes_before == 0
        assert catalog.conversation_fingerprint == binding.conversation_fingerprint
        assert catalog.manifest_fingerprint == binding.manifest_fingerprint
        assert catalog.runtime_epoch == store.runtime_epoch
        assert catalog.mode_generation == store.mode_generation
        assert catalog.declarations == TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        assert catalog.manifest is TASK_FENCE_SELECTED_LAUNCH_MANIFEST

        selected = TASK_FENCE_SELECTED_LAUNCH_MANIFEST.routes[0]
        exact = catalog.classify_route(selected)
        assert (exact.verified, exact.reason, exact.route_id) == (
            True,
            "verified",
            selected.route_id,
        )

        unknown = catalog.classify_route(
            TaskFenceLaunchRoute(
                kind=selected.kind,
                route_id="provider:future.unreviewed",
                capability_version=selected.capability_version,
            )
        )
        assert (unknown.verified, unknown.reason) == (
            False,
            "unknown_reachable_route",
        )

        changed = catalog.classify_route(
            TaskFenceLaunchRoute(
                kind=selected.kind,
                route_id=selected.route_id,
                capability_version=selected.capability_version + ".changed",
            )
        )
        assert (changed.verified, changed.reason) == (
            False,
            "changed_reachable_route",
        )

        unsupported_declaration = next(
            declaration
            for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
            if declaration.state is TaskFenceCapabilityState.UNSUPPORTED
        )
        unsupported = catalog.classify_route(
            TaskFenceLaunchRoute(
                kind=unsupported_declaration.kind,
                route_id=unsupported_declaration.capability_id,
                capability_version=unsupported_declaration.capability_version,
            )
        )
        assert (unsupported.verified, unsupported.reason) == (
            False,
            "unsupported_reachable_route",
        )

        expected_ids = {
            route.route_id for route in TASK_FENCE_SELECTED_LAUNCH_MANIFEST.routes
        }
        extra_declaration = next(
            declaration
            for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
            if declaration.state is TaskFenceCapabilityState.SUPPORTED
            and declaration.capability_id not in expected_ids
        )
        extra = catalog.classify_route(
            TaskFenceLaunchRoute(
                kind=extra_declaration.kind,
                route_id=extra_declaration.capability_id,
                capability_version=extra_declaration.capability_version,
            )
        )
        assert (extra.verified, extra.reason) == (
            False,
            "extra_reachable_route",
        )

        with pytest.raises(FrozenInstanceError):
            catalog.runtime_epoch = 99
        with pytest.raises(TaskFenceProtocolRejected) as exc_info:
            replace(catalog, manifest_fingerprint="d" * 64)
        assert (
            exc_info.value.reason
            == "launch_manifest_fingerprint_mismatch"
        )

        database._conn.execute("BEGIN")
        try:
            with pytest.raises(TaskFenceLaunchBindingUnavailable) as exc_info:
                database.load_task_fence_selected_launch_catalog(
                    shadow_conversation_key=conversation_key,
                    manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
                    expected_runtime_epoch=store.runtime_epoch,
                    expected_mode_generation=store.mode_generation,
                )
            assert exc_info.value.reason == "launch_snapshot_unavailable"
        finally:
            database._conn.rollback()
    finally:
        database.close()


@pytest.mark.parametrize(
    ("epoch_delta", "generation_delta", "reason"),
    [
        (1, 0, "runtime_epoch_changed"),
        (0, 1, "mode_generation_changed"),
    ],
)
def test_launch_catalog_refuses_stale_control_expectations_without_dml(
    tmp_path,
    epoch_delta: int,
    generation_delta: int,
    reason: str,
) -> None:
    conversation_key = "agent:main:slack:dm:workspace:channel:user"
    path = tmp_path / "state.db"
    seed = SessionDB(path)
    store = _ready_store(seed)
    seed.materialize_task_fence_selected_launch_binding(
        shadow_conversation_key=conversation_key,
        manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )
    seed.close()

    database = SessionDB(path, read_only=True)
    try:
        with pytest.raises(TaskFenceLaunchBindingUnavailable) as exc_info:
            database.load_task_fence_selected_launch_catalog(
                shadow_conversation_key=conversation_key,
                manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
                expected_runtime_epoch=store.runtime_epoch + epoch_delta,
                expected_mode_generation=(
                    store.mode_generation + generation_delta
                ),
            )
        assert exc_info.value.reason == reason
        assert database._conn.total_changes == 0
    finally:
        database.close()


def test_launch_catalog_refuses_valid_hash_retarget_without_repair(
    tmp_path,
) -> None:
    conversation_key = "agent:main:slack:dm:workspace:channel:user"
    path = tmp_path / "state.db"
    seed = SessionDB(path)
    store = _ready_store(seed)
    seed.materialize_task_fence_selected_launch_binding(
        shadow_conversation_key=conversation_key,
        manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )
    seed._conn.execute(
        "UPDATE task_fence_cohort_launch_bindings "
        "SET conversation_fingerprint = ?",
        ("d" * 64,),
    )
    seed._conn.commit()
    row_before = tuple(
        seed._conn.execute(
            "SELECT * FROM task_fence_cohort_launch_bindings"
        ).fetchone()
    )
    seed.close()

    database = SessionDB(path, read_only=True)
    try:
        with pytest.raises(TaskFenceLaunchBindingUnavailable) as exc_info:
            database.load_task_fence_selected_launch_catalog(
                shadow_conversation_key=conversation_key,
                manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
                expected_runtime_epoch=store.runtime_epoch,
                expected_mode_generation=store.mode_generation,
            )
        row_after = tuple(
            database._conn.execute(
                "SELECT * FROM task_fence_cohort_launch_bindings"
            ).fetchone()
        )
        assert exc_info.value.reason == "launch_binding_conflict"
        assert row_after == row_before
        assert database._conn.total_changes == 0
    finally:
        database.close()


def test_launch_binding_creates_distinct_inactive_candidate_and_exact_replays(
    tmp_path,
) -> None:
    database = SessionDB(tmp_path / "state.db")
    store = _ready_store(database)

    created = database.materialize_task_fence_selected_launch_binding(
        shadow_conversation_key="agent:main:slack:dm:workspace:channel:user",
        manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )
    changes_after_create = database._conn.total_changes
    replayed = database.materialize_task_fence_selected_launch_binding(
        shadow_conversation_key="agent:main:slack:dm:workspace:channel:user",
        manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )

    assert created.created is True
    assert replayed.created is False
    assert replayed == created.__class__(
        conversation_fingerprint=created.conversation_fingerprint,
        manifest_fingerprint=created.manifest_fingerprint,
        created=False,
    )
    assert database._conn.total_changes == changes_after_create
    assert (
        created.conversation_fingerprint
        == hashlib.sha256(
            b"hermes.task_fence.conversation.v1\0"
            b"agent:main:slack:dm:workspace:channel:user"
        ).hexdigest()
    )
    assert created.manifest_fingerprint == (
        task_fence_launch_manifest_fingerprint(TASK_FENCE_SELECTED_LAUNCH_MANIFEST)
    )
    assert tuple(
        database._conn.execute(
            "SELECT mode, mode_generation, activation_state, audit_degraded "
            "FROM task_fence_cohorts WHERE cohort_key = ?",
            (_SELECTED_COHORT_KEY,),
        ).fetchone()
    ) == ("audit", 0, "inactive", 0)
    assert tuple(
        database._conn.execute(
            "SELECT conversation_fingerprint, manifest_fingerprint "
            "FROM task_fence_cohort_launch_bindings WHERE cohort_key = ?",
            (_SELECTED_COHORT_KEY,),
        ).fetchone()
    ) == (
        created.conversation_fingerprint,
        created.manifest_fingerprint,
    )
    database.close()


def test_launch_binding_requires_materialized_inventory_without_partial_create(
    tmp_path,
) -> None:
    database = SessionDB(tmp_path / "state.db")
    store = database.inspect_task_fence_store(include_counts=False)

    with pytest.raises(TaskFenceLaunchBindingUnavailable) as exc_info:
        database.materialize_task_fence_selected_launch_binding(
            shadow_conversation_key="agent:main:slack:dm:workspace:channel:user",
            manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
            expected_runtime_epoch=store.runtime_epoch,
            expected_mode_generation=store.mode_generation,
        )

    assert exc_info.value.reason == "capabilities_not_materialized"
    assert (
        database._conn.execute(
            "SELECT 1 FROM task_fence_cohorts WHERE cohort_key = ?",
            (_SELECTED_COHORT_KEY,),
        ).fetchone()
        is None
    )
    assert (
        database._conn.execute(
            "SELECT 1 FROM task_fence_cohort_launch_bindings"
        ).fetchone()
        is None
    )
    database.close()


def test_launch_binding_refuses_retarget_or_manifest_change_without_repair(
    tmp_path,
) -> None:
    database = SessionDB(tmp_path / "state.db")
    store = _ready_store(database)
    selected = TASK_FENCE_SELECTED_LAUNCH_MANIFEST
    created = database.materialize_task_fence_selected_launch_binding(
        shadow_conversation_key="agent:main:slack:dm:workspace:channel:user",
        manifest=selected,
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )
    row_before = tuple(
        database._conn.execute(
            "SELECT * FROM task_fence_cohort_launch_bindings"
        ).fetchone()
    )

    for conversation_key, manifest in (
        ("agent:main:slack:dm:workspace:other", selected),
        (
            "agent:main:slack:dm:workspace:channel:user",
            TaskFenceLaunchManifest(
                manifest_version=selected.manifest_version,
                routes=selected.routes[:-1],
            ),
        ),
    ):
        with pytest.raises(TaskFenceLaunchBindingUnavailable) as exc_info:
            database.materialize_task_fence_selected_launch_binding(
                shadow_conversation_key=conversation_key,
                manifest=manifest,
                expected_runtime_epoch=store.runtime_epoch,
                expected_mode_generation=store.mode_generation,
            )
        assert exc_info.value.reason == "launch_binding_conflict"

    assert (
        tuple(
            database._conn.execute(
                "SELECT * FROM task_fence_cohort_launch_bindings"
            ).fetchone()
        )
        == row_before
    )
    assert created.created is True
    database.close()


def test_launch_binding_inspection_rejects_valid_hash_retarget_without_repair(
    tmp_path,
) -> None:
    conversation_key = "agent:main:slack:dm:workspace:channel:user"
    database = SessionDB(tmp_path / "state.db")
    store = _ready_store(database)
    database.materialize_task_fence_selected_launch_binding(
        shadow_conversation_key=conversation_key,
        manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )
    database._conn.execute(
        "UPDATE task_fence_cohort_launch_bindings SET conversation_fingerprint = ?",
        ("d" * 64,),
    )
    database._conn.commit()
    row_before = tuple(
        database._conn.execute(
            "SELECT * FROM task_fence_cohort_launch_bindings"
        ).fetchone()
    )

    inspection = database.inspect_task_fence_selected_launch_binding(
        shadow_conversation_key=conversation_key,
    )
    with pytest.raises(TaskFenceLaunchBindingUnavailable) as exc_info:
        database.materialize_task_fence_selected_launch_binding(
            shadow_conversation_key=conversation_key,
            manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
            expected_runtime_epoch=store.runtime_epoch,
            expected_mode_generation=store.mode_generation,
        )

    assert inspection.verified is False
    assert inspection.reason == "launch_binding_conflict"
    assert exc_info.value.reason == "launch_binding_conflict"
    assert (
        tuple(
            database._conn.execute(
                "SELECT * FROM task_fence_cohort_launch_bindings"
            ).fetchone()
        )
        == row_before
    )
    database.close()


def test_launch_binding_rejects_foreign_row_without_repair(tmp_path) -> None:
    conversation_key = "agent:main:slack:dm:workspace:channel:user"
    database = SessionDB(tmp_path / "state.db")
    store = _ready_store(database)
    binding = database.materialize_task_fence_selected_launch_binding(
        shadow_conversation_key=conversation_key,
        manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )
    database._conn.execute(
        "INSERT INTO task_fence_cohorts ("
        "cohort_key, mode, mode_generation, activation_state, "
        "audit_degraded, created_at, updated_at"
        ") VALUES ('foreign-candidate', 'audit', 0, 'inactive', 0, 1.0, 1.0)"
    )
    foreign_binding = (
        "foreign-candidate",
        "e" * 64,
        binding.manifest_fingerprint,
        1.0,
    )
    with pytest.raises(sqlite3.IntegrityError):
        database._conn.execute(
            "INSERT INTO task_fence_cohort_launch_bindings VALUES (?, ?, ?, ?)",
            foreign_binding,
        )
    database._conn.execute("PRAGMA ignore_check_constraints=ON")
    try:
        database._conn.execute(
            "INSERT INTO task_fence_cohort_launch_bindings VALUES (?, ?, ?, ?)",
            foreign_binding,
        )
    finally:
        database._conn.execute("PRAGMA ignore_check_constraints=OFF")
    database._conn.commit()
    rows_before = database._conn.execute(
        "SELECT * FROM task_fence_cohort_launch_bindings ORDER BY cohort_key"
    ).fetchall()

    inspection = database.inspect_task_fence_selected_launch_binding(
        shadow_conversation_key=conversation_key,
    )
    with pytest.raises(TaskFenceLaunchBindingUnavailable) as exc_info:
        database.materialize_task_fence_selected_launch_binding(
            shadow_conversation_key=conversation_key,
            manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
            expected_runtime_epoch=store.runtime_epoch,
            expected_mode_generation=store.mode_generation,
        )

    assert inspection.verified is False
    assert inspection.reason == "launch_binding_conflict"
    assert exc_info.value.reason == "launch_binding_conflict"
    assert (
        database._conn.execute(
            "SELECT * FROM task_fence_cohort_launch_bindings ORDER BY cohort_key"
        ).fetchall()
        == rows_before
    )
    database.close()


def test_launch_binding_preserves_unsupported_reachability_without_activation(
    tmp_path,
) -> None:
    database = SessionDB(tmp_path / "state.db")
    store = _ready_store(database)
    unsupported = next(
        declaration
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        if declaration.state is TaskFenceCapabilityState.UNSUPPORTED
    )
    manifest = TaskFenceLaunchManifest(
        manifest_version="task-fence-test-unsupported-v1",
        routes=(
            TaskFenceLaunchRoute(
                kind=unsupported.kind,
                route_id=unsupported.capability_id,
                capability_version=unsupported.capability_version,
            ),
        ),
    )

    binding = database.materialize_task_fence_selected_launch_binding(
        shadow_conversation_key="agent:main:slack:dm:workspace:channel:user",
        manifest=manifest,
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )

    assert binding.created is True
    assert tuple(
        database._conn.execute(
            "SELECT mode, activation_state FROM task_fence_cohorts "
            "WHERE cohort_key = ?",
            (_SELECTED_COHORT_KEY,),
        ).fetchone()
    ) == ("audit", "inactive")
    database.close()


def test_launch_binding_fault_rolls_back_candidate_and_binding(tmp_path) -> None:
    database = SessionDB(tmp_path / "state.db")
    store = _ready_store(database)

    def deny_binding_insert(action, table, _column, _database, _trigger):
        if (
            action == sqlite3.SQLITE_INSERT
            and table == "task_fence_cohort_launch_bindings"
        ):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    database._conn.set_authorizer(deny_binding_insert)
    try:
        with pytest.raises(TaskFenceLaunchBindingUnavailable) as exc_info:
            database.materialize_task_fence_selected_launch_binding(
                shadow_conversation_key=("agent:main:slack:dm:workspace:channel:user"),
                manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
                expected_runtime_epoch=store.runtime_epoch,
                expected_mode_generation=store.mode_generation,
            )
    finally:
        database._conn.set_authorizer(None)

    assert exc_info.value.reason == "launch_binding_database_error"
    assert (
        database._conn.execute(
            "SELECT 1 FROM task_fence_cohorts WHERE cohort_key = ?",
            (_SELECTED_COHORT_KEY,),
        ).fetchone()
        is None
    )
    assert (
        database._conn.execute(
            "SELECT 1 FROM task_fence_cohort_launch_bindings"
        ).fetchone()
        is None
    )
    database.close()


def test_concurrent_exact_launch_binding_has_one_create_and_one_replay(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    seed = SessionDB(path)
    store = _ready_store(seed)
    seed.close()

    def bind() -> bool:
        database = SessionDB(path)
        try:
            result = database.materialize_task_fence_selected_launch_binding(
                shadow_conversation_key=("agent:main:slack:dm:workspace:channel:user"),
                manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
                expected_runtime_epoch=store.runtime_epoch,
                expected_mode_generation=store.mode_generation,
            )
            return result.created
        finally:
            database.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _index: bind(), range(2)))

    assert sorted(results) == [False, True]
    check = SessionDB(path)
    try:
        assert (
            check.inspect_task_fence_selected_launch_binding(
                shadow_conversation_key=("agent:main:slack:dm:workspace:channel:user"),
            ).verified
            is True
        )
        assert (
            check._conn.execute(
                "SELECT COUNT(*) FROM task_fence_cohort_launch_bindings"
            ).fetchone()[0]
            == 1
        )
    finally:
        check.close()


def test_concurrent_different_launch_selectors_have_one_winner(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    seed = SessionDB(path)
    store = _ready_store(seed)
    seed.close()
    selectors = (
        "agent:main:slack:dm:workspace:channel:user-a",
        "agent:main:slack:dm:workspace:channel:user-b",
    )

    def bind(selector: str) -> tuple[str, str]:
        database = SessionDB(path)
        try:
            result = database.materialize_task_fence_selected_launch_binding(
                shadow_conversation_key=selector,
                manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
                expected_runtime_epoch=store.runtime_epoch,
                expected_mode_generation=store.mode_generation,
            )
            return "created", result.conversation_fingerprint
        except TaskFenceLaunchBindingUnavailable as exc:
            return exc.reason, ""
        finally:
            database.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(bind, selectors))

    assert sorted(result[0] for result in results) == [
        "created",
        "launch_binding_conflict",
    ]
    winner_fingerprint = next(
        fingerprint for outcome, fingerprint in results if outcome == "created"
    )
    winner_selector = next(
        selector
        for selector, result in zip(selectors, results)
        if result[0] == "created"
    )
    check = SessionDB(path)
    try:
        inspection = check.inspect_task_fence_selected_launch_binding(
            shadow_conversation_key=winner_selector,
        )
        assert inspection.verified is True
        assert inspection.conversation_fingerprint == winner_fingerprint
    finally:
        check.close()


def test_populated_wp11_1a_v6_store_migrates_without_losing_authority(
    tmp_path,
) -> None:
    path = tmp_path / "state.db"
    envelope, original = _seed_wp11_1a_store(path)
    _downgrade_current_store_to_exact_v6(path)

    migrated = SessionDB(path)
    store = migrated.inspect_task_fence_store(include_counts=False)
    capabilities = migrated.inspect_task_fence_selected_cohort_capabilities()
    replay = migrated.accept_task_fence_ingress(envelope)
    task_version = migrated._conn.execute(
        "SELECT store_schema_version FROM task_fence_tasks WHERE task_id = ?",
        (original.task_id,),
    ).fetchone()[0]

    assert store.compatible is True
    assert store.observed_store_schema_version == 7
    assert store.runtime_epoch == 1
    assert store.tested_artifact_commit == (_MIGRATION_IDENTITY.tested_artifact_commit)
    assert store.tested_artifact_checksum == (
        _MIGRATION_IDENTITY.tested_artifact_checksum
    )
    assert store.dependency_lock_fingerprint == (
        _MIGRATION_IDENTITY.dependency_lock_fingerprint
    )
    assert capabilities.verified is True
    assert capabilities.declarations == TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    assert replay.replayed is True
    assert replay.event_id == original.event_id
    assert task_version == 7
    assert (
        migrated._conn.execute(
            "SELECT 1 FROM task_fence_cohort_launch_bindings"
        ).fetchone()
        is None
    )
    migrated.close()


def test_read_only_v6_open_never_migrates(tmp_path) -> None:
    path = tmp_path / "state.db"
    _seed_wp11_1a_store(path)
    _downgrade_current_store_to_exact_v6(path)

    reader = SessionDB(path, read_only=True)
    inspection = reader.inspect_task_fence_store(include_counts=False)
    reader.close()

    assert inspection.compatible is False
    assert inspection.observed_store_schema_version == 6
    check = sqlite3.connect(path)
    try:
        assert (
            hermes_state._read_task_fence_schema_objects(check)
            == hermes_state._expected_task_fence_v6_schema_objects()
        )
    finally:
        check.close()


@pytest.mark.parametrize("corruption", ("candidate_key", "capability"))
def test_unsafe_v6_source_is_left_untouched(tmp_path, corruption: str) -> None:
    path = tmp_path / "state.db"
    _seed_wp11_1a_store(path)
    _downgrade_current_store_to_exact_v6(path)
    connection = sqlite3.connect(path)
    if corruption == "candidate_key":
        connection.execute(
            "INSERT INTO task_fence_cohorts ("
            "cohort_key, mode, mode_generation, activation_state, "
            "audit_degraded, created_at, updated_at"
            ") VALUES (?, 'audit', 0, 'inactive', 0, 1.0, 1.0)",
            (_SELECTED_COHORT_KEY,),
        )
    else:
        declaration = TASK_FENCE_SELECTED_COHORT_CAPABILITIES[0]
        connection.execute(
            "UPDATE task_fence_cohort_capabilities "
            "SET declaration_state = CASE declaration_state "
            "WHEN 'supported' THEN 'unsupported' ELSE 'supported' END "
            "WHERE cohort_key = '__task_fence_shadow_v1__' "
            "AND capability_kind = ? AND capability_id = ? "
            "AND capability_version = ?",
            (
                declaration.kind.value,
                declaration.capability_id,
                declaration.capability_version,
            ),
        )
    connection.commit()
    connection.close()

    opened = SessionDB(path)
    inspection = opened.inspect_task_fence_store(include_counts=False)
    opened.close()

    assert inspection.compatible is False
    assert inspection.observed_store_schema_version == 6
    check = sqlite3.connect(path)
    try:
        assert (
            hermes_state._read_task_fence_schema_objects(check)
            == hermes_state._expected_task_fence_v6_schema_objects()
        )
        assert (
            check.execute(
                "SELECT store_schema_version FROM task_fence_control "
                "WHERE singleton = 1"
            ).fetchone()[0]
            == 6
        )
    finally:
        check.close()


def test_v6_migration_ddl_fault_rolls_back_exact_source(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.db"
    _seed_wp11_1a_store(path)
    _downgrade_current_store_to_exact_v6(path)
    connection = sqlite3.connect(path)
    control_before = tuple(
        connection.execute(
            "SELECT * FROM task_fence_control WHERE singleton = 1"
        ).fetchone()
    )
    connection.close()
    original = SessionDB._execute_task_fence_schema_sql

    def fail_after_extension(cursor, schema_sql=None):
        original(cursor, schema_sql)
        if schema_sql == hermes_state.TASK_FENCE_SCHEMA_V7_EXTENSION_SQL:
            raise sqlite3.DatabaseError("injected v7 extension failure")

    monkeypatch.setattr(
        SessionDB,
        "_execute_task_fence_schema_sql",
        staticmethod(fail_after_extension),
    )

    opened = SessionDB(path)
    inspection = opened.inspect_task_fence_store(include_counts=False)
    opened.close()

    assert inspection.compatible is False
    assert inspection.reason == "schema_initialization_failed"
    check = sqlite3.connect(path)
    try:
        assert (
            hermes_state._read_task_fence_schema_objects(check)
            == hermes_state._expected_task_fence_v6_schema_objects()
        )
        assert (
            check.execute(
                "SELECT store_schema_version FROM task_fence_control "
                "WHERE singleton = 1"
            ).fetchone()[0]
            == 6
        )
        assert (
            tuple(
                check.execute(
                    "SELECT * FROM task_fence_control WHERE singleton = 1"
                ).fetchone()
            )
            == control_before
        )
    finally:
        check.close()


def test_concurrent_v6_migrators_publish_only_complete_v7(tmp_path) -> None:
    path = tmp_path / "state.db"
    _seed_wp11_1a_store(path)
    _downgrade_current_store_to_exact_v6(path)

    def open_store() -> tuple[bool, int | None]:
        database = SessionDB(path)
        inspection = database.inspect_task_fence_store(include_counts=False)
        database.close()
        return inspection.compatible, inspection.observed_store_schema_version

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _index: open_store(), range(2)))

    assert results == ((True, 7), (True, 7))
    check = sqlite3.connect(path)
    try:
        assert (
            hermes_state._read_task_fence_schema_objects(check)
            == hermes_state._expected_task_fence_schema_objects()
        )
    finally:
        check.close()
