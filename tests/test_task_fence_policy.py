from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
import threading

import pytest

import hermes_state
from hermes_state import SessionDB
from task_fence import (
    AttemptTerminal,
    DecisionOutcome,
    DecisionReason,
    DispatchDecision,
    IngressEnvelope,
    OperationDescriptor,
    OperationKind,
    TASK_FENCE_ACTIONS,
    TASK_FENCE_POLICY_VERSION,
    TaskFencePolicy,
    TaskFencePolicyRejected,
    TaskFencePolicyUnavailable,
    TaskFenceProtocolRejected,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ingress(
    action: str,
    source_event_id: str,
    *,
    task_id: str | None = None,
) -> IngressEnvelope:
    return IngressEnvelope(
        source="gateway:test:policy",
        source_event_id=source_event_id,
        conversation_id="policy-conversation",
        action=TASK_FENCE_ACTIONS[action],
        payload_hash=_hash(source_event_id),
        task_id=task_id,
    )


def _live_lane(
    path,
    *,
    invocation_id: str = "tfiv_policy",
    kind: OperationKind = OperationKind.TOOL,
    generation_state: str = "committed",
):
    db = SessionDB(path)
    acceptance = db.accept_task_fence_ingress(
        _ingress("initial_submit", "policy-initial")
    )
    generation = db.reserve_task_fence_generation(acceptance)
    if generation_state == "committed":
        assert db.finish_task_fence_generation(generation, state="committed")
    else:
        assert generation_state == "started"
    envelope = generation.for_invocation(invocation_id)
    operation = OperationDescriptor(
        invocation_id=invocation_id,
        kind=kind,
        adapter={
            OperationKind.MODEL: "provider:test-model",
            OperationKind.TOOL: "registry:write_file",
            OperationKind.DELIVERY: "gateway:slack:chat_post_message",
        }[kind],
        invocation_fingerprint=_hash("post-default-operation"),
    )
    return db, acceptance, envelope, operation


def _multi_input_model_lane(path, *, invocation_id: str):
    db = SessionDB(path)
    initial = db.accept_task_fence_ingress(
        _ingress("initial_submit", f"{invocation_id}-initial")
    )
    db.accept_task_fence_ingress(
        _ingress(
            "comment_hold",
            f"{invocation_id}-held",
            task_id=initial.task_id,
        )
    )
    resumed = db.accept_task_fence_ingress(
        _ingress(
            "change_and_run",
            f"{invocation_id}-resumed",
            task_id=initial.task_id,
        )
    )
    envelope = db.reserve_task_fence_generation(resumed).for_invocation(
        invocation_id
    )
    operation = OperationDescriptor(
        invocation_id=invocation_id,
        kind=OperationKind.MODEL,
        adapter="provider:test-multi-input-model",
        invocation_fingerprint=_hash(invocation_id),
    )
    return db, envelope, operation


def _count(db: SessionDB, table: str) -> int:
    assert table in {
        "task_fence_dispatch_permits",
        "task_fence_attempts",
        "task_fence_attempt_transitions",
        "task_fence_policy_decisions",
    }
    return db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _hold_next_write(
    db: SessionDB,
    entered: threading.Event,
    release: threading.Event,
    monkeypatch,
) -> None:
    original = db._execute_write
    first = True

    def held(fn):
        nonlocal first
        if not first:
            return original(fn)
        first = False

        def after_begin(conn):
            entered.set()
            assert release.wait(timeout=5)
            return fn(conn)

        return original(after_begin)

    monkeypatch.setattr(db, "_execute_write", held)


def test_operation_descriptor_and_decisions_are_closed_and_bounded():
    operation = OperationDescriptor(
        invocation_id="tfiv_closed",
        kind=OperationKind.TOOL,
        adapter="registry:terminal",
        invocation_fingerprint=_hash("operation"),
    )
    with pytest.raises(FrozenInstanceError):
        operation.adapter = "other"
    with pytest.raises(TaskFenceProtocolRejected, match="invalid_operation_kind"):
        OperationDescriptor(
            invocation_id="tfiv_closed",
            kind="tool",  # type: ignore[arg-type]
            adapter="registry:terminal",
            invocation_fingerprint=_hash("operation"),
        )
    with pytest.raises(
        TaskFenceProtocolRejected,
        match="invalid_invocation_fingerprint",
    ):
        replace(operation, invocation_fingerprint="not-a-hash")
    with pytest.raises(TaskFenceProtocolRejected, match="adapter_too_large"):
        replace(operation, adapter="x" * 513)
    with pytest.raises(TaskFenceProtocolRejected, match="permit_id_too_large"):
        DispatchDecision(
            DecisionOutcome.WOULD_RESERVE,
            DecisionReason.CURRENT_AUTHORITY,
            permit_id="x" * 513,
        )
    with pytest.raises(
        TaskFenceProtocolRejected,
        match="invalid_dispatch_decision_reason",
    ):
        DispatchDecision(
            DecisionOutcome.WOULD_ALLOW,
            DecisionReason.MISSING_PROVENANCE,
            permit_id="tfp_invalid",
            attempt_id="tfa_invalid",
        )
    with pytest.raises(
        TaskFenceProtocolRejected,
        match="invalid_dispatch_decision_reason",
    ):
        DispatchDecision(
            DecisionOutcome.WOULD_BLOCK,
            DecisionReason.CURRENT_AUTHORITY,
        )


def test_real_policy_lane_conversation_inspection_is_read_only(tmp_path):
    path = tmp_path / "state.db"
    db, acceptance, envelope, operation = _live_lane(path)
    try:
        policy = TaskFencePolicy(db)
        admitted = policy.admit_operation(envelope, operation)
        assert admitted.outcome is DecisionOutcome.WOULD_RESERVE
        started = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )
        assert started.outcome is DecisionOutcome.WOULD_ALLOW
        assert started.attempt_id is not None
    finally:
        db.close()

    before_bytes = path.read_bytes()
    before_stat = path.stat()
    read_only = SessionDB(path, read_only=True)
    try:
        inspection = read_only.inspect_task_fence_conversation(
            "policy-conversation"
        )
        unknown = read_only.inspect_task_fence_conversation(
            "unknown-policy-conversation"
        )
        assert read_only._conn.total_changes == 0
    finally:
        read_only.close()
    after_stat = path.stat()

    assert inspection.compatible is True
    assert inspection.reason == "compatible"
    assert inspection.store.compatible is True
    assert inspection.conversation_fingerprint is not None
    assert len(inspection.conversation_fingerprint) == 64
    assert inspection.task is not None
    assert inspection.task.task_id == acceptance.task_id
    assert inspection.task.status == "running"
    assert inspection.task.intent_epoch == 1
    assert inspection.task.control_revision == 1
    assert inspection.active_run is not None
    assert inspection.active_run.run_id == acceptance.opened_run_id
    assert inspection.active_run.authority_event_id == acceptance.event_id
    assert inspection.current_generation is not None
    assert inspection.current_generation.generation_id == envelope.generation_id
    assert inspection.current_generation.state == "committed"
    assert inspection.current_generation.reserved_permit_ids == ()
    assert inspection.pending_input_ids == ()
    assert inspection.pending_inputs_truncated is False
    assert len(inspection.started_attempts) == 1
    assert inspection.started_attempts[0].attempt_id == started.attempt_id
    assert inspection.started_attempts[0].run_id == acceptance.opened_run_id
    assert inspection.started_attempts_truncated is False
    assert inspection.open_incident is None
    assert inspection.cohort is not None
    assert inspection.cohort.binding == "implicit"
    assert len(inspection.cohort.cohort_fingerprint) == 64
    assert inspection.cohort.mode == "audit"
    assert inspection.cohort.mode_generation == 0
    assert inspection.cohort.activation_state == "inactive"
    assert inspection.cohort.audit_degraded is False
    assert "policy-conversation" not in repr(inspection)
    assert hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT not in repr(inspection)
    assert unknown.compatible is True
    assert unknown.reason == "no_active_task"
    assert unknown.task is None
    assert unknown.cohort is None
    assert unknown.active_run is None
    assert unknown.current_generation is None
    assert unknown.pending_input_ids == ()
    assert unknown.pending_inputs_truncated is False
    assert unknown.started_attempts == ()
    assert unknown.started_attempts_truncated is False
    assert unknown.open_incident is None
    assert path.read_bytes() == before_bytes
    assert (
        after_stat.st_dev,
        after_stat.st_ino,
        after_stat.st_size,
        after_stat.st_mtime_ns,
    ) == (
        before_stat.st_dev,
        before_stat.st_ino,
        before_stat.st_size,
        before_stat.st_mtime_ns,
    )


def test_conversation_inspection_bounds_pending_and_started_frontiers(
    tmp_path,
    monkeypatch,
):
    db, acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    started_ids = []
    pending_ids = []
    try:
        for index in range(hermes_state._TASK_FENCE_MAX_INSPECTION_ITEMS + 1):
            invocation_id = f"tfiv-inspection-bound-{index}"
            invocation = envelope.for_invocation(invocation_id)
            bounded_operation = replace(
                operation,
                invocation_id=invocation_id,
                invocation_fingerprint=_hash(invocation_id),
            )
            admitted = policy.admit_operation(invocation, bounded_operation)
            started = policy.authorize_and_start(
                invocation,
                bounded_operation,
                admitted.permit_id,
            )
            assert started.attempt_id is not None
            started_ids.append(started.attempt_id)

        inspection_work_limit = hermes_state._TASK_FENCE_MAX_INSPECTION_WORK
        monkeypatch.setattr(
            hermes_state,
            "_TASK_FENCE_MAX_INSPECTION_WORK",
            hermes_state._TASK_FENCE_MAX_INSPECTION_ITEMS,
        )
        started_work_limited = db.inspect_task_fence_conversation(
            "policy-conversation"
        )
        monkeypatch.setattr(
            hermes_state,
            "_TASK_FENCE_MAX_INSPECTION_WORK",
            inspection_work_limit,
        )
        for index in range(hermes_state._TASK_FENCE_MAX_INSPECTION_ITEMS + 1):
            held = db.accept_task_fence_ingress(
                _ingress(
                    "comment_hold",
                    f"inspection-pending-secret-{index}",
                    task_id=acceptance.task_id,
                )
            )
            pending_ids.append(held.event_id)

        expected_started = tuple(
            row[0]
            for row in db._conn.execute(
                "SELECT attempt.attempt_id "
                "FROM task_fence_attempts AS attempt "
                "JOIN task_fence_dispatch_permits AS permit "
                "ON permit.permit_id = attempt.permit_id "
                "WHERE attempt.state = 'STARTED' AND permit.task_id = ? "
                "ORDER BY attempt.started_at, attempt.attempt_id LIMIT ?",
                (
                    acceptance.task_id,
                    hermes_state._TASK_FENCE_MAX_INSPECTION_ITEMS,
                ),
            )
        )
        inspected = db.inspect_task_fence_conversation("policy-conversation")
        monkeypatch.setattr(
            hermes_state,
            "_TASK_FENCE_MAX_INSPECTION_WORK",
            hermes_state._TASK_FENCE_MAX_INSPECTION_ITEMS,
        )
        pending_work_limited = db.inspect_task_fence_conversation(
            "policy-conversation"
        )
    finally:
        db.close()

    assert inspected.compatible is True
    assert inspected.task is not None
    assert inspected.task.status == "paused"
    assert inspected.active_run is None
    assert inspected.pending_input_ids == tuple(
        pending_ids[: hermes_state._TASK_FENCE_MAX_INSPECTION_ITEMS]
    )
    assert inspected.pending_inputs_truncated is True
    assert tuple(
        attempt.attempt_id for attempt in inspected.started_attempts
    ) == expected_started
    assert len(set(started_ids)) == len(started_ids)
    assert inspected.started_attempts_truncated is True
    assert inspected.open_incident is None
    assert "inspection-pending-secret" not in repr(inspected)
    assert started_work_limited.compatible is False
    assert started_work_limited.reason == "started_attempt_work_limit_exceeded"
    assert started_work_limited.task is None
    assert started_work_limited.started_attempts == ()
    assert pending_work_limited.compatible is False
    assert pending_work_limited.reason == "pending_input_limit_exceeded"
    assert pending_work_limited.task is None
    assert pending_work_limited.pending_input_ids == ()


def test_admission_reserves_without_consuming_then_authorization_starts(tmp_path):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    try:
        admitted = policy.admit_operation(envelope, operation)
        replayed = policy.admit_operation(envelope, operation)

        assert admitted.outcome is DecisionOutcome.WOULD_RESERVE
        assert admitted.reason is DecisionReason.CURRENT_AUTHORITY
        assert replayed == admitted
        assert _count(db, "task_fence_dispatch_permits") == 1
        assert _count(db, "task_fence_attempts") == 0
        permit = db._conn.execute(
            "SELECT state, invocation_envelope_id, invocation_fingerprint, "
            "executor, tool_name, method, audience, policy_decision, "
            "policy_version, consumed_at FROM task_fence_dispatch_permits"
        ).fetchone()
        assert tuple(permit) == (
            "reserved",
            operation.invocation_id,
            hermes_state.operation_binding_fingerprint(envelope, operation),
            operation.adapter,
            None,
            None,
            "tool",
            "would_reserve:current_authority",
            TASK_FENCE_POLICY_VERSION,
            None,
        )
        assert "post-default-operation" not in "|".join(
            "" if value is None else str(value) for value in permit
        )
        reserved = db.inspect_task_fence_conversation("policy-conversation")
        assert reserved.compatible is True
        assert reserved.current_generation is not None
        assert reserved.current_generation.generation_id == envelope.generation_id
        assert reserved.current_generation.state == "committed"
        assert reserved.current_generation.reserved_permit_ids == (
            admitted.permit_id,
        )
        assert reserved.started_attempts == ()

        started = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )
        assert started.outcome is DecisionOutcome.WOULD_ALLOW
        assert started.reason is DecisionReason.CURRENT_AUTHORITY
        assert started.permit_id == admitted.permit_id
        assert started.attempt_id is not None
        assert tuple(
            db._conn.execute(
                "SELECT state, policy_decision, consumed_at IS NOT NULL "
                "FROM task_fence_dispatch_permits WHERE permit_id = ?",
                (admitted.permit_id,),
            ).fetchone()
        ) == ("consumed", "would_allow:current_authority", 1)
        assert tuple(
            db._conn.execute(
                "SELECT state, recovery_classification, handoff_ref, "
                "prepared_at = started_at "
                "FROM task_fence_attempts WHERE attempt_id = ?",
                (started.attempt_id,),
            ).fetchone()
        ) == (
            "STARTED",
            "may_effect",
            None,
            1,
        )
        assert tuple(
            db._conn.execute(
                "SELECT from_state, to_state, disposition "
                "FROM task_fence_attempt_transitions WHERE attempt_id = ?",
                (started.attempt_id,),
            ).fetchone()
        ) == (None, "STARTED", "would_allow")
        consumed = db.inspect_task_fence_conversation("policy-conversation")
        assert consumed.compatible is True
        assert consumed.current_generation is not None
        assert consumed.current_generation.reserved_permit_ids == ()
        assert tuple(
            attempt.attempt_id for attempt in consumed.started_attempts
        ) == (started.attempt_id,)
    finally:
        db.close()


def test_current_generation_reserved_permits_are_exact_bounded_and_indexed(
    tmp_path,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    permit_ids = []
    limit = hermes_state._TASK_FENCE_MAX_CURRENT_GENERATION_RESERVED_PERMITS
    try:
        for index in range(limit):
            invocation_id = f"tfiv-current-generation-permit-{index:02d}"
            invocation = envelope.for_invocation(invocation_id)
            candidate = replace(
                operation,
                invocation_id=invocation_id,
                invocation_fingerprint=_hash(invocation_id),
            )
            admitted = policy.admit_operation(invocation, candidate)
            permit_ids.append(admitted.permit_id)

        complete = db.inspect_task_fence_conversation("policy-conversation")
        explain_query = (
            "EXPLAIN QUERY PLAN "
            + hermes_state._TASK_FENCE_CURRENT_GENERATION_RESERVED_PERMITS_SQL
        )
        query_plan = tuple(
            row[-1]
            for row in db._conn.execute(
                explain_query,
                (envelope.generation_id, limit + 1),
            )
        )

        overflow_invocation_id = "tfiv-current-generation-permit-overflow"
        overflow_invocation = envelope.for_invocation(overflow_invocation_id)
        overflow_operation = replace(
            operation,
            invocation_id=overflow_invocation_id,
            invocation_fingerprint=_hash(overflow_invocation_id),
        )
        overflow_permit = policy.admit_operation(
            overflow_invocation,
            overflow_operation,
        )
        overflow = db.inspect_task_fence_conversation("policy-conversation")
    finally:
        db.close()

    assert complete.compatible is True
    assert complete.current_generation is not None
    assert complete.current_generation.reserved_permit_ids == tuple(
        sorted(permit_ids)
    )
    assert any(
        "idx_task_fence_permits_generation_state" in detail
        for detail in query_plan
    )
    assert all("USE TEMP B-TREE" not in detail for detail in query_plan)
    assert overflow.compatible is False
    assert overflow.reason == "current_generation_reserved_permit_limit_exceeded"
    assert overflow.task is None
    assert overflow.active_run is None
    assert overflow.current_generation is None
    assert overflow.pending_input_ids == ()
    assert overflow.started_attempts == ()
    assert overflow.open_incident is None
    assert overflow.terminal_open_incidents == ()
    assert overflow_permit.permit_id not in repr(overflow)


def test_current_generation_and_reserved_permit_corruption_fail_without_partial(
    tmp_path,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    try:
        original_manifest = envelope.input_manifest_hash
        db._conn.execute(
            "UPDATE task_fence_model_generations SET input_manifest_hash = ? "
            "WHERE generation_id = ?",
            (_hash("forged-generation-manifest"), envelope.generation_id),
        )
        malformed_generation = db.inspect_task_fence_conversation(
            "policy-conversation"
        )
        db._conn.execute(
            "UPDATE task_fence_model_generations SET input_manifest_hash = ? "
            "WHERE generation_id = ?",
            (original_manifest, envelope.generation_id),
        )

        admitted = policy.admit_operation(envelope, operation)
        db._conn.execute(
            "UPDATE task_fence_dispatch_permits "
            "SET control_revision = control_revision + 1 WHERE permit_id = ?",
            (admitted.permit_id,),
        )
        malformed_permit = db.inspect_task_fence_conversation(
            "policy-conversation"
        )
    finally:
        db.close()

    assert malformed_generation.compatible is False
    assert malformed_generation.reason == "incompatible_current_generation_projection"
    assert malformed_generation.task is None
    assert malformed_generation.current_generation is None
    assert malformed_permit.compatible is False
    assert (
        malformed_permit.reason
        == "incompatible_current_generation_reserved_permit_projection"
    )
    assert malformed_permit.task is None
    assert malformed_permit.active_run is None
    assert malformed_permit.current_generation is None
    assert admitted.permit_id not in repr(malformed_permit)


def test_inspection_excludes_old_generation_reserved_permits_without_expiry_policy(
    tmp_path,
):
    db, acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    try:
        admitted = policy.admit_operation(envelope, operation)
        db._conn.execute(
            "UPDATE task_fence_dispatch_permits "
            "SET reserved_at = 1.0, expires_at = 2.0 WHERE permit_id = ?",
            (admitted.permit_id,),
        )
        expired_by_clock = db.inspect_task_fence_conversation(
            "policy-conversation"
        )

        replacement = db.reserve_task_fence_generation(acceptance)
        assert db.finish_task_fence_generation(replacement, state="committed")
        superseded = db.inspect_task_fence_conversation("policy-conversation")
        old_permit_state = db._conn.execute(
            "SELECT generation_id, state FROM task_fence_dispatch_permits "
            "WHERE permit_id = ?",
            (admitted.permit_id,),
        ).fetchone()
    finally:
        db.close()

    assert expired_by_clock.compatible is True
    assert expired_by_clock.current_generation is not None
    assert expired_by_clock.current_generation.reserved_permit_ids == (
        admitted.permit_id,
    )
    assert superseded.compatible is True
    assert superseded.current_generation is not None
    assert superseded.current_generation.generation_id == replacement.generation_id
    assert superseded.current_generation.state == "committed"
    assert superseded.current_generation.reserved_permit_ids == ()
    assert tuple(old_permit_state) == (envelope.generation_id, "reserved")


def test_reserved_permit_and_started_attempt_share_one_inspection_snapshot(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "state.db"
    owner, _acceptance, envelope, operation = _live_lane(path)
    admitted = TaskFencePolicy(owner).admit_operation(envelope, operation)
    owner.close()

    reader = SessionDB(path, read_only=True)
    store_read = threading.Event()
    writer_done = threading.Event()
    writer_errors = []
    decisions = []
    original_inspect = reader._inspect_task_fence_store_unlocked

    def pause_after_store(*, include_counts):
        inspected = original_inspect(include_counts=include_counts)
        store_read.set()
        if not writer_done.wait(timeout=5):
            raise RuntimeError("concurrent permit authorization timed out")
        return inspected

    def authorize_permit():
        try:
            if not store_read.wait(timeout=5):
                raise RuntimeError("inspection did not establish its snapshot")
            writer = SessionDB(path)
            try:
                decisions.append(
                    TaskFencePolicy(writer).authorize_and_start(
                        envelope,
                        operation,
                        admitted.permit_id,
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
    writer_thread = threading.Thread(target=authorize_permit)
    writer_thread.start()
    try:
        before = reader.inspect_task_fence_conversation("policy-conversation")
        writer_thread.join(timeout=5)
        after = reader.inspect_task_fence_conversation("policy-conversation")
    finally:
        reader.close()

    assert writer_thread.is_alive() is False
    assert writer_errors == []
    assert len(decisions) == 1
    assert decisions[0].attempt_id is not None
    assert before.compatible is True
    assert before.current_generation is not None
    assert before.current_generation.reserved_permit_ids == (admitted.permit_id,)
    assert before.started_attempts == ()
    assert after.compatible is True
    assert after.current_generation is not None
    assert after.current_generation.reserved_permit_ids == ()
    assert tuple(attempt.attempt_id for attempt in after.started_attempts) == (
        decisions[0].attempt_id,
    )


def test_delivery_kind_uses_committed_generation_and_persists_exactly(tmp_path):
    db, _acceptance, envelope, operation = _live_lane(
        tmp_path / "state.db",
        kind=OperationKind.DELIVERY,
    )
    policy = TaskFencePolicy(db)
    try:
        admitted = policy.admit_operation(envelope, operation)
        started = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )

        assert admitted.outcome is DecisionOutcome.WOULD_RESERVE
        assert started.outcome is DecisionOutcome.WOULD_ALLOW
        assert tuple(
            db._conn.execute(
                "SELECT audience, executor FROM task_fence_dispatch_permits "
                "WHERE permit_id = ?",
                (admitted.permit_id,),
            ).fetchone()
        ) == ("delivery", operation.adapter)
        assert {
            row[0]
            for row in db._conn.execute(
                "SELECT DISTINCT operation_kind FROM task_fence_policy_decisions"
            )
        } == {"delivery"}
        inspection = db.inspect_task_fence_policy_decision(started.decision_id)
        assert inspection.compatible is True
        assert inspection.decision is not None
        assert inspection.decision.operation == operation
        assert inspection.decision.operation.kind is OperationKind.DELIVERY
    finally:
        db.close()


def test_policy_journal_is_deterministic_across_reserve_allow_and_consumed_replay(
    tmp_path,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    try:
        admitted = policy.admit_operation(envelope, operation)
        replayed = policy.admit_operation(envelope, operation)

        assert admitted.decision_id is not None
        assert replayed == admitted
        assert _count(db, "task_fence_policy_decisions") == 1

        started = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )
        consumed = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )

        assert started.outcome is DecisionOutcome.WOULD_ALLOW
        assert consumed.outcome is DecisionOutcome.WOULD_BLOCK
        assert consumed.reason is DecisionReason.PERMIT_ALREADY_CONSUMED
        assert all(
            decision.decision_id is not None
            for decision in (admitted, started, consumed)
        )
        assert len(
            {
                admitted.decision_id,
                started.decision_id,
                consumed.decision_id,
            }
        ) == 3
        assert tuple(
            tuple(row)
            for row in db._conn.execute(
                "SELECT decision_id, decision_point, outcome, reason_code, "
                "permit_id, attempt_id FROM task_fence_policy_decisions "
                "ORDER BY decision_order"
            )
        ) == (
            (
                admitted.decision_id,
                "admission",
                "would_reserve",
                "current_authority",
                admitted.permit_id,
                None,
            ),
            (
                started.decision_id,
                "authorization",
                "would_allow",
                "current_authority",
                admitted.permit_id,
                started.attempt_id,
            ),
            (
                consumed.decision_id,
                "authorization",
                "would_block",
                "permit_already_consumed",
                admitted.permit_id,
                None,
            ),
        )
    finally:
        db.close()


def test_policy_journal_exact_reservation_replays_after_reopen(tmp_path):
    path = tmp_path / "state.db"
    db, _acceptance, envelope, operation = _live_lane(path)
    admitted = TaskFencePolicy(db).admit_operation(envelope, operation)
    db.close()

    reopened = SessionDB(path)
    try:
        replayed = TaskFencePolicy(reopened).admit_operation(envelope, operation)

        assert replayed == admitted
        assert replayed.decision_id is not None
        assert _count(reopened, "task_fence_policy_decisions") == 1
        assert reopened._conn.execute(
            "SELECT decision_id FROM task_fence_policy_decisions"
        ).fetchone()[0] == admitted.decision_id
        inspection = reopened.inspect_task_fence_policy_decision(
            admitted.decision_id
        )
        assert inspection.compatible is True
        assert inspection.reason == "compatible"
        assert inspection.decision is not None
        assert inspection.decision.decision_id == admitted.decision_id
        assert inspection.decision.outcome is DecisionOutcome.WOULD_RESERVE
        assert inspection.decision.operation == operation
        with pytest.raises(FrozenInstanceError):
            inspection.decision.reason = DecisionReason.STALE_AUTHORITY
    finally:
        reopened.close()


def test_started_model_generation_with_bound_inputs_can_start(tmp_path):
    db, _acceptance, envelope, operation = _live_lane(
        tmp_path / "state.db",
        kind=OperationKind.MODEL,
        generation_state="started",
    )
    policy = TaskFencePolicy(db)
    try:
        assert tuple(
            db._conn.execute(
                "SELECT state FROM task_fence_task_inputs "
                "WHERE task_id = ? ORDER BY event_id",
                (envelope.task_id,),
            ).fetchone()
        ) == ("bound",)

        admitted = policy.admit_operation(envelope, operation)
        started = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )

        assert admitted.outcome is DecisionOutcome.WOULD_RESERVE
        assert started.outcome is DecisionOutcome.WOULD_ALLOW
        assert tuple(
            db._conn.execute(
                "SELECT executor, tool_name, method, audience "
                "FROM task_fence_dispatch_permits WHERE permit_id = ?",
                (admitted.permit_id,),
            ).fetchone()
        ) == (operation.adapter, None, None, "model")
        assert db._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0] == "STARTED"
    finally:
        db.close()


def test_post_commit_model_continuation_with_presented_inputs_can_start(
    tmp_path,
    monkeypatch,
):
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "policy-model-continuation")
        )
        first_generation = db.reserve_task_fence_generation(acceptance)
        assert db.finish_task_fence_generation(
            first_generation,
            state="committed",
        )
        continuation = db.reserve_task_fence_generation(acceptance).for_invocation(
            "tfiv_policy_model_continuation"
        )
        operation = OperationDescriptor(
            invocation_id=continuation.invocation_id,
            kind=OperationKind.MODEL,
            adapter="provider:test-model-continuation",
            invocation_fingerprint=_hash("post-commit-model-continuation"),
        )

        assert tuple(
            db._conn.execute(
                "SELECT state FROM task_fence_task_inputs "
                "WHERE task_id = ? ORDER BY event_id",
                (continuation.task_id,),
            ).fetchone()
        ) == ("presented",)

        policy = TaskFencePolicy(db)

        def committed_generation_scan_is_not_on_the_policy_hot_path(
            *_args,
            **_kwargs,
        ):
            raise AssertionError("unexpected global generation scan")

        monkeypatch.setattr(
            SessionDB,
            "_task_fence_run_has_committed_generation_unlocked",
            staticmethod(committed_generation_scan_is_not_on_the_policy_hot_path),
        )
        admitted = policy.admit_operation(continuation, operation)
        started = policy.authorize_and_start(
            continuation,
            operation,
            admitted.permit_id,
        )

        assert admitted.outcome is DecisionOutcome.WOULD_RESERVE
        assert started.outcome is DecisionOutcome.WOULD_ALLOW
        assert db._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0] == "STARTED"
    finally:
        db.close()


def test_model_admission_rejects_mixed_bound_and_presented_inputs(tmp_path):
    db, envelope, operation = _multi_input_model_lane(
        tmp_path / "state.db",
        invocation_id="tfiv_policy_mixed_admission",
    )
    try:
        input_rows = db._conn.execute(
            "SELECT event_id, state FROM task_fence_task_inputs "
            "WHERE task_id = ? AND bound_run_id = ? ORDER BY event_id",
            (envelope.task_id, envelope.run_id),
        ).fetchall()
        assert len(input_rows) >= 2
        assert {row["state"] for row in input_rows} == {"bound"}
        db._conn.execute(
            "UPDATE task_fence_task_inputs SET state = 'presented' "
            "WHERE event_id = ?",
            (input_rows[0]["event_id"],),
        )

        admitted = TaskFencePolicy(db).admit_operation(envelope, operation)

        assert admitted.outcome is DecisionOutcome.WOULD_BLOCK
        assert admitted.permit_id is None
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
    finally:
        db.close()


def test_model_authorization_revalidates_uniform_input_state(tmp_path):
    db, envelope, operation = _multi_input_model_lane(
        tmp_path / "state.db",
        invocation_id="tfiv_policy_mixed_authorization",
    )
    policy = TaskFencePolicy(db)
    try:
        admitted = policy.admit_operation(envelope, operation)
        assert admitted.outcome is DecisionOutcome.WOULD_RESERVE
        input_rows = db._conn.execute(
            "SELECT event_id, state FROM task_fence_task_inputs "
            "WHERE task_id = ? AND bound_run_id = ? ORDER BY event_id",
            (envelope.task_id, envelope.run_id),
        ).fetchall()
        assert len(input_rows) >= 2
        assert {row["state"] for row in input_rows} == {"bound"}
        db._conn.execute(
            "UPDATE task_fence_task_inputs SET state = 'presented' "
            "WHERE event_id = ?",
            (input_rows[0]["event_id"],),
        )

        started = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )

        assert started.outcome is DecisionOutcome.WOULD_BLOCK
        assert _count(db, "task_fence_attempts") == 0
        assert db._conn.execute(
            "SELECT state FROM task_fence_dispatch_permits WHERE permit_id = ?",
            (admitted.permit_id,),
        ).fetchone()[0] != "consumed"
    finally:
        db.close()


@pytest.mark.parametrize(
    ("kind", "generation_state"),
    (
        (OperationKind.MODEL, "committed"),
        (OperationKind.TOOL, "started"),
        (OperationKind.DELIVERY, "started"),
    ),
)
def test_operation_kind_must_match_generation_phase(
    tmp_path,
    kind,
    generation_state,
):
    db, _acceptance, envelope, operation = _live_lane(
        tmp_path / "state.db",
        kind=kind,
        generation_state=generation_state,
    )
    try:
        blocked = TaskFencePolicy(db).admit_operation(envelope, operation)

        assert blocked.outcome is DecisionOutcome.WOULD_BLOCK
        assert blocked.reason is DecisionReason.STALE_AUTHORITY
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
        assert _count(db, "task_fence_attempt_transitions") == 0
    finally:
        db.close()


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("missing", DecisionReason.MISSING_PROVENANCE),
        ("generation", DecisionReason.STALE_AUTHORITY),
        ("runtime", DecisionReason.STALE_AUTHORITY),
        ("control_runtime", DecisionReason.STALE_AUTHORITY),
        ("invocation", DecisionReason.PERMIT_OPERATION_MISMATCH),
    ),
)
def test_admission_rejects_missing_stale_or_mismatched_provenance_without_dispatch_lifecycle_rows(
    tmp_path,
    case,
    reason,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    candidate = envelope
    candidate_operation = operation
    if case == "missing":
        candidate = None
    elif case == "generation":
        candidate = replace(envelope, generation_id="tfg_stale")
    elif case == "runtime":
        candidate = replace(
            envelope,
            runtime_epoch=envelope.runtime_epoch + 1,
        )
    elif case == "control_runtime":
        db._conn.execute(
            "UPDATE task_fence_control SET runtime_epoch = runtime_epoch + 1 "
            "WHERE singleton = 1"
        )
    else:
        assert case == "invocation"
        candidate_operation = replace(
            operation,
            invocation_id="tfiv_mismatch",
        )
    try:
        blocked = policy.admit_operation(candidate, candidate_operation)

        assert blocked.outcome is DecisionOutcome.WOULD_BLOCK
        assert blocked.reason is reason
        if case == "missing":
            authorization = policy.authorize_and_start(
                None,
                candidate_operation,
                "tfp_missing",
            )
            assert authorization.outcome is DecisionOutcome.WOULD_BLOCK
            assert authorization.reason is DecisionReason.MISSING_PROVENANCE
            assert authorization.permit_id == "tfp_missing"
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
        assert _count(db, "task_fence_attempt_transitions") == 0
    finally:
        db.close()


def test_permitless_missing_and_stale_policy_decisions_are_durable(tmp_path):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    stale_envelope = replace(envelope, generation_id="tfg_stale")
    try:
        missing_admission = policy.admit_operation(None, operation)
        missing_authorization = policy.authorize_and_start(
            None,
            operation,
            "tfp_missing",
        )
        stale = policy.admit_operation(stale_envelope, operation)

        assert missing_admission.reason is DecisionReason.MISSING_PROVENANCE
        assert missing_authorization.reason is DecisionReason.MISSING_PROVENANCE
        assert stale.reason is DecisionReason.STALE_AUTHORITY
        assert all(
            decision.decision_id is not None
            for decision in (missing_admission, missing_authorization, stale)
        )
        assert _count(db, "task_fence_policy_decisions") == 3
        assert tuple(
            tuple(row)
            for row in db._conn.execute(
                "SELECT decision_id, decision_point, outcome, reason_code, "
                "candidate_task_id, candidate_generation_id, "
                "operation_invocation_id, envelope_invocation_id, "
                "causal_binding_fingerprint, permit_id, attempt_id "
                "FROM task_fence_policy_decisions ORDER BY decision_order"
            )
        ) == (
            (
                missing_admission.decision_id,
                "admission",
                "would_block",
                "missing_provenance",
                None,
                None,
                operation.invocation_id,
                None,
                None,
                None,
                None,
            ),
            (
                missing_authorization.decision_id,
                "authorization",
                "would_block",
                "missing_provenance",
                None,
                None,
                operation.invocation_id,
                None,
                None,
                "tfp_missing",
                None,
            ),
            (
                stale.decision_id,
                "admission",
                "would_block",
                "stale_authority",
                envelope.task_id,
                stale_envelope.generation_id,
                operation.invocation_id,
                envelope.invocation_id,
                hermes_state.operation_binding_fingerprint(
                    stale_envelope,
                    operation,
                ),
                None,
                None,
            ),
        )
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
        assert _count(db, "task_fence_attempt_transitions") == 0
    finally:
        db.close()


def test_admission_rejects_invocation_rebinding_without_new_dispatch_lifecycle_rows(
    tmp_path,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    try:
        admitted = policy.admit_operation(envelope, operation)
        conflict = policy.admit_operation(
            envelope,
            replace(operation, adapter="registry:other"),
        )

        assert admitted.outcome is DecisionOutcome.WOULD_RESERVE
        assert conflict.outcome is DecisionOutcome.WOULD_BLOCK
        assert conflict.reason is DecisionReason.INVOCATION_CONFLICT
        assert conflict.permit_id == admitted.permit_id
        assert _count(db, "task_fence_dispatch_permits") == 1
        assert _count(db, "task_fence_attempts") == 0
        assert _count(db, "task_fence_attempt_transitions") == 0
    finally:
        db.close()


def test_non_authority_ingress_does_not_stale_current_generation(tmp_path):
    db, acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    try:
        before = db.inspect_task_fence_task(envelope.task_id).task
        note = db.accept_task_fence_ingress(
            _ingress(
                "explicit_note",
                "policy-note",
                task_id=acceptance.task_id,
            )
        )
        synthetic = db.accept_task_fence_ingress(
            _ingress(
                "synthetic_notice",
                "policy-synthetic",
                task_id=acceptance.task_id,
            )
        )
        after = db.inspect_task_fence_task(envelope.task_id).task

        assert before is not None and after is not None
        authority_fields = (
            "intent_epoch",
            "control_revision",
            "status",
            "active_authority_event_id",
            "active_execution_run_id",
            "current_generation_id",
            "current_runtime_epoch",
        )
        assert tuple(getattr(after, field) for field in authority_fields) == tuple(
            getattr(before, field) for field in authority_fields
        )
        assert envelope.accepted_order < note.accepted_order < synthetic.accepted_order
        assert after.last_accepted_order == synthetic.accepted_order
        assert after.current_generation_id == envelope.generation_id

        admitted = policy.admit_operation(envelope, operation)
        started = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )
        assert admitted.outcome is DecisionOutcome.WOULD_RESERVE
        assert started.outcome is DecisionOutcome.WOULD_ALLOW
    finally:
        db.close()


def test_healthy_ever_enforced_audit_halts_without_dispatch_lifecycle_rows(
    tmp_path,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    db._conn.execute(
        "UPDATE task_fence_control SET ever_enforced = 1 WHERE singleton = 1"
    )
    try:
        halted = TaskFencePolicy(db).admit_operation(envelope, operation)

        assert halted.outcome is DecisionOutcome.HALT_DISPATCH
        assert halted.reason is DecisionReason.COHORT_HALTED
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
        assert _count(db, "task_fence_attempt_transitions") == 0
    finally:
        db.close()


def test_read_only_never_enforced_audit_reports_unavailable_without_halt(
    tmp_path,
):
    path = tmp_path / "state.db"
    owner, _acceptance, envelope, operation = _live_lane(path)
    read_only = SessionDB(path, read_only=True)
    try:
        blocked = TaskFencePolicy(read_only).admit_operation(
            envelope,
            operation,
        )

        assert blocked.outcome is DecisionOutcome.WOULD_BLOCK
        assert blocked.reason is DecisionReason.STORE_UNAVAILABLE
        assert blocked.decision_id is None
        assert _count(owner, "task_fence_dispatch_permits") == 0
        assert _count(owner, "task_fence_policy_decisions") == 0
        assert owner._conn.execute(
            "SELECT COUNT(*) FROM task_fence_cohorts WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        ).fetchone()[0] == 0
    finally:
        read_only.close()
        owner.close()


def test_read_only_halted_implicit_cohort_does_not_fall_back(tmp_path):
    path = tmp_path / "state.db"
    owner, _acceptance, envelope, operation = _live_lane(path)
    owner._conn.execute(
        "INSERT INTO task_fence_cohorts ("
        "cohort_key, mode, mode_generation, activation_state, "
        "audit_degraded, created_at, updated_at"
        ") VALUES (?, 'halt_dispatch', 0, 'halted', 0, 1.0, 1.0)",
        (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
    )
    read_only = SessionDB(path, read_only=True)
    try:
        halted = TaskFencePolicy(read_only).admit_operation(
            envelope,
            operation,
        )

        assert halted.outcome is DecisionOutcome.HALT_DISPATCH
        assert halted.reason is DecisionReason.STORE_UNAVAILABLE
        assert _count(owner, "task_fence_dispatch_permits") == 0
    finally:
        read_only.close()
        owner.close()


def test_closed_store_with_unknown_mode_halts(tmp_path):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    db.close()

    halted = policy.admit_operation(envelope, operation)

    assert halted.outcome is DecisionOutcome.HALT_DISPATCH
    assert halted.reason is DecisionReason.STORE_UNAVAILABLE


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        (
            lambda envelope, operation: (
                envelope,
                replace(operation, adapter="registry:other"),
            ),
            DecisionReason.PERMIT_OPERATION_MISMATCH,
        ),
        (
            lambda envelope, operation: (
                replace(envelope, generation_id="tfg_wrong_generation"),
                operation,
            ),
            DecisionReason.PERMIT_GENERATION_MISMATCH,
        ),
        (
            lambda envelope, operation: (
                replace(envelope, runtime_epoch=envelope.runtime_epoch + 1),
                operation,
            ),
            DecisionReason.PERMIT_RUNTIME_EPOCH_MISMATCH,
        ),
    ),
)
def test_permit_rejects_wrong_operation_generation_and_runtime(
    tmp_path,
    mutation,
    reason,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    try:
        admitted = policy.admit_operation(envelope, operation)
        mutated_envelope, mutated_operation = mutation(envelope, operation)
        blocked = policy.authorize_and_start(
            mutated_envelope,
            mutated_operation,
            admitted.permit_id,
        )

        assert blocked.outcome is DecisionOutcome.WOULD_BLOCK
        assert blocked.reason is reason
        assert _count(db, "task_fence_attempts") == 0
        assert db._conn.execute(
            "SELECT state FROM task_fence_dispatch_permits WHERE permit_id = ?",
            (admitted.permit_id,),
        ).fetchone()[0] == "reserved"
    finally:
        db.close()


def test_permit_is_single_use_across_independent_writers(tmp_path):
    path = tmp_path / "state.db"
    first, _acceptance, envelope, operation = _live_lane(path)
    admitted = TaskFencePolicy(first).admit_operation(envelope, operation)
    second = SessionDB(path)
    try:
        policies = (TaskFencePolicy(first), TaskFencePolicy(second))
        barrier = threading.Barrier(2)

        def authorize(policy):
            barrier.wait(timeout=5)
            return policy.authorize_and_start(
                envelope,
                operation,
                admitted.permit_id,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            decisions = tuple(pool.map(authorize, policies))

        assert {decision.outcome for decision in decisions} == {
            DecisionOutcome.WOULD_ALLOW,
            DecisionOutcome.WOULD_BLOCK,
        }
        assert {decision.reason for decision in decisions} == {
            DecisionReason.CURRENT_AUTHORITY,
            DecisionReason.PERMIT_ALREADY_CONSUMED,
        }
        assert all(decision.decision_id is not None for decision in decisions)
        assert len({decision.decision_id for decision in decisions}) == 2
        assert _count(first, "task_fence_attempts") == 1
        assert _count(first, "task_fence_attempt_transitions") == 1
        assert tuple(
            tuple(row)
            for row in first._conn.execute(
                "SELECT decision_point, outcome, reason_code, permit_id, "
                "attempt_id FROM task_fence_policy_decisions "
                "ORDER BY decision_order"
            )
        ) == (
            (
                "admission",
                "would_reserve",
                "current_authority",
                admitted.permit_id,
                None,
            ),
            (
                "authorization",
                "would_allow",
                "current_authority",
                admitted.permit_id,
                next(
                    decision.attempt_id
                    for decision in decisions
                    if decision.outcome is DecisionOutcome.WOULD_ALLOW
                ),
            ),
            (
                "authorization",
                "would_block",
                "permit_already_consumed",
                admitted.permit_id,
                None,
            ),
        )
    finally:
        second.close()
        first.close()


def test_expired_permit_is_not_consumed(tmp_path):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    db._conn.execute(
        "UPDATE task_fence_dispatch_permits "
        "SET reserved_at = 1.0, expires_at = 2.0 WHERE permit_id = ?",
        (admitted.permit_id,),
    )
    try:
        blocked = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )

        assert blocked.outcome is DecisionOutcome.WOULD_BLOCK
        assert blocked.reason is DecisionReason.PERMIT_EXPIRED
        assert tuple(
            db._conn.execute(
                "SELECT state, policy_decision, consumed_at "
                "FROM task_fence_dispatch_permits WHERE permit_id = ?",
                (admitted.permit_id,),
            ).fetchone()
        ) == ("expired", "would_block:permit_expired", None)
        assert _count(db, "task_fence_attempts") == 0
    finally:
        db.close()


@pytest.mark.parametrize(
    ("corruption", "expected_attempts"),
    (
        ("nonnumeric_expiry", 0),
        ("reserved_with_attempt", 1),
        ("consumed_without_attempt", 0),
        ("consumed_without_transition", 1),
    ),
)
def test_corrupt_permit_lifecycle_latches_audit_degraded(
    tmp_path,
    corruption,
    expected_attempts,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    permit = db._conn.execute(
        "SELECT reserved_at FROM task_fence_dispatch_permits "
        "WHERE permit_id = ?",
        (admitted.permit_id,),
    ).fetchone()
    reserved_at = permit["reserved_at"]

    if corruption == "nonnumeric_expiry":
        db._conn.execute(
            "UPDATE task_fence_dispatch_permits SET expires_at = 'corrupt' "
            "WHERE permit_id = ?",
            (admitted.permit_id,),
        )
    else:
        if corruption.startswith("consumed_"):
            db._conn.execute(
                "UPDATE task_fence_dispatch_permits "
                "SET state = 'consumed', consumed_at = ?, "
                "policy_decision = 'would_allow:current_authority' "
                "WHERE permit_id = ?",
                (reserved_at, admitted.permit_id),
            )
        if corruption in {"reserved_with_attempt", "consumed_without_transition"}:
            db._conn.execute(
                "INSERT INTO task_fence_attempts ("
                "attempt_id, permit_id, recovery_classification, state, "
                "prepared_at, started_at"
                ") VALUES ('tfa_corrupt', ?, 'may_effect', 'STARTED', ?, ?)",
                (admitted.permit_id, reserved_at, reserved_at),
            )
    try:
        degraded = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )

        assert degraded.outcome is DecisionOutcome.WOULD_BLOCK
        assert degraded.reason is DecisionReason.AUDIT_DEGRADED
        assert _count(db, "task_fence_attempts") == expected_attempts
        assert _count(db, "task_fence_attempt_transitions") == 0
        assert db._conn.execute(
            "SELECT audit_degraded FROM task_fence_cohorts WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_acceptance_first_revokes_old_permit_without_starting_attempt(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "state.db"
    dispatch_db, acceptance, envelope, operation = _live_lane(path)
    policy = TaskFencePolicy(dispatch_db)
    admitted = policy.admit_operation(envelope, operation)
    ingress_db = SessionDB(path)
    entered = threading.Event()
    second_started = threading.Event()
    release = threading.Event()
    _hold_next_write(ingress_db, entered, release, monkeypatch)

    def authorize_after_start():
        second_started.set()
        return policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            accepted_future = pool.submit(
                ingress_db.accept_task_fence_ingress,
                _ingress(
                    "comment_hold",
                    "policy-hold-first",
                    task_id=acceptance.task_id,
                ),
            )
            assert entered.wait(timeout=5)
            authorized_future = pool.submit(authorize_after_start)
            assert second_started.wait(timeout=5)
            release.set()
            held = accepted_future.result(timeout=10)
            blocked = authorized_future.result(timeout=10)

        assert held.task_projection.status == "paused"
        assert blocked.outcome is DecisionOutcome.WOULD_BLOCK
        assert blocked.reason is DecisionReason.PERMIT_REVOKED
        assert _count(dispatch_db, "task_fence_attempts") == 0
        assert tuple(
            dispatch_db._conn.execute(
                "SELECT state, policy_decision FROM task_fence_dispatch_permits "
                "WHERE permit_id = ?",
                (admitted.permit_id,),
            ).fetchone()
        ) == ("revoked", "would_block:permit_revoked")
    finally:
        ingress_db.close()
        dispatch_db.close()


def test_authorization_first_preserves_started_attempt_then_blocks_old_run(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "state.db"
    dispatch_db, acceptance, envelope, operation = _live_lane(path)
    policy = TaskFencePolicy(dispatch_db)
    admitted = policy.admit_operation(envelope, operation)
    ingress_db = SessionDB(path)
    entered = threading.Event()
    second_started = threading.Event()
    release = threading.Event()
    _hold_next_write(dispatch_db, entered, release, monkeypatch)

    def accept_after_start():
        second_started.set()
        return ingress_db.accept_task_fence_ingress(
            _ingress(
                "comment_hold",
                "policy-hold-second",
                task_id=acceptance.task_id,
            )
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            authorized_future = pool.submit(
                policy.authorize_and_start,
                envelope,
                operation,
                admitted.permit_id,
            )
            assert entered.wait(timeout=5)
            accepted_future = pool.submit(accept_after_start)
            assert second_started.wait(timeout=5)
            release.set()
            started = authorized_future.result(timeout=10)
            held = accepted_future.result(timeout=10)

        assert started.outcome is DecisionOutcome.WOULD_ALLOW
        assert held.task_projection.status == "paused"
        assert dispatch_db._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0] == "STARTED"
        assert dispatch_db._conn.execute(
            "SELECT state FROM task_fence_dispatch_permits WHERE permit_id = ?",
            (admitted.permit_id,),
        ).fetchone()[0] == "consumed"
        frontier = dispatch_db.inspect_task_fence_conversation(
            "policy-conversation"
        )
        assert frontier.compatible is True
        assert frontier.task is not None
        assert frontier.task.status == "paused"
        assert frontier.active_run is None
        assert frontier.pending_input_ids == (held.event_id,)
        assert frontier.pending_inputs_truncated is False
        assert len(frontier.started_attempts) == 1
        assert frontier.started_attempts[0].attempt_id == started.attempt_id
        assert frontier.started_attempts[0].run_id == acceptance.opened_run_id
        assert frontier.started_attempts_truncated is False
        assert frontier.open_incident is None

        later = TaskFencePolicy(dispatch_db).admit_operation(
            envelope.for_invocation("tfiv_old_retry"),
            replace(operation, invocation_id="tfiv_old_retry"),
        )
        assert later.outcome is DecisionOutcome.WOULD_BLOCK
        assert later.reason is DecisionReason.TASK_NOT_RUNNABLE
    finally:
        ingress_db.close()
        dispatch_db.close()


def test_policy_journal_fault_rolls_back_authorization_and_latches_audit_degraded(
    tmp_path,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_policy_decision_insert "
        "BEFORE INSERT ON task_fence_policy_decisions "
        "WHEN NEW.decision_point = 'authorization' BEGIN "
        "SELECT RAISE(ABORT, 'secret policy journal fault'); END"
    )
    try:
        degraded = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )

        assert degraded.outcome is DecisionOutcome.WOULD_BLOCK
        assert degraded.reason is DecisionReason.AUDIT_DEGRADED
        assert degraded.decision_id is None
        assert tuple(
            db._conn.execute(
                "SELECT state, consumed_at FROM task_fence_dispatch_permits "
                "WHERE permit_id = ?",
                (admitted.permit_id,),
            ).fetchone()
        ) == ("reserved", None)
        assert _count(db, "task_fence_attempts") == 0
        assert _count(db, "task_fence_attempt_transitions") == 0
        assert _count(db, "task_fence_policy_decisions") == 1
        assert db._conn.execute(
            "SELECT decision_id FROM task_fence_policy_decisions"
        ).fetchone()[0] == admitted.decision_id
        assert db._conn.execute(
            "SELECT audit_degraded FROM task_fence_cohorts WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        ).fetchone()[0] == 1

        db._conn.execute("DROP TRIGGER fail_policy_decision_insert")
        started = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )
        assert started.outcome is DecisionOutcome.WOULD_ALLOW
        assert started.decision_id is not None
        assert _count(db, "task_fence_policy_decisions") == 2
    finally:
        db.close()


def test_policy_journal_fault_rolls_back_reservation_and_latches_audit_degraded(
    tmp_path,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_policy_decision_insert "
        "BEFORE INSERT ON task_fence_policy_decisions BEGIN "
        "SELECT RAISE(ABORT, 'secret policy journal fault'); END"
    )
    try:
        degraded = TaskFencePolicy(db).admit_operation(envelope, operation)

        assert degraded.outcome is DecisionOutcome.WOULD_BLOCK
        assert degraded.reason is DecisionReason.AUDIT_DEGRADED
        assert degraded.decision_id is None
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
        assert _count(db, "task_fence_attempt_transitions") == 0
        assert _count(db, "task_fence_policy_decisions") == 0
        assert db._conn.execute(
            "SELECT audit_degraded FROM task_fence_cohorts WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_missing_provenance_journal_fault_preserves_reason_and_latches_degraded(
    tmp_path,
):
    db, _acceptance, _envelope, operation = _live_lane(tmp_path / "state.db")
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_missing_policy_decision_insert "
        "BEFORE INSERT ON task_fence_policy_decisions BEGIN "
        "SELECT RAISE(ABORT, 'secret missing journal fault'); END"
    )
    policy = TaskFencePolicy(db)
    try:
        admission = policy.admit_operation(None, operation)
        authorization = policy.authorize_and_start(
            None,
            operation,
            "tfp_missing",
        )

        assert admission.reason is DecisionReason.MISSING_PROVENANCE
        assert authorization.reason is DecisionReason.MISSING_PROVENANCE
        assert admission.decision_id is None
        assert authorization.decision_id is None
        assert _count(db, "task_fence_policy_decisions") == 0
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert db._conn.execute(
            "SELECT audit_degraded FROM task_fence_cohorts WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_missing_provenance_journal_fault_latches_permit_cohort(tmp_path):
    db, acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    db._conn.execute(
        "INSERT INTO task_fence_cohorts ("
        "cohort_key, mode, mode_generation, activation_state, "
        "audit_degraded, created_at, updated_at"
        ") VALUES ('explicit-review', 'audit', 0, 'inactive', 0, 1.0, 1.0)"
    )
    db._conn.execute(
        "UPDATE task_fence_tasks SET cohort_key = 'explicit-review' WHERE task_id = ?",
        (acceptance.task_projection.task_id,),
    )
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_missing_policy_decision_insert "
        "BEFORE INSERT ON task_fence_policy_decisions BEGIN "
        "SELECT RAISE(ABORT, 'secret missing journal fault'); END"
    )
    try:
        authorization = policy.authorize_and_start(
            None,
            operation,
            admitted.permit_id,
        )

        assert authorization.reason is DecisionReason.MISSING_PROVENANCE
        assert authorization.decision_id is None
        assert tuple(
            tuple(row)
            for row in db._conn.execute(
                "SELECT cohort_key, audit_degraded FROM task_fence_cohorts "
                "WHERE cohort_key IN ('explicit-review', ?) ORDER BY cohort_key",
                (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
            )
        ) == (
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT, 0),
            ("explicit-review", 1),
        )
    finally:
        db.close()


def test_late_authorization_fault_rolls_back_and_latches_audit_degraded(
    tmp_path,
):
    path = tmp_path / "state.db"
    db, _acceptance, envelope, operation = _live_lane(path)
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    db._conn.execute(
        "INSERT INTO task_fence_cohorts ("
        "cohort_key, mode, mode_generation, activation_state, "
        "audit_degraded, created_at, updated_at"
        ") VALUES ('unrelated-shadow', 'audit', 0, 'inactive', 0, 1.0, 1.0)"
    )
    db._conn.commit()
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_policy_started_transition "
        "BEFORE INSERT ON task_fence_attempt_transitions "
        "WHEN NEW.to_state = 'STARTED' BEGIN "
        "SELECT RAISE(ABORT, 'secret injected policy fault'); END"
    )
    try:
        degraded = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )

        assert degraded.outcome is DecisionOutcome.WOULD_BLOCK
        assert degraded.reason is DecisionReason.AUDIT_DEGRADED
        assert "secret" not in degraded.reason.value
        assert db._conn.execute(
            "SELECT state FROM task_fence_dispatch_permits WHERE permit_id = ?",
            (admitted.permit_id,),
        ).fetchone()[0] == "reserved"
        assert _count(db, "task_fence_attempts") == 0
        assert _count(db, "task_fence_attempt_transitions") == 0
        assert tuple(
            db._conn.execute(
                "SELECT mode, activation_state, audit_degraded "
                "FROM task_fence_cohorts WHERE cohort_key = ?",
                (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
            ).fetchone()
        ) == ("audit", "inactive", 1)
        assert db._conn.execute(
            "SELECT audit_degraded FROM task_fence_cohorts "
            "WHERE cohort_key = 'unrelated-shadow'"
        ).fetchone()[0] == 0

        db._conn.execute("DROP TRIGGER fail_policy_started_transition")
        started = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )
        assert started.outcome is DecisionOutcome.WOULD_ALLOW
        assert db._conn.execute(
            "SELECT audit_degraded FROM task_fence_cohorts WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        ).fetchone()[0] == 1
    finally:
        db.close()

    reopened = SessionDB(path)
    try:
        assert reopened._conn.execute(
            "SELECT audit_degraded FROM task_fence_cohorts WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        ).fetchone()[0] == 1
    finally:
        reopened.close()


@pytest.mark.parametrize("terminal", tuple(AttemptTerminal))
def test_terminal_recording_is_exact_idempotent_and_survives_superseding_ingress(
    tmp_path,
    terminal,
):
    db, acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    started = policy.authorize_and_start(
        envelope,
        operation,
        admitted.permit_id,
    )
    try:
        db.accept_task_fence_ingress(
            _ingress(
                "comment_hold",
                "policy-finish-after-hold",
                task_id=acceptance.task_id,
            )
        )
        evidence = "adapter:evidence-1"
        policy.finish_attempt(started.attempt_id, terminal, evidence)
        policy.finish_attempt(started.attempt_id, terminal, evidence)

        assert tuple(
            db._conn.execute(
                "SELECT state, disposition, acknowledgement_ref, "
                "terminal_at IS NOT NULL FROM task_fence_attempts "
                "WHERE attempt_id = ?",
                (started.attempt_id,),
            ).fetchone()
        ) == (terminal.value, terminal.value, evidence, 1)
        assert tuple(
            tuple(row)
            for row in db._conn.execute(
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
                    envelope,
                    operation,
                ),
            ),
            ("STARTED", terminal.value, evidence),
        )
        with pytest.raises(
            TaskFencePolicyRejected,
            match="attempt_terminal_conflict",
        ):
            policy.finish_attempt(
                started.attempt_id,
                terminal,
                "adapter:different-evidence",
            )
    finally:
        db.close()


@pytest.mark.parametrize(
    "terminal",
    (AttemptTerminal.SUCCEEDED, AttemptTerminal.FAILED_DEFINITE),
)
def test_definitive_terminal_requires_evidence(tmp_path, terminal):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    started = policy.authorize_and_start(
        envelope,
        operation,
        admitted.permit_id,
    )
    try:
        with pytest.raises(
            TaskFencePolicyRejected,
            match="missing_terminal_evidence",
        ):
            policy.finish_attempt(started.attempt_id, terminal, None)

        assert db._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0] == "STARTED"
        assert _count(db, "task_fence_attempt_transitions") == 1
    finally:
        db.close()


def test_terminal_fault_rolls_back_and_latches_only_target_cohort(tmp_path):
    path = tmp_path / "state.db"
    db, _acceptance, envelope, operation = _live_lane(path)
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    db._conn.execute(
        "INSERT INTO task_fence_cohorts ("
        "cohort_key, mode, mode_generation, activation_state, "
        "audit_degraded, created_at, updated_at"
        ") VALUES ('unrelated-shadow', 'audit', 0, 'inactive', 0, 1.0, 1.0)"
    )
    started = policy.authorize_and_start(
        envelope,
        operation,
        admitted.permit_id,
    )
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_policy_terminal_transition "
        "BEFORE INSERT ON task_fence_attempt_transitions "
        "WHEN NEW.from_state = 'STARTED' BEGIN "
        "SELECT RAISE(ABORT, 'secret terminal fault'); END"
    )
    try:
        with pytest.raises(
            TaskFencePolicyUnavailable,
            match="audit_degraded",
        ) as raised:
            policy.finish_attempt(
                started.attempt_id,
                AttemptTerminal.SUCCEEDED,
                "adapter:ack-1",
            )

        assert raised.value.reason == "audit_degraded"
        assert raised.value.__cause__ is None
        assert "secret" not in str(raised.value)
        assert db._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0] == "STARTED"
        assert _count(db, "task_fence_attempt_transitions") == 1
        assert tuple(
            tuple(row)
            for row in db._conn.execute(
                "SELECT cohort_key, audit_degraded FROM task_fence_cohorts "
                "WHERE cohort_key IN (?, 'unrelated-shadow') ORDER BY cohort_key",
                (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
            )
        ) == (
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT, 1),
            ("unrelated-shadow", 0),
        )
    finally:
        db.close()

    reopened = SessionDB(path)
    try:
        assert tuple(
            tuple(row)
            for row in reopened._conn.execute(
                "SELECT cohort_key, audit_degraded FROM task_fence_cohorts "
                "WHERE cohort_key IN (?, 'unrelated-shadow') ORDER BY cohort_key",
                (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
            )
        ) == (
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT, 1),
            ("unrelated-shadow", 0),
        )
    finally:
        reopened.close()


def test_corrupt_attempt_transition_chain_is_not_accepted_as_terminal(
    tmp_path,
):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    started = policy.authorize_and_start(
        envelope,
        operation,
        admitted.permit_id,
    )
    db._conn.execute(
        "INSERT INTO task_fence_attempt_transitions ("
        "attempt_id, from_state, to_state, disposition, transitioned_at"
        ") VALUES (?, 'STARTED', 'OUTCOME_UNKNOWN', 'OUTCOME_UNKNOWN', 1.0)",
        (started.attempt_id,),
    )
    try:
        with pytest.raises(TaskFencePolicyUnavailable, match="audit_degraded"):
            policy.finish_attempt(
                started.attempt_id,
                AttemptTerminal.SUCCEEDED,
                "adapter:ack-1",
            )

        assert db._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0] == "STARTED"
        assert _count(db, "task_fence_attempt_transitions") == 2
        assert db._conn.execute(
            "SELECT audit_degraded FROM task_fence_cohorts WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        ).fetchone()[0] == 1
    finally:
        db.close()


@pytest.mark.parametrize(
    "corruption",
    ("permit_state", "binding_fingerprint", "consumed_timestamp"),
)
def test_terminal_rejects_corrupt_linked_permit(tmp_path, corruption):
    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    started = policy.authorize_and_start(
        envelope,
        operation,
        admitted.permit_id,
    )
    if corruption == "permit_state":
        db._conn.execute(
            "UPDATE task_fence_dispatch_permits "
            "SET state = 'reserved', consumed_at = NULL, "
            "policy_decision = 'would_reserve:current_authority' "
            "WHERE permit_id = ?",
            (admitted.permit_id,),
        )
    elif corruption == "binding_fingerprint":
        db._conn.execute(
            "UPDATE task_fence_dispatch_permits "
            "SET invocation_fingerprint = ? WHERE permit_id = ?",
            (_hash("corrupt-binding"), admitted.permit_id),
        )
    else:
        assert corruption == "consumed_timestamp"
        db._conn.execute(
            "UPDATE task_fence_dispatch_permits "
            "SET consumed_at = consumed_at + 1.0 WHERE permit_id = ?",
            (admitted.permit_id,),
        )
    try:
        with pytest.raises(TaskFencePolicyUnavailable, match="audit_degraded"):
            policy.finish_attempt(
                started.attempt_id,
                AttemptTerminal.SUCCEEDED,
                "adapter:ack-1",
            )

        assert db._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0] == "STARTED"
        assert _count(db, "task_fence_attempt_transitions") == 1
        assert db._conn.execute(
            "SELECT audit_degraded FROM task_fence_cohorts WHERE cohort_key = ?",
            (hermes_state._TASK_FENCE_IMPLICIT_AUDIT_COHORT,),
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_started_attempt_survives_reopen_as_unknown_candidate(tmp_path):
    path = tmp_path / "state.db"
    db, _acceptance, envelope, operation = _live_lane(path)
    admitted = TaskFencePolicy(db).admit_operation(envelope, operation)
    started = TaskFencePolicy(db).authorize_and_start(
        envelope,
        operation,
        admitted.permit_id,
    )
    db.close()

    reopened = SessionDB(path)
    policy = TaskFencePolicy(reopened)
    try:
        assert reopened._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0] == "STARTED"
        reused = policy.authorize_and_start(
            envelope,
            operation,
            admitted.permit_id,
        )
        assert reused.reason is DecisionReason.PERMIT_ALREADY_CONSUMED
        assert _count(reopened, "task_fence_attempts") == 1

        policy.finish_attempt(
            started.attempt_id,
            AttemptTerminal.OUTCOME_UNKNOWN,
            None,
        )
        assert reopened._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (started.attempt_id,),
        ).fetchone()[0] == "OUTCOME_UNKNOWN"
    finally:
        reopened.close()


def test_shadow_block_does_not_change_legacy_middleware_result(
    tmp_path,
    monkeypatch,
):
    from hermes_cli import middleware

    db, acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    db.accept_task_fence_ingress(
        _ingress(
            "comment_hold",
            "policy-legacy-hold",
            task_id=acceptance.task_id,
        )
    )
    try:
        decision = policy.admit_operation(envelope, operation)
        assert decision.outcome is DecisionOutcome.WOULD_BLOCK
        assert _count(db, "task_fence_attempts") == 0

        monkeypatch.setattr(middleware, "_get_middleware_callbacks", lambda _kind: [])
        handled = []
        result = middleware.run_tool_execution_middleware(
            "write_file",
            {"path": "artifact.txt", "content": "sanitized"},
            lambda args: handled.append(args) or "legacy-result",
        )
        assert result == "legacy-result"
        assert handled == [{"path": "artifact.txt", "content": "sanitized"}]
    finally:
        db.close()


def test_shadow_reservation_does_not_mask_legacy_handler_exception(
    tmp_path,
    monkeypatch,
):
    from hermes_cli import middleware

    db, _acceptance, envelope, operation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(envelope, operation)
    assert admitted.outcome is DecisionOutcome.WOULD_RESERVE
    monkeypatch.setattr(middleware, "_get_middleware_callbacks", lambda _kind: [])
    handler_calls = []

    class LegacyRejected(RuntimeError):
        pass

    def reject(args):
        handler_calls.append(args)
        raise LegacyRejected("legacy rejected")

    try:
        with pytest.raises(LegacyRejected, match="legacy rejected"):
            middleware.run_tool_execution_middleware(
                "write_file",
                {"path": "artifact.txt"},
                reject,
            )

        assert handler_calls == [{"path": "artifact.txt"}]
        assert db._conn.execute(
            "SELECT state FROM task_fence_dispatch_permits WHERE permit_id = ?",
            (admitted.permit_id,),
        ).fetchone()[0] == "reserved"
        assert _count(db, "task_fence_attempts") == 0
    finally:
        db.close()
