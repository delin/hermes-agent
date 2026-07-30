from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import FrozenInstanceError
import hashlib
import json
import logging
import queue
import shlex
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from task_fence import (
    DecisionOutcome,
    DecisionReason,
    IngressEnvelope,
    TASK_FENCE_ACTIONS,
    TaskFencePolicy,
    TaskFenceIngressRejected,
    TaskFenceIngressUnavailable,
    bind_causal_envelope,
    bind_task_fence_policy,
    current_causal_envelope,
    current_task_fence_policy,
)
from tools.registry import _task_fence_tool_fingerprint


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ingress(
    action: str,
    source_event_id: str,
    *,
    task_id: str | None = None,
) -> IngressEnvelope:
    return IngressEnvelope(
        source="gateway:test:delegation",
        source_event_id=source_event_id,
        conversation_id="delegation-conversation",
        action=TASK_FENCE_ACTIONS[action],
        payload_hash=_hash(source_event_id),
        task_id=task_id,
    )


def _live_lane(path):
    db = SessionDB(path)
    acceptance = db.accept_task_fence_ingress(
        _ingress("initial_submit", "delegation-initial")
    )
    generation = db.reserve_task_fence_generation(acceptance)
    assert db.finish_task_fence_generation(generation, state="committed")
    return db, acceptance, generation


def _context_carries_task_fence_authority() -> bool:
    return any(
        isinstance(value, (SessionDB, TaskFencePolicy))
        for _variable, value in copy_context().items()
    )


def _attempt_for_invocation(db: SessionDB, invocation_id: str):
    return db._conn.execute(
        "SELECT a.state, d.invocation_fingerprint, p.executor "
        "FROM task_fence_attempts AS a "
        "JOIN task_fence_dispatch_permits AS p ON p.permit_id = a.permit_id "
        "JOIN task_fence_policy_decisions AS d ON d.attempt_id = a.attempt_id "
        "WHERE p.invocation_envelope_id = ?",
        (invocation_id,),
    ).fetchone()


def _task_fence_dispatch_dump(db: SessionDB) -> str:
    rows = []
    for statement in (
        "SELECT * FROM task_fence_dispatch_permits",
        "SELECT * FROM task_fence_attempts",
        "SELECT * FROM task_fence_attempt_transitions",
        "SELECT * FROM task_fence_policy_decisions",
    ):
        rows.extend(tuple(row) for row in db._conn.execute(statement))
    return repr(rows)


def _task_fence_execution_counts(db: SessionDB) -> tuple[int, ...]:
    return tuple(
        db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "task_fence_execution_runs",
            "task_fence_model_generations",
            "task_fence_dispatch_permits",
            "task_fence_attempts",
        )
    )


def _task_fence_authority_projection(task) -> tuple:
    return (
        task.task_id,
        task.conversation_id,
        task.cohort_key,
        task.store_schema_version,
        task.control_protocol_version,
        task.intent_epoch,
        task.control_revision,
        task.status,
        task.active_authority_event_id,
        task.active_execution_run_id,
        task.current_generation_id,
        task.current_runtime_epoch,
        task.last_transition_event_id,
        task.created_at,
    )


def _await_completion(delegation_id: str, *, timeout: float = 10.0):
    from tools.process_registry import process_registry

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = process_registry.completion_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.02)
            continue
        if event.get("delegation_id") == delegation_id:
            return event
    raise AssertionError(f"completion not received for {delegation_id}")


def _dispatch_completed_batch(
    async_delegation,
    *,
    delegation_id: str,
    parent_generation_id=None,
    parent_runtime_epoch=None,
):
    dispatch = async_delegation.dispatch_async_delegation_batch(
        goals=["completion evidence"],
        context=None,
        toolsets=None,
        role="leaf",
        model="test/model",
        session_key="legacy-session-key",
        runner=lambda: {
            "results": [{"status": "completed", "summary": "done"}],
            "total_duration_seconds": 0.0,
        },
        max_async_children=1,
        delegation_id=delegation_id,
        _task_fence_parent_generation_id=parent_generation_id,
        _task_fence_parent_runtime_epoch=parent_runtime_epoch,
    )
    assert dispatch == {
        "status": "dispatched",
        "delegation_id": delegation_id,
    }
    return _await_completion(delegation_id)


class _FakeChild:
    def __init__(self, subagent_id: str, run_conversation):
        self._subagent_id = subagent_id
        self._parent_subagent_id = None
        self._delegate_depth = 1
        self._delegate_role = "leaf"
        self._delegate_saved_tool_names = []
        self._credential_pool = None
        self.tool_progress_callback = None
        self.model = "test/model"
        self.session_id = f"session-{subagent_id}"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self._run_conversation = run_conversation
        self.closed = False
        self.interrupted = False

    def run_conversation(self, user_message, task_id=None, stream_callback=None):
        return self._run_conversation(
            user_message=user_message,
            task_id=task_id,
            stream_callback=stream_callback,
        )

    def get_activity_summary(self):
        return {
            "api_call_count": 0,
            "max_iterations": 1,
            "current_tool": None,
            "last_activity_ts": time.monotonic(),
        }

    def interrupt(self, *_args):
        self.interrupted = True

    def close(self):
        self.closed = True


def _parent(db: SessionDB, *, active_children=None):
    return SimpleNamespace(
        _active_children=[] if active_children is None else active_children,
        _active_children_lock=threading.Lock(),
        _current_task_id="runtime-parent-task",
        _current_turn_id="turn-delegation",
        _delegate_depth=0,
        _delegate_spinner=None,
        _interrupt_requested=False,
        _memory_manager=None,
        _print_fn=None,
        _session_db=db,
        session_id="parent-session",
        tool_progress_callback=None,
    )


def _drain_completion_queue() -> None:
    from tools.process_registry import process_registry

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _spawn_gated_local_process(registry, tmp_path, *, label: str):
    marker = tmp_path / f"{label}.release"
    toxic_text = f"TASK_FENCE_PROCESS_PAYLOAD_{label}"
    source = "\n".join(
        (
            "from pathlib import Path",
            "import time",
            f"marker = Path({str(marker)!r})",
            "while not marker.exists():",
            "    time.sleep(0.01)",
            f"print({toxic_text!r})",
        )
    )
    command = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"
    )
    session = registry.spawn_local(
        command=command,
        cwd=str(tmp_path),
        task_id="runtime-process-task",
        session_key="process-owner-session",
    )
    return session, marker, toxic_text


def _await_process_completion(
    registry,
    session_id: str,
    *,
    timeout: float = 10.0,
):
    deadline = time.monotonic() + timeout
    deferred = []
    try:
        while time.monotonic() < deadline:
            try:
                event = registry.completion_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if (
                event.get("type") == "completion"
                and event.get("session_id") == session_id
            ):
                return event
            deferred.append(event)
    finally:
        for event in deferred:
            registry.completion_queue.put(event)
    raise AssertionError(f"process completion not received for {session_id}")


@pytest.fixture
def _clean_async_registry():
    from tools import async_delegation

    async_delegation._reset_for_tests()
    _drain_completion_queue()
    yield
    deadline = time.monotonic() + 5.0
    while async_delegation.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)
    async_delegation._reset_for_tests()
    _drain_completion_queue()


def test_exact_child_launch_is_started_before_run_conversation(
    tmp_path,
    monkeypatch,
):
    from tools.delegate_tool import _run_single_child

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation("tfiv_delegate_handler")
    secret_goal = "delegated-secret-must-not-be-durable"
    observed = []

    def run_conversation(*, user_message, task_id, stream_callback):
        envelope = current_causal_envelope()
        attempt = _attempt_for_invocation(db, envelope.invocation_id)
        observed.append(
            (
                user_message,
                task_id,
                stream_callback,
                envelope,
                current_task_fence_policy(),
                _context_carries_task_fence_authority(),
                tuple(attempt) if attempt else None,
            )
        )
        return {
            "final_response": "finished",
            "completed": True,
            "api_calls": 1,
        }

    child = _FakeChild("sa-task-fence-exact", run_conversation)
    parent = _parent(db, active_children=[child])
    try:
        result = _run_single_child(
            0,
            secret_goal,
            child,
            parent,
            _task_fence_parent=parent_envelope,
            _task_fence_policy=policy,
        )

        assert result["status"] == "completed"
        assert result["summary"] == "finished"
        assert len(observed) == 1
        (
            user_message,
            task_id,
            stream_callback,
            child_envelope,
            child_policy,
            carries_authority,
            attempt,
        ) = observed[0]
        assert user_message == secret_goal
        assert task_id == child._subagent_id
        assert callable(stream_callback)
        assert child_envelope.generation_id == generation.generation_id
        assert child_envelope.invocation_id != parent_envelope.invocation_id
        assert child_envelope.parent_invocation_id == parent_envelope.invocation_id
        assert child_policy is None
        assert carries_authority is False
        assert attempt == (
            "STARTED",
            _task_fence_tool_fingerprint(
                "delegate_child_launch",
                {"goal": secret_goal},
                {"task_id": child._subagent_id},
            ),
            "delegate:run_conversation",
        )
        with pytest.raises(FrozenInstanceError):
            child_envelope.task_id = "forged"
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 1
        assert secret_goal not in _task_fence_dispatch_dump(db)
    finally:
        db.close()


def test_child_launch_without_explicit_capability_preserves_legacy_identity(
    tmp_path,
):
    from tools.delegate_tool import _run_task_fence_child_launch

    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    ambient_policy = TaskFencePolicy(db)
    ambient_envelope = generation.for_invocation("tfiv_ambient_not_authority")
    result_sentinel = object()
    callback = lambda _delta: None
    observed = []

    def run_conversation(*, user_message, task_id, stream_callback):
        observed.append(
            (
                user_message,
                task_id,
                stream_callback,
                current_causal_envelope(),
                current_task_fence_policy(),
                _context_carries_task_fence_authority(),
            )
        )
        return result_sentinel

    child = _FakeChild("sa-task-fence-legacy", run_conversation)
    try:
        with (
            bind_causal_envelope(ambient_envelope),
            bind_task_fence_policy(ambient_policy),
        ):
            result = _run_task_fence_child_launch(
                child=child,
                goal="legacy goal",
                child_task_id="legacy-task-id",
                stream_callback=callback,
                parent_envelope=None,
                policy=None,
            )

        assert result is result_sentinel
        assert observed == [
            (
                "legacy goal",
                "legacy-task-id",
                callback,
                ambient_envelope,
                None,
                False,
            )
        ]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_child_launch_observation_failure_is_unprivileged_and_fail_open(
    tmp_path,
):
    from tools.delegate_tool import _run_task_fence_child_launch

    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation("tfiv_delegate_audit_failure")
    result_sentinel = object()
    calls = []

    def run_conversation(*, user_message, task_id, stream_callback):
        calls.append(
            (
                user_message,
                task_id,
                stream_callback,
                current_causal_envelope(),
                current_task_fence_policy(),
                _context_carries_task_fence_authority(),
            )
        )
        return result_sentinel

    child = _FakeChild("sa-task-fence-audit-failure", run_conversation)
    callback = lambda _delta: None
    try:
        with patch(
            "tools.registry._audit_task_fence_tool_start",
            side_effect=RuntimeError("audit unavailable"),
        ):
            result = _run_task_fence_child_launch(
                child=child,
                goal="fail-open goal",
                child_task_id=child._subagent_id,
                stream_callback=callback,
                parent_envelope=parent_envelope,
                policy=policy,
            )

        assert result is result_sentinel
        assert len(calls) == 1
        (
            user_message,
            task_id,
            stream_callback,
            child_envelope,
            child_policy,
            carries_authority,
        ) = calls[0]
        assert (user_message, task_id, stream_callback) == (
            "fail-open goal",
            child._subagent_id,
            callback,
        )
        assert child_envelope.parent_invocation_id == parent_envelope.invocation_id
        assert child_policy is None
        assert carries_authority is False
    finally:
        db.close()


def test_superseded_child_launch_is_shadow_only(tmp_path):
    from tools.delegate_tool import _run_task_fence_child_launch

    db, acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation("tfiv_delegate_race")
    original_admit = TaskFencePolicy.admit_operation
    result_sentinel = object()
    secret_goal = "race-secret-must-not-be-durable"
    calls = []
    advanced = False

    def admit_then_advance(self, envelope, operation):
        nonlocal advanced
        decision = original_admit(self, envelope, operation)
        if self is policy and not advanced:
            advanced = True
            db.accept_task_fence_ingress(
                _ingress(
                    "comment_hold",
                    "delegation-newer-input",
                    task_id=acceptance.task_id,
                )
            )
        return decision

    def run_conversation(*, user_message, task_id, stream_callback):
        calls.append(
            (
                user_message,
                task_id,
                stream_callback,
                current_causal_envelope(),
                current_task_fence_policy(),
            )
        )
        return result_sentinel

    child = _FakeChild("sa-task-fence-race", run_conversation)
    callback = lambda _delta: None
    try:
        with patch.object(
            TaskFencePolicy,
            "admit_operation",
            new=admit_then_advance,
        ):
            result = _run_task_fence_child_launch(
                child=child,
                goal=secret_goal,
                child_task_id=child._subagent_id,
                stream_callback=callback,
                parent_envelope=parent_envelope,
                policy=policy,
            )

        assert result is result_sentinel
        assert len(calls) == 1
        user_message, task_id, stream_callback, child_envelope, child_policy = calls[0]
        assert (user_message, task_id, stream_callback) == (
            secret_goal,
            child._subagent_id,
            callback,
        )
        assert child_envelope.parent_invocation_id == parent_envelope.invocation_id
        assert child_policy is None
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT decision_point, outcome, reason_code "
                "FROM task_fence_policy_decisions ORDER BY decision_order"
            )
        ] == [
            (
                "admission",
                DecisionOutcome.WOULD_RESERVE.value,
                DecisionReason.CURRENT_AUTHORITY.value,
            ),
            (
                "authorization",
                DecisionOutcome.WOULD_BLOCK.value,
                DecisionReason.PERMIT_REVOKED.value,
            ),
        ]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
        assert secret_goal not in _task_fence_dispatch_dump(db)
    finally:
        db.close()


def test_background_batch_separates_exact_launch_from_synthetic_completion(
    tmp_path,
    monkeypatch,
    _clean_async_registry,
):
    from tools import async_delegation
    import tools.delegate_tool as delegate_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation("tfiv_background_delegate_handler")
    goals = ["background-secret-alpha", "background-secret-beta"]
    observed = []
    observed_lock = threading.Lock()
    all_entered = threading.Event()
    release = threading.Event()
    finalized_contexts = []
    children = []

    def build_child(**kwargs):
        index = kwargs["task_index"]
        subagent_id = f"sa-task-fence-bg-{index}"

        def run_conversation(*, user_message, task_id, stream_callback):
            envelope = current_causal_envelope()
            attempt = _attempt_for_invocation(db, envelope.invocation_id)
            with observed_lock:
                observed.append(
                    (
                        user_message,
                        task_id,
                        stream_callback,
                        envelope,
                        current_task_fence_policy(),
                        _context_carries_task_fence_authority(),
                        tuple(attempt) if attempt else None,
                    )
                )
                if len(observed) == len(goals):
                    all_entered.set()
            assert release.wait(timeout=10.0)
            return {
                "final_response": f"done: {user_message}",
                "completed": True,
                "api_calls": 1,
            }

        child = _FakeChild(subagent_id, run_conversation)
        kwargs["parent_agent"]._active_children.append(child)
        children.append(child)
        return child

    def finalize_results(*_args, **_kwargs):
        finalized_contexts.append(
            (
                current_causal_envelope(),
                current_task_fence_policy(),
                _context_carries_task_fence_authority(),
            )
        )

    credentials = {
        "model": "test/model",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "request_overrides": None,
        "max_output_tokens": None,
        "command": None,
        "args": None,
    }
    parent = _parent(db)
    try:
        monkeypatch.setattr(delegate_tool, "_build_child_agent", build_child)
        monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
        monkeypatch.setattr(
            delegate_tool,
            "_resolve_delegation_credentials",
            lambda *_args, **_kwargs: credentials,
        )
        monkeypatch.setattr(
            delegate_tool,
            "_finalize_child_results",
            finalize_results,
        )

        with (
            bind_causal_envelope(parent_envelope),
            bind_task_fence_policy(policy),
        ):
            dispatch = json.loads(
                delegate_tool.delegate_task(
                    tasks=[{"goal": goal} for goal in goals],
                    background=True,
                    parent_agent=parent,
                )
            )

        assert dispatch["status"] == "dispatched"
        delegation_id = dispatch["delegation_id"]
        assert all_entered.wait(timeout=10.0)
        assert len(observed) == 2
        child_envelopes = [item[3] for item in observed]
        assert len({envelope.invocation_id for envelope in child_envelopes}) == 2
        assert {
            envelope.parent_invocation_id for envelope in child_envelopes
        } == {parent_envelope.invocation_id}
        assert {envelope.generation_id for envelope in child_envelopes} == {
            generation.generation_id
        }
        child_task_ids = {child._subagent_id for child in children}
        for (
            user_message,
            task_id,
            stream_callback,
            _envelope,
            child_policy,
            carries_authority,
            attempt,
        ) in observed:
            assert user_message in goals
            assert task_id in child_task_ids
            assert callable(stream_callback)
            assert child_policy is None
            assert carries_authority is False
            assert attempt is not None
            assert attempt[0] == "STARTED"
            assert attempt[2] == "delegate:run_conversation"

        durable_parent = db._conn.execute(
            "SELECT causal_parent_generation_id, causal_parent_runtime_epoch "
            "FROM async_delegations WHERE delegation_id = ?",
            (delegation_id,),
        ).fetchone()
        assert tuple(durable_parent) == (
            generation.generation_id,
            generation.runtime_epoch,
        )

        held = db.accept_task_fence_ingress(
            _ingress(
                "comment_hold",
                "delegation-late-hold",
                task_id=acceptance.task_id,
            )
        )
        held_task = db.inspect_task_fence_task(acceptance.task_id).task
        assert held_task is not None
        assert held_task.status == "paused"
        assert held_task.active_authority_event_id == held.event_id
        assert held_task.active_execution_run_id is None
        assert held_task.current_generation_id is None

        release.set()
        deadline = time.monotonic() + 10.0
        while async_delegation.active_count() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert async_delegation.active_count() == 0
        assert finalized_contexts == [(None, None, False)]
        event = _await_completion(delegation_id)
        durable = db._conn.execute(
            "SELECT dispatched_at, event_json, delivery_state, "
            "delivery_attempts, causal_parent_generation_id, "
            "causal_parent_runtime_epoch FROM async_delegations "
            "WHERE delegation_id = ?",
            (delegation_id,),
        ).fetchone()
        assert durable is not None
        assert json.loads(durable["event_json"]) == event
        assert durable["delivery_state"] == "pending"
        assert durable["delivery_attempts"] == 0
        assert durable["causal_parent_generation_id"] == generation.generation_id
        assert durable["causal_parent_runtime_epoch"] == generation.runtime_epoch
        assert "causal_parent_generation_id" not in event
        assert "causal_parent_runtime_epoch" not in event
        assert not any(key.startswith("_task_fence_") for key in event)

        execution_counts = _task_fence_execution_counts(db)
        ingress_count = db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress"
        ).fetchone()[0]
        snapshot_count = db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_acceptance_snapshots"
        ).fetchone()[0]

        claim_a = "consumer-a:wake"
        claimed, wake_acceptance = (
            async_delegation.claim_completion_delivery_with_acceptance(
                delegation_id,
                claim_a,
            )
        )
        assert claimed
        assert wake_acceptance is not None
        assert async_delegation.claim_event_delivery(event, "consumer-loser") is None

        source_identity = json.dumps(
            (delegation_id, durable["dispatched_at"]),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        expected_source_event_id = "tfadc_" + _hash(source_identity)
        synthetic = db._conn.execute(
            "SELECT i.event_id, i.source_event_id, i.conversation_id, "
            "i.task_id, i.origin, i.ingress_class, i.intent, i.execution, "
            "i.input_effect, i.correlation_kind, i.payload_hash, "
            "i.opaque_payload_ref, i.causal_parent_generation_id, "
            "i.accepted_order, s.opened_run_id, s.closed_run_id, "
            "s.task_status, s.task_active_authority_event_id, "
            "s.task_active_execution_run_id, s.task_current_generation_id "
            "FROM task_fence_ingress AS i "
            "JOIN task_fence_acceptance_snapshots AS s "
            "ON s.event_id = i.event_id "
            "WHERE i.source = 'runtime:async_delegation'"
        ).fetchall()
        assert len(synthetic) == 1
        evidence = synthetic[0]
        assert wake_acceptance.event_id == evidence["event_id"]
        assert (
            evidence["source_event_id"],
            evidence["conversation_id"],
            evidence["task_id"],
            evidence["origin"],
            evidence["ingress_class"],
            evidence["intent"],
            evidence["execution"],
            evidence["input_effect"],
            evidence["correlation_kind"],
            evidence["payload_hash"],
            evidence["opaque_payload_ref"],
            evidence["causal_parent_generation_id"],
            evidence["opened_run_id"],
            evidence["closed_run_id"],
        ) == (
            expected_source_event_id,
            "delegation-conversation",
            acceptance.task_id,
            "runtime",
            "synthetic",
            "keep",
            "none",
            "none",
            "none",
            _hash(durable["event_json"]),
            f"async-delegation:{delegation_id}:{expected_source_event_id}",
            generation.generation_id,
            None,
            None,
        )
        assert (
            evidence["task_status"],
            evidence["task_active_authority_event_id"],
            evidence["task_active_execution_run_id"],
            evidence["task_current_generation_id"],
        ) == ("paused", held.event_id, None, None)

        after_claim = db.inspect_task_fence_task(acceptance.task_id).task
        assert after_claim is not None
        assert _task_fence_authority_projection(after_claim) == (
            _task_fence_authority_projection(held_task)
        )
        assert after_claim.last_accepted_order == evidence["accepted_order"]
        assert _task_fence_execution_counts(db) == execution_counts
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress"
        ).fetchone()[0] == ingress_count + 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_acceptance_snapshots"
        ).fetchone()[0] == snapshot_count + 1

        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress_collisions"
        ).fetchone()[0] == 0
        assert _task_fence_execution_counts(db) == execution_counts
        assert async_delegation.complete_completion_delivery(
            delegation_id,
            claim_a,
        )
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 2
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 4
        durable_dump = _task_fence_dispatch_dump(db)
        assert all(goal not in durable_dump for goal in goals)
    finally:
        release.set()
        deadline = time.monotonic() + 10.0
        while async_delegation.active_count() and time.monotonic() < deadline:
            time.sleep(0.02)
        db.close()


def test_synthetic_evidence_replays_before_epoch_gate(tmp_path):
    db, acceptance, generation = _live_lane(tmp_path / "state.db")
    source_event_id = "tfadc_" + _hash("one-producer-incarnation")
    opaque_payload_ref = "async-delegation:feedbeef:one-producer-incarnation"
    common = {
        "source_event_id": source_event_id,
        "parent_generation_id": generation.generation_id,
        "parent_runtime_epoch": generation.runtime_epoch,
        "opaque_payload_ref": opaque_payload_ref,
    }
    before = db.inspect_task_fence_task(acceptance.task_id).task
    execution_counts = _task_fence_execution_counts(db)
    try:
        assert before is not None
        accepted = db.accept_task_fence_synthetic_evidence(
            **common,
            payload_hash=_hash("completion"),
        )
        assert accepted.opened_run_id is None
        assert accepted.closed_run_id is None
        assert accepted.replayed is False

        db._conn.execute(
            "UPDATE task_fence_control SET runtime_epoch = 1 WHERE singleton = 1"
        )
        replay = db.accept_task_fence_synthetic_evidence(
            **common,
            payload_hash=_hash("completion"),
        )
        assert replay.event_id == accepted.event_id
        assert replay.replayed is True

        with pytest.raises(
            TaskFenceIngressRejected,
            match="source_event_id_collision",
        ):
            db.accept_task_fence_synthetic_evidence(
                **common,
                payload_hash=_hash("mutated-completion"),
            )
        with pytest.raises(
            TaskFenceIngressUnavailable,
            match="runtime_epoch_mismatch",
        ):
            db.accept_task_fence_synthetic_evidence(
                **{
                    **common,
                    "source_event_id": "tfadc_" + _hash("stale-new-event"),
                },
                payload_hash=_hash("completion"),
            )

        after = db.inspect_task_fence_task(acceptance.task_id).task
        assert after is not None
        assert _task_fence_authority_projection(after) == (
            _task_fence_authority_projection(before)
        )
        assert _task_fence_execution_counts(db) == execution_counts
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress "
            "WHERE source = 'runtime:async_delegation'"
        ).fetchone()[0] == 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress_collisions"
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_completion_claim_never_falls_back_to_ambient_authority(
    tmp_path,
    monkeypatch,
    _clean_async_registry,
):
    from tools import async_delegation

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    ambient = generation.for_invocation("tfiv_completion_ambient_only")
    before_task = db.inspect_task_fence_task(acceptance.task_id).task
    table_names = tuple(
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name GLOB 'task_fence_*' "
            "ORDER BY name"
        )
    )
    before_counts = tuple(
        db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in table_names
    )
    cases = (
        (None, None),
        (None, generation.runtime_epoch),
        (generation.generation_id, None),
        ("unknown-parent-generation", generation.runtime_epoch),
        (generation.generation_id, generation.runtime_epoch + 1),
    )
    try:
        assert before_task is not None
        for index, (parent_generation_id, parent_runtime_epoch) in enumerate(cases):
            delegation_id = f"badf00{index:02d}"
            event = _dispatch_completed_batch(
                async_delegation,
                delegation_id=delegation_id,
                parent_generation_id=parent_generation_id,
                parent_runtime_epoch=parent_runtime_epoch,
            )
            with (
                bind_causal_envelope(ambient),
                bind_task_fence_policy(policy),
            ):
                claim = async_delegation.claim_event_delivery(
                    event,
                    f"negative-{index}",
                )
            assert claim is not None
            assert async_delegation.complete_completion_delivery(
                delegation_id,
                claim,
            )

        malformed_id = "badf0099"
        malformed_event = _dispatch_completed_batch(
            async_delegation,
            delegation_id=malformed_id,
            parent_generation_id=generation.generation_id,
            parent_runtime_epoch=generation.runtime_epoch,
        )
        db._conn.execute(
            "UPDATE async_delegations SET event_json = '{' "
            "WHERE delegation_id = ?",
            (malformed_id,),
        )
        db._conn.commit()
        malformed_claim = async_delegation.claim_event_delivery(
            malformed_event,
            "malformed-durable-event",
        )
        assert malformed_claim is not None
        assert async_delegation.complete_completion_delivery(
            malformed_id,
            malformed_claim,
        )

        after_task = db.inspect_task_fence_task(acceptance.task_id).task
        after_counts = tuple(
            db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in table_names
        )
        assert after_task == before_task
        assert after_counts == before_counts
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress "
            "WHERE source = 'runtime:async_delegation'"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_restored_completion_uses_exact_durable_evidence(
    tmp_path,
    monkeypatch,
    _clean_async_registry,
):
    from tools import async_delegation

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, acceptance, generation = _live_lane(tmp_path / "state.db")
    delegation_id = "feedbeef"
    try:
        live_event = _dispatch_completed_batch(
            async_delegation,
            delegation_id=delegation_id,
            parent_generation_id=generation.generation_id,
            parent_runtime_epoch=generation.runtime_epoch,
        )
        durable = db._conn.execute(
            "SELECT dispatched_at, event_json FROM async_delegations "
            "WHERE delegation_id = ?",
            (delegation_id,),
        ).fetchone()
        assert durable is not None
        exact_event = json.loads(durable["event_json"])
        assert live_event == exact_event
        execution_counts = _task_fence_execution_counts(db)

        async_delegation._reset_for_tests()
        restored_queue = queue.Queue()
        assert async_delegation.restore_undelivered_completions(restored_queue) == 1
        restored = restored_queue.get_nowait()
        assert restored == {**exact_event, "restored": True}
        assert "restored" not in exact_event
        assert "causal_parent_generation_id" not in restored
        assert "causal_parent_runtime_epoch" not in restored

        restored["causal_parent_generation_id"] = "forged-in-memory-parent"
        restored["causal_parent_runtime_epoch"] = generation.runtime_epoch + 99
        claim_a = async_delegation.claim_event_delivery(restored, "restore-a")
        assert claim_a is not None

        source_identity = json.dumps(
            (delegation_id, durable["dispatched_at"]),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        expected_source_event_id = "tfadc_" + _hash(source_identity)
        evidence = db._conn.execute(
            "SELECT source_event_id, task_id, payload_hash, "
            "causal_parent_generation_id FROM task_fence_ingress "
            "WHERE source = 'runtime:async_delegation'"
        ).fetchone()
        assert tuple(evidence) == (
            expected_source_event_id,
            acceptance.task_id,
            _hash(durable["event_json"]),
            generation.generation_id,
        )
        assert _task_fence_execution_counts(db) == execution_counts

        assert async_delegation.release_completion_delivery(
            delegation_id,
            claim_a,
        )
        replay_queue = queue.Queue()
        assert async_delegation.restore_undelivered_completions(replay_queue) == 1
        replay_event = replay_queue.get_nowait()
        claim_b = async_delegation.claim_event_delivery(replay_event, "restore-b")
        assert claim_b is not None and claim_b != claim_a
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress "
            "WHERE source = 'runtime:async_delegation'"
        ).fetchone()[0] == 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress_collisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT delivery_attempts FROM async_delegations "
            "WHERE delegation_id = ?",
            (delegation_id,),
        ).fetchone()[0] == 2
        assert async_delegation.complete_completion_delivery(
            delegation_id,
            claim_b,
        )
        assert async_delegation.restore_undelivered_completions(queue.Queue()) == 0
        assert _task_fence_execution_counts(db) == execution_counts
    finally:
        db.close()


@pytest.mark.parametrize(
    ("with_parent", "with_policy", "warning_count"),
    (
        (False, False, 0),
        (True, False, 1),
        (False, True, 1),
    ),
)
def test_process_parent_capture_requires_complete_context(
    tmp_path,
    caplog,
    with_parent,
    with_policy,
    warning_count,
):
    from tools.process_registry import _capture_task_fence_process_parent

    db_path = tmp_path / "state.db"
    db, _acceptance, generation = _live_lane(db_path)
    policy = TaskFencePolicy(db)
    try:
        caplog.clear()
        with (
            caplog.at_level(logging.WARNING, logger="tools.process_registry"),
            bind_causal_envelope(generation) if with_parent else nullcontext(),
            bind_task_fence_policy(policy) if with_policy else nullcontext(),
        ):
            captured = _capture_task_fence_process_parent()

        assert captured == (None, None, None)
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith(
                "Task Fence shadow process-parent capture failed:"
            )
        ]
        assert warnings == [
            "Task Fence shadow process-parent capture failed: ValueError"
        ] * warning_count
        warning_text = "\n".join(warnings)
        assert generation.generation_id not in warning_text
        assert str(db_path) not in warning_text
    finally:
        db.close()


def test_live_process_completion_records_exact_synthetic_evidence(
    tmp_path,
    monkeypatch,
    caplog,
):
    from tools import process_registry as process_registry_module

    registry_home = tmp_path / "registry-home"
    registry_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(registry_home))
    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(
        process_registry_module,
        "CHECKPOINT_PATH",
        checkpoint,
    )
    registry = process_registry_module.ProcessRegistry()
    monkeypatch.setattr(
        process_registry_module,
        "process_registry",
        registry,
    )

    owner_db_path = tmp_path / "owner-home" / "state.db"
    db, acceptance, generation = _live_lane(owner_db_path)
    policy = TaskFencePolicy(db)
    ambient_home = tmp_path / "ambient-home"
    ambient_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(ambient_home))

    session = None
    marker = None
    try:
        with (
            bind_causal_envelope(generation),
            bind_task_fence_policy(policy),
        ):
            session, marker, toxic_text = _spawn_gated_local_process(
                registry,
                tmp_path,
                label="live",
            )
        assert session._task_fence_parent_generation_id == generation.generation_id
        assert session._task_fence_parent_runtime_epoch == generation.runtime_epoch
        assert session._task_fence_store_path == str(owner_db_path.resolve())

        checkpoint_rows = json.loads(checkpoint.read_text(encoding="utf-8"))
        checkpoint_row = next(
            row for row in checkpoint_rows
            if row["session_id"] == session.id
        )
        assert not any(
            "task_fence" in key
            or "causal_parent" in key
            or key == "store_path"
            for key in checkpoint_row
        )

        session.notify_on_complete = True
        held = db.accept_task_fence_ingress(
            _ingress(
                "comment_hold",
                "process-completion-late-hold",
                task_id=acceptance.task_id,
            )
        )
        held_task = db.inspect_task_fence_task(acceptance.task_id).task
        assert held_task is not None
        assert held_task.status == "paused"
        assert held_task.active_execution_run_id is None
        assert held_task.current_generation_id is None
        execution_counts = _task_fence_execution_counts(db)
        ingress_count = db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress"
        ).fetchone()[0]

        marker.touch()
        event = _await_process_completion(registry, session.id)
        assert toxic_text in event["output"]
        assert not any(key.startswith("_task_fence_") for key in event)
        assert "causal_parent_generation_id" not in event
        assert "causal_parent_runtime_epoch" not in event
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress "
            "WHERE source = 'runtime:process_completion'"
        ).fetchone()[0] == 0

        registry.completion_queue.put(event)
        drained = registry.drain_notifications(
            session_key=session.session_key,
            owns_event=lambda candidate: (
                candidate.get("session_id") == session.id
            ),
        )
        assert len(drained) == 1
        delivered_event, synthetic_message = drained[0]
        assert delivered_event == event
        assert toxic_text in synthetic_message
        from tools.async_delegation import (
            claim_event_delivery,
            complete_event_delivery,
        )
        from tools.process_registry import (
            observe_task_fence_process_completion,
        )

        claim = claim_event_delivery(delivered_event, "process-test")
        assert claim == ""
        wake_acceptance = observe_task_fence_process_completion(
            delivered_event
        )
        assert wake_acceptance is not None
        complete_event_delivery(delivered_event, claim)

        identity = json.dumps(
            (session.id, session.started_at),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        source_event_id = "tfpc_" + _hash(identity)
        event_json = json.dumps(
            delivered_event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        evidence = db._conn.execute(
            "SELECT i.source_event_id, i.conversation_id, i.task_id, "
            "i.origin, i.ingress_class, i.intent, i.execution, "
            "i.input_effect, i.correlation_kind, i.payload_hash, "
            "i.opaque_payload_ref, i.causal_parent_generation_id, "
            "i.accepted_order, s.opened_run_id, s.closed_run_id, "
            "s.task_status, s.task_active_authority_event_id, "
            "s.task_active_execution_run_id, s.task_current_generation_id "
            "FROM task_fence_ingress AS i "
            "JOIN task_fence_acceptance_snapshots AS s "
            "ON s.event_id = i.event_id "
            "WHERE i.source = 'runtime:process_completion'"
        ).fetchone()
        assert evidence is not None
        assert wake_acceptance.event_id == db._conn.execute(
            "SELECT event_id FROM task_fence_ingress "
            "WHERE source = 'runtime:process_completion'"
        ).fetchone()[0]
        assert tuple(evidence[:12]) == (
            source_event_id,
            "delegation-conversation",
            acceptance.task_id,
            "runtime",
            "synthetic",
            "keep",
            "none",
            "none",
            "none",
            _hash(event_json),
            f"process-completion:{session.id}:{source_event_id}",
            generation.generation_id,
        )
        assert evidence["accepted_order"] == held.accepted_order + 1
        assert tuple(evidence[13:]) == (
            None,
            None,
            "paused",
            held.event_id,
            None,
            None,
        )

        after = db.inspect_task_fence_task(acceptance.task_id).task
        assert after is not None
        assert _task_fence_authority_projection(after) == (
            _task_fence_authority_projection(held_task)
        )
        assert after.last_accepted_order == evidence["accepted_order"]
        assert _task_fence_execution_counts(db) == execution_counts
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress"
        ).fetchone()[0] == ingress_count + 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress_collisions"
        ).fetchone()[0] == 0
        stored = repr(
            [
                tuple(row)
                for row in db._conn.execute(
                    "SELECT * FROM task_fence_ingress "
                    "WHERE source = 'runtime:process_completion'"
                )
            ]
        )
        assert toxic_text not in stored
        assert not (ambient_home / "state.db").exists()

        complete_event_delivery(delivered_event, "")
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress "
            "WHERE source = 'runtime:process_completion'"
        ).fetchone()[0] == 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress_collisions"
        ).fetchone()[0] == 0
        mutated_event = {
            **delivered_event,
            "output": delivered_event["output"] + "\nmutated",
        }
        with caplog.at_level(
            logging.WARNING,
            logger="tools.process_registry",
        ):
            complete_event_delivery(mutated_event, "")
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress_collisions"
        ).fetchone()[0] == 1
        assert any(
            "TaskFenceIngressRejected" in record.getMessage()
            for record in caplog.records
        )
    finally:
        if marker is not None:
            marker.touch(exist_ok=True)
        if session is not None and not session.exited:
            registry.kill_process(session.id)
        db.close()


def test_stale_pre_registration_process_notification_cannot_open_task_fence_run(
    tmp_path,
    caplog,
    monkeypatch,
):
    from tools import process_registry as process_registry_module

    runtime_home = tmp_path / "runtime-home"
    runtime_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(runtime_home))
    monkeypatch.setattr(
        process_registry_module,
        "CHECKPOINT_PATH",
        tmp_path / "processes.json",
    )
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    registry = process_registry_module.ProcessRegistry()
    monkeypatch.setattr(
        process_registry_module,
        "process_registry",
        registry,
    )
    session = process_registry_module.ProcessSession(
        id="proc_pre_registration",
        command="ignored toxic command",
        started_at=1234.5,
        exited=True,
        exit_code=0,
        completion_reason="exited",
        termination_source="",
        _task_fence_parent_generation_id=generation.generation_id,
        _task_fence_parent_runtime_epoch=generation.runtime_epoch,
        _task_fence_store_path=str((tmp_path / "state.db").resolve()),
    )
    with registry._lock:
        registry._running[session.id] = session
    event = {
        "type": "completion",
        "session_id": session.id,
        "session_key": "process-owner-session",
        "command": session.command,
        "exit_code": session.exit_code,
        "completion_reason": session.completion_reason,
        "termination_source": session.termination_source,
        "output": "ignored toxic output",
        "started_at": session.started_at,
    }
    execution_counts = _task_fence_execution_counts(db)
    try:
        db._conn.execute(
            "UPDATE task_fence_control SET runtime_epoch = ? "
            "WHERE singleton = 1",
            (generation.runtime_epoch + 1,),
        )
        with caplog.at_level(
            logging.WARNING,
            logger="tools.process_registry",
        ):
            from tools.async_delegation import complete_event_delivery

            complete_event_delivery(event, "")

        assert registry.completion_queue.empty()
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress "
            "WHERE source = 'runtime:process_completion'"
        ).fetchone()[0] == 0
        assert _task_fence_execution_counts(db) == execution_counts
        assert any(
            "TaskFenceIngressUnavailable" in record.getMessage()
            for record in caplog.records
        )
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress_collisions"
        ).fetchone()[0] == 0
    finally:
        db.close()
