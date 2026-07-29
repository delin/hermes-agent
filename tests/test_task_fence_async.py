from contextvars import copy_context
from dataclasses import FrozenInstanceError
import hashlib
import json
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


def test_background_batch_owns_each_exact_child_launch_only(
    tmp_path,
    monkeypatch,
    _clean_async_registry,
):
    from tools import async_delegation
    import tools.delegate_tool as delegate_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
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

        release.set()
        deadline = time.monotonic() + 10.0
        while async_delegation.active_count() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert async_delegation.active_count() == 0
        assert finalized_contexts == [(None, None, False)]
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
