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
    TaskFenceArtifactIdentity,
    TaskFencePolicy,
    TaskFenceIngressRejected,
    TaskFenceIngressUnavailable,
    TaskFenceRecoveryUnavailable,
    bind_causal_envelope,
    bind_task_fence_policy,
    current_causal_envelope,
    current_task_fence_policy,
)
from tools.registry import _task_fence_tool_fingerprint


_SHADOW_SESSION_KEY = "slack:workspace:channel:user"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact_identity() -> TaskFenceArtifactIdentity:
    return TaskFenceArtifactIdentity(
        tested_artifact_commit="d" * 40,
        tested_artifact_checksum="sha256:" + "e" * 64,
        dependency_lock_fingerprint="sha256:" + "f" * 64,
    )


def _async_rows(db: SessionDB) -> tuple[tuple, ...]:
    return tuple(
        tuple(row)
        for row in db._conn.execute(
            "SELECT * FROM async_delegations ORDER BY delegation_id"
        )
    )


def _task_fence_rows(db: SessionDB) -> tuple:
    tables = tuple(
        row[0]
        for row in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name GLOB 'task_fence_*' ORDER BY name"
        )
    )
    rows = tuple(
        (
            table,
            tuple(
                tuple(row)
                for row in db._conn.execute(f"SELECT * FROM {table} ORDER BY rowid")
            ),
        )
        for table in tables
    )
    sequence = tuple(
        tuple(row)
        for row in db._conn.execute(
            "SELECT name, seq FROM sqlite_sequence "
            "WHERE name GLOB 'task_fence_*' ORDER BY name"
        )
    )
    return rows, sequence


def _async_restore_invocation(delegation_id: str, dispatched_at: float) -> str:
    identity = json.dumps(
        (delegation_id, float(dispatched_at)),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "tfqr_" + _hash(identity)


def _ingress(
    action: str,
    source_event_id: str,
    *,
    task_id: str | None = None,
    conversation_id: str = "delegation-conversation",
) -> IngressEnvelope:
    return IngressEnvelope(
        source="gateway:test:delegation",
        source_event_id=source_event_id,
        conversation_id=conversation_id,
        action=TASK_FENCE_ACTIONS[action],
        payload_hash=_hash(source_event_id),
        task_id=task_id,
    )


def _live_lane(path, *, conversation_id: str = "delegation-conversation"):
    db = SessionDB(path)
    acceptance = db.accept_task_fence_ingress(
        _ingress(
            "initial_submit",
            "delegation-initial",
            conversation_id=conversation_id,
        )
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
    session_key: str = "legacy-session-key",
):
    dispatch = async_delegation.dispatch_async_delegation_batch(
        goals=["completion evidence"],
        context=None,
        toolsets=None,
        role="leaf",
        model="test/model",
        session_key=session_key,
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
        terminal = db._conn.execute(
            "SELECT a.attempt_id, a.state, a.acknowledgement_ref, "
            "a.terminal_at IS NOT NULL "
            "FROM task_fence_attempts AS a "
            "JOIN task_fence_dispatch_permits AS p "
            "ON p.permit_id = a.permit_id "
            "WHERE p.invocation_envelope_id = ?",
            (child_envelope.invocation_id,),
        ).fetchone()
        evidence = f"delegate:run_conversation:completed:v1:{terminal['attempt_id']}"
        assert tuple(terminal[1:4]) == ("SUCCEEDED", evidence, 1)
        transitions = [
            tuple(row)
            for row in db._conn.execute(
                "SELECT from_state, to_state, disposition, evidence_ref "
                "FROM task_fence_attempt_transitions WHERE attempt_id = ? "
                "ORDER BY transition_order",
                (terminal["attempt_id"],),
            )
        ]
        assert len(transitions) == 2
        assert transitions[0][:3] == (None, "STARTED", "would_allow")
        assert len(transitions[0][3]) == 64
        assert set(transitions[0][3]) <= set("0123456789abcdef")
        assert transitions[1] == (
            "STARTED",
            "SUCCEEDED",
            "SUCCEEDED",
            evidence,
        )
        assert secret_goal not in _task_fence_dispatch_dump(db)
    finally:
        db.close()


@pytest.mark.parametrize(
    "result_sentinel",
    (
        {"completed": False, "final_response": "not complete"},
        {"completed": 1, "final_response": "not an exact boolean"},
        {"final_response": "missing acknowledgement"},
    ),
)
def test_child_launch_requires_exact_completed_acknowledgement(
    tmp_path,
    monkeypatch,
    result_sentinel,
):
    from tools.delegate_tool import _run_single_child

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation("tfiv_delegate_non_success")
    calls = []

    def run_conversation(**kwargs):
        calls.append(kwargs)
        return result_sentinel

    child = _FakeChild("sa-task-fence-non-success", run_conversation)
    parent = _parent(db, active_children=[child])
    try:
        result = _run_single_child(
            0,
            "conservative terminal evidence",
            child,
            parent,
            _task_fence_parent=parent_envelope,
            _task_fence_policy=policy,
        )

        assert result["status"] == "completed"
        assert result["summary"] == result_sentinel["final_response"]
        assert len(calls) == 1
        assert calls[0]["user_message"] == "conservative terminal evidence"
        assert calls[0]["task_id"] == child._subagent_id
        assert callable(calls[0]["stream_callback"])
        assert child.closed
        assert tuple(
            db._conn.execute(
                "SELECT state, acknowledgement_ref, terminal_at "
                "FROM task_fence_attempts"
            ).fetchone()
        ) == ("STARTED", None, None)
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_attempt_transitions"
            ).fetchone()[0]
            == 1
        )
    finally:
        db.close()


def test_child_launch_exception_is_ambiguous_and_preserves_identity(tmp_path):
    from tools.delegate_tool import _run_task_fence_child_launch

    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation("tfiv_delegate_exception")
    expected = ConnectionError("ambiguous child connection")
    calls = 0

    def run_conversation(**_kwargs):
        nonlocal calls
        calls += 1
        raise expected

    child = _FakeChild("sa-task-fence-exception", run_conversation)
    try:
        with pytest.raises(ConnectionError) as raised:
            _run_task_fence_child_launch(
                child=child,
                goal="ambiguous exception",
                child_task_id=child._subagent_id,
                stream_callback=lambda _delta: None,
                parent_envelope=parent_envelope,
                policy=policy,
            )

        assert raised.value is expected
        assert calls == 1
        assert tuple(
            db._conn.execute(
                "SELECT state, acknowledgement_ref, terminal_at "
                "FROM task_fence_attempts"
            ).fetchone()
        ) == ("STARTED", None, None)
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_attempt_transitions"
            ).fetchone()[0]
            == 1
        )
    finally:
        db.close()


def test_child_launch_terminal_evidence_fault_is_fail_open(
    tmp_path,
    monkeypatch,
    caplog,
):
    from tools.delegate_tool import _run_single_child

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation("tfiv_delegate_terminal_fault")
    result_sentinel = {"completed": True, "final_response": "done"}
    calls = 0

    def run_conversation(**_kwargs):
        nonlocal calls
        calls += 1
        return result_sentinel

    child = _FakeChild("sa-task-fence-terminal-fault", run_conversation)
    parent = _parent(db, active_children=[child])
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_delegate_terminal_transition "
        "BEFORE INSERT ON task_fence_attempt_transitions "
        "WHEN NEW.from_state = 'STARTED' BEGIN "
        "SELECT RAISE(ABORT, 'secret delegate terminal fault'); END"
    )
    try:
        with caplog.at_level(logging.WARNING):
            result = _run_single_child(
                0,
                "terminal fault",
                child,
                parent,
                _task_fence_parent=parent_envelope,
                _task_fence_policy=policy,
            )

        assert result["status"] == "completed"
        assert result["summary"] == result_sentinel["final_response"]
        assert calls == 1
        assert child.closed
        assert tuple(
            db._conn.execute(
                "SELECT state, acknowledgement_ref, terminal_at "
                "FROM task_fence_attempts"
            ).fetchone()
        ) == ("STARTED", None, None)
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_attempt_transitions"
            ).fetchone()[0]
            == 1
        )
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_cohorts WHERE audit_degraded = 1"
            ).fetchone()[0]
            == 1
        )
        assert "TaskFencePolicyUnavailable" in caplog.text
        assert "secret delegate terminal fault" not in caplog.text
    finally:
        db.close()


def test_child_launch_terminal_observer_runs_on_caller_after_child_returns(
    tmp_path,
    monkeypatch,
):
    from tools import delegate_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 5.0)
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation("tfiv_delegate_timeout_boundary")
    caller_thread = threading.current_thread()
    child_threads = []
    finish_threads = []
    original_finish = TaskFencePolicy.finish_attempt

    def run_conversation(**_kwargs):
        child_threads.append(threading.current_thread())
        return {"completed": True, "final_response": "done"}

    def observe_finish(self, *args, **kwargs):
        finish_threads.append(threading.current_thread())
        return original_finish(self, *args, **kwargs)

    child = _FakeChild("sa-task-fence-timeout-boundary", run_conversation)
    parent = _parent(db, active_children=[child])
    try:
        monkeypatch.setattr(TaskFencePolicy, "finish_attempt", observe_finish)
        result = delegate_tool._run_single_child(
            0,
            "timeout boundary",
            child,
            parent,
            _task_fence_parent=parent_envelope,
            _task_fence_policy=policy,
        )

        assert result["status"] == "completed"
        assert len(child_threads) == 1
        assert child_threads[0] is not caller_thread
        assert finish_threads == [caller_thread]
        assert not child.interrupted
        assert (
            db._conn.execute("SELECT state FROM task_fence_attempts").fetchone()[0]
            == "SUCCEEDED"
        )
    finally:
        db.close()


def test_child_launch_timeout_does_not_accept_late_success(
    tmp_path,
    monkeypatch,
):
    from tools import delegate_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 1.0)
    monkeypatch.setattr(
        delegate_tool,
        "_dump_subagent_timeout_diagnostic",
        lambda **_kwargs: None,
    )
    db, acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation("tfiv_delegate_real_timeout")
    entered = threading.Event()
    release = threading.Event()
    worker_exited = threading.Event()
    result_ready = threading.Event()
    original_launch = delegate_tool._run_task_fence_child_launch
    results = []
    errors = []

    def run_conversation(**_kwargs):
        entered.set()
        assert release.wait(timeout=10.0)
        return {"completed": True, "final_response": "late success"}

    def observe_worker_exit(**kwargs):
        try:
            return original_launch(**kwargs)
        finally:
            worker_exited.set()

    child = _FakeChild("sa-task-fence-real-timeout", run_conversation)
    parent = _parent(db, active_children=[child])

    def run_parent():
        try:
            results.append(
                delegate_tool._run_single_child(
                    0,
                    "timeout must stay ambiguous",
                    child,
                    parent,
                    _task_fence_parent=parent_envelope,
                    _task_fence_policy=policy,
                )
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            result_ready.set()

    runner = threading.Thread(target=run_parent)
    try:
        monkeypatch.setattr(
            delegate_tool,
            "_run_task_fence_child_launch",
            observe_worker_exit,
        )
        runner.start()
        assert entered.wait(timeout=10.0)
        assert not result_ready.is_set()
        assert result_ready.wait(timeout=10.0)
        runner.join(timeout=10.0)
        assert not runner.is_alive()
        assert errors == []
        assert len(results) == 1
        result = results[0]
        assert result["status"] == "timeout"
        assert result["exit_reason"] == "timeout"
        assert result["summary"] is None
        assert result["timeout_seconds"] == 1.0
        assert child.interrupted
        assert child.closed
        attempt_id = db._conn.execute(
            "SELECT attempt_id FROM task_fence_attempts"
        ).fetchone()[0]
        assert tuple(
            db._conn.execute(
                "SELECT state, acknowledgement_ref, terminal_at "
                "FROM task_fence_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        ) == ("STARTED", None, None)

        release.set()
        assert worker_exited.wait(timeout=10.0)
        assert tuple(
            db._conn.execute(
                "SELECT state, acknowledgement_ref, terminal_at "
                "FROM task_fence_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        ) == ("STARTED", None, None)
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_attempt_transitions "
                "WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()[0]
            == 1
        )

        inspection = db.inspect_task_fence_store()
        recovery = db.recover_task_fence_state(
            expected_runtime_epoch=inspection.runtime_epoch,
            expected_mode_generation=inspection.mode_generation,
        )
        assert recovery.runtime_epoch == generation.runtime_epoch + 1
        assert tuple(
            db._conn.execute(
                "SELECT state, disposition, acknowledgement_ref, "
                "terminal_at IS NOT NULL FROM task_fence_attempts "
                "WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        ) == ("OUTCOME_UNKNOWN", "OUTCOME_UNKNOWN", None, 1)
        assert [
            row["attempt_id"]
            for row in db._conn.execute(
                "SELECT link.attempt_id FROM task_fence_incident_attempts AS link "
                "JOIN task_fence_incidents AS incident "
                "ON incident.incident_id = link.incident_id "
                "WHERE incident.task_id = ?",
                (acceptance.task_id,),
            )
        ] == [attempt_id]
    finally:
        release.set()
        runner.join(timeout=10.0)
        worker_exited.wait(timeout=10.0)
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
        attempts = db._conn.execute(
            "SELECT attempt_id, state, acknowledgement_ref, "
            "terminal_at IS NOT NULL FROM task_fence_attempts"
        ).fetchall()
        assert len(attempts) == 2
        assert {row["state"] for row in attempts} == {"SUCCEEDED"}
        for attempt in attempts:
            evidence = f"delegate:run_conversation:completed:v1:{attempt['attempt_id']}"
            assert tuple(attempt[1:]) == ("SUCCEEDED", evidence, 1)
            assert (
                db._conn.execute(
                    "SELECT COUNT(*) FROM task_fence_attempt_transitions "
                    "WHERE attempt_id = ?",
                    (attempt["attempt_id"],),
                ).fetchone()[0]
                == 2
            )
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


def test_recovery_incidents_only_ambiguous_child_launch(tmp_path, monkeypatch):
    from tools.delegate_tool import _run_single_child

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    db, acceptance, generation = _live_lane(path)
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation("tfiv_delegate_recovery")
    success_result = {"completed": True, "final_response": "done"}
    ambiguous_result = {"completed": False, "final_response": "incomplete"}

    def run_child(subagent_id, result):
        child = _FakeChild(
            subagent_id,
            lambda **_kwargs: result,
        )
        parent = _parent(db, active_children=[child])
        observed = _run_single_child(
            0,
            f"recovery {subagent_id}",
            child,
            parent,
            _task_fence_parent=parent_envelope,
            _task_fence_policy=policy,
        )
        assert observed["summary"] == result["final_response"]
        assert child.closed

    try:
        run_child("sa-task-fence-success", success_result)
        run_child("sa-task-fence-ambiguous", ambiguous_result)
        attempts_before = db._conn.execute(
            "SELECT attempt_id, state, acknowledgement_ref, terminal_at "
            "FROM task_fence_attempts ORDER BY prepared_at, attempt_id"
        ).fetchall()
        assert {row["state"] for row in attempts_before} == {
            "STARTED",
            "SUCCEEDED",
        }
        succeeded_before = next(
            tuple(row) for row in attempts_before if row["state"] == "SUCCEEDED"
        )
        ambiguous_id = next(
            row["attempt_id"]
            for row in attempts_before
            if row["state"] == "STARTED"
        )
        succeeded_id = succeeded_before[0]
        succeeded_transitions_before = tuple(
            tuple(row)
            for row in db._conn.execute(
                "SELECT from_state, to_state, disposition, evidence_ref, "
                "transitioned_at FROM task_fence_attempt_transitions "
                "WHERE attempt_id = ? ORDER BY transition_order",
                (succeeded_id,),
            )
        )
    finally:
        db.close()

    reopened = SessionDB(path)
    try:
        inspection = reopened.inspect_task_fence_store()
        recovery = reopened.recover_task_fence_state(
            expected_runtime_epoch=inspection.runtime_epoch,
            expected_mode_generation=inspection.mode_generation,
        )
        assert (recovery.previous_runtime_epoch, recovery.runtime_epoch) == (0, 1)
        assert tuple(
            reopened._conn.execute(
                "SELECT attempt_id, state, acknowledgement_ref, terminal_at "
                "FROM task_fence_attempts WHERE attempt_id = ?",
                (succeeded_id,),
            ).fetchone()
        ) == succeeded_before
        assert tuple(
            reopened._conn.execute(
                "SELECT state, disposition, acknowledgement_ref, "
                "terminal_at IS NOT NULL FROM task_fence_attempts "
                "WHERE attempt_id = ?",
                (ambiguous_id,),
            ).fetchone()
        ) == ("OUTCOME_UNKNOWN", "OUTCOME_UNKNOWN", None, 1)
        assert tuple(
            tuple(row)
            for row in reopened._conn.execute(
                "SELECT from_state, to_state, disposition, evidence_ref, "
                "transitioned_at FROM task_fence_attempt_transitions "
                "WHERE attempt_id = ? ORDER BY transition_order",
                (succeeded_id,),
            )
        ) == succeeded_transitions_before
        incident_attempts = reopened._conn.execute(
            "SELECT link.attempt_id FROM task_fence_incident_attempts AS link "
            "JOIN task_fence_incidents AS incident "
            "ON incident.incident_id = link.incident_id "
            "WHERE incident.task_id = ?",
            (acceptance.task_id,),
        ).fetchall()
        assert [row["attempt_id"] for row in incident_attempts] == [ambiguous_id]
        assert [
            tuple(row)
            for row in reopened._conn.execute(
                "SELECT state, reason_code FROM task_fence_incidents "
                "WHERE task_id = ?",
                (acceptance.task_id,),
            )
        ] == [("open", "outcome_unknown")]
    finally:
        reopened.close()


def test_recovery_first_rejects_late_child_success_without_erasing_incident(
    tmp_path,
    monkeypatch,
    caplog,
):
    from tools.delegate_tool import _run_single_child

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    db, acceptance, generation = _live_lane(path)
    policy = TaskFencePolicy(db)
    parent_envelope = generation.for_invocation(
        "tfiv_delegate_recovery_first"
    )
    entered = threading.Event()
    release = threading.Event()
    secret_goal = "recovery-first-secret-goal"
    secret_response = "recovery-first-secret-response"
    results = []
    errors = []

    def run_conversation(**_kwargs):
        entered.set()
        assert release.wait(timeout=10.0)
        return {"completed": True, "final_response": secret_response}

    child = _FakeChild("sa-task-fence-recovery-first", run_conversation)
    parent = _parent(db, active_children=[child])

    def run_parent():
        try:
            results.append(
                _run_single_child(
                    0,
                    secret_goal,
                    child,
                    parent,
                    _task_fence_parent=parent_envelope,
                    _task_fence_policy=policy,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    runner = threading.Thread(target=run_parent)
    reopened = None
    try:
        with caplog.at_level(logging.WARNING):
            runner.start()
            assert entered.wait(timeout=10.0)
            attempt = db._conn.execute(
                "SELECT attempt_id, state FROM task_fence_attempts"
            ).fetchone()
            assert tuple(attempt) == (attempt["attempt_id"], "STARTED")

            reopened = SessionDB(path)
            inspection = reopened.inspect_task_fence_store()
            recovery = reopened.recover_task_fence_state(
                expected_runtime_epoch=inspection.runtime_epoch,
                expected_mode_generation=inspection.mode_generation,
            )
            assert recovery.runtime_epoch == generation.runtime_epoch + 1

            recovered_attempt = tuple(
                reopened._conn.execute(
                    "SELECT state, disposition, recovery_classification, "
                    "handoff_ref, acknowledgement_ref, prepared_at, started_at, "
                    "terminal_at FROM task_fence_attempts WHERE attempt_id = ?",
                    (attempt["attempt_id"],),
                ).fetchone()
            )
            assert recovered_attempt[:3] == (
                "OUTCOME_UNKNOWN",
                "OUTCOME_UNKNOWN",
                "may_effect",
            )
            assert recovered_attempt[4] is None
            assert recovered_attempt[7] is not None
            recovered_transitions = tuple(
                tuple(row)
                for row in reopened._conn.execute(
                    "SELECT transition_order, from_state, to_state, disposition, "
                    "evidence_ref, transitioned_at "
                    "FROM task_fence_attempt_transitions WHERE attempt_id = ? "
                    "ORDER BY transition_order",
                    (attempt["attempt_id"],),
                )
            )
            assert [row[2] for row in recovered_transitions] == [
                "STARTED",
                "OUTCOME_UNKNOWN",
            ]
            recovered_incidents = tuple(
                tuple(row)
                for row in reopened._conn.execute(
                    "SELECT incident_id, task_id, source_run_id, reason_code, "
                    "state, opened_at, resolved_at FROM task_fence_incidents "
                    "WHERE task_id = ? ORDER BY incident_id",
                    (acceptance.task_id,),
                )
            )
            recovered_links = tuple(
                tuple(row)
                for row in reopened._conn.execute(
                    "SELECT incident_id, attempt_id "
                    "FROM task_fence_incident_attempts ORDER BY incident_id, attempt_id"
                )
            )
            assert len(recovered_incidents) == 1
            assert recovered_incidents[0][3:5] == (
                "outcome_unknown",
                "open",
            )
            assert recovered_links == (
                (recovered_incidents[0][0], attempt["attempt_id"]),
            )

            release.set()
            runner.join(timeout=10.0)

        assert not runner.is_alive()
        assert errors == []
        assert len(results) == 1
        assert results[0]["status"] == "completed"
        assert results[0]["summary"] == secret_response
        assert results[0]["exit_reason"] == "completed"
        assert not child.interrupted
        assert child.closed
        assert tuple(
            reopened._conn.execute(
                "SELECT state, disposition, recovery_classification, "
                "handoff_ref, acknowledgement_ref, prepared_at, started_at, "
                "terminal_at FROM task_fence_attempts WHERE attempt_id = ?",
                (attempt["attempt_id"],),
            ).fetchone()
        ) == recovered_attempt
        assert tuple(
            tuple(row)
            for row in reopened._conn.execute(
                "SELECT transition_order, from_state, to_state, disposition, "
                "evidence_ref, transitioned_at "
                "FROM task_fence_attempt_transitions WHERE attempt_id = ? "
                "ORDER BY transition_order",
                (attempt["attempt_id"],),
            )
        ) == recovered_transitions
        assert tuple(
            tuple(row)
            for row in reopened._conn.execute(
                "SELECT incident_id, task_id, source_run_id, reason_code, "
                "state, opened_at, resolved_at FROM task_fence_incidents "
                "WHERE task_id = ? ORDER BY incident_id",
                (acceptance.task_id,),
            )
        ) == recovered_incidents
        assert tuple(
            tuple(row)
            for row in reopened._conn.execute(
                "SELECT incident_id, attempt_id "
                "FROM task_fence_incident_attempts ORDER BY incident_id, attempt_id"
            )
        ) == recovered_links
        assert "TaskFencePolicyRejected" in caplog.text
        assert secret_goal not in caplog.text
        assert secret_response not in caplog.text
        durable_dump = _task_fence_dispatch_dump(reopened)
        assert secret_goal not in durable_dump
        assert secret_response not in durable_dump
    finally:
        release.set()
        runner.join(timeout=10.0)
        if reopened is not None:
            reopened.close()
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


def test_startup_recovery_observes_stale_async_completions_without_changing_delivery(
    tmp_path,
    monkeypatch,
    _clean_async_registry,
):
    from tools import async_delegation

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    db, acceptance, generation = _live_lane(
        path,
        conversation_id=_SHADOW_SESSION_KEY,
    )
    foreign_acceptance = db.accept_task_fence_ingress(
        _ingress(
            "initial_submit",
            "foreign-delegation-initial",
            conversation_id="foreign-conversation",
        )
    )
    foreign_generation = db.reserve_task_fence_generation(foreign_acceptance)
    assert db.finish_task_fence_generation(
        foreign_generation,
        state="committed",
    )
    cases = (
        (
            "f00d0001",
            _SHADOW_SESSION_KEY,
            generation.generation_id,
            generation.runtime_epoch,
            DecisionReason.STALE_AUTHORITY.value,
        ),
        (
            "f00d0002",
            _SHADOW_SESSION_KEY,
            None,
            None,
            DecisionReason.MISSING_PROVENANCE.value,
        ),
        (
            "f00d0003",
            _SHADOW_SESSION_KEY,
            generation.generation_id,
            None,
            DecisionReason.MISSING_PROVENANCE.value,
        ),
        (
            "f00d0004",
            _SHADOW_SESSION_KEY,
            "unknown-parent-generation",
            generation.runtime_epoch,
            DecisionReason.MISSING_PROVENANCE.value,
        ),
        (
            "f00d0005",
            _SHADOW_SESSION_KEY,
            foreign_generation.generation_id,
            foreign_generation.runtime_epoch,
            DecisionReason.MISSING_PROVENANCE.value,
        ),
        (
            "f00d0006",
            "foreign-session",
            generation.generation_id,
            generation.runtime_epoch,
            None,
        ),
    )
    secret = "raw-queued-result-must-not-enter-task-fence-journal"
    active_secret = "raw-running-task-must-not-enter-task-fence-journal"
    recovery_pending = (
        (
            "f00d0007",
            "running",
            generation.generation_id,
            generation.runtime_epoch,
            DecisionReason.STALE_AUTHORITY.value,
        ),
        (
            "f00d0008",
            "finalizing",
            None,
            None,
            DecisionReason.MISSING_PROVENANCE.value,
        ),
    )
    live_running_id = "f00d0009"
    foreign_running_id = "f00d0010"
    reopened_delivery_running_id = "f00d0011"
    observed_recovery_pending = (
        *recovery_pending,
        (
            live_running_id,
            "running",
            generation.generation_id,
            generation.runtime_epoch,
            DecisionReason.STALE_AUTHORITY.value,
        ),
        (
            reopened_delivery_running_id,
            "running",
            generation.generation_id,
            generation.runtime_epoch,
            DecisionReason.STALE_AUTHORITY.value,
        ),
    )
    try:
        for (
            delegation_id,
            session_key,
            parent_generation_id,
            parent_runtime_epoch,
            _reason,
        ) in cases:
            _dispatch_completed_batch(
                async_delegation,
                delegation_id=delegation_id,
                session_key=session_key,
                parent_generation_id=parent_generation_id,
                parent_runtime_epoch=parent_runtime_epoch,
            )
        monkeypatch.setattr(
            "gateway.status.get_process_start_time",
            lambda _pid: 202,
        )
        async_delegation._persist_dispatch({
            "delegation_id": recovery_pending[0][0],
            "session_key": _SHADOW_SESSION_KEY,
            "dispatched_at": 7.0,
            "goal": active_secret,
            "_task_fence_parent_generation_id": generation.generation_id,
            "_task_fence_parent_runtime_epoch": generation.runtime_epoch,
        })
        for ordinal, (
            delegation_id,
            state,
            parent_generation_id,
            parent_runtime_epoch,
            _reason,
        ) in enumerate(recovery_pending[1:], start=8):
            db._conn.execute(
                "INSERT INTO async_delegations ("
                "delegation_id, origin_session, state, dispatched_at, updated_at, "
                "delivery_state, task_json, causal_parent_generation_id, "
                "causal_parent_runtime_epoch"
                ") VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
                (
                    delegation_id,
                    _SHADOW_SESSION_KEY,
                    state,
                    float(ordinal),
                    float(ordinal),
                    json.dumps({"goal": active_secret}),
                    parent_generation_id,
                    parent_runtime_epoch,
                ),
            )
        db._conn.execute(
            "INSERT INTO async_delegations ("
            "delegation_id, origin_session, state, dispatched_at, updated_at, "
            "delivery_state, task_json, causal_parent_generation_id, "
            "causal_parent_runtime_epoch, owner_pid, owner_started_at"
            ") VALUES (?, ?, 'running', 9.0, 9.0, 'pending', ?, ?, ?, 4242, 101)",
            (
                live_running_id,
                _SHADOW_SESSION_KEY,
                json.dumps({"goal": active_secret}),
                generation.generation_id,
                generation.runtime_epoch,
            ),
        )
        db._conn.execute(
            "INSERT INTO async_delegations ("
            "delegation_id, origin_session, state, dispatched_at, updated_at, "
            "delivery_state, delivered_at, task_json, "
            "causal_parent_generation_id, causal_parent_runtime_epoch"
            ") VALUES (?, ?, 'running', 11.0, 11.0, 'delivered', 11.0, ?, ?, ?)",
            (
                reopened_delivery_running_id,
                _SHADOW_SESSION_KEY,
                json.dumps({"goal": active_secret}),
                generation.generation_id,
                generation.runtime_epoch,
            ),
        )
        db._conn.execute(
            "INSERT INTO async_delegations ("
            "delegation_id, origin_session, state, dispatched_at, updated_at, "
            "delivery_state, task_json, causal_parent_generation_id, "
            "causal_parent_runtime_epoch"
            ") VALUES (?, 'foreign-session', 'running', 10.0, 10.0, "
            "'pending', '{}', ?, ?)",
            (
                foreign_running_id,
                generation.generation_id,
                generation.runtime_epoch,
            ),
        )
        secret_event = json.loads(
            db._conn.execute(
                "SELECT event_json FROM async_delegations "
                "WHERE delegation_id = 'f00d0002'"
            ).fetchone()[0]
        )
        secret_event["summary"] = secret
        db._conn.execute(
            "UPDATE async_delegations SET event_json = ?, result_json = ? "
            "WHERE delegation_id = 'f00d0002'",
            (
                json.dumps(secret_event),
                json.dumps({"status": "completed", "summary": secret}),
            ),
        )
        db._conn.commit()
        before_async = _async_rows(db)
    finally:
        db.close()

    def unexpected_pid_probe(_pid):
        raise AssertionError("startup transaction must not probe OS process state")

    monkeypatch.setattr("gateway.status._pid_exists", unexpected_pid_probe)
    monkeypatch.setattr(
        "gateway.status.get_process_start_time",
        unexpected_pid_probe,
    )
    reopened = SessionDB(path)
    try:
        recovery = reopened.recover_task_fence_state(
            expected_runtime_epoch=0,
            expected_mode_generation=0,
            tested_artifact_identity=_artifact_identity(),
            shadow_session_key=_SHADOW_SESSION_KEY,
        )
        assert (recovery.previous_runtime_epoch, recovery.runtime_epoch) == (0, 1)
        assert _async_rows(reopened) == before_async

        durable_identity = {
            row["delegation_id"]: row["dispatched_at"]
            for row in reopened._conn.execute(
                "SELECT delegation_id, dispatched_at FROM async_delegations"
            )
        }
        decisions = {
            row["operation_invocation_id"]: row
            for row in reopened._conn.execute(
                "SELECT operation_invocation_id, outcome, reason_code, "
                "decision_point, operation_kind, adapter, "
                "invocation_fingerprint, candidate_task_id, "
                "candidate_generation_id, candidate_runtime_epoch, "
                "permit_id, attempt_id FROM task_fence_policy_decisions "
                "WHERE adapter = 'runtime:async_delegation_restore_ready'"
            )
        }
        assert len(decisions) == 5
        for (
            delegation_id,
            session_key,
            _parent_generation_id,
            _parent_runtime_epoch,
            reason,
        ) in cases:
            invocation_id = _async_restore_invocation(
                delegation_id,
                durable_identity[delegation_id],
            )
            if session_key != _SHADOW_SESSION_KEY:
                assert invocation_id not in decisions
                continue
            decision = decisions[invocation_id]
            assert (
                decision["outcome"],
                decision["reason_code"],
                decision["decision_point"],
                decision["operation_kind"],
                decision["adapter"],
                decision["permit_id"],
                decision["attempt_id"],
            ) == (
                DecisionOutcome.WOULD_BLOCK.value,
                reason,
                "admission",
                "delivery",
                "runtime:async_delegation_restore_ready",
                None,
                None,
            )
            event_json = reopened._conn.execute(
                "SELECT event_json FROM async_delegations WHERE delegation_id = ?",
                (delegation_id,),
            ).fetchone()[0]
            assert (
                decision["invocation_fingerprint"]
                == hashlib.sha256(event_json.encode("utf-8")).hexdigest()
            )
            if reason == DecisionReason.STALE_AUTHORITY.value:
                assert (
                    decision["candidate_task_id"],
                    decision["candidate_generation_id"],
                    decision["candidate_runtime_epoch"],
                ) == (
                    acceptance.task_id,
                    generation.generation_id,
                    generation.runtime_epoch,
                )
            else:
                assert (
                    decision["candidate_task_id"],
                    decision["candidate_generation_id"],
                    decision["candidate_runtime_epoch"],
                ) == (None, None, None)

        recovery_decisions = {
            row["operation_invocation_id"]: row
            for row in reopened._conn.execute(
                "SELECT operation_invocation_id, outcome, reason_code, "
                "decision_point, operation_kind, adapter, "
                "invocation_fingerprint, candidate_task_id, "
                "candidate_generation_id, candidate_runtime_epoch, "
                "permit_id, attempt_id FROM task_fence_policy_decisions "
                "WHERE adapter = 'runtime:async_delegation_recovery_pending'"
            )
        }
        assert len(recovery_decisions) == 4
        assert (
            _async_restore_invocation(
                foreign_running_id,
                durable_identity[foreign_running_id],
            )
            not in recovery_decisions
        )
        for (
            delegation_id,
            state,
            _parent_generation_id,
            _parent_runtime_epoch,
            reason,
        ) in observed_recovery_pending:
            invocation_id = _async_restore_invocation(
                delegation_id,
                durable_identity[delegation_id],
            )
            decision = recovery_decisions[invocation_id]
            assert (
                decision["outcome"],
                decision["reason_code"],
                decision["decision_point"],
                decision["operation_kind"],
                decision["adapter"],
                decision["permit_id"],
                decision["attempt_id"],
            ) == (
                DecisionOutcome.WOULD_BLOCK.value,
                reason,
                "admission",
                "delivery",
                "runtime:async_delegation_recovery_pending",
                None,
                None,
            )
            active_row = reopened._conn.execute(
                "SELECT state, delivery_state, owner_pid, owner_started_at, task_json "
                "FROM async_delegations WHERE delegation_id = ?",
                (delegation_id,),
            ).fetchone()
            assert active_row["state"] == state
            observation = json.dumps(
                tuple(active_row),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            assert (
                decision["invocation_fingerprint"]
                == hashlib.sha256(observation).hexdigest()
            )
            if reason == DecisionReason.STALE_AUTHORITY.value:
                assert (
                    decision["candidate_task_id"],
                    decision["candidate_generation_id"],
                    decision["candidate_runtime_epoch"],
                ) == (
                    acceptance.task_id,
                    generation.generation_id,
                    generation.runtime_epoch,
                )
            else:
                assert (
                    decision["candidate_task_id"],
                    decision["candidate_generation_id"],
                    decision["candidate_runtime_epoch"],
                ) == (None, None, None)

        journal_dump = repr(
            tuple(
                tuple(row)
                for row in (*decisions.values(), *recovery_decisions.values())
            )
        )
        assert secret not in journal_dump
        assert active_secret not in journal_dump
        assert _SHADOW_SESSION_KEY not in journal_dump
        assert all(delegation_id not in journal_dump for delegation_id, *_ in cases)
        assert all(
            delegation_id not in journal_dump
            for delegation_id, *_ in observed_recovery_pending
        )
        assert foreign_running_id not in journal_dump

        first_decision_ids = tuple(
            row[0]
            for row in reopened._conn.execute(
                "SELECT decision_id FROM task_fence_policy_decisions "
                "WHERE adapter IN ("
                "'runtime:async_delegation_restore_ready', "
                "'runtime:async_delegation_recovery_pending') "
                "ORDER BY decision_id"
            )
        )
        replay = reopened.recover_task_fence_state(
            expected_runtime_epoch=1,
            expected_mode_generation=0,
            tested_artifact_identity=_artifact_identity(),
            shadow_session_key=_SHADOW_SESSION_KEY,
        )
        assert (replay.previous_runtime_epoch, replay.runtime_epoch) == (1, 2)
        assert _async_rows(reopened) == before_async
        assert (
            tuple(
                row[0]
                for row in reopened._conn.execute(
                    "SELECT decision_id FROM task_fence_policy_decisions "
                    "WHERE adapter IN ("
                    "'runtime:async_delegation_restore_ready', "
                    "'runtime:async_delegation_recovery_pending') "
                    "ORDER BY decision_id"
                )
            )
            == first_decision_ids
        )

        changed_event = json.loads(
            reopened._conn.execute(
                "SELECT event_json FROM async_delegations "
                "WHERE delegation_id = 'f00d0002'"
            ).fetchone()[0]
        )
        changed_event["summary"] = "changed-exact-queued-payload"
        reopened._conn.execute(
            "UPDATE async_delegations SET event_json = ? "
            "WHERE delegation_id = 'f00d0002'",
            (json.dumps(changed_event),),
        )
        reopened._conn.execute(
            "UPDATE async_delegations SET task_json = ? WHERE delegation_id = ?",
            (
                json.dumps({"goal": "changed-exact-running-task"}),
                live_running_id,
            ),
        )
        reopened._conn.commit()
        changed_async = _async_rows(reopened)
        changed = reopened.recover_task_fence_state(
            expected_runtime_epoch=2,
            expected_mode_generation=0,
            tested_artifact_identity=_artifact_identity(),
            shadow_session_key=_SHADOW_SESSION_KEY,
        )
        assert (changed.previous_runtime_epoch, changed.runtime_epoch) == (2, 3)
        assert _async_rows(reopened) == changed_async
        changed_identity = _async_restore_invocation(
            "f00d0002",
            durable_identity["f00d0002"],
        )
        changed_decisions = reopened._conn.execute(
            "SELECT decision_id, invocation_fingerprint "
            "FROM task_fence_policy_decisions "
            "WHERE operation_invocation_id = ? ORDER BY decision_id",
            (changed_identity,),
        ).fetchall()
        assert len(changed_decisions) == 2
        assert len({row["decision_id"] for row in changed_decisions}) == 2
        assert len({row["invocation_fingerprint"] for row in changed_decisions}) == 2
        live_changed_decisions = reopened._conn.execute(
            "SELECT decision_id, invocation_fingerprint "
            "FROM task_fence_policy_decisions "
            "WHERE adapter = 'runtime:async_delegation_recovery_pending' "
            "AND operation_invocation_id = ? ORDER BY decision_id",
            (
                _async_restore_invocation(
                    live_running_id,
                    durable_identity[live_running_id],
                ),
            ),
        ).fetchall()
        assert len(live_changed_decisions) == 2
        assert len({row["decision_id"] for row in live_changed_decisions}) == 2
        assert (
            len({row["invocation_fingerprint"] for row in live_changed_decisions}) == 2
        )
    finally:
        reopened.close()

    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: pid == 4242)
    monkeypatch.setattr(
        "gateway.status.get_process_start_time",
        lambda pid: 101 if pid == 4242 else None,
    )
    restored_queue = queue.Queue()
    assert async_delegation.restore_undelivered_completions(restored_queue) == 10
    restored = [restored_queue.get_nowait() for _ in range(10)]
    assert {event["delegation_id"] for event in restored} == {
        *(case[0] for case in cases),
        *(case[0] for case in recovery_pending),
        foreign_running_id,
        reopened_delivery_running_id,
    }
    for event in restored:
        claim_id = async_delegation.claim_event_delivery(
            event,
            "shadow-recovery-legacy",
        )
        assert claim_id is not None
        assert async_delegation.complete_completion_delivery(
            event["delegation_id"],
            claim_id,
        )
    live_row = async_delegation.get_durable_delegation(live_running_id)
    assert live_row is not None
    assert live_row["state"] == "running"
    assert live_row["delivery_state"] == "pending"
    assert async_delegation.restore_undelivered_completions(queue.Queue()) == 0


def test_materialized_finalizing_event_keeps_restore_ready_observation(
    tmp_path,
) -> None:
    db = SessionDB(tmp_path / "state.db")
    event = json.dumps({
        "type": "async_delegation",
        "delegation_id": "f00d0012",
        "status": "completed",
    })
    db._conn.execute(
        "INSERT INTO async_delegations ("
        "delegation_id, origin_session, state, dispatched_at, completed_at, "
        "updated_at, event_json, result_json, delivery_state, task_json"
        ") VALUES (?, ?, 'finalizing', 1.0, 2.0, 2.0, ?, ?, 'pending', '{}')",
        ("f00d0012", _SHADOW_SESSION_KEY, event, event),
    )
    db._conn.commit()
    before_async = _async_rows(db)

    recovery = db.recover_task_fence_state(
        expected_runtime_epoch=0,
        expected_mode_generation=0,
        tested_artifact_identity=_artifact_identity(),
        shadow_session_key=_SHADOW_SESSION_KEY,
    )

    assert recovery.runtime_epoch == 1
    decision = db._conn.execute(
        "SELECT adapter, invocation_fingerprint FROM task_fence_policy_decisions"
    ).fetchone()
    assert tuple(decision) == (
        "runtime:async_delegation_restore_ready",
        hashlib.sha256(event.encode("utf-8")).hexdigest(),
    )
    assert _async_rows(db) == before_async
    db.close()


@pytest.mark.parametrize(
    "corruption",
    ("generation-state", "manifest", "acceptance-snapshot"),
)
def test_async_restore_corrupt_historical_parent_is_missing_provenance(
    tmp_path,
    monkeypatch,
    _clean_async_registry,
    corruption,
):
    from tools import async_delegation

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, acceptance, generation = _live_lane(
        tmp_path / "state.db",
        conversation_id=_SHADOW_SESSION_KEY,
    )
    delegation_id = {
        "generation-state": "cafe0001",
        "manifest": "cafe0002",
        "acceptance-snapshot": "cafe0003",
    }[corruption]
    _dispatch_completed_batch(
        async_delegation,
        delegation_id=delegation_id,
        session_key=_SHADOW_SESSION_KEY,
        parent_generation_id=generation.generation_id,
        parent_runtime_epoch=generation.runtime_epoch,
    )
    held = db.accept_task_fence_ingress(
        _ingress(
            "comment_hold",
            f"hold-{corruption}",
            task_id=acceptance.task_id,
            conversation_id=_SHADOW_SESSION_KEY,
        )
    )
    assert held.task_projection is not None
    assert held.task_projection.status == "paused"

    if corruption == "generation-state":
        db._conn.execute(
            "UPDATE task_fence_model_generations SET state = 'failed' "
            "WHERE generation_id = ?",
            (generation.generation_id,),
        )
    elif corruption == "manifest":
        db._conn.execute(
            "UPDATE task_fence_model_generations "
            "SET input_manifest_hash = ? WHERE generation_id = ?",
            ("0" * 64, generation.generation_id),
        )
    else:
        trigger_sql = db._conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'trigger' "
            "AND name = 'task_fence_acceptance_snapshots_no_delete'"
        ).fetchone()[0]
        db._conn.execute("DROP TRIGGER task_fence_acceptance_snapshots_no_delete")
        db._conn.execute(
            "DELETE FROM task_fence_acceptance_snapshots WHERE event_id = ?",
            (generation.snapshot_event_id,),
        )
        db._conn.execute(trigger_sql)
    db._conn.commit()
    before_async = _async_rows(db)

    recovery = db.recover_task_fence_state(
        expected_runtime_epoch=0,
        expected_mode_generation=0,
        tested_artifact_identity=_artifact_identity(),
        shadow_session_key=_SHADOW_SESSION_KEY,
    )

    assert recovery.runtime_epoch == 1
    decision = db._conn.execute(
        "SELECT outcome, reason_code, candidate_task_id, "
        "candidate_generation_id, candidate_runtime_epoch "
        "FROM task_fence_policy_decisions "
        "WHERE adapter = 'runtime:async_delegation_restore_ready'"
    ).fetchone()
    assert tuple(decision) == (
        DecisionOutcome.WOULD_BLOCK.value,
        DecisionReason.MISSING_PROVENANCE.value,
        None,
        None,
        None,
    )
    assert _async_rows(db) == before_async
    db.close()


@pytest.mark.parametrize("fault_target", ("decision", "control"))
@pytest.mark.parametrize("candidate_state", ("completed", "running"))
def test_async_restore_observation_rolls_back_with_startup_recovery(
    tmp_path,
    monkeypatch,
    _clean_async_registry,
    fault_target,
    candidate_state,
):
    from tools import async_delegation

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    db, _acceptance, generation = _live_lane(
        path,
        conversation_id=_SHADOW_SESSION_KEY,
    )
    delegation_id = (
        f"fade{int(candidate_state == 'running')}00{int(fault_target == 'control')}"
    )
    if candidate_state == "completed":
        _dispatch_completed_batch(
            async_delegation,
            delegation_id=delegation_id,
            session_key=_SHADOW_SESSION_KEY,
            parent_generation_id=generation.generation_id,
            parent_runtime_epoch=generation.runtime_epoch,
        )
    else:
        db._conn.execute(
            "INSERT INTO async_delegations ("
            "delegation_id, origin_session, state, dispatched_at, updated_at, "
            "delivery_state, task_json, causal_parent_generation_id, "
            "causal_parent_runtime_epoch"
            ") VALUES (?, ?, 'running', 1.0, 1.0, 'pending', '{}', ?, ?)",
            (
                delegation_id,
                _SHADOW_SESSION_KEY,
                generation.generation_id,
                generation.runtime_epoch,
            ),
        )
        db._conn.commit()
    trigger = {
        "decision": (
            "BEFORE INSERT ON main.task_fence_policy_decisions",
            "private async observation fault",
        ),
        "control": (
            "BEFORE UPDATE OF runtime_epoch ON main.task_fence_control "
            "WHEN NEW.runtime_epoch != OLD.runtime_epoch",
            "private recovery control fault",
        ),
    }[fault_target]
    db._conn.execute(
        "CREATE TEMP TRIGGER fail_async_restore_recovery "
        f"{trigger[0]} BEGIN SELECT RAISE(ABORT, '{trigger[1]}'); END"
    )
    before_task_fence = _task_fence_rows(db)
    before_async = _async_rows(db)

    with pytest.raises(
        TaskFenceRecoveryUnavailable,
        match="recovery_database_error",
    ) as exc:
        db.recover_task_fence_state(
            expected_runtime_epoch=0,
            expected_mode_generation=0,
            tested_artifact_identity=_artifact_identity(),
            shadow_session_key=_SHADOW_SESSION_KEY,
        )

    assert exc.value.reason == "recovery_database_error"
    assert exc.value.__cause__ is None
    assert "private" not in str(exc.value)
    assert _task_fence_rows(db) == before_task_fence
    assert _async_rows(db) == before_async
    db.close()

    reopened = SessionDB(path)
    try:
        assert _task_fence_rows(reopened) == before_task_fence
        assert _async_rows(reopened) == before_async
    finally:
        reopened.close()


@pytest.mark.parametrize("candidate_state", ("completed", "running"))
def test_async_restore_observation_shares_recovery_authority_bound(
    tmp_path,
    monkeypatch,
    candidate_state,
) -> None:
    import hermes_state

    db = SessionDB(tmp_path / "state.db")
    event = json.dumps({
        "type": "async_delegation",
        "delegation_id": "face0001",
        "status": "completed",
    })
    if candidate_state == "completed":
        db._conn.execute(
            "INSERT INTO async_delegations ("
            "delegation_id, origin_session, state, dispatched_at, completed_at, "
            "updated_at, event_json, result_json, delivery_state"
            ") VALUES (?, ?, 'completed', 1.0, 2.0, 2.0, ?, ?, 'pending')",
            ("face0001", _SHADOW_SESSION_KEY, event, event),
        )
    else:
        db._conn.execute(
            "INSERT INTO async_delegations ("
            "delegation_id, origin_session, state, dispatched_at, updated_at, "
            "task_json, delivery_state"
            ") VALUES (?, ?, 'running', 1.0, 1.0, '{}', 'pending')",
            ("face0001", _SHADOW_SESSION_KEY),
        )
    db._conn.commit()
    before_task_fence = _task_fence_rows(db)
    before_async = _async_rows(db)
    monkeypatch.setattr(hermes_state, "_TASK_FENCE_MAX_RECOVERY_AUTHORITIES", 0)

    with pytest.raises(
        TaskFenceRecoveryUnavailable,
        match="recovery_authority_limit_exceeded",
    ):
        db.recover_task_fence_state(
            expected_runtime_epoch=0,
            expected_mode_generation=0,
            tested_artifact_identity=_artifact_identity(),
            shadow_session_key=_SHADOW_SESSION_KEY,
        )

    assert _task_fence_rows(db) == before_task_fence
    assert _async_rows(db) == before_async
    db.close()


def test_async_recovery_pending_serialization_failure_is_closed(tmp_path) -> None:
    db = SessionDB(tmp_path / "state.db")
    db._conn.execute(
        "INSERT INTO async_delegations ("
        "delegation_id, origin_session, state, dispatched_at, updated_at, "
        "task_json, delivery_state, owner_pid"
        ") VALUES (?, ?, 'running', 1.0, 1.0, '{}', 'pending', ?)",
        ("face0002", _SHADOW_SESSION_KEY, b"not-an-integer-owner"),
    )
    db._conn.commit()
    before_task_fence = _task_fence_rows(db)
    before_async = _async_rows(db)

    with pytest.raises(
        TaskFenceRecoveryUnavailable,
        match="incompatible_recovery_projection",
    ) as exc:
        db.recover_task_fence_state(
            expected_runtime_epoch=0,
            expected_mode_generation=0,
            tested_artifact_identity=_artifact_identity(),
            shadow_session_key=_SHADOW_SESSION_KEY,
        )

    assert exc.value.reason == "incompatible_recovery_projection"
    assert exc.value.__cause__ is None
    assert _task_fence_rows(db) == before_task_fence
    assert _async_rows(db) == before_async
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
