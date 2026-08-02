import ast
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest

import cli as classic_cli
from hermes_state import SessionDB
from task_fence import (
    TaskFenceIngressUnavailable,
    TaskFencePolicy,
    bind_task_fence_policy,
    current_task_fence_runtime_conversation_key,
    task_fence_sidecar_for_plain_text,
)


_ROOT = "cli-root"
_TIP = "cli-tip"


class _RecordingQueue:
    def __init__(self, database: SessionDB):
        self.database = database
        self.values = []

    def put(self, value) -> None:
        with self.database._lock:
            durable_count = self.database._conn.execute(
                "SELECT COUNT(*) FROM task_fence_ingress"
            ).fetchone()[0]
        assert durable_count == 1
        self.values.append(value)


class _FailOpenQueue:
    def __init__(self):
        self.values = []

    def put(self, value) -> None:
        self.values.append(value)


def _seed_compression_conversation(database: SessionDB) -> None:
    database.create_session(_ROOT, source="cli")
    database.end_session(_ROOT, "compression")
    database.create_session(
        _TIP,
        source="cli",
        parent_session_id=_ROOT,
    )


def _cli_for(
    database: SessionDB,
    pending_queue,
    *,
    selector: str = _ROOT,
    session_id: str = _TIP,
):
    instance = object.__new__(classic_cli.HermesCLI)
    instance._session_db = database
    instance._pending_input = pending_queue
    instance._task_fence_shadow_conversation_key = selector
    instance._task_fence_cli_startup_resume = True
    instance.session_id = session_id
    instance._agent_running = False
    instance._command_running = False
    instance._pending_resume_sessions = None
    instance._voice_mode = False
    instance._voice_continuous = False
    return instance


def test_cli_plain_text_is_durable_before_single_queue_exposure(tmp_path) -> None:
    database = SessionDB(tmp_path / "state.db")
    try:
        _seed_compression_conversation(database)
        pending_queue = _RecordingQueue(database)
        instance = _cli_for(database, pending_queue)
        payload = "ship the bounded increment"

        instance._queue_task_fence_foreground_input(payload)

        assert len(pending_queue.values) == 1
        queued = pending_queue.values[0]
        assert isinstance(queued, classic_cli._TaskFenceCLIInput)
        assert queued.text == payload
        assert queued.task_fence_acceptance.task_projection.conversation_id == _ROOT
        assert payload not in repr(queued)
        with pytest.raises(FrozenInstanceError):
            queued.text = "changed"

        with database._lock:
            row = database._conn.execute(
                "SELECT source, source_event_id, conversation_id, payload_hash, "
                "intent, execution, input_effect "
                "FROM task_fence_ingress"
            ).fetchone()
        assert tuple(row[:1]) == ("cli:foreground",)
        assert row["source_event_id"].startswith("submit:")
        assert row["conversation_id"] == _ROOT
        assert row["payload_hash"] == hashlib.sha256(
            b"text\x00ship the bounded increment"
        ).hexdigest()
        assert (
            row["intent"],
            row["execution"],
            row["input_effect"],
        ) == ("replace", "run", "append")
    finally:
        database.close()


def test_cli_known_acceptance_failure_preserves_exact_legacy_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = SessionDB(tmp_path / "state.db")
    try:
        _seed_compression_conversation(database)
        pending_queue = _FailOpenQueue()
        instance = _cli_for(database, pending_queue)
        payload = "unchanged legacy input"

        def fail_acceptance(self, sidecar, *, conversation_id):
            raise TaskFenceIngressUnavailable("test_unavailable")

        monkeypatch.setattr(
            SessionDB,
            "accept_task_fence_ingress_sidecar",
            fail_acceptance,
        )

        instance._queue_task_fence_foreground_input(payload)

        assert pending_queue.values == [payload]
        with database._lock:
            assert database._conn.execute(
                "SELECT COUNT(*) FROM task_fence_ingress"
            ).fetchone()[0] == 0
    finally:
        database.close()


def test_cli_excluded_shapes_never_attempt_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = SessionDB(tmp_path / "state.db")
    try:
        _seed_compression_conversation(database)
        dropped_file = tmp_path / "attached.txt"
        dropped_file.write_text("content", encoding="utf-8")
        attempted = 0

        def count_acceptance(self, sidecar, *, conversation_id):
            nonlocal attempted
            attempted += 1
            raise AssertionError("excluded CLI input reached acceptance")

        monkeypatch.setattr(
            SessionDB,
            "accept_task_fence_ingress_sidecar",
            count_acceptance,
        )

        for payload in (
            " leading space",
            "/help",
            "inspect @file:README.md",
            "[Pasted text #1: 2 lines → /tmp/paste.txt]",
            str(dropped_file),
            "\x1b[<0;10;10Mmouse",
        ):
            pending_queue = _FailOpenQueue()
            instance = _cli_for(database, pending_queue)
            instance._queue_task_fence_foreground_input(payload)
            assert pending_queue.values == [payload]

        pending_queue = _FailOpenQueue()
        instance = _cli_for(database, pending_queue)
        instance._agent_running = True
        instance._queue_task_fence_foreground_input("busy submit")
        assert pending_queue.values == ["busy submit"]
        assert attempted == 0
    finally:
        database.close()


def test_plain_text_classifier_is_surface_neutral_and_runtime_key_is_scoped() -> None:
    payload = "same payload"
    cli_sidecar = task_fence_sidecar_for_plain_text(
        source="cli:foreground",
        source_event_id="submit:1",
        payload_text=payload,
    )
    gateway_sidecar = task_fence_sidecar_for_plain_text(
        source="gateway:slack",
        source_event_id="event:1",
        payload_text=payload,
    )

    assert cli_sidecar.payload_hash == gateway_sidecar.payload_hash
    assert cli_sidecar.action == gateway_sidecar.action
    assert cli_sidecar.active_lane_action == gateway_sidecar.active_lane_action

    policy = TaskFencePolicy(object(), runtime_conversation_key=_ROOT)
    assert current_task_fence_runtime_conversation_key() is None
    with bind_task_fence_policy(policy):
        assert current_task_fence_runtime_conversation_key() == _ROOT
    assert current_task_fence_runtime_conversation_key() is None


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_cli_ingress_wiring_preserves_acceptance_and_legacy_task_id() -> None:
    tree = ast.parse(Path(classic_cli.__file__).read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HermesCLI"
    )
    helper = _method(class_node, "_queue_task_fence_foreground_input")
    queue_puts = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "put"
    ]
    accept_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "accept_task_fence_ingress_sidecar"
    ]
    assert len(queue_puts) == 1
    assert len(accept_calls) == 1
    assert accept_calls[0].lineno < queue_puts[0].lineno

    run_method = _method(class_node, "run")
    handle_enter = next(
        node
        for node in ast.walk(run_method)
        if isinstance(node, ast.FunctionDef) and node.name == "handle_enter"
    )
    producer_calls = [
        node
        for node in ast.walk(class_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_queue_task_fence_foreground_input"
    ]
    assert len(producer_calls) == 1
    assert producer_calls[0] in ast.walk(handle_enter)

    process_loop = next(
        node
        for node in ast.walk(run_method)
        if isinstance(node, ast.FunctionDef) and node.name == "process_loop"
    )
    chat_call = next(
        node
        for node in ast.walk(process_loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "chat"
    )
    assert {
        keyword.arg for keyword in chat_call.keywords
    } >= {"task_fence_acceptance"}

    chat_method = _method(class_node, "chat")
    run_conversation_call = next(
        node
        for node in ast.walk(chat_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_conversation"
    )
    keyword_values = {
        keyword.arg: keyword.value
        for keyword in run_conversation_call.keywords
    }
    assert isinstance(keyword_values["task_fence_acceptance"], ast.Name)
    task_id = keyword_values["task_id"]
    assert isinstance(task_id, ast.Attribute)
    assert isinstance(task_id.value, ast.Name)
    assert (task_id.value.id, task_id.attr) == ("self", "session_id")
