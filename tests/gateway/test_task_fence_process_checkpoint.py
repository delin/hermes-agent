import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from gateway import task_fence_startup as startup


_SESSION_KEY = "slack:workspace:channel:user"


def _candidate(
    session_id: str = "proc_selected",
    **overrides,
) -> dict:
    entry = {
        "session_id": session_id,
        "command": "printf 'toxic-command-secret'",
        "pid": 999_999_999,
        "pid_scope": "host",
        "host_start_time": 123,
        "cwd": "/private/toxic-cwd",
        "started_at": 456.5,
        "task_id": "private-task",
        "session_key": _SESSION_KEY,
        "watcher_platform": "slack",
        "watcher_chat_id": "private-chat",
        "watcher_user_id": "private-user",
        "watcher_user_name": "private-name",
        "watcher_thread_id": "private-thread",
        "watcher_message_id": "private-message",
        "watcher_interval": 5,
        "notify_on_complete": True,
        "watch_patterns": ["toxic-pattern-secret"],
    }
    entry.update(overrides)
    return entry


def _write_checkpoint(path: Path, entries: list[dict]) -> bytes:
    raw = json.dumps(
        entries,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_process_checkpoint_snapshot_absent_and_empty_are_read_only(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "processes.json"

    assert startup._read_process_checkpoint_snapshot(
        checkpoint,
        shadow_session_key=_SESSION_KEY,
    ) == ()
    assert not checkpoint.exists()

    checkpoint.write_text("[]", encoding="utf-8")
    before = checkpoint.stat()
    assert startup._read_process_checkpoint_snapshot(
        checkpoint,
        shadow_session_key=_SESSION_KEY,
    ) == ()
    after = checkpoint.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert not (tmp_path / "state.db").exists()


def test_process_checkpoint_snapshot_reads_real_registry_writer_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import process_registry as process_registry_module

    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(
        process_registry_module,
        "CHECKPOINT_PATH",
        checkpoint,
    )
    registry = process_registry_module.ProcessRegistry()
    session = process_registry_module.ProcessSession(
        id="proc_real_writer",
        command="printf 'real-writer-toxic-secret'",
        pid=999_999_999,
        pid_scope="host",
        host_start_time=123,
        cwd="/private/real-writer-cwd",
        started_at=456.5,
        task_id="private-real-writer-task",
        session_key=_SESSION_KEY,
        watcher_platform="slack",
        watcher_chat_id="private-real-writer-chat",
        watcher_interval=5,
        notify_on_complete=True,
        watch_patterns=["real-writer-toxic-pattern"],
    )
    registry._running[session.id] = session
    registry._write_checkpoint()
    before_bytes = checkpoint.read_bytes()
    before_running = tuple(registry._running)
    before_watchers = tuple(registry.pending_watchers)
    before_queue_size = registry.completion_queue.qsize()

    observations = startup._read_process_checkpoint_snapshot(
        checkpoint,
        shadow_session_key=_SESSION_KEY,
    )

    assert len(observations) == 1
    assert "toxic" not in repr(observations)
    assert checkpoint.read_bytes() == before_bytes
    assert tuple(registry._running) == before_running
    assert tuple(registry.pending_watchers) == before_watchers
    assert registry.completion_queue.qsize() == before_queue_size


def test_process_checkpoint_snapshot_is_exact_lane_pre_liveness_hash_only(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "processes.json"
    selected = _candidate()
    _write_checkpoint(
        checkpoint,
        [
            selected,
            _candidate("proc_case", session_key=_SESSION_KEY.upper()),
            _candidate("proc_foreign", session_key="slack:other"),
            _candidate("proc_sandbox", pid_scope="sandbox"),
            _candidate(
                "proc_manual",
                watcher_interval=0,
                notify_on_complete=False,
            ),
            _candidate("proc_no_pid", pid=None),
        ],
    )
    before_bytes = checkpoint.read_bytes()
    before_stat = checkpoint.stat()

    observations = startup._read_process_checkpoint_snapshot(
        checkpoint,
        shadow_session_key=_SESSION_KEY,
    )

    assert len(observations) == 2
    observation = observations[0]
    assert observation.invocation_id.startswith("tfqp_")
    assert len(observation.invocation_id) == len("tfqp_") + 64
    assert len(observation.invocation_fingerprint) == 64
    assert "toxic" not in repr(observation)
    assert "private-" not in repr(observation)
    with pytest.raises(FrozenInstanceError):
        observation.invocation_id = "changed"
    after_stat = checkpoint.stat()
    assert checkpoint.read_bytes() == before_bytes
    assert (after_stat.st_dev, after_stat.st_ino, after_stat.st_mtime_ns) == (
        before_stat.st_dev,
        before_stat.st_ino,
        before_stat.st_mtime_ns,
    )


def test_process_checkpoint_snapshot_is_canonical_and_closed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "processes.json"
    first = _candidate()
    checkpoint.write_text(
        json.dumps([first], indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    baseline = startup._read_process_checkpoint_snapshot(
        checkpoint,
        shadow_session_key=_SESSION_KEY,
    )

    reordered = dict(reversed(first.items()))
    checkpoint.write_text(
        json.dumps([reordered], separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    assert startup._read_process_checkpoint_snapshot(
        checkpoint,
        shadow_session_key=_SESSION_KEY,
    ) == baseline

    changed = dict(reordered)
    changed["command"] = "different command"
    _write_checkpoint(checkpoint, [changed])
    changed_observation = startup._read_process_checkpoint_snapshot(
        checkpoint,
        shadow_session_key=_SESSION_KEY,
    )[0]
    assert changed_observation.invocation_id == baseline[0].invocation_id
    assert (
        changed_observation.invocation_fingerprint
        != baseline[0].invocation_fingerprint
    )

    _write_checkpoint(checkpoint, [{**changed, "future_field": True}])
    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )
    assert exc_info.value.reason == "incompatible_process_checkpoint"


def test_process_checkpoint_snapshot_legacy_missing_started_at_is_stable(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "processes.json"
    legacy = _candidate()
    legacy.pop("started_at")
    _write_checkpoint(checkpoint, [legacy])
    missing = startup._read_process_checkpoint_snapshot(
        checkpoint,
        shadow_session_key=_SESSION_KEY,
    )

    _write_checkpoint(checkpoint, [{**legacy, "started_at": None}])
    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )
    assert exc_info.value.reason == "incompatible_process_checkpoint"

    _write_checkpoint(
        checkpoint,
        [{
            **legacy,
            "started_at": 999.0,
            "host_start_time": 888,
        }],
    )
    backfilled = startup._read_process_checkpoint_snapshot(
        checkpoint,
        shadow_session_key=_SESSION_KEY,
    )
    assert backfilled[0].invocation_id == missing[0].invocation_id
    assert (
        backfilled[0].invocation_fingerprint
        != missing[0].invocation_fingerprint
    )


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"\xff", "incompatible_process_checkpoint"),
        (b"{", "incompatible_process_checkpoint"),
        (b"{}", "incompatible_process_checkpoint"),
        (b"[1]", "incompatible_process_checkpoint"),
        (
            b'[{"session_key":"' + _SESSION_KEY.encode() + b'",'
            b'"session_key":"duplicate"}]',
            "incompatible_process_checkpoint",
        ),
        (
            b'[{"session_key":"' + _SESSION_KEY.encode() + b'",'
            b'"pid":NaN}]',
            "incompatible_process_checkpoint",
        ),
        (b'[{"future":1e9999}]', "incompatible_process_checkpoint"),
        (b"[" * 2_000 + b"]" * 2_000, "incompatible_process_checkpoint"),
    ],
    ids=[
        "invalid-utf8",
        "truncated-json",
        "wrong-top-level",
        "non-object-row",
        "duplicate-key",
        "non-finite-number",
        "overflowing-float",
        "deeply-nested-json",
    ],
)
def test_process_checkpoint_snapshot_rejects_noncanonical_json(
    tmp_path: Path,
    raw: bytes,
    reason: str,
) -> None:
    checkpoint = tmp_path / "processes.json"
    checkpoint.write_bytes(raw)

    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )

    assert exc_info.value.reason == reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"session_id": ""},
        {"session_id": "x" * 513},
        {"pid": True},
        {"pid": -1},
        {"host_start_time": "123"},
        {"host_start_time": -1},
        {"started_at": "456"},
        {"started_at": -1},
        {"command": None},
        {"cwd": 123},
        {"task_id": None},
        {"watcher_platform": None},
        {"watcher_interval": True},
        {"watcher_interval": -1},
        {"notify_on_complete": 1},
        {"watch_patterns": "pattern"},
        {"watch_patterns": [1]},
    ],
)
def test_process_checkpoint_snapshot_rejects_invalid_selected_projection(
    tmp_path: Path,
    overrides: dict,
) -> None:
    checkpoint = tmp_path / "processes.json"
    _write_checkpoint(checkpoint, [_candidate(**overrides)])

    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )

    assert exc_info.value.reason == "incompatible_process_checkpoint"


def test_process_checkpoint_snapshot_ignores_malformed_foreign_projection(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "processes.json"
    _write_checkpoint(
        checkpoint,
        [
            _candidate(
                "proc_foreign",
                session_key="foreign",
                command=None,
                future_field=True,
            ),
            _candidate(),
        ],
    )

    assert len(
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )
    ) == 1


def test_process_checkpoint_snapshot_rejects_duplicate_selected_identity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "processes.json"
    _write_checkpoint(
        checkpoint,
        [
            _candidate(),
            _candidate(pid=123, host_start_time=456),
        ],
    )

    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )

    assert exc_info.value.reason == "incompatible_process_checkpoint"


def test_process_checkpoint_snapshot_enforces_entry_bound(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "processes.json"
    entries = [
        _candidate(f"proc_{index}", pid=index + 1)
        for index in range(startup._MAX_PROCESS_CHECKPOINT_ENTRIES)
    ]
    _write_checkpoint(checkpoint, entries)
    assert len(
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )
    ) == startup._MAX_PROCESS_CHECKPOINT_ENTRIES

    _write_checkpoint(
        checkpoint,
        entries + [_candidate("proc_overflow", pid=len(entries) + 1)],
    )
    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )
    assert exc_info.value.reason == "process_checkpoint_limit_exceeded"

    foreign_entries = [
        _candidate(
            f"proc_foreign_{index}",
            session_key="foreign",
            pid=index + 1,
        )
        for index in range(startup._MAX_PROCESS_CHECKPOINT_ENTRIES + 1)
    ]
    _write_checkpoint(checkpoint, [*foreign_entries, _candidate()])
    assert len(
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )
    ) == 1


def test_process_checkpoint_snapshot_enforces_exact_byte_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "processes.json"
    raw = _write_checkpoint(checkpoint, [_candidate()])
    monkeypatch.setattr(startup, "_MAX_PROCESS_CHECKPOINT_BYTES", len(raw))
    assert len(
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )
    ) == 1

    monkeypatch.setattr(startup, "_MAX_PROCESS_CHECKPOINT_BYTES", len(raw) - 1)
    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )
    assert exc_info.value.reason == "process_checkpoint_limit_exceeded"


@pytest.mark.parametrize(
    "unsafe_kind",
    ["directory", "hardlink", "fifo"],
)
def test_process_checkpoint_snapshot_rejects_unsafe_inode(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    checkpoint = tmp_path / "processes.json"
    if unsafe_kind == "directory":
        checkpoint.mkdir()
    elif unsafe_kind == "fifo":
        os.mkfifo(checkpoint)
    else:
        target = tmp_path / "target.json"
        _write_checkpoint(target, [_candidate()])
        os.link(target, checkpoint)

    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )

    assert exc_info.value.reason == "process_checkpoint_unavailable"


def test_process_checkpoint_snapshot_preserves_writer_supported_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    checkpoint = tmp_path / "processes.json"
    _write_checkpoint(target, [_candidate()])
    checkpoint.symlink_to(target)

    observations = startup._read_process_checkpoint_snapshot(
        checkpoint,
        shadow_session_key=_SESSION_KEY,
    )

    assert len(observations) == 1
    assert checkpoint.is_symlink()
    assert checkpoint.resolve() == target


def test_process_checkpoint_snapshot_rejects_in_place_read_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "processes.json"
    _write_checkpoint(checkpoint, [_candidate()])
    real_fstat = os.fstat
    call_count = 0

    def changed_fstat(descriptor: int) -> os.stat_result:
        nonlocal call_count
        call_count += 1
        metadata = real_fstat(descriptor)
        if call_count == 2:
            values = list(metadata)
            values[8] += 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr(startup.os, "fstat", changed_fstat)

    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )

    assert exc_info.value.reason == "process_checkpoint_unavailable"


def test_process_checkpoint_snapshot_errors_do_not_expose_raw_values(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint = tmp_path / "processes.json"
    _write_checkpoint(
        checkpoint,
        [_candidate(command=None, watcher_chat_id="toxic-route-secret")],
    )

    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_process_checkpoint_snapshot(
            checkpoint,
            shadow_session_key=_SESSION_KEY,
        )

    combined = repr(exc_info.value) + "\n" + caplog.text
    assert "toxic-command-secret" not in combined
    assert "toxic-route-secret" not in combined
