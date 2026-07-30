import json
import os
from pathlib import Path
from typing import Any

import pytest

from gateway import task_fence_startup as startup
from gateway.config import GatewayConfig
from hermes_state import SessionDB


_PLATFORM = "linux/amd64"
_SHADOW_SESSION_KEY = "slack:workspace:channel:user"
_COMMIT = "a" * 40
_ARTIFACT_DIGEST = "sha256:" + "b" * 64
_LOCK_DIGEST = "sha256:" + "c" * 64


def _receipt_payload(*, platform: str = _PLATFORM) -> dict[str, str]:
    return {
        "schema": "hermes.task-fence.tested-artifact-receipt/v1",
        "target_platform": platform,
        "tested_artifact_commit": _COMMIT,
        "tested_artifact_checksum": _ARTIFACT_DIGEST,
        "dependency_lock_fingerprint": _LOCK_DIGEST,
    }


def _receipt_bytes(payload: dict[str, str] | None = None) -> bytes:
    return (
        json.dumps(
            payload or _receipt_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _write_receipt(path: Path, raw: bytes | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_receipt_bytes() if raw is None else raw)
    path.chmod(0o444)


def _pretend_receipt_is_root_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    _pretend_receipt_has_uid(monkeypatch, 0)


def _pretend_receipt_has_uid(
    monkeypatch: pytest.MonkeyPatch,
    uid: int,
) -> None:
    real_fstat = os.fstat

    def root_owned_fstat(descriptor: int) -> os.stat_result:
        values = list(real_fstat(descriptor))
        values[4] = uid
        return os.stat_result(values)

    monkeypatch.setattr(startup.os, "fstat", root_owned_fstat)


def _install_start_gateway_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[str],
    runner_factory: Any,
):
    import atexit

    from gateway import run as gateway_run
    from gateway import status as gateway_status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr("gateway.code_skew.record_boot_fingerprint", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr(
        "hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path
    )
    monkeypatch.setattr(
        "hermes_cli.security_audit_startup.log_startup_security_warnings",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "hermes_cli.nous_auth_keepalive.start_nous_auth_keepalive",
        lambda: None,
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: None)
    monkeypatch.setattr(gateway_run, "_run_planned_stop_watcher", lambda *args: None)
    monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: None)

    monkeypatch.setattr(gateway_status, "get_running_pid", lambda: None)

    def acquire_lock() -> bool:
        events.append("lock")
        return True

    def write_pid() -> None:
        events.append("pid")

    def remove_pid() -> None:
        events.append("remove_pid")

    def release_lock() -> None:
        events.append("release_lock")

    monkeypatch.setattr(gateway_status, "acquire_gateway_runtime_lock", acquire_lock)
    monkeypatch.setattr(gateway_status, "write_pid_file", write_pid)
    monkeypatch.setattr(gateway_status, "remove_pid_file", remove_pid)
    monkeypatch.setattr(gateway_status, "release_gateway_runtime_lock", release_lock)
    monkeypatch.setattr(gateway_run, "GatewayRunner", runner_factory)
    return gateway_run


def test_parse_receipt_returns_typed_identity() -> None:
    identity = startup._parse_receipt(_receipt_bytes(), expected_platform=_PLATFORM)

    assert identity.tested_artifact_commit == _COMMIT
    assert identity.tested_artifact_checksum == _ARTIFACT_DIGEST
    assert identity.dependency_lock_fingerprint == _LOCK_DIGEST


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (
            _receipt_bytes({**_receipt_payload(), "unexpected": "field"}),
            "invalid_receipt_fields",
        ),
        (
            _receipt_bytes().replace(
                b'"schema":', b'"schema":"duplicate","schema":', 1
            ),
            "invalid_receipt_json",
        ),
        (
            _receipt_bytes(_receipt_payload(platform="linux/arm64")),
            "receipt_platform_mismatch",
        ),
        (
            _receipt_bytes({**_receipt_payload(), "schema": "unsupported-receipt/v2"}),
            "unsupported_receipt_schema",
        ),
        (
            _receipt_bytes({**_receipt_payload(), "tested_artifact_commit": "0" * 40}),
            "invalid_tested_artifact_commit",
        ),
        (b"{", "invalid_receipt_json"),
        (b"x" * (startup._MAX_RECEIPT_BYTES + 1), "invalid_receipt_size"),
    ],
    ids=[
        "closed-fields",
        "duplicate-key",
        "wrong-platform",
        "wrong-schema",
        "zero-identity",
        "malformed",
        "oversize",
    ],
)
def test_parse_receipt_rejects_noncanonical_input(raw: bytes, reason: str) -> None:
    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._parse_receipt(raw, expected_platform=_PLATFORM)

    assert exc_info.value.reason == reason


def test_read_receipt_uses_fixed_root_owned_read_only_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert startup._TESTED_ARTIFACT_RECEIPT_PATH == Path(
        "/run/hermes/task-fence/tested-artifact.json"
    )
    receipt_path = tmp_path / "tested-artifact.json"
    platform = startup._runtime_platform()
    _write_receipt(receipt_path, _receipt_bytes(_receipt_payload(platform=platform)))
    _pretend_receipt_is_root_owned(monkeypatch)
    monkeypatch.setattr(startup, "_TESTED_ARTIFACT_RECEIPT_PATH", receipt_path)

    identity = startup._read_receipt()

    assert identity.tested_artifact_commit == _COMMIT
    assert receipt_path.stat().st_mode & 0o777 == 0o444


def test_read_receipt_does_not_follow_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_path = tmp_path / "target.json"
    receipt_path = tmp_path / "tested-artifact.json"
    _write_receipt(target_path)
    receipt_path.symlink_to(target_path)
    monkeypatch.setattr(startup, "_TESTED_ARTIFACT_RECEIPT_PATH", receipt_path)

    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_receipt()

    assert exc_info.value.reason == "receipt_unavailable"


@pytest.mark.parametrize(
    "unsafe_kind",
    ["directory", "foreign-owner", "writable", "hardlink"],
)
def test_read_receipt_rejects_unsafe_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    receipt_path = tmp_path / "tested-artifact.json"
    if unsafe_kind == "directory":
        receipt_path.mkdir()
        _pretend_receipt_is_root_owned(monkeypatch)
    else:
        _write_receipt(receipt_path)
        if unsafe_kind == "foreign-owner":
            _pretend_receipt_has_uid(monkeypatch, 12345)
        else:
            _pretend_receipt_is_root_owned(monkeypatch)
        if unsafe_kind == "writable":
            receipt_path.chmod(0o644)
        elif unsafe_kind == "hardlink":
            os.link(receipt_path, tmp_path / "receipt-alias.json")
    monkeypatch.setattr(startup, "_TESTED_ARTIFACT_RECEIPT_PATH", receipt_path)

    with pytest.raises(startup.TaskFenceStartupUnavailable) as exc_info:
        startup._read_receipt()

    assert exc_info.value.reason == "unsafe_receipt_file"


@pytest.mark.asyncio
async def test_start_gateway_commits_recovery_before_runner_reopens_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    platform = startup._runtime_platform()
    receipt_path = tmp_path / "tested-artifact.json"
    _write_receipt(receipt_path, _receipt_bytes(_receipt_payload(platform=platform)))
    _pretend_receipt_is_root_owned(monkeypatch)
    monkeypatch.setattr(startup, "_TESTED_ARTIFACT_RECEIPT_PATH", receipt_path)

    real_parse = startup._parse_receipt

    def traced_parse(raw: bytes, *, expected_platform: str):
        events.append("receipt")
        return real_parse(raw, expected_platform=expected_platform)

    monkeypatch.setattr(startup, "_parse_receipt", traced_parse)

    real_recover = SessionDB.recover_task_fence_state

    def traced_recover(self: SessionDB, **kwargs):
        events.append("composite")
        return real_recover(self, **kwargs)

    monkeypatch.setattr(SessionDB, "recover_task_fence_state", traced_recover)

    queued_event = json.dumps({
        "type": "async_delegation",
        "delegation_id": "deadbeef",
        "status": "completed",
    })
    seed = SessionDB(tmp_path / "state.db")
    seed._conn.execute(
        "INSERT INTO async_delegations ("
        "delegation_id, origin_session, state, dispatched_at, completed_at, "
        "updated_at, event_json, result_json, delivery_state"
        ") VALUES (?, ?, 'completed', 1.0, 2.0, 2.0, ?, ?, 'pending')",
        ("deadbeef", _SHADOW_SESSION_KEY, queued_event, queued_event),
    )
    seed._conn.execute(
        "INSERT INTO async_delegations ("
        "delegation_id, origin_session, state, dispatched_at, updated_at, "
        "task_json, delivery_state"
        ") VALUES ('deadbee0', ?, 'running', 3.0, 3.0, '{}', 'pending')",
        (_SHADOW_SESSION_KEY,),
    )
    seed._conn.commit()
    seed.close()
    from gateway import delivery_ledger

    monkeypatch.setattr(
        delivery_ledger,
        "_db_path",
        lambda: tmp_path / "state.db",
    )
    delivery_ledger.record_obligation(
        obligation_id="cafe00000000000000000001",
        session_key=_SHADOW_SESSION_KEY,
        platform="slack",
        chat_id="private-chat",
        thread_id="private-thread",
        content="private pending final response",
    )
    delivery_seed = SessionDB(tmp_path / "state.db")
    try:
        delivery_before = tuple(
            tuple(row)
            for row in delivery_seed._conn.execute(
                "SELECT * FROM delivery_obligations ORDER BY obligation_id"
            )
        )
    finally:
        delivery_seed.close()

    class ReopeningRunner:
        def __init__(self, config: GatewayConfig):
            events.append("runner")
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

            database = SessionDB(tmp_path / "state.db")
            try:
                store = database.inspect_task_fence_store()
            finally:
                database.close()
            assert store.compatible is True
            assert store.runtime_epoch == 1
            assert store.mode_generation == 0
            assert store.ever_enforced is False
            assert store.tested_artifact_commit == _COMMIT
            assert store.tested_artifact_checksum == _ARTIFACT_DIGEST
            assert store.dependency_lock_fingerprint == _LOCK_DIGEST
            database = SessionDB(tmp_path / "state.db")
            try:
                decisions = {
                    row["adapter"]: tuple(row)
                    for row in database._conn.execute(
                        "SELECT outcome, reason_code, decision_point, "
                        "operation_kind, adapter FROM task_fence_policy_decisions"
                    )
                }
                queued = tuple(
                    tuple(row)
                    for row in database._conn.execute(
                        "SELECT delegation_id, state, delivery_state, event_json "
                        "FROM async_delegations "
                        "WHERE delegation_id IN ('deadbee0', 'deadbeef') "
                        "ORDER BY delegation_id"
                    )
                )
                delivery_after = tuple(
                    tuple(row)
                    for row in database._conn.execute(
                        "SELECT * FROM delivery_obligations ORDER BY obligation_id"
                    )
                )
            finally:
                database.close()
            assert decisions == {
                "runtime:async_delegation_recovery_pending": (
                    "would_block",
                    "missing_provenance",
                    "admission",
                    "delivery",
                    "runtime:async_delegation_recovery_pending",
                ),
                "runtime:async_delegation_restore_ready": (
                    "would_block",
                    "missing_provenance",
                    "admission",
                    "delivery",
                    "runtime:async_delegation_restore_ready",
                ),
                "runtime:delivery_obligation_recovery_pending": (
                    "would_block",
                    "missing_provenance",
                    "admission",
                    "delivery",
                    "runtime:delivery_obligation_recovery_pending",
                ),
            }
            assert queued == (
                ("deadbee0", "running", "pending", None),
                ("deadbeef", "completed", "pending", queued_event),
            )
            assert delivery_after == delivery_before

        async def start(self) -> bool:
            events.append("start")
            assert self._platform_lock_takeover_on_start is False
            return True

    gateway_run = _install_start_gateway_shell(
        monkeypatch, tmp_path, events, ReopeningRunner
    )

    ok = await gateway_run.start_gateway(
        config=GatewayConfig(
            task_fence_shadow_session_key=_SHADOW_SESSION_KEY,
            sessions_dir=tmp_path / "sessions",
        ),
        verbosity=None,
    )

    assert ok is True
    assert events[:6] == [
        "lock",
        "pid",
        "receipt",
        "composite",
        "runner",
        "start",
    ]


@pytest.mark.asyncio
async def test_active_startup_failure_releases_claim_without_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    receipt_path = tmp_path / "tested-artifact.json"
    _write_receipt(receipt_path, b"{")
    _pretend_receipt_is_root_owned(monkeypatch)
    monkeypatch.setattr(startup, "_TESTED_ARTIFACT_RECEIPT_PATH", receipt_path)

    class RunnerMustNotOpen:
        def __init__(self, config: GatewayConfig):
            raise AssertionError("runner opened after failed startup barrier")

    gateway_run = _install_start_gateway_shell(
        monkeypatch, tmp_path, events, RunnerMustNotOpen
    )

    ok = await gateway_run.start_gateway(
        config=GatewayConfig(
            task_fence_shadow_session_key=_SHADOW_SESSION_KEY,
            sessions_dir=tmp_path / "sessions",
        ),
        verbosity=None,
    )

    assert ok is False
    assert events == ["lock", "pid", "remove_pid", "release_lock"]


@pytest.mark.asyncio
async def test_empty_shadow_key_preserves_legacy_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        startup,
        "_read_receipt",
        lambda: (_ for _ in ()).throw(
            AssertionError("legacy startup read Task Fence receipt")
        ),
    )

    class LegacyRunner:
        def __init__(self, config: GatewayConfig):
            events.append("runner")
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self) -> bool:
            events.append("start")
            return True

    gateway_run = _install_start_gateway_shell(
        monkeypatch, tmp_path, events, LegacyRunner
    )

    ok = await gateway_run.start_gateway(
        config=GatewayConfig(sessions_dir=tmp_path / "sessions"),
        verbosity=None,
    )

    assert ok is True
    assert events[:4] == ["lock", "pid", "runner", "start"]
    assert not (tmp_path / "state.db").exists()


@pytest.mark.asyncio
async def test_multiplex_shadow_key_refuses_before_receipt_or_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        startup,
        "_read_receipt",
        lambda: (_ for _ in ()).throw(
            AssertionError("multiplex refusal read Task Fence receipt")
        ),
    )

    class RunnerMustNotOpen:
        def __init__(self, config: GatewayConfig):
            raise AssertionError("runner opened for unsupported multiplex cohort")

    gateway_run = _install_start_gateway_shell(
        monkeypatch, tmp_path, events, RunnerMustNotOpen
    )

    ok = await gateway_run.start_gateway(
        config=GatewayConfig(
            multiplex_profiles=True,
            task_fence_shadow_session_key=_SHADOW_SESSION_KEY,
            sessions_dir=tmp_path / "sessions",
        ),
        verbosity=None,
    )

    assert ok is False
    assert events == ["lock", "pid", "remove_pid", "release_lock"]


@pytest.mark.asyncio
async def test_pid_claim_error_releases_lock_before_barrier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []

    class RunnerMustNotOpen:
        def __init__(self, config: GatewayConfig):
            raise AssertionError("runner opened after failed PID claim")

    gateway_run = _install_start_gateway_shell(
        monkeypatch, tmp_path, events, RunnerMustNotOpen
    )

    def fail_pid_claim() -> None:
        events.append("pid")
        raise OSError("private PID write fault")

    monkeypatch.setattr("gateway.status.write_pid_file", fail_pid_claim)

    with pytest.raises(OSError, match="private PID write fault"):
        await gateway_run.start_gateway(
            config=GatewayConfig(sessions_dir=tmp_path / "sessions"),
            verbosity=None,
        )

    assert events == ["lock", "pid", "remove_pid", "release_lock"]
