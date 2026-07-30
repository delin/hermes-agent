"""Bounded external Task Fence shadow status rendering."""

import hashlib
import json
import os
from types import SimpleNamespace

import hermes_cli.gateway as gateway
import hermes_state
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from task_fence import (
    IngressEnvelope,
    OperationDescriptor,
    OperationKind,
    TASK_FENCE_ACTIONS,
    TaskFencePolicy,
    TerminalReason,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ingress(
    conversation_id: str,
    action: str,
    source_event_id: str,
    *,
    task_id: str | None = None,
) -> IngressEnvelope:
    return IngressEnvelope(
        source="gateway:test:task-fence-status",
        source_event_id=source_event_id,
        conversation_id=conversation_id,
        action=TASK_FENCE_ACTIONS[action],
        payload_hash=_hash(source_event_id),
        task_id=task_id,
        terminal_reason=TerminalReason.STOPPED if action == "stop" else None,
    )


def _started_lane(path, conversation_id: str, marker: str):
    db = SessionDB(path)
    accepted = db.accept_task_fence_ingress(
        _ingress(conversation_id, "initial_submit", f"{marker}-initial")
    )
    generation = db.reserve_task_fence_generation(accepted)
    assert db.finish_task_fence_generation(generation, state="committed")
    invocation = generation.for_invocation(f"tfiv-{marker}")
    operation = OperationDescriptor(
        invocation_id=invocation.invocation_id,
        kind=OperationKind.TOOL,
        adapter="registry:write_file",
        invocation_fingerprint=_hash(marker),
    )
    policy = TaskFencePolicy(db)
    admitted = policy.admit_operation(invocation, operation)
    started = policy.authorize_and_start(
        invocation,
        operation,
        admitted.permit_id,
    )
    assert started.attempt_id is not None
    return db, accepted, started


def _recover(db: SessionDB):
    store = db.inspect_task_fence_store()
    assert store.runtime_epoch is not None
    assert store.mode_generation is not None
    return db.recover_task_fence_state(
        expected_runtime_epoch=store.runtime_epoch,
        expected_mode_generation=store.mode_generation,
    )


def _write_shadow_config(home, conversation_id: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "gateway:\n"
        "  task_fence:\n"
        f"    shadow_session_key: {json.dumps(conversation_id)}\n",
        encoding="utf-8",
    )


def _stub_manual_status(monkeypatch) -> None:
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_windows", lambda: False)
    monkeypatch.setattr(gateway, "is_termux", lambda: False)
    monkeypatch.setattr(gateway, "is_wsl", lambda: False)
    monkeypatch.setattr(
        gateway,
        "get_gateway_runtime_snapshot",
        lambda system=False: gateway.GatewayRuntimeSnapshot(manager="manual process"),
    )
    monkeypatch.setattr(gateway, "_runtime_health_lines", lambda: [])
    monkeypatch.setattr(gateway, "_print_other_profiles_gateway_status", lambda: None)


def _run_status(home) -> None:
    token = set_hermes_home_override(home)
    try:
        gateway.gateway_command(
            SimpleNamespace(
                gateway_command="status",
                deep=False,
                full=False,
                system=False,
            )
        )
    finally:
        reset_hermes_home_override(token)


def _db_identity(path):
    stat = path.stat()
    return path.read_bytes(), stat.st_mtime_ns, stat.st_size


def test_gateway_status_default_off_does_not_open_or_create_store(
    tmp_path,
    monkeypatch,
    capsys,
):
    _stub_manual_status(monkeypatch)
    (tmp_path / "config.yaml").write_text("gateway: {}\n", encoding="utf-8")

    def unexpected_plugin_discovery():
        raise SystemExit("status must not discover plugins")

    monkeypatch.setattr(
        "hermes_cli.plugins.discover_plugins",
        unexpected_plugin_discovery,
    )

    def unexpected_session_db(*args, **kwargs):
        raise AssertionError("default-off status must not open SessionDB")

    monkeypatch.setattr(hermes_state, "SessionDB", unexpected_session_db)
    _run_status(tmp_path)

    output = capsys.readouterr().out
    assert "Gateway is not running" in output
    assert "Task Fence shadow (active non-terminal task only):" in output
    assert "State: not configured" in output
    assert not (tmp_path / "state.db").exists()


def test_gateway_status_renders_bounded_current_profile_without_writes_or_leaks(
    tmp_path,
    monkeypatch,
    capsys,
):
    _stub_manual_status(monkeypatch)
    conversation_id = "profile-a:raw-secret-lane"
    _write_shadow_config(tmp_path, conversation_id)
    db, accepted, started = _started_lane(
        tmp_path / "state.db",
        conversation_id,
        "status-secret-source",
    )
    held = []
    for index in range(65):
        held.append(
            db.accept_task_fence_ingress(
                _ingress(
                    conversation_id,
                    "comment_hold",
                    f"status-secret-held-{index}",
                    task_id=accepted.task_id,
                )
            )
        )
    expected = db.inspect_task_fence_conversation(conversation_id)
    assert expected.pending_inputs_truncated is True
    assert expected.started_attempts_truncated is False
    db.close()
    db_path = tmp_path / "state.db"
    before = _db_identity(db_path)

    _run_status(tmp_path)

    output = capsys.readouterr().out
    assert f"Conversation: sha256:{expected.conversation_fingerprint}" in output
    assert expected.task is not None
    assert expected.task.task_id in output
    assert started.attempt_id in output
    assert f"{started.attempt_id}@{accepted.opened_run_id}" in output
    assert expected.pending_input_ids[0] == held[0].event_id
    assert held[0].event_id in output
    assert "showing first 64; truncated" in output
    assert "Open incident for active task: none" in output
    assert "terminal tasks, history, and their incidents are not inspected" in output
    assert conversation_id not in output
    assert "status-secret-source" not in output
    assert "status-secret-held" not in output
    assert "healthy" not in output.lower()
    assert "safe" not in output.lower()
    assert _db_identity(db_path) == before


def test_gateway_status_scopes_incident_and_terminal_history_honestly(
    tmp_path,
    monkeypatch,
    capsys,
):
    _stub_manual_status(monkeypatch)
    conversation_id = "profile-incident:raw-secret-lane"
    _write_shadow_config(tmp_path, conversation_id)
    db, accepted, started = _started_lane(
        tmp_path / "state.db",
        conversation_id,
        "status-incident-secret",
    )
    _recover(db)
    active = db.inspect_task_fence_conversation(conversation_id)
    assert active.open_incident is not None
    incident_id = active.open_incident.incident_id
    db.close()

    _run_status(tmp_path)
    active_output = capsys.readouterr().out
    assert f"Open incident for active task: {incident_id}" in active_output
    assert "reason=outcome_unknown" in active_output
    assert started.attempt_id in active_output
    assert conversation_id not in active_output

    db = SessionDB(tmp_path / "state.db")
    stopped = db.accept_task_fence_ingress(
        _ingress(
            conversation_id,
            "stop",
            "status-terminal-secret",
            task_id=accepted.task_id,
        )
    )
    assert stopped.task_projection is not None
    assert stopped.task_projection.status == "stopped"
    db.close()

    _run_status(tmp_path)
    terminal_output = capsys.readouterr().out
    assert "Inspection: no_active_task" in terminal_output
    assert "Active task: none" in terminal_output
    assert (
        "terminal tasks, history, and their incidents are not inspected"
        in terminal_output
    )
    assert "Cohort: not materialized" not in terminal_output
    assert incident_id not in terminal_output
    assert "Open incident for active task: none" not in terminal_output
    assert conversation_id not in terminal_output


def test_gateway_status_configured_missing_store_is_nonfatal_and_secret_free(
    tmp_path,
    monkeypatch,
    capsys,
):
    _stub_manual_status(monkeypatch)
    conversation_id = "missing-store:raw-secret-lane"
    _write_shadow_config(tmp_path, conversation_id)

    _run_status(tmp_path)

    output = capsys.readouterr().out
    assert "Gateway is not running" in output
    assert "State: inspection unavailable (state_db_not_found)" in output
    assert conversation_id not in output
    assert not (tmp_path / "state.db").exists()


def test_gateway_status_incompatible_store_is_nonfatal_without_partial_state(
    tmp_path,
    monkeypatch,
    capsys,
):
    _stub_manual_status(monkeypatch)
    conversation_id = "incompatible-store:raw-secret-lane"
    _write_shadow_config(tmp_path, conversation_id)
    db, accepted, _ = _started_lane(
        tmp_path / "state.db",
        conversation_id,
        "incompatible-secret",
    )
    db._conn.execute(
        "UPDATE task_fence_control SET store_schema_version = "
        "store_schema_version + 1 WHERE singleton = 1"
    )
    db._conn.commit()
    db.close()

    _run_status(tmp_path)

    output = capsys.readouterr().out
    assert "Gateway is not running" in output
    assert "State: inspection unavailable (unsupported_store_schema)" in output
    assert accepted.task_id not in output
    assert conversation_id not in output


def test_gateway_status_system_scope_adopts_unit_pinned_profile(
    tmp_path,
    monkeypatch,
    capsys,
):
    caller_home = tmp_path / "caller"
    target_home = tmp_path / "target"
    caller_key = "caller:raw-secret-lane"
    target_key = "target:raw-secret-lane"
    _write_shadow_config(caller_home, caller_key)
    _write_shadow_config(target_home, target_key)
    caller_db, caller_accepted, _ = _started_lane(
        caller_home / "state.db",
        caller_key,
        "caller-secret",
    )
    caller_db.close()
    target_db, target_accepted, _ = _started_lane(
        target_home / "state.db",
        target_key,
        "target-secret",
    )
    target_inspection = target_db.inspect_task_fence_conversation(target_key)
    target_db.close()

    system_unit = tmp_path / "hermes-gateway.service"
    system_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={target_home}"\n',
        encoding="utf-8",
    )
    user_unit = tmp_path / "missing-user.service"
    monkeypatch.setenv("HERMES_HOME", str(caller_home))
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_windows", lambda: False)
    monkeypatch.setattr(
        gateway,
        "get_systemd_unit_path",
        lambda system=False: system_unit if system else user_unit,
    )

    def snapshot(system=False):
        assert os.environ["HERMES_HOME"] == str(target_home)
        return gateway.GatewayRuntimeSnapshot(
            manager="systemd (system)",
            service_installed=True,
            service_running=True,
            service_scope="system",
        )

    monkeypatch.setattr(gateway, "get_gateway_runtime_snapshot", snapshot)
    monkeypatch.setattr(
        gateway,
        "systemd_status",
        lambda deep=False, system=False, full=False: print(
            "system gateway service status"
        ),
    )
    monkeypatch.setattr(
        gateway, "_print_gateway_process_mismatch", lambda snapshot: None
    )
    monkeypatch.setattr(gateway, "_print_other_profiles_gateway_status", lambda: None)

    gateway.gateway_command(
        SimpleNamespace(
            gateway_command="status",
            deep=False,
            full=False,
            system=True,
        )
    )

    output = capsys.readouterr().out
    assert "system gateway service status" in output
    assert (
        f"Conversation: sha256:{target_inspection.conversation_fingerprint}" in output
    )
    assert target_accepted.task_id in output
    assert target_accepted.opened_run_id in output
    assert target_accepted.event_id in output
    assert caller_accepted.task_id not in output
    assert target_key not in output
    assert caller_key not in output
