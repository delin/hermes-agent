"""Regression tests for task/session cwd propagation in terminal_tool."""

import json
from types import SimpleNamespace

import tools.terminal_tool as terminal_tool
from task_fence import TaskFencePolicy, bind_task_fence_policy


def _minimal_terminal_config(cwd="/default"):
    return {
        "env_type": "local",
        "cwd": cwd,
        "timeout": 60,
        "lifetime_seconds": 3600,
    }


def test_foreground_command_uses_registered_task_cwd_for_existing_environment(monkeypatch):
    """ACP can update task cwd after the local env exists; foreground must honor it."""
    calls = []

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "ok", "returncode": 0}

    task_id = "acp-session-1"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/acp"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))

    assert result["exit_code"] == 0
    assert calls == [("pwd", {"timeout": 60, "cwd": "/workspace/acp", "bounded_capture": True})]


def test_explicit_workdir_still_wins_over_registered_task_cwd(monkeypatch):
    calls = []

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append(kwargs)
            return {"output": "ok", "returncode": 0}

    task_id = "acp-session-1"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/acp"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            command="pwd",
            task_id=task_id,
            workdir="/explicit/workdir",
        )
    )

    assert result["exit_code"] == 0
    assert calls == [{"timeout": 60, "cwd": "/explicit/workdir", "bounded_capture": True}]


def test_background_command_prefers_recorded_session_cwd_over_init_time_cwd(monkeypatch):
    """Background process launches must also use the recorded session cwd."""

    class FakeEnv:
        env = {}
        cwd = "/workspace/live"

    class FakeRegistry:
        def __init__(self):
            self.calls = []
            self.pending_watchers = []

        def spawn_local(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(id="proc_test", pid=1234)

    import tools.process_registry as process_registry_mod

    registry = FakeRegistry()
    task_id = "session-live-cwd-bg"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/init"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config(cwd="/workspace/init"))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: value or "default")
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(process_registry_mod, "process_registry", registry)
    terminal_tool.record_session_cwd(task_id, "/workspace/live")

    result = json.loads(
        terminal_tool.terminal_tool(
            command="sleep 1",
            task_id=task_id,
            background=True,
        )
    )

    assert result["exit_code"] == 0
    # session_key falls back to the raw task_id when no gateway contextvar is set
    # (it doesn't propagate to tool-worker threads), so process.kill / stop can
    # still find and terminate this background process.
    assert registry.calls == [{
        "command": "sleep 1",
        "cwd": "/workspace/live",
        "task_id": task_id,
        "session_key": task_id,
        "env_vars": {},
        "use_pty": False,
    }]


def test_safe_getcwd_falls_back_to_home_when_no_terminal_cwd(monkeypatch):
    def _boom():
        raise FileNotFoundError()

    monkeypatch.setattr(terminal_tool.os, "getcwd", _boom)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool.os.path, "expanduser", lambda p: "/home/me")
    assert terminal_tool._safe_getcwd() == "/home/me"


def test_task_fence_background_routes_root_without_changing_tip_cwd(
    monkeypatch,
):
    class FakeEnv:
        env = {}
        cwd = "/workspace/live-tip"

    class FakeRegistry:
        def __init__(self):
            self.calls = []
            self.pending_watchers = []

        def spawn_local(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(id="proc_task_fence", pid=4321)

    import tools.process_registry as process_registry_mod

    registry = FakeRegistry()
    tip = "compression-tip"
    root = "compression-root"
    monkeypatch.setattr(terminal_tool, "_active_environments", {tip: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(
        terminal_tool,
        "_task_env_overrides",
        {tip: {"cwd": "/workspace/init-tip"}},
    )
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: _minimal_terminal_config(cwd="/workspace/default"),
    )
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_resolve_container_task_id",
        lambda value: value or "default",
    )
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(process_registry_mod, "process_registry", registry)
    terminal_tool.record_session_cwd(tip, "/workspace/live-tip")
    terminal_tool.record_session_cwd(root, "/workspace/wrong-root")

    with bind_task_fence_policy(
        TaskFencePolicy(object(), runtime_conversation_key=root)
    ):
        result = json.loads(
            terminal_tool.terminal_tool(
                command="sleep 1",
                task_id=tip,
                background=True,
            )
        )

    assert result["exit_code"] == 0
    assert registry.calls[0]["cwd"] == "/workspace/live-tip"
    assert registry.calls[0]["session_key"] == root
    assert registry.calls[0]["task_id"] == tip
