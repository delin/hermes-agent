import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.main as cli_main
import task_fence_config
import task_fence_runtime
from hermes_state import SessionDB


_SELECTOR = "selected-root"


def _args(**overrides):
    values = {
        "command": "chat",
        "resume": _SELECTOR,
        "continue_last": None,
        "query": None,
        "image": None,
        "oneshot": None,
        "tui": False,
        "cli": True,
        "skills": None,
        "source": None,
        "ignore_user_config": False,
        "safe_mode": False,
        "worktree": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_active_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[str],
) -> None:
    import atexit
    import gateway.status
    import hermes_cli.config
    import hermes_constants

    monkeypatch.setattr(
        hermes_constants,
        "get_process_hermes_home",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        hermes_constants,
        "get_default_hermes_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        task_fence_config,
        "load_task_fence_shadow_conversation_key",
        lambda home=None: _SELECTOR,
    )
    monkeypatch.setattr(
        hermes_cli.config,
        "load_config_readonly",
        lambda: {"worktree": False},
    )
    monkeypatch.setattr(
        cli_main,
        "_resolve_use_tui",
        lambda args: bool(getattr(args, "tui", False)),
    )
    monkeypatch.setattr(
        gateway.status,
        "is_gateway_runtime_lock_active",
        lambda: events.append("gateway_lock") or False,
    )
    monkeypatch.setattr(
        task_fence_runtime,
        "acquire_task_fence_owner_lock",
        lambda home=None: events.append("owner") or True,
    )
    monkeypatch.setattr(
        task_fence_runtime,
        "prepare_task_fence_shadow_startup",
        lambda key, **kwargs: events.append("recover"),
    )
    monkeypatch.setattr(
        task_fence_runtime,
        "release_task_fence_owner_lock",
        lambda: events.append("release"),
    )
    monkeypatch.setattr(
        atexit,
        "register",
        lambda callback: events.append("atexit"),
    )


def test_management_and_default_off_do_not_probe_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        task_fence_config,
        "load_task_fence_shadow_conversation_key",
        lambda home=None: (_ for _ in ()).throw(
            AssertionError("management command loaded Task Fence config")
        ),
    )
    assert cli_main._prepare_task_fence_cli_startup(
        _args(command="gateway")
    ) == ""

    import hermes_constants

    monkeypatch.setattr(
        hermes_constants,
        "get_process_hermes_home",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        task_fence_config,
        "load_task_fence_shadow_conversation_key",
        lambda home=None: "",
    )
    monkeypatch.setattr(
        cli_main,
        "_resolve_use_tui",
        lambda args: (_ for _ in ()).throw(
            AssertionError("default-off startup probed runtime mode")
        ),
    )
    args = _args()

    assert cli_main._prepare_task_fence_cli_startup(args) == ""
    assert args.task_fence_shadow_conversation_key == ""


def test_active_cli_recovers_before_exact_compression_root_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = SessionDB(tmp_path / "state.db")
    database.create_session(_SELECTOR, source="cli")
    database.end_session(_SELECTOR, "compression")
    database.create_session(
        "selected-tip",
        source="cli",
        parent_session_id=_SELECTOR,
    )
    database.close()

    events: list[str] = []
    _install_active_preflight(monkeypatch, tmp_path, events)
    original_get_session = SessionDB.get_session
    original_get_root = SessionDB.get_task_fence_compression_root

    def traced_get_session(self, session_id):
        events.append("session")
        return original_get_session(self, session_id)

    def traced_get_root(self, session_id):
        events.append("root")
        return original_get_root(self, session_id)

    monkeypatch.setattr(SessionDB, "get_session", traced_get_session)
    monkeypatch.setattr(
        SessionDB,
        "get_task_fence_compression_root",
        traced_get_root,
    )
    args = _args(resume="selected-tip")

    assert cli_main._prepare_task_fence_cli_startup(args) == _SELECTOR
    assert args.task_fence_shadow_conversation_key == _SELECTOR
    assert events == [
        "gateway_lock",
        "owner",
        "recover",
        "session",
        "root",
        "atexit",
    ]


def test_active_cli_mismatch_releases_owner_and_refuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = SessionDB(tmp_path / "state.db")
    database.create_session("other", source="cli")
    database.close()

    events: list[str] = []
    _install_active_preflight(monkeypatch, tmp_path, events)

    with pytest.raises(SystemExit, match="1"):
        cli_main._prepare_task_fence_cli_startup(_args(resume="other"))

    assert events[-1] == "release"
    assert "selected_conversation_mismatch" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"resume": None}, "explicit_resume_required"),
        ({"continue_last": True}, "explicit_resume_required"),
        ({"query": "one shot"}, "noninteractive_cli_unsupported"),
        ({"tui": True}, "tui_unsupported"),
        ({"skills": ["custom"]}, "extended_cli_ingress_unsupported"),
    ],
)
def test_active_cli_refuses_unbounded_ingress_before_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    overrides: dict,
    reason: str,
) -> None:
    events: list[str] = []
    _install_active_preflight(monkeypatch, tmp_path, events)

    with pytest.raises(SystemExit, match="1"):
        cli_main._prepare_task_fence_cli_startup(_args(**overrides))

    assert "owner" not in events
    assert reason in capsys.readouterr().err


def test_gateway_runtime_refusal_does_not_attempt_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gateway.status

    events: list[str] = []
    _install_active_preflight(monkeypatch, tmp_path, events)
    monkeypatch.setattr(
        gateway.status,
        "is_gateway_runtime_lock_active",
        lambda: events.append("gateway_lock") or True,
    )

    with pytest.raises(SystemExit, match="1"):
        cli_main._prepare_task_fence_cli_startup(_args())

    assert "owner" not in events
    assert events[-1] == "release"
    assert "gateway_runtime_active" in capsys.readouterr().err


def _direct_call_lines(function_name: str, callee: str) -> list[int]:
    tree = ast.parse(Path(cli_main.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return sorted(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callee
    )


def test_all_cli_entrypoints_fence_before_consumers() -> None:
    assert _direct_call_lines(
        "main",
        "_prepare_task_fence_cli_startup",
    )[0] < _direct_call_lines("main", "_prepare_agent_startup")[0]
    assert _direct_call_lines(
        "_try_termux_fast_cli_launch",
        "_prepare_task_fence_cli_startup",
    )[0] < _direct_call_lines(
        "_try_termux_fast_cli_launch",
        "_prepare_agent_startup",
    )[0]
    assert _direct_call_lines(
        "_try_termux_fast_tui_launch",
        "_prepare_task_fence_cli_startup",
    )[0] < _direct_call_lines(
        "_try_termux_fast_tui_launch",
        "cmd_chat",
    )[0]
