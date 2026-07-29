import asyncio
from collections import OrderedDict
from dataclasses import replace
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    load_gateway_config,
)
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    task_fence_sidecar_for_human_message,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_state import AsyncSessionDB, SessionDB
from task_fence import (
    TASK_FENCE_ACTIONS,
    TaskFenceIngressRejected,
    TaskFenceIngressSidecar,
    TaskFenceIngressUnavailable,
)


class _ShadowSlackAdapter(BasePlatformAdapter):
    def __init__(self, db: SessionDB | None = None):
        super().__init__(
            PlatformConfig(enabled=True, token="test"),
            Platform.SLACK,
        )
        self.db = db
        self.observations: list[tuple[str, int, str | None, int]] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, text, **kwargs):
        return SendResult(success=True, message_id="reply")

    async def get_chat_info(self, chat_id):
        return {}

    def observe(self, label: str) -> None:
        if self.db is None:
            self.observations.append((label, 0, None, 0))
            return
        with self.db._lock:
            ingress_count = self.db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_ingress"
            ).fetchone()[0]
            task = self.db._conn.execute(
                "SELECT status FROM task_fence_tasks "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            pending_count = self.db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_task_inputs "
                "WHERE state = 'pending'"
            ).fetchone()[0]
        self.observations.append(
            (
                label,
                ingress_count,
                task[0] if task is not None else None,
                pending_count,
            )
        )

    async def _keep_typing(self, chat_id, **kwargs):
        self.observe("typing")

    async def on_processing_start(self, event):
        self.observe("processing_start")


@pytest.fixture
def task_fence_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield db
    finally:
        db.close()


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="D123",
        chat_type="dm",
        user_id="U123",
        scope_id="T123",
    )


def _session_key() -> str:
    return build_session_key(_source())


def _event(
    text: str,
    event_id: str,
    *,
    message_type: MessageType = MessageType.TEXT,
) -> MessageEvent:
    event = MessageEvent(
        text=text,
        message_type=message_type,
        source=_source(),
        message_id=event_id,
    )
    event.task_fence_ingress = task_fence_sidecar_for_human_message(
        event,
        source="gateway:slack",
        source_event_id=f"event:T123:D123:{event_id}",
    )
    return event


def _runner(
    db,
    *,
    configured_key: str | None = None,
    authorized: bool = True,
) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(enabled=True, token="test")
        },
        task_fence_shadow_session_key=(
            _session_key() if configured_key is None else configured_key
        ),
    )
    runner._session_db = db
    runner.session_store = None
    runner._is_user_authorized = lambda _source: authorized
    return runner


def _count(db: SessionDB, table: str) -> int:
    with db._lock:
        return db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.mark.asyncio
async def test_gateway_passes_exact_acceptance_per_turn_without_cached_leak(
    task_fence_db,
    monkeypatch,
    tmp_path,
):
    runner = _runner(AsyncSessionDB(task_fence_db))
    adapter = _ShadowSlackAdapter(task_fence_db)
    runner.adapters = {Platform.SLACK: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    runner._refresh_fallback_model = lambda: None
    runner._apply_fallback_chain_to_agent = lambda *_args: None
    runner._init_cached_agent_for_turn = lambda *_args: None
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)

    event = _event("start", "1700000000.000000")
    await runner._accept_task_fence_gateway_ingress(event, _session_key())
    acceptance = event.task_fence_acceptance
    assert acceptance is not None

    observed = []
    created = []

    class FakeAgent:
        tools = []

        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            created.append(self)

        def run_conversation(
            self,
            message,
            conversation_history=None,
            task_id=None,
            **kwargs,
        ):
            observed.append(kwargs.get("task_fence_acceptance"))
            return {
                "final_response": "done",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"model": {"default": "test/model"}},
    )
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )

    kwargs = {
        "context_prompt": "",
        "history": [],
        "source": _source(),
        "session_id": "task-fence-session",
        "session_key": _session_key(),
    }
    first = await runner._run_agent(
        message="first",
        task_fence_acceptance=acceptance,
        **kwargs,
    )
    second = await runner._run_agent(message="second", **kwargs)

    assert first["final_response"] == "done"
    assert second["final_response"] == "done"
    assert len(created) == 1
    assert observed == [acceptance, None]


@pytest.mark.asyncio
async def test_cold_acceptance_commits_before_typing_hook_and_handler(
    task_fence_db,
):
    runner = _runner(AsyncSessionDB(task_fence_db))
    adapter = _ShadowSlackAdapter(task_fence_db)
    adapter.set_task_fence_ingress_handler(
        runner._accept_task_fence_gateway_ingress
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(event):
        adapter.observe("handler")
        started.set()
        await release.wait()
        return None

    adapter.set_message_handler(handler)
    event = _event("start the task", "1700000000.000001")

    await adapter.handle_message(event)
    task = adapter._session_tasks[_session_key()]
    await asyncio.wait_for(started.wait(), timeout=1)
    for _ in range(20):
        if len(adapter.observations) >= 3:
            break
        await asyncio.sleep(0)

    assert event.task_fence_acceptance is not None
    assert {item[0] for item in adapter.observations[:3]} == {
        "typing",
        "processing_start",
        "handler",
    }
    assert all(item[1:] == (1, "running", 0) for item in adapter.observations[:3])

    release.set()
    await task


@pytest.mark.asyncio
async def test_busy_plain_text_commits_hold_before_legacy_queue(
    task_fence_db,
):
    runner = _runner(AsyncSessionDB(task_fence_db))
    adapter = _ShadowSlackAdapter(task_fence_db)
    adapter.set_task_fence_ingress_handler(
        runner._accept_task_fence_gateway_ingress
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def handler(event):
        if event.text == "first":
            first_started.set()
            await release_first.wait()
        return None

    async def busy_handler(event, session_key):
        adapter.observe("busy_handler")
        assert session_key not in adapter._pending_messages
        return False

    adapter.set_message_handler(handler)
    adapter.set_busy_session_handler(busy_handler)
    first = _event("first", "1700000000.000010")
    follow_up = _event("follow up", "1700000000.000011")

    await adapter.handle_message(first)
    first_task = adapter._session_tasks[_session_key()]
    await asyncio.wait_for(first_started.wait(), timeout=1)
    await adapter.handle_message(follow_up)

    assert follow_up.task_fence_acceptance is not None
    assert adapter.observations[-1] == ("busy_handler", 2, "paused", 2)
    assert adapter._pending_messages[_session_key()] is follow_up

    release_first.set()
    await first_task
    await adapter.cancel_background_tasks()


@pytest.mark.parametrize("busy_text_mode", ("", "queue"))
@pytest.mark.asyncio
async def test_busy_text_coalescing_carries_latest_exact_acceptance(
    task_fence_db,
    busy_text_mode,
):
    runner = _runner(AsyncSessionDB(task_fence_db))
    adapter = _ShadowSlackAdapter(task_fence_db)
    adapter._busy_text_mode = busy_text_mode
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    adapter.set_task_fence_ingress_handler(
        runner._accept_task_fence_gateway_ingress
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def handler(event):
        if event.text == "first":
            first_started.set()
            await release_first.wait()
        return None

    async def busy_handler(_event, _session_key):
        return False

    adapter.set_message_handler(handler)
    adapter.set_busy_session_handler(busy_handler)
    first = _event("first", "1700000000.000012")
    earlier = _event("earlier", "1700000000.000013")
    latest = _event("latest", "1700000000.000014")

    await adapter.handle_message(first)
    first_task = adapter._session_tasks[_session_key()]
    await asyncio.wait_for(first_started.wait(), timeout=1)
    await adapter.handle_message(earlier)
    await adapter.handle_message(latest)

    if busy_text_mode == "queue":
        merged = adapter._text_debounce[_session_key()].event
        assert _session_key() not in adapter._pending_messages
    else:
        merged = adapter._pending_messages[_session_key()]

    assert earlier.task_fence_acceptance is not None
    assert latest.task_fence_acceptance is not None
    assert latest.task_fence_acceptance.accepted_order == 3
    assert merged is earlier
    assert merged.text == "earlier\nlatest"
    assert merged.task_fence_ingress is latest.task_fence_ingress
    assert merged.task_fence_acceptance is latest.task_fence_acceptance
    assert merged.task_fence_acceptance_attempted is True

    failed = _event("failed", "1700000000.000015")
    failed.task_fence_acceptance_attempted = True
    await adapter.handle_message(failed)

    assert failed.task_fence_acceptance is None
    assert merged.text == "earlier\nlatest\nfailed"
    assert merged.task_fence_ingress is failed.task_fence_ingress
    assert merged.task_fence_acceptance is None
    assert merged.task_fence_acceptance_attempted is True
    assert _count(task_fence_db, "task_fence_ingress") == 3

    release_first.set()
    await first_task
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_exact_redelivery_reuses_recorded_action_after_lane_changes(
    task_fence_db,
):
    runner = _runner(AsyncSessionDB(task_fence_db))
    first = _event("same payload", "1700000000.000020")
    redelivery = _event("same payload", "1700000000.000020")

    await runner._accept_task_fence_gateway_ingress(first, _session_key())
    await runner._accept_task_fence_gateway_ingress(redelivery, _session_key())

    assert first.task_fence_acceptance is not None
    assert redelivery.task_fence_acceptance is not None
    assert redelivery.task_fence_acceptance.replayed is True
    assert redelivery.task_fence_acceptance.event_id == (
        first.task_fence_acceptance.event_id
    )
    assert _count(task_fence_db, "task_fence_ingress") == 1


def test_changed_sidecar_action_is_a_durable_source_event_collision(
    task_fence_db,
):
    original = _event("same payload", "1700000000.000025").task_fence_ingress
    assert original is not None
    task_fence_db.accept_task_fence_ingress_sidecar(
        original,
        conversation_id=_session_key(),
    )
    changed = TaskFenceIngressSidecar(
        source=original.source,
        source_event_id=original.source_event_id,
        action=TASK_FENCE_ACTIONS["explicit_note"],
        payload_hash=original.payload_hash,
    )

    with pytest.raises(
        TaskFenceIngressRejected,
        match="source_event_id_collision",
    ):
        task_fence_db.accept_task_fence_ingress_sidecar(
            changed,
            conversation_id=_session_key(),
        )

    assert _count(task_fence_db, "task_fence_ingress") == 1
    assert _count(task_fence_db, "task_fence_ingress_collisions") == 1
    with task_fence_db._lock:
        projection = task_fence_db._conn.execute(
            "SELECT status, control_revision, intent_epoch "
            "FROM task_fence_tasks"
        ).fetchone()
    assert tuple(projection) == ("running", 1, 1)


@pytest.mark.asyncio
async def test_concurrent_plain_messages_select_empty_then_active_atomically(
    task_fence_db,
):
    second_db = SessionDB(db_path=task_fence_db.db_path)
    runners = (
        _runner(AsyncSessionDB(task_fence_db)),
        _runner(AsyncSessionDB(second_db)),
    )
    events = [
        _event("first concurrent", "1700000000.000030"),
        _event("second concurrent", "1700000000.000031"),
    ]

    try:
        await asyncio.gather(
            *(
                runner._accept_task_fence_gateway_ingress(
                    event,
                    _session_key(),
                )
                for runner, event in zip(runners, events)
            )
        )
    finally:
        second_db.close()

    with task_fence_db._lock:
        actions = task_fence_db._conn.execute(
            "SELECT intent, execution, input_effect "
            "FROM task_fence_ingress ORDER BY accepted_order"
        ).fetchall()
        task = task_fence_db._conn.execute(
            "SELECT status FROM task_fence_tasks"
        ).fetchone()
    assert [tuple(row) for row in actions] == [
        ("replace", "run", "append"),
        ("keep", "hold", "append"),
    ]
    assert task[0] == "paused"
    assert _count(task_fence_db, "task_fence_task_inputs") == 2


@pytest.mark.asyncio
async def test_store_failure_is_telemetry_and_legacy_handler_still_runs():
    broken_db = AsyncMock()
    broken_db.accept_task_fence_ingress_sidecar.side_effect = (
        TaskFenceIngressUnavailable("test_unavailable")
    )
    runner = _runner(broken_db)
    adapter = _ShadowSlackAdapter()
    adapter.set_task_fence_ingress_handler(
        runner._accept_task_fence_gateway_ingress
    )
    handled = asyncio.Event()

    async def handler(event):
        handled.set()
        return None

    adapter.set_message_handler(handler)
    event = _event("still legacy", "1700000000.000040")
    await adapter.handle_message(event)
    await asyncio.wait_for(handled.wait(), timeout=1)
    await adapter.cancel_background_tasks()
    broken_db.accept_task_fence_ingress_sidecar.assert_awaited_once()
    broken_db.accept_task_fence_ingress_sidecar.side_effect = None
    await runner._accept_task_fence_gateway_ingress(event, _session_key())
    broken_db.accept_task_fence_ingress_sidecar.assert_awaited_once()
    assert event.task_fence_acceptance is None


@pytest.mark.asyncio
async def test_auth_probe_failure_cannot_accept_on_late_handler_retry():
    db = AsyncMock()
    runner = _runner(db)
    auth_calls = 0

    def unstable_auth(_source):
        nonlocal auth_calls
        auth_calls += 1
        if auth_calls == 1:
            raise RuntimeError("auth probe unavailable")
        return True

    runner._is_user_authorized = unstable_auth
    adapter = _ShadowSlackAdapter()
    adapter.set_task_fence_ingress_handler(
        runner._accept_task_fence_gateway_ingress
    )
    handled = asyncio.Event()

    async def handler(event):
        await runner._accept_task_fence_gateway_ingress(event, _session_key())
        handled.set()
        return None

    adapter.set_message_handler(handler)
    event = _event("legacy after auth failure", "1700000000.000041")
    await adapter.handle_message(event)
    await asyncio.wait_for(handled.wait(), timeout=1)
    await adapter.cancel_background_tasks()

    assert auth_calls == 1
    assert event.task_fence_acceptance_attempted is True
    assert event.task_fence_acceptance is None
    db.accept_task_fence_ingress_sidecar.assert_not_awaited()


@pytest.mark.asyncio
async def test_payload_collision_is_journaled_without_blocking_legacy_handler(
    task_fence_db,
):
    runner = _runner(AsyncSessionDB(task_fence_db))
    await runner._accept_task_fence_gateway_ingress(
        _event("original", "1700000000.000045"),
        _session_key(),
    )
    adapter = _ShadowSlackAdapter(task_fence_db)
    adapter.set_task_fence_ingress_handler(
        runner._accept_task_fence_gateway_ingress
    )
    handled = asyncio.Event()

    async def handler(event):
        handled.set()
        return None

    adapter.set_message_handler(handler)
    collision = _event("changed", "1700000000.000045")
    await adapter.handle_message(collision)
    await asyncio.wait_for(handled.wait(), timeout=1)
    await adapter.cancel_background_tasks()

    assert collision.task_fence_acceptance is None
    assert _count(task_fence_db, "task_fence_ingress") == 1
    assert _count(task_fence_db, "task_fence_ingress_collisions") == 1
    with task_fence_db._lock:
        task_status = task_fence_db._conn.execute(
            "SELECT status FROM task_fence_tasks"
        ).fetchone()[0]
    assert task_status == "running"


@pytest.mark.asyncio
async def test_default_off_and_wrong_lane_do_not_touch_writer():
    db = AsyncMock()
    for configured_key in ("", "agent:main:slack:dm:T123:OTHER"):
        runner = _runner(db, configured_key=configured_key)
        event = _event("legacy", f"off-{configured_key!r}")
        await runner._accept_task_fence_gateway_ingress(event, _session_key())
        assert event.task_fence_acceptance is None
    db.accept_task_fence_ingress_sidecar.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_sender_does_not_touch_writer():
    db = AsyncMock()
    runner = _runner(db, authorized=False)

    await runner._accept_task_fence_gateway_ingress(
        _event("unauthorized", "1700000000.000047"),
        _session_key(),
    )

    db.accept_task_fence_ingress_sidecar.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pending_kind",
    ["update", "approval", "clarify", "slash_confirm"],
)
async def test_pending_uncorrelated_prompt_is_excluded_once(
    pending_kind,
    monkeypatch,
):
    from tools import approval, clarify_gateway, slash_confirm

    monkeypatch.setattr(
        approval,
        "has_blocking_approval",
        lambda _key: pending_kind == "approval",
    )
    monkeypatch.setattr(
        clarify_gateway,
        "get_pending_for_session",
        lambda _key, include_choice_prompts: (
            object() if pending_kind == "clarify" else None
        ),
    )
    monkeypatch.setattr(
        slash_confirm,
        "get_pending",
        lambda _key: {} if pending_kind == "slash_confirm" else None,
    )
    db = AsyncMock()
    runner = _runner(db)
    runner._update_prompt_pending = {
        _session_key(): pending_kind == "update"
    }
    event = _event("yes", "1700000000.000048")

    await runner._accept_task_fence_gateway_ingress(event, _session_key())
    runner._update_prompt_pending.clear()
    await runner._accept_task_fence_gateway_ingress(event, _session_key())

    assert event.task_fence_acceptance_attempted is True
    assert event.task_fence_acceptance is None
    db.accept_task_fence_ingress_sidecar.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_slack_or_mismatched_sidecar_source_is_excluded():
    db = AsyncMock()
    discord_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="D123",
        chat_type="dm",
        user_id="U123",
        scope_id="T123",
    )
    discord_event = _event("not slack", "1700000000.000049")
    discord_event.source = discord_source
    discord_runner = _runner(
        db,
        configured_key=build_session_key(discord_source),
    )
    await discord_runner._accept_task_fence_gateway_ingress(
        discord_event,
        build_session_key(discord_source),
    )

    wrong_source_event = _event("wrong source", "1700000000.000050")
    wrong_source_event.task_fence_ingress = replace(
        wrong_source_event.task_fence_ingress,
        source="gateway:discord",
    )
    await _runner(db)._accept_task_fence_gateway_ingress(
        wrong_source_event,
        _session_key(),
    )

    db.accept_task_fence_ingress_sidecar.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_secondary_profile_adapter_never_touches_primary_writer():
    runner = object.__new__(GatewayRunner)
    runner.session_store = None
    runner._busy_text_mode = "queue"
    runner._draining = True
    runner._is_user_authorized = lambda _source: True
    runner._adapter_for_source = lambda _source: None
    runner._session_db = AsyncMock()
    runner._make_profile_message_handler = lambda _profile: AsyncMock()
    runner._make_profile_fatal_error_handler = (
        lambda _profile, _platform: AsyncMock()
    )
    runner._handle_reaction_event = AsyncMock()
    runner._recover_telegram_topic_thread_id = MagicMock()
    runner._make_adapter_auth_check = (
        lambda _platform, profile_name=None: MagicMock(return_value=True)
    )
    adapter = _ShadowSlackAdapter()

    runner._configure_profile_adapter(
        adapter,
        "secondary",
        Platform.SLACK,
    )

    assert adapter._task_fence_ingress_handler is None
    event = _event("secondary busy", "1700000000.000051")
    adapter._active_sessions[_session_key()] = asyncio.Event()
    await adapter.handle_message(event)
    runner._session_db.accept_task_fence_ingress_sidecar.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected_reason"),
    [("/stop", "stopped"), ("/new", "cancelled")],
)
async def test_session_closing_command_commits_before_adapter_dispatch(
    task_fence_db,
    command,
    expected_reason,
):
    runner = _runner(AsyncSessionDB(task_fence_db))
    initial = _event("initial", "1700000000.000050")
    await runner._accept_task_fence_gateway_ingress(initial, _session_key())
    runner._update_prompt_pending = {_session_key(): True}

    adapter = _ShadowSlackAdapter(task_fence_db)
    adapter.set_task_fence_ingress_handler(
        runner._accept_task_fence_gateway_ingress
    )
    adapter._active_sessions[_session_key()] = asyncio.Event()
    observed = asyncio.Event()

    async def handler(event):
        adapter.observe("command_handler")
        observed.set()
        return None

    adapter.set_message_handler(handler)
    event = _event(
        command,
        f"1700000000.00005{1 if command == '/stop' else 2}",
        message_type=MessageType.COMMAND,
    )

    await adapter.handle_message(event)
    await asyncio.wait_for(observed.wait(), timeout=1)

    assert adapter.observations[-1][2] == "stopped"
    with task_fence_db._lock:
        reason = task_fence_db._conn.execute(
            "SELECT terminal_reason FROM task_fence_ingress "
            "ORDER BY accepted_order DESC LIMIT 1"
        ).fetchone()[0]
    assert reason == expected_reason


def test_gateway_config_has_one_exact_default_off_shadow_lane():
    assert GatewayConfig().task_fence_shadow_session_key == ""
    config = GatewayConfig.from_dict(
        {
            "gateway": {
                "task_fence": {
                    "shadow_session_key": _session_key(),
                }
            }
        }
    )
    assert config.task_fence_shadow_session_key == _session_key()
    assert config.to_dict()["task_fence"]["shadow_session_key"] == _session_key()
    assert (
        GatewayConfig.from_dict(
            {"task_fence": {"shadow_session_key": ["not", "a", "key"]}}
        ).task_fence_shadow_session_key
        == ""
    )
    assert (
        GatewayConfig.from_dict(
            {
                "task_fence": False,
                "gateway": {
                    "task_fence": {
                        "shadow_session_key": _session_key(),
                    }
                },
            }
        ).task_fence_shadow_session_key
        == ""
    )


def test_real_config_loader_latches_exact_shadow_lane(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        "  task_fence:\n"
        f"    shadow_session_key: {_session_key()}\n",
        encoding="utf-8",
    )
    home_token = set_hermes_home_override(str(tmp_path))
    try:
        config = load_gateway_config()
    finally:
        reset_hermes_home_override(home_token)

    assert config.task_fence_shadow_session_key == _session_key()


def test_malformed_yaml_cannot_preserve_legacy_json_shadow_lane(tmp_path):
    (tmp_path / "gateway.json").write_text(
        '{"task_fence": {"shadow_session_key": "legacy-enabled"}}',
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  task_fence: false\n",
        encoding="utf-8",
    )
    home_token = set_hermes_home_override(str(tmp_path))
    try:
        config = load_gateway_config()
    finally:
        reset_hermes_home_override(home_token)

    assert config.task_fence_shadow_session_key == ""


def test_legacy_gateway_json_cannot_activate_shadow_lane(tmp_path):
    (tmp_path / "gateway.json").write_text(
        '{"task_fence": {"shadow_session_key": "legacy-enabled"}}',
        encoding="utf-8",
    )
    home_token = set_hermes_home_override(str(tmp_path))
    try:
        config = load_gateway_config()
    finally:
        reset_hermes_home_override(home_token)

    assert config.task_fence_shadow_session_key == ""


def test_invalid_shadow_lane_warning_does_not_echo_value(caplog):
    secret = "do-not-log-this-value"

    config = GatewayConfig.from_dict(
        {"task_fence": {"shadow_session_key": [secret]}}
    )

    assert config.task_fence_shadow_session_key == ""
    assert secret not in caplog.text
    assert "got list" in caplog.text
