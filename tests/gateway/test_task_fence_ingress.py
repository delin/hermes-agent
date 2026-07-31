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
    load_task_fence_shadow_session_key,
)
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    task_fence_sidecar_for_human_message,
)
from gateway.run import (
    GatewayRunner,
    _TASK_FENCE_FINAL_TURN_GENERATION_KEY,
)
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

    async def send(self, chat_id, text=None, content=None, **kwargs):
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


def _text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=None,
                    reasoning=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        model="test/model",
        usage=None,
    )


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


def _configure_agent_run(runner, adapter) -> None:
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
    runner._queued_events = {}


def _count(db: SessionDB, table: str) -> int:
    with db._lock:
        return db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.mark.asyncio
async def test_gateway_queued_turn_returns_latest_generation_without_cached_leak(
    task_fence_db,
    monkeypatch,
    tmp_path,
):
    runner = _runner(AsyncSessionDB(task_fence_db))
    adapter = _ShadowSlackAdapter(task_fence_db)
    _configure_agent_run(runner, adapter)

    first_event = _event("first", "1700000000.000000")
    await runner._accept_task_fence_gateway_ingress(
        first_event,
        _session_key(),
    )
    first_acceptance = first_event.task_fence_acceptance
    assert first_acceptance is not None

    queued_event = _event("queued", "1700000000.000001")
    await runner._accept_task_fence_gateway_ingress(
        queued_event,
        _session_key(),
    )
    queued_acceptance = queued_event.task_fence_acceptance
    assert queued_acceptance is not None

    observed = []
    created = []
    generations = {
        "first": object(),
        "queued": object(),
    }

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
            turn_acceptance = kwargs.get("task_fence_acceptance")
            observed.append((message, turn_acceptance))
            result = {
                "final_response": f"done:{message}",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }
            if turn_acceptance is not None:
                result[_TASK_FENCE_FINAL_TURN_GENERATION_KEY] = (
                    generations[message]
                )
            return result

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
    adapter._pending_messages[_session_key()] = queued_event
    terminal = await runner._run_agent(
        message="first",
        task_fence_acceptance=first_acceptance,
        **kwargs,
    )
    untyped = await runner._run_agent(message="untyped", **kwargs)

    assert terminal["final_response"] == "done:queued"
    assert (
        terminal[_TASK_FENCE_FINAL_TURN_GENERATION_KEY]
        is generations["queued"]
    )
    assert untyped["final_response"] == "done:untyped"
    assert untyped[_TASK_FENCE_FINAL_TURN_GENERATION_KEY] is None
    assert len(created) == 1
    assert observed == [
        ("first", first_acceptance),
        ("queued", queued_acceptance),
        ("untyped", None),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "supersede_before_drain",
    [False, True],
    ids=["exact-descendant", "stale-before-drain"],
)
async def test_goal_continuation_drains_into_real_descendant_or_fails_open(
    task_fence_db,
    monkeypatch,
    tmp_path,
    supersede_before_drain,
):
    import hashlib
    from unittest.mock import patch

    import run_agent
    from gateway.run import _move_task_fence_final_turn_generation
    from task_fence import (
        IngressEnvelope,
        TASK_FENCE_FINAL_GENERATION_KEY,
        current_causal_envelope,
    )

    runner = _runner(AsyncSessionDB(task_fence_db))
    adapter = _ShadowSlackAdapter(task_fence_db)
    _configure_agent_run(runner, adapter)
    adapter.set_task_fence_ingress_handler(
        runner._accept_task_fence_gateway_ingress
    )
    runner._defer_goal_status_notice_after_delivery = AsyncMock()
    session_id = "goal-task-fence-session"

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = run_agent.AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id=session_id,
            session_db=task_fence_db,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = False

    provider_envelopes = []
    responses = iter(
        [
            _text_response("parent complete"),
            _text_response("child complete"),
        ]
    )

    def provider_create(**_kwargs):
        provider_envelopes.append(current_causal_envelope())
        return next(responses)

    agent.client.chat.completions.create.side_effect = provider_create
    real_agent_type = run_agent.AIAgent

    # Runtime helpers read class-level AIAgent constants during the turn.
    class ExistingAgentFactory(real_agent_type):
        def __new__(cls, **_kwargs):
            return agent

    monkeypatch.setattr(run_agent, "AIAgent", ExistingAgentFactory)
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"model": {"default": "test/model"}},
    )
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )

    session_entry = SimpleNamespace(session_id=session_id)
    goal_manager = MagicMock()
    goal_manager.is_active.return_value = True
    goal_manager.evaluate_after_turn.return_value = {
        "should_continue": True,
        "continuation_prompt": (
            "[Continuing toward your standing goal]\nGoal: ship it"
        ),
        "message": "",
    }
    history = []
    turn_generations = []
    turn_acceptances = []
    turn_responses = []
    race_projection = None
    child_done = asyncio.Event()

    async def handler(event):
        nonlocal history, race_projection
        result = await runner._run_agent(
            message=event.text,
            context_prompt="",
            history=history,
            source=_source(),
            session_id=session_id,
            session_key=_session_key(),
            task_fence_acceptance=event.task_fence_acceptance,
        )
        _move_task_fence_final_turn_generation(event, result)
        generation = getattr(event, TASK_FENCE_FINAL_GENERATION_KEY)
        turn_generations.append(generation)
        turn_acceptances.append(event.task_fence_acceptance)
        turn_responses.append(result["final_response"])
        history = result["messages"]
        if len(turn_generations) == 1:
            await runner._post_turn_goal_continuation(
                session_entry=session_entry,
                source=_source(),
                final_response=result["final_response"],
                task_fence_parent_generation=generation,
            )
            if supersede_before_drain:
                task_fence_db.accept_task_fence_ingress(
                    IngressEnvelope(
                        source="gateway:slack",
                        source_event_id="event:T123:D123:race",
                        conversation_id=_session_key(),
                        action=TASK_FENCE_ACTIONS["comment_hold"],
                        payload_hash=hashlib.sha256(
                            b"newer human input"
                        ).hexdigest(),
                        task_id=generation.task_id,
                    )
                )
                race_projection = task_fence_db.inspect_task_fence_task(
                    generation.task_id
                ).task
        else:
            child_done.set()
        return result["final_response"]

    adapter.set_message_handler(handler)
    initial = _event("start the goal", "1700000000.000010")

    try:
        with (
            patch("hermes_cli.goals.GoalManager", return_value=goal_manager),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            await adapter.handle_message(initial)
            await asyncio.wait_for(child_done.wait(), timeout=5)

        assert turn_responses == ["parent complete", "child complete"]
        assert len(turn_acceptances) == 2
        assert turn_acceptances[0] is not None
        synthetic_acceptance = turn_acceptances[1]
        assert synthetic_acceptance is not None
        assert provider_envelopes[0] is not None

        parent_generation = turn_generations[0]
        synthetic_row = task_fence_db._conn.execute(
            "SELECT source, causal_parent_generation_id "
            "FROM task_fence_ingress WHERE event_id = ?",
            (synthetic_acceptance.event_id,),
        ).fetchone()
        assert tuple(synthetic_row) == (
            "runtime:goal_continuation",
            parent_generation.generation_id,
        )

        generation_count = task_fence_db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_model_generations"
        ).fetchone()[0]
        if not supersede_before_drain:
            assert generation_count == 2
            child_generation = turn_generations[1]
            assert child_generation is not None
            assert provider_envelopes[1] is not None
            child_snapshot = task_fence_db._conn.execute(
                "SELECT snapshot_event_id "
                "FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (child_generation.generation_id,),
            ).fetchone()
            assert child_snapshot[0] == synthetic_acceptance.event_id
        else:
            assert generation_count == 1
            assert turn_generations[1] is None
            assert provider_envelopes[1] is None
            assert race_projection is not None
            assert task_fence_db.inspect_task_fence_task(
                parent_generation.task_id
            ).task == race_projection
            assert task_fence_db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_dispatch_permits"
            ).fetchone()[0] == 1
            assert task_fence_db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_attempts"
            ).fetchone()[0] == 1
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "supersede_before_drain",
    [False, True],
    ids=["exact-wake-descendant", "stale-wake-before-drain"],
)
async def test_process_wake_runs_with_exact_synthetic_parent_or_fails_open(
    task_fence_db,
    supersede_before_drain,
):
    import hashlib
    from unittest.mock import patch

    import run_agent
    from gateway.wake import deliver_wake
    from task_fence import (
        IngressEnvelope,
        TASK_FENCE_FINAL_GENERATION_KEY,
        current_causal_envelope,
    )

    parent_acceptance = task_fence_db.accept_task_fence_ingress(
        IngressEnvelope(
            source="gateway:slack",
            source_event_id="wake-parent-event",
            conversation_id=_session_key(),
            action=TASK_FENCE_ACTIONS["initial_submit"],
            payload_hash=hashlib.sha256(b"parent input").hexdigest(),
            opaque_payload_ref="slack:wake-parent-event",
        )
    )
    parent_generation = task_fence_db.reserve_task_fence_generation(
        parent_acceptance
    )
    assert task_fence_db.finish_task_fence_generation(
        parent_generation,
        state="committed",
    )
    wake_acceptance = (
        task_fence_db.accept_task_fence_process_completion_evidence(
            source_event_id="wake-process-completion",
            parent_generation_id=parent_generation.generation_id,
            parent_runtime_epoch=parent_generation.runtime_epoch,
            payload_hash=hashlib.sha256(b"completion evidence").hexdigest(),
            opaque_payload_ref="process-completion:proc-wake",
        )
    )

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = run_agent.AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="wake-task-fence-session",
            session_db=task_fence_db,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = False

    provider_envelopes = []

    def provider_create(**_kwargs):
        provider_envelopes.append(current_causal_envelope())
        return _text_response("wake child complete")

    agent.client.chat.completions.create.side_effect = provider_create
    adapter = _ShadowSlackAdapter(task_fence_db)
    blocker_started = asyncio.Event()
    blocker_release = asyncio.Event()
    child_done = asyncio.Event()
    child_results = []
    child_acceptances = []

    async def handler(event):
        if event.text == "blocker":
            blocker_started.set()
            await blocker_release.wait()
            return None
        result = agent.run_conversation(
            event.text,
            conversation_history=[],
            task_id="wake-task-fence-session",
            task_fence_acceptance=event.task_fence_acceptance,
        )
        child_results.append(result)
        child_acceptances.append(event.task_fence_acceptance)
        child_done.set()
        return result["final_response"]

    adapter.set_message_handler(handler)
    race_projection = None
    try:
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            if supersede_before_drain:
                await adapter.handle_message(
                    MessageEvent(
                        text="blocker",
                        message_type=MessageType.TEXT,
                        source=_source(),
                        internal=True,
                    )
                )
                await asyncio.wait_for(blocker_started.wait(), timeout=5)

            await deliver_wake(
                adapter,
                text="[IMPORTANT: process completed]",
                source=_source(),
                task_fence_acceptance=wake_acceptance,
            )

            if supersede_before_drain:
                task_fence_db.accept_task_fence_ingress(
                    IngressEnvelope(
                        source="gateway:slack",
                        source_event_id="wake-race-human",
                        conversation_id=_session_key(),
                        task_id=parent_generation.task_id,
                        action=TASK_FENCE_ACTIONS["comment_hold"],
                        payload_hash=hashlib.sha256(
                            b"newer human input"
                        ).hexdigest(),
                    )
                )
                race_projection = task_fence_db.inspect_task_fence_task(
                    parent_generation.task_id
                ).task
                blocker_release.set()

            await asyncio.wait_for(child_done.wait(), timeout=5)

        assert child_acceptances == [wake_acceptance]
        assert child_results[0]["final_response"] == "wake child complete"
        assert len(provider_envelopes) == 1
        recorded = task_fence_db._conn.execute(
            "SELECT source, origin, ingress_class, "
            "causal_parent_generation_id FROM task_fence_ingress "
            "WHERE event_id = ?",
            (wake_acceptance.event_id,),
        ).fetchone()
        assert tuple(recorded) == (
            "runtime:process_completion",
            "runtime",
            "synthetic",
            parent_generation.generation_id,
        )
        generation_count = task_fence_db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_model_generations"
        ).fetchone()[0]
        child_generation = child_results[0][
            TASK_FENCE_FINAL_GENERATION_KEY
        ]
        if not supersede_before_drain:
            assert generation_count == 2
            assert child_generation is not None
            assert provider_envelopes[0] is not None
            snapshot = task_fence_db._conn.execute(
                "SELECT snapshot_event_id "
                "FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (child_generation.generation_id,),
            ).fetchone()
            assert snapshot[0] == wake_acceptance.event_id
        else:
            assert generation_count == 1
            assert child_generation is None
            assert provider_envelopes[0] is None
            assert race_projection is not None
            assert task_fence_db.inspect_task_fence_task(
                parent_generation.task_id
            ).task == race_projection
            assert _count(task_fence_db, "task_fence_dispatch_permits") == 0
            assert _count(task_fence_db, "task_fence_attempts") == 0
    finally:
        blocker_release.set()
        await adapter.cancel_background_tasks()


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
    update_state = runner._session_state(_session_key()).persistent
    update_state.update_prompt_pending = pending_kind == "update"
    event = _event("yes", "1700000000.000048")

    await runner._accept_task_fence_gateway_ingress(event, _session_key())
    update_state.update_prompt_pending = False
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
    runner._session_state(_session_key()).persistent.update_prompt_pending = True

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
        status_session_key = load_task_fence_shadow_session_key()
    finally:
        reset_hermes_home_override(home_token)

    assert config.task_fence_shadow_session_key == _session_key()
    assert status_session_key == config.task_fence_shadow_session_key


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
        status_session_key = load_task_fence_shadow_session_key()
    finally:
        reset_hermes_home_override(home_token)

    assert config.task_fence_shadow_session_key == ""
    assert status_session_key == ""


def test_legacy_gateway_json_cannot_activate_shadow_lane(tmp_path):
    (tmp_path / "gateway.json").write_text(
        '{"task_fence": {"shadow_session_key": "legacy-enabled"}}',
        encoding="utf-8",
    )
    home_token = set_hermes_home_override(str(tmp_path))
    try:
        config = load_gateway_config()
        status_session_key = load_task_fence_shadow_session_key()
    finally:
        reset_hermes_home_override(home_token)

    assert config.task_fence_shadow_session_key == ""
    assert status_session_key == ""


def test_shadow_status_config_keeps_literal_env_references(tmp_path, monkeypatch):
    monkeypatch.setenv("TASK_FENCE_TEST_LANE", "expanded-lane")
    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        "  task_fence:\n"
        "    shadow_session_key: \"${TASK_FENCE_TEST_LANE}\"\n",
        encoding="utf-8",
    )
    home_token = set_hermes_home_override(str(tmp_path))
    try:
        startup_session_key = load_gateway_config().task_fence_shadow_session_key
        status_session_key = load_task_fence_shadow_session_key()
    finally:
        reset_hermes_home_override(home_token)

    assert startup_session_key == "${TASK_FENCE_TEST_LANE}"
    assert status_session_key == startup_session_key


def test_shadow_status_config_does_not_repair_malformed_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("gateway:\n  task_fence: [\n", encoding="utf-8")
    before = config_path.read_bytes(), tuple(tmp_path.iterdir())
    home_token = set_hermes_home_override(str(tmp_path))
    try:
        assert load_task_fence_shadow_session_key() == ""
    finally:
        reset_hermes_home_override(home_token)

    assert (config_path.read_bytes(), tuple(tmp_path.iterdir())) == before


def test_invalid_shadow_lane_warning_does_not_echo_value(caplog):
    secret = "do-not-log-this-value"

    config = GatewayConfig.from_dict(
        {"task_fence": {"shadow_session_key": [secret]}}
    )

    assert config.task_fence_shadow_session_key == ""
    assert secret not in caplog.text
    assert "got list" in caplog.text
