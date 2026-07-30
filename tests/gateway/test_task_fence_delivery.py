"""Real-path invariants for the bounded Slack final-delivery shadow cohort."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import (
    GatewayRunner,
    _TASK_FENCE_DELIVERY_EXCLUDE_QUEUED,
    _TASK_FENCE_FINAL_TURN_GENERATION_KEY,
    _mark_task_fence_delivery_exclusion,
    _move_task_fence_final_turn_generation,
    _task_fence_delivery_capability_for_event,
)
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.task_fence_delivery import (
    TASK_FENCE_DELIVERY_CAPABILITY_ATTR,
    _CURRENT_DELIVERY_CAPABILITY,
)
from hermes_state import AsyncSessionDB, SessionDB
from plugins.platforms.slack.adapter import SlackAdapter
from task_fence import (
    CausalEnvelope,
    TaskFencePolicy,
    bind_task_fence_policy,
    current_causal_envelope,
    current_task_fence_policy,
)


class _SlackRejectedBlocks(Exception):
    def __init__(self) -> None:
        super().__init__("invalid_blocks")
        self.response = {"error": "invalid_blocks"}


def _source(*, profile: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="D123",
        chat_type="dm",
        user_id="U123",
        scope_id="T123",
        profile=profile,
    )


def _event(text: str, message_id: str) -> MessageEvent:
    from gateway.platforms.base import task_fence_sidecar_for_human_message

    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(),
        message_id=message_id,
    )
    event.task_fence_ingress = task_fence_sidecar_for_human_message(
        event,
        source="gateway:slack",
        source_event_id=f"event:T123:D123:{message_id}",
    )
    return event


def _runner(db: SessionDB) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    session_key = build_session_key(_source())
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(enabled=True, token="test")
        },
        task_fence_shadow_session_key=session_key,
    )
    runner._session_db = AsyncSessionDB(db)
    runner._update_prompt_pending = {}
    runner._is_user_authorized = lambda _source: True
    return runner


async def _accepted_lane(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    runner = _runner(db)
    event = _event("start bounded delivery", "1700000000.000001")
    session_key = build_session_key(event.source)
    await runner._accept_task_fence_gateway_ingress(event, session_key)
    acceptance = event.task_fence_acceptance
    assert acceptance is not None
    generation = db.reserve_task_fence_generation(acceptance)
    assert db.finish_task_fence_generation(generation, state="committed")
    return db, runner, event, session_key, generation


def _stage_capability(
    runner: GatewayRunner,
    event: MessageEvent,
    session_key: str,
    response: str,
    parent: CausalEnvelope | None,
) -> None:
    capability = _task_fence_delivery_capability_for_event(
        runner,
        event,
        session_key,
        response,
        parent,
        queued_followup=False,
    )
    assert capability is not None
    setattr(event, TASK_FENCE_DELIVERY_CAPABILITY_ATTR, capability)


def _slack_adapter(*, rich_blocks: bool = False) -> tuple[SlackAdapter, AsyncMock]:
    adapter = SlackAdapter(
        PlatformConfig(
            enabled=True,
            token="xoxb-test",
            typing_indicator=False,
            extra={"rich_blocks": rich_blocks},
        )
    )
    adapter._app = MagicMock()
    adapter._running = True

    async def stop_typing(*_args, **_kwargs):
        assert current_task_fence_policy() is None
        assert _CURRENT_DELIVERY_CAPABILITY.get() is None

    adapter.stop_typing = AsyncMock(side_effect=stop_typing)
    client = AsyncMock()
    adapter._get_client = MagicMock(return_value=client)
    return adapter, client


def _configure_real_runner_dispatch(
    runner: GatewayRunner,
    adapter: SlackAdapter,
    event: MessageEvent,
    parent: CausalEnvelope,
    response: str,
) -> None:
    session_key = build_session_key(event.source)
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="task-fence-delivery-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="dm",
    )
    runner.adapters = {Platform.SLACK: adapter}
    adapter.gateway_runner = runner
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._busy_input_mode = "interrupt"
    runner._draining = False
    runner._external_drain_active = False
    runner._session_run_generation = {}
    runner._session_sources = {}
    runner._pending_native_image_paths_by_session = {}
    runner._background_tasks = {}
    runner._background_task_counter = 0
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._service_tier = None
    runner._fast_mode_by_session = {}
    runner._goal_state_by_session = {}
    runner._goal_runs_in_progress = set()
    runner._goal_queued_by_session = set()
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_telegram_topic_root_lobby = lambda _source: False
    runner._should_send_telegram_lobby_reminder = lambda _source: False
    runner._check_slash_access = lambda _source, _command: None
    runner._claim_active_session_slot = lambda *_args: (None, None)
    runner._begin_session_run_generation = lambda _key: 1
    runner._persist_active_agents = lambda: None
    runner._restore_moa_one_shot = lambda *_args: None
    runner._restore_pending_one_turn_model_override = lambda *_args: None
    runner._release_running_agent_state = (
        lambda key, *_args, **_kwargs: runner._running_agents.pop(key, None)
    )
    runner._release_turn_lease = lambda *_args: None
    runner._post_turn_goal_continuation = AsyncMock()

    async def handle_with_agent(current_event, _source, _key, _generation):
        agent_result = {_TASK_FENCE_FINAL_TURN_GENERATION_KEY: parent}
        _move_task_fence_final_turn_generation(current_event, agent_result)
        return response

    runner._handle_message_with_agent = handle_with_agent


async def _run_base_final(
    adapter: SlackAdapter,
    event: MessageEvent,
    session_key: str,
    response: str,
    *,
    handler=None,
) -> None:
    adapter._message_handler = handler or AsyncMock(return_value=response)
    adapter._active_sessions[session_key] = asyncio.Event()
    with (
        patch("gateway.delivery_ledger.ledger_enabled", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        await adapter._process_message_background(event, session_key)


def _count(db: SessionDB, table: str) -> int:
    with db._lock:
        return db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["single", "chunks", "base-retry", "block-fallback"],
)
async def test_real_slack_final_response_audits_every_physical_post(
    tmp_path,
    case,
):
    db, runner, event, session_key, parent = await _accepted_lane(tmp_path)
    rich_blocks = case == "block-fallback"
    adapter, client = _slack_adapter(rich_blocks=rich_blocks)
    adapter.gateway_runner = runner
    if case == "chunks":
        adapter.MAX_MESSAGE_LENGTH = 120
        response = "\n".join(f"distinct line {index}: {'x' * 24}" for index in range(12))
    elif case == "block-fallback":
        response = "| Item | Status |\n|---|---|\n| Hermes | ready |"
    else:
        response = "bounded final response"

    real_runner_handler = None
    if case == "single":
        _configure_real_runner_dispatch(runner, adapter, event, parent, response)
        real_runner_handler = runner._handle_message
    else:
        _stage_capability(runner, event, session_key, response, parent)
    physical_calls: list[dict] = []
    sdk_envelopes: list[CausalEnvelope] = []

    async def post_message(**kwargs):
        # Admission and authorization are durable before physical SDK entry.
        assert _count(db, "task_fence_attempts") == len(physical_calls) + 1
        assert current_task_fence_policy() is None
        assert _CURRENT_DELIVERY_CAPABILITY.get() is None
        envelope = current_causal_envelope()
        assert isinstance(envelope, CausalEnvelope)
        sdk_envelopes.append(envelope)
        physical_calls.append(dict(kwargs))
        if case == "base-retry" and len(physical_calls) == 1:
            raise ConnectionError("connection reset")
        if case == "block-fallback" and len(physical_calls) == 1:
            raise _SlackRejectedBlocks()
        return {"ts": f"1710000000.{len(physical_calls):06d}"}

    client.chat_postMessage.side_effect = post_message
    try:
        with patch("gateway.platforms.base.asyncio.sleep", new=AsyncMock()):
            await _run_base_final(
                adapter,
                event,
                session_key,
                response,
                handler=real_runner_handler,
            )

        expected_calls = 2 if case in {"base-retry", "block-fallback"} else 1
        if case == "chunks":
            assert len(physical_calls) > 1
        else:
            assert len(physical_calls) == expected_calls

        with db._lock:
            permits = db._conn.execute(
                "SELECT invocation_envelope_id, generation_id, "
                "invocation_fingerprint, audience, executor, state "
                "FROM task_fence_dispatch_permits ORDER BY reserved_at"
            ).fetchall()
            attempts = db._conn.execute(
                "SELECT state FROM task_fence_attempts ORDER BY prepared_at"
            ).fetchall()
            operation_fingerprints = db._conn.execute(
                "SELECT invocation_fingerprint "
                "FROM task_fence_policy_decisions "
                "WHERE operation_kind = 'delivery' "
                "AND decision_point = 'admission' "
                "ORDER BY decision_order"
            ).fetchall()
            dump = "\n".join(db._conn.iterdump())

        assert len(permits) == len(physical_calls)
        assert len({row[0] for row in permits}) == len(physical_calls)
        assert {row[1] for row in permits} == {parent.generation_id}
        assert {(row[3], row[4], row[5]) for row in permits} == {
            ("delivery", "gateway:slack:chat_post_message", "consumed")
        }
        assert [row[0] for row in attempts] == ["STARTED"] * len(physical_calls)
        assert len({envelope.invocation_id for envelope in sdk_envelopes}) == len(
            physical_calls
        )
        assert all(
            envelope.generation_id == parent.generation_id
            and envelope.parent_invocation_id == parent.invocation_id
            for envelope in sdk_envelopes
        )
        fingerprints = {row[0] for row in operation_fingerprints}
        if case == "base-retry":
            assert len(fingerprints) == 1
        elif case in {"chunks", "block-fallback"}:
            assert len(fingerprints) == len(physical_calls)
        assert response not in dump
        assert not hasattr(event, TASK_FENCE_DELIVERY_CAPABILITY_ATTR)
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["missing-parent", "misbound-parent", "child-parent", "stale-parent"],
)
async def test_shadow_block_preserves_the_exact_legacy_slack_post(tmp_path, case):
    db, runner, event, session_key, parent = await _accepted_lane(tmp_path)
    adapter, client = _slack_adapter()
    adapter.gateway_runner = runner
    response = "legacy delivery remains unchanged"
    candidate_parent = parent
    if case == "missing-parent":
        candidate_parent = None
    elif case == "misbound-parent":
        candidate_parent = replace(parent, snapshot_event_id="wrong-event")
    elif case == "child-parent":
        candidate_parent = parent.for_invocation()
    _stage_capability(
        runner,
        event,
        session_key,
        response,
        candidate_parent,
    )

    if case == "stale-parent":
        newer = _event("newer human input", "1700000000.000002")
        await runner._accept_task_fence_gateway_ingress(newer, session_key)
        assert newer.task_fence_acceptance is not None

    client.chat_postMessage.return_value = {"ts": "1710000000.000001"}
    try:
        await _run_base_final(adapter, event, session_key, response)

        assert client.chat_postMessage.await_count == 1
        assert client.chat_postMessage.await_args.kwargs == {
            "channel": "D123",
            "text": response,
            "mrkdwn": True,
            "thread_ts": "1700000000.000001",
        }
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
        with db._lock:
            decisions = db._conn.execute(
                "SELECT outcome, reason_code FROM task_fence_policy_decisions "
                "WHERE operation_kind = 'delivery'"
            ).fetchall()
        assert len(decisions) == 1
        assert decisions[0][0] == "would_block"
        if case in {"missing-parent", "misbound-parent", "child-parent"}:
            assert decisions[0][1] == "missing_provenance"
        else:
            assert decisions[0][1] in {
                "newer_input_pending",
                "stale_authority",
                "task_not_runnable",
            }
    finally:
        db.close()


@pytest.mark.asyncio
async def test_audit_fault_is_fail_open_and_clears_context(
    tmp_path,
    monkeypatch,
):
    db, runner, event, session_key, parent = await _accepted_lane(tmp_path)
    response = "fail-open final response"
    _stage_capability(runner, event, session_key, response, parent)
    capability = getattr(event, TASK_FENCE_DELIVERY_CAPABILITY_ATTR)
    assert not hasattr(capability, "policy")
    monkeypatch.setattr(
        db,
        "_admit_task_fence_operation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("disk fault")),
    )
    adapter, client = _slack_adapter()
    adapter.gateway_runner = runner
    client.chat_postMessage.return_value = {"ts": "1710000000.000003"}

    try:
        await _run_base_final(adapter, event, session_key, response)
        assert client.chat_postMessage.await_count == 1
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None

    finally:
        db.close()


@pytest.mark.asyncio
async def test_direct_slack_send_is_outside_delivery_audit(tmp_path):
    db, runner, _event, _session_key, _parent = await _accepted_lane(tmp_path)
    adapter, client = _slack_adapter()
    adapter.gateway_runner = runner

    async def post_message(**_kwargs):
        assert current_task_fence_policy() is None
        assert _CURRENT_DELIVERY_CAPABILITY.get() is None
        return {"ts": "1710000000.000004"}

    async def conversations_open(**_kwargs):
        assert current_task_fence_policy() is None
        assert _CURRENT_DELIVERY_CAPABILITY.get() is None
        return {"channel": {"id": "D123"}}

    client.chat_postMessage.side_effect = post_message
    client.conversations_open.side_effect = conversations_open
    try:
        with bind_task_fence_policy(TaskFencePolicy(db)):
            result = await adapter.send(
                "U123",
                "direct status message",
                reply_to="1700000000.000004",
            )

        assert result.success
        client.conversations_open.assert_awaited_once_with(users="U123")
        assert client.chat_postMessage.await_count == 1
        adapter.stop_typing.assert_awaited_once()
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
        assert current_task_fence_policy() is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_mixed_final_response_stays_outside_delivery_audit(tmp_path):
    db, runner, event, session_key, parent = await _accepted_lane(tmp_path)
    response = "bounded text\n\n![diagram](https://example.test/diagram.png)"
    _stage_capability(runner, event, session_key, response, parent)
    adapter, client = _slack_adapter()
    adapter.gateway_runner = runner
    adapter.send_multiple_images = AsyncMock()

    async def post_message(**_kwargs):
        assert current_task_fence_policy() is None
        assert _CURRENT_DELIVERY_CAPABILITY.get() is None
        return {"ts": "1710000000.000005"}

    client.chat_postMessage.side_effect = post_message
    try:
        await _run_base_final(adapter, event, session_key, response)

        assert client.chat_postMessage.await_count == 1
        adapter.send_multiple_images.assert_awaited_once()
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
        assert not hasattr(event, TASK_FENCE_DELIVERY_CAPABILITY_ATTR)
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "failure", "failed_calls", "audited_calls"),
    [
        ("plain-fallback", ValueError("invalid_arguments"), 1, 1),
        ("failure-notice", ConnectionError("connection reset"), 3, 3),
    ],
)
async def test_generic_retry_fallbacks_are_not_delivery_operations(
    tmp_path,
    case,
    failure,
    failed_calls,
    audited_calls,
):
    db, runner, event, session_key, parent = await _accepted_lane(tmp_path)
    response = "bounded final response"
    _stage_capability(runner, event, session_key, response, parent)
    adapter, client = _slack_adapter()
    adapter.gateway_runner = runner
    physical_calls: list[dict] = []

    async def post_message(**kwargs):
        assert current_task_fence_policy() is None
        assert _CURRENT_DELIVERY_CAPABILITY.get() is None
        physical_calls.append(dict(kwargs))
        if len(physical_calls) <= failed_calls:
            raise failure
        return {"ts": "1710000000.000006"}

    client.chat_postMessage.side_effect = post_message
    try:
        with patch("gateway.platforms.base.asyncio.sleep", new=AsyncMock()):
            await _run_base_final(adapter, event, session_key, response)

        assert len(physical_calls) == failed_calls + 1
        assert physical_calls[-1]["text"] != response
        assert _count(db, "task_fence_dispatch_permits") == audited_calls
        assert _count(db, "task_fence_attempts") == audited_calls
        assert not hasattr(event, TASK_FENCE_DELIVERY_CAPABILITY_ATTR)
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["proxy-exception", "queued-exception", "stream-failure"])
async def test_excluded_error_route_preserves_legacy_send_without_audit(
    tmp_path,
    case,
):
    db, runner, event, session_key, parent = await _accepted_lane(tmp_path)
    response = f"{case} response"
    adapter, client = _slack_adapter()
    _configure_real_runner_dispatch(runner, adapter, event, parent, response)

    if case == "proxy-exception":
        runner._get_proxy_url = lambda: "http://proxy.test"
        runner._run_agent_via_proxy = AsyncMock(
            side_effect=RuntimeError("proxy failed")
        )

        async def excluded_handler(_event, source, key, generation):
            try:
                await runner._run_agent_inner(
                    message="proxy request",
                    context_prompt="",
                    history=[],
                    source=source,
                    session_id="task-fence-delivery-session",
                    session_key=key,
                    run_generation=generation,
                )
            except RuntimeError:
                return response

    elif case == "queued-exception":

        async def excluded_handler(_event, _source, _key, _generation):
            _mark_task_fence_delivery_exclusion(
                _TASK_FENCE_DELIVERY_EXCLUDE_QUEUED
            )
            return response

    else:

        async def excluded_handler(current_event, _source, _key, _generation):
            _move_task_fence_final_turn_generation(
                current_event,
                {
                    _TASK_FENCE_FINAL_TURN_GENERATION_KEY: parent,
                    "already_sent": True,
                    "failed": True,
                },
            )
            return response

    runner._handle_message_with_agent = excluded_handler

    async def post_message(**_kwargs):
        assert current_task_fence_policy() is None
        assert _CURRENT_DELIVERY_CAPABILITY.get() is None
        return {"ts": "1710000000.000007"}

    client.chat_postMessage.side_effect = post_message
    try:
        await _run_base_final(
            adapter,
            event,
            session_key,
            response,
            handler=runner._handle_message,
        )

        assert client.chat_postMessage.await_count == 1
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
        assert not hasattr(event, TASK_FENCE_DELIVERY_CAPABILITY_ATTR)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_queued_result_and_terminal_command_cannot_stage_delivery(tmp_path):
    db, runner, event, session_key, parent = await _accepted_lane(tmp_path)
    try:
        assert _task_fence_delivery_capability_for_event(
            runner,
            event,
            session_key,
            "ordinary final response",
            parent,
            queued_followup=False,
        ) is not None
        assert _task_fence_delivery_capability_for_event(
            runner,
            event,
            session_key,
            "proxy response",
            parent,
            queued_followup=False,
            proxy_result=True,
        ) is None
        assert _task_fence_delivery_capability_for_event(
            runner,
            event,
            session_key,
            "failed streaming response",
            parent,
            queued_followup=False,
            streaming_result=True,
        ) is None
        assert _task_fence_delivery_capability_for_event(
            runner,
            event,
            session_key,
            "response with runtime footer",
            parent,
            queued_followup=False,
            runtime_footer=True,
        ) is None

        assert _task_fence_delivery_capability_for_event(
            runner,
            event,
            session_key,
            "queued terminal response",
            parent,
            queued_followup=True,
        ) is None

        terminal = _event("/stop", "1700000000.000010")
        await runner._accept_task_fence_gateway_ingress(terminal, session_key)
        assert terminal.task_fence_acceptance is not None
        assert _task_fence_delivery_capability_for_event(
            runner,
            terminal,
            session_key,
            "stopped",
            None,
            queued_followup=False,
        ) is None
    finally:
        db.close()
