"""Real-path tests for bounded Telegram Task Fence shadow ingress/delivery."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, _task_fence_delivery_capability_for_event
from gateway.session import SessionSource, build_session_key
from gateway.task_fence_delivery import (
    TASK_FENCE_DELIVERY_CAPABILITY_ATTR,
    TaskFenceDeliveryCapability,
    _CURRENT_DELIVERY_CAPABILITY,
    _telegram_send_message_acknowledgement_ref,
    _telegram_send_message_fingerprint,
    bind_task_fence_delivery_capability,
)
from hermes_state import AsyncSessionDB, SessionDB
from plugins.platforms.telegram.adapter import TelegramAdapter
from task_fence import (
    CausalEnvelope,
    TaskFenceIngressUnavailable,
    TaskFencePolicy,
    bind_task_fence_policy,
    current_causal_envelope,
    current_task_fence_policy,
)
from telegram.error import NetworkError


def _source(*, thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="111",
        thread_id=thread_id,
    )


def _session_key(*, thread_id: str | None = None) -> str:
    return build_session_key(_source(thread_id=thread_id))


def _message(
    text: str,
    *,
    message_id: int = 42,
    thread_id: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        message_id=message_id,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        chat=SimpleNamespace(
            id=12345,
            type="private",
            title=None,
            full_name="Test User",
            is_forum=False,
        ),
        from_user=SimpleNamespace(
            id=111,
            full_name="Test User",
            first_name="Test",
            is_bot=False,
        ),
        reply_to_message=None,
        date=None,
        location=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
    )


def _update(
    text: str,
    *,
    update_id: int,
    message_id: int = 42,
    thread_id: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        update_id=update_id,
        message=_message(
            text,
            message_id=message_id,
            thread_id=thread_id,
        ),
        effective_message=None,
    )


def _stack(
    tmp_path,
    *,
    configured_key: str,
    topic_recovery=None,
):
    platform_config = PlatformConfig(
        enabled=True,
        token="test-token",
        typing_indicator=False,
        extra={"allow_from": ["111"]},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: platform_config},
        task_fence_shadow_conversation_key=configured_key,
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    runner._session_db = AsyncSessionDB(db)
    runner.session_store = None
    runner._update_prompt_pending = {}
    runner._is_user_authorized = lambda source: (
        source is not None and source.user_id == "111"
    )

    adapter = TelegramAdapter(platform_config)
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._text_batch_delay_seconds = 0.03
    adapter._text_batch_split_delay_seconds = 0.03
    adapter.set_task_fence_ingress_handler(
        runner._accept_task_fence_gateway_ingress
    )
    if topic_recovery is not None:
        adapter.set_topic_recovery_fn(topic_recovery)
    return adapter, db


async def _cleanup(adapter: TelegramAdapter, db: SessionDB) -> None:
    batch_tasks = tuple(adapter._pending_text_batch_tasks.values())
    for task in batch_tasks:
        task.cancel()
    await asyncio.gather(*batch_tasks, return_exceptions=True)
    await adapter.cancel_background_tasks()
    db.close()


def _count(db: SessionDB, table: str) -> int:
    with db._lock:
        return db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.mark.asyncio
async def test_two_raw_dm_updates_commit_before_batch_without_laundering_run(
    tmp_path,
):
    adapter, db = _stack(tmp_path, configured_key=_session_key())
    handled = asyncio.Event()
    handled_events = []
    handled_generations = []
    enqueue_counts = []
    real_enqueue = adapter._enqueue_text_event

    async def handler(event):
        generation = db.reserve_task_fence_generation(
            event.task_fence_acceptance
        )
        assert db.finish_task_fence_generation(
            generation,
            state="committed",
        ) is True
        handled_events.append(event)
        handled_generations.append(generation)
        handled.set()
        return None

    def traced_enqueue(event, **kwargs):
        enqueue_counts.append(_count(db, "task_fence_ingress"))
        assert not handled.is_set()
        return real_enqueue(event, **kwargs)

    adapter.set_message_handler(handler)
    adapter._enqueue_text_event = traced_enqueue
    try:
        await adapter._handle_text_message(
            _update("part one", update_id=1001),
            SimpleNamespace(),
        )
        await adapter._handle_text_message(
            _update("part two", update_id=1002, message_id=43),
            SimpleNamespace(),
        )

        pending = adapter._pending_text_batches[_session_key()]
        latest = pending.task_fence_acceptance
        assert enqueue_counts == [1, 2]
        assert pending.text == "part one\npart two"
        assert latest is not None
        assert latest.accepted_order == 2
        assert latest.pending_input_ids == ()
        assert latest.task_projection is not None
        assert latest.task_projection.status == "running"
        assert latest.opened_run_id is not None
        assert latest.closed_run_id is not None
        with db._lock:
            bound_updates = db._conn.execute(
                "SELECT ingress.source_event_id "
                "FROM task_fence_task_inputs AS inputs "
                "JOIN task_fence_ingress AS ingress "
                "ON ingress.event_id = inputs.event_id "
                "WHERE inputs.bound_run_id = ? "
                "ORDER BY ingress.accepted_order",
                (latest.opened_run_id,),
            ).fetchall()
        assert [row[0] for row in bound_updates] == [
            "update:999:1001",
            "update:999:1002",
        ]

        await asyncio.wait_for(handled.wait(), timeout=1)
        await adapter.cancel_background_tasks()
        with db._lock:
            rows = db._conn.execute(
                "SELECT source, source_event_id, intent, execution "
                "FROM task_fence_ingress ORDER BY accepted_order"
            ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("gateway:telegram", "update:999:1001", "replace", "run"),
            ("gateway:telegram", "update:999:1002", "replace", "run"),
        ]
        assert handled_events == [pending]
        assert handled_events[0].task_fence_acceptance is latest
        assert handled_generations[0].snapshot_event_id == latest.event_id
    finally:
        await _cleanup(adapter, db)


@pytest.mark.asyncio
async def test_long_command_text_batch_remains_untyped(tmp_path):
    adapter, db = _stack(tmp_path, configured_key=_session_key())
    handled = asyncio.Event()
    handled_events = []

    async def handler(event):
        handled_events.append(event)
        handled.set()
        return None

    adapter.set_message_handler(handler)
    first = "/queue " + "x" * 4089
    continuation = "y" * 500
    try:
        before = db._conn.total_changes
        await adapter._handle_command(
            _update(first, update_id=2001),
            SimpleNamespace(),
        )
        await adapter._handle_text_message(
            _update(continuation, update_id=2002, message_id=43),
            SimpleNamespace(),
        )
        await asyncio.wait_for(handled.wait(), timeout=1)
        await adapter.cancel_background_tasks()

        assert _count(db, "task_fence_ingress") == 0
        assert _count(db, "task_fence_ingress_collisions") == 0
        assert db._conn.total_changes == before
        assert len(handled_events) == 1
        event = handled_events[0]
        assert event.message_type.value == "command"
        assert event.text == f"{first}\n{continuation}"
        assert event.task_fence_ingress is None
        assert event.task_fence_acceptance is None
        assert event.task_fence_acceptance_attempted is True
    finally:
        await _cleanup(adapter, db)


@pytest.mark.asyncio
async def test_update_identity_replays_and_changed_payload_collides(tmp_path):
    adapter, db = _stack(tmp_path, configured_key=_session_key())
    handled = asyncio.Queue()

    async def handler(event):
        await handled.put(event)
        return None

    adapter.set_message_handler(handler)

    async def deliver(text, *, update_id, message_id):
        await adapter._handle_text_message(
            _update(text, update_id=update_id, message_id=message_id),
            SimpleNamespace(),
        )
        event = await asyncio.wait_for(handled.get(), timeout=1)
        await adapter.cancel_background_tasks()
        return event

    try:
        first = await deliver("same", update_id=3001, message_id=42)
        replay = await deliver("same", update_id=3001, message_id=999)
        changed = await deliver("changed", update_id=3001, message_id=1000)

        assert first.task_fence_acceptance is not None
        assert replay.task_fence_acceptance is not None
        assert replay.task_fence_acceptance.replayed is True
        assert replay.task_fence_acceptance.event_id == (
            first.task_fence_acceptance.event_id
        )
        assert changed.task_fence_acceptance is None
        assert _count(db, "task_fence_ingress") == 1
        assert _count(db, "task_fence_ingress_collisions") == 1
    finally:
        await _cleanup(adapter, db)


@pytest.mark.asyncio
async def test_dm_topic_recovery_uses_one_acceptance_and_batch_key(tmp_path):
    expected_key = _session_key(thread_id="222")
    adapter, db = _stack(
        tmp_path,
        configured_key=expected_key,
        topic_recovery=lambda _source: "222",
    )
    handled = asyncio.Event()

    async def handler(_event):
        handled.set()
        return None

    adapter.set_message_handler(handler)
    try:
        await adapter._handle_text_message(
            _update("topic input", update_id=4001),
            SimpleNamespace(),
        )
        pending = adapter._pending_text_batches[expected_key]
        assert pending.source.thread_id == "222"
        acceptance = pending.task_fence_acceptance
        assert acceptance is not None
        with db._lock:
            conversation_id = db._conn.execute(
                "SELECT conversation_id FROM task_fence_ingress"
            ).fetchone()[0]
        assert conversation_id == expected_key
        generation = db.reserve_task_fence_generation(acceptance)
        assert generation.snapshot_event_id == acceptance.event_id
        assert db.finish_task_fence_generation(
            generation,
            state="committed",
        ) is True

        await asyncio.wait_for(handled.wait(), timeout=1)
        await adapter.cancel_background_tasks()
        assert _count(db, "task_fence_ingress") == 1
    finally:
        await _cleanup(adapter, db)


@pytest.mark.asyncio
async def test_terminal_and_arbitrary_short_commands_keep_closed_boundary(
    tmp_path,
):
    adapter, db = _stack(tmp_path, configured_key=_session_key())
    handled = asyncio.Queue()

    async def handler(event):
        with db._lock:
            task = db._conn.execute(
                "SELECT status FROM task_fence_tasks"
            ).fetchone()
        await handled.put((event, None if task is None else task[0]))
        return None

    adapter.set_message_handler(handler)
    try:
        await adapter._handle_text_message(
            _update("start", update_id=5001),
            SimpleNamespace(),
        )
        await asyncio.wait_for(handled.get(), timeout=1)
        await adapter.cancel_background_tasks()

        await adapter._handle_command(
            _update("/stop", update_id=5002, message_id=43),
            SimpleNamespace(),
        )
        terminal_event, status = await asyncio.wait_for(handled.get(), timeout=1)
        await adapter.cancel_background_tasks()
        assert status == "stopped"
        assert terminal_event.task_fence_acceptance is not None
        with db._lock:
            reason = db._conn.execute(
                "SELECT terminal_reason FROM task_fence_ingress "
                "ORDER BY accepted_order DESC LIMIT 1"
            ).fetchone()[0]
            before = db._conn.total_changes
        assert reason == "stopped"

        await adapter._handle_command(
            _update("/model test", update_id=5003, message_id=44),
            SimpleNamespace(),
        )
        arbitrary_event, _status = await asyncio.wait_for(
            handled.get(),
            timeout=1,
        )
        await adapter.cancel_background_tasks()
        assert arbitrary_event.task_fence_ingress is None
        assert arbitrary_event.task_fence_acceptance is None
        assert db._conn.total_changes == before
        assert _count(db, "task_fence_ingress") == 2
    finally:
        await _cleanup(adapter, db)


@pytest.mark.asyncio
async def test_command_between_text_chunks_clears_batch_authority(tmp_path):
    adapter, db = _stack(tmp_path, configured_key=_session_key())
    adapter._text_batch_delay_seconds = 0.1
    handled = asyncio.Queue()

    async def handler(event):
        await handled.put(event)
        return None

    adapter.set_message_handler(handler)
    try:
        await adapter._handle_text_message(
            _update("pending text", update_id=5501),
            SimpleNamespace(),
        )
        pending = adapter._pending_text_batches[_session_key()]
        assert pending.task_fence_acceptance is not None

        await adapter._handle_command(
            _update("/stop", update_id=5502, message_id=43),
            SimpleNamespace(),
        )
        command = await asyncio.wait_for(handled.get(), timeout=1)
        assert command.text == "/stop"
        assert pending.text == "pending text"
        assert pending.task_fence_ingress is None
        assert pending.task_fence_acceptance is None
        assert pending._task_fence_mixed_origin is True

        delayed = await asyncio.wait_for(handled.get(), timeout=1)
        await adapter.cancel_background_tasks()
        assert delayed is pending
        assert delayed.text == "pending text"
        assert _count(db, "task_fence_ingress") == 2
    finally:
        await _cleanup(adapter, db)


@pytest.mark.asyncio
async def test_default_off_telegram_dispatches_without_task_fence_write(
    tmp_path,
):
    adapter, db = _stack(tmp_path, configured_key="")
    handled = asyncio.Event()

    async def handler(_event):
        handled.set()
        return None

    adapter.set_message_handler(handler)
    try:
        await adapter._handle_text_message(
            _update("legacy", update_id=6001),
            SimpleNamespace(),
        )
        await asyncio.wait_for(handled.wait(), timeout=1)
        await adapter.cancel_background_tasks()
        assert _count(db, "task_fence_ingress") == 0
    finally:
        await _cleanup(adapter, db)


@pytest.mark.asyncio
async def test_wrong_lane_dispatches_without_task_fence_write(tmp_path):
    adapter, db = _stack(
        tmp_path,
        configured_key=_session_key(thread_id="other"),
    )
    handled = asyncio.Event()

    async def handler(_event):
        handled.set()
        return None

    adapter.set_message_handler(handler)
    try:
        await adapter._handle_text_message(
            _update("wrong lane", update_id=7001),
            SimpleNamespace(),
        )
        await asyncio.wait_for(handled.wait(), timeout=1)
        await adapter.cancel_background_tasks()
        assert _count(db, "task_fence_ingress") == 0
        assert _count(db, "task_fence_ingress_collisions") == 0
    finally:
        await _cleanup(adapter, db)


@pytest.mark.asyncio
async def test_store_failure_preserves_exact_legacy_batch(tmp_path):
    adapter, db = _stack(tmp_path, configured_key=_session_key())
    runner = adapter._task_fence_ingress_handler.__self__
    broken_store = AsyncMock()
    broken_store.accept_task_fence_ingress_sidecar.side_effect = (
        TaskFenceIngressUnavailable("test_unavailable")
    )
    runner._session_db = broken_store
    handled = asyncio.Event()
    handled_events = []

    async def handler(event):
        handled_events.append(event)
        handled.set()
        return None

    adapter.set_message_handler(handler)
    try:
        await adapter._handle_text_message(
            _update("legacy on failure", update_id=8001),
            SimpleNamespace(),
        )
        pending = adapter._pending_text_batches[_session_key()]
        await asyncio.wait_for(handled.wait(), timeout=1)
        await adapter.cancel_background_tasks()

        assert handled_events == [pending]
        assert pending.text == "legacy on failure"
        assert pending.task_fence_acceptance is None
        broken_store.accept_task_fence_ingress_sidecar.assert_awaited_once()
        assert _count(db, "task_fence_ingress") == 0
        assert _count(db, "task_fence_ingress_collisions") == 0
    finally:
        await _cleanup(adapter, db)


async def _await_delivery_task(
    adapter: TelegramAdapter,
    physical_entered: asyncio.Event,
) -> None:
    await asyncio.wait_for(physical_entered.wait(), timeout=1)
    tasks = tuple(adapter._background_tasks)
    assert tasks
    results = await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True),
        timeout=2,
    )
    assert not [result for result in results if isinstance(result, BaseException)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "single",
        "chunks",
        "internal-retry",
        "markdown-fallback",
        "missing-ack",
        "misbound-ack",
    ],
)
async def test_real_telegram_final_audits_every_physical_send_message(
    tmp_path,
    case,
):
    adapter, db = _stack(tmp_path, configured_key=_session_key())
    runner = adapter._task_fence_ingress_handler.__self__
    adapter.gateway_runner = runner
    if case == "chunks":
        adapter.MAX_MESSAGE_LENGTH = 96
        response = "\n".join(
            f"distinct line {index}: {'x' * 24}" for index in range(12)
        )
    else:
        response = "bounded Telegram final response"

    generations: list[CausalEnvelope] = []
    physical_calls: list[dict] = []
    physical_results: list[object | None] = []
    sdk_envelopes: list[CausalEnvelope] = []
    physical_entered = asyncio.Event()

    async def handler(event):
        acceptance = event.task_fence_acceptance
        assert acceptance is not None
        generation = db.reserve_task_fence_generation(acceptance)
        assert db.finish_task_fence_generation(generation, state="committed")
        generations.append(generation)
        capability = _task_fence_delivery_capability_for_event(
            runner,
            event,
            _session_key(),
            response,
            generation,
            queued_followup=False,
        )
        assert capability is not None
        setattr(event, TASK_FENCE_DELIVERY_CAPABILITY_ATTR, capability)
        return response

    async def send_message(**kwargs):
        assert _count(db, "task_fence_attempts") == len(physical_calls) + 1
        with db._lock:
            latest_state = db._conn.execute(
                "SELECT state FROM task_fence_attempts "
                "ORDER BY prepared_at DESC LIMIT 1"
            ).fetchone()[0]
        assert latest_state == "STARTED"
        assert current_task_fence_policy() is None
        assert _CURRENT_DELIVERY_CAPABILITY.get() is None
        envelope = current_causal_envelope()
        assert isinstance(envelope, CausalEnvelope)
        sdk_envelopes.append(envelope)
        physical_calls.append(dict(kwargs))
        physical_results.append(None)
        physical_entered.set()

        if case == "internal-retry" and len(physical_calls) == 1:
            raise NetworkError("connection reset before acknowledgement")
        if case == "markdown-fallback" and len(physical_calls) == 1:
            raise ValueError("can't parse MarkdownV2")

        chat_id = 99999 if case == "misbound-ack" else kwargs["chat_id"]
        result = SimpleNamespace(
            message_id=7000 + len(physical_calls),
            chat=(
                None
                if case == "missing-ack"
                else SimpleNamespace(id=chat_id)
            ),
        )
        physical_results[-1] = result
        return result

    adapter._bot = SimpleNamespace(
        id=999,
        username="test_bot",
        send_message=send_message,
    )
    adapter.set_message_handler(handler)
    try:
        with (
            patch("gateway.delivery_ledger.ledger_enabled", return_value=False),
            patch("hermes_cli.plugins.invoke_hook", return_value=[]),
            patch(
                "plugins.platforms.telegram.adapter.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            await adapter._handle_text_message(
                _update("start bounded delivery", update_id=9001),
                SimpleNamespace(),
            )
            await _await_delivery_task(adapter, physical_entered)

        expected_calls = 2 if case in {
            "internal-retry",
            "markdown-fallback",
        } else 1
        if case == "chunks":
            assert len(physical_calls) > 1
        else:
            assert len(physical_calls) == expected_calls
        assert len(generations) == 1
        parent = generations[0]

        with db._lock:
            permits = db._conn.execute(
                "SELECT invocation_envelope_id, generation_id, "
                "invocation_fingerprint, audience, executor, state "
                "FROM task_fence_dispatch_permits ORDER BY reserved_at"
            ).fetchall()
            attempts = db._conn.execute(
                "SELECT state, acknowledgement_ref, terminal_at IS NOT NULL "
                "FROM task_fence_attempts ORDER BY prepared_at"
            ).fetchall()
            decision_fingerprints = db._conn.execute(
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
            ("delivery", "gateway:telegram:send_message", "consumed")
        }
        expected_evidence = [
            (
                _telegram_send_message_acknowledgement_ref(
                    bot_id=999,
                    request=request,
                    result=result,
                )
                if result is not None
                else None
            )
            for request, result in zip(
                physical_calls,
                physical_results,
                strict=True,
            )
        ]
        assert [row[0] for row in attempts] == [
            "SUCCEEDED" if evidence is not None else "STARTED"
            for evidence in expected_evidence
        ]
        assert [row[1] for row in attempts] == expected_evidence
        assert [row[2] for row in attempts] == [
            evidence is not None for evidence in expected_evidence
        ]
        terminal_evidence = [
            evidence for evidence in expected_evidence if evidence is not None
        ]
        assert len(terminal_evidence) == len(set(terminal_evidence))
        assert all(
            evidence.startswith("telegram:send_message:ack:sha256:")
            and len(evidence.encode("utf-8")) < 128
            for evidence in terminal_evidence
        )
        assert len({envelope.invocation_id for envelope in sdk_envelopes}) == len(
            physical_calls
        )
        assert all(
            envelope.generation_id == parent.generation_id
            and envelope.parent_invocation_id == parent.invocation_id
            for envelope in sdk_envelopes
        )
        fingerprints = {row[0] for row in decision_fingerprints}
        if case == "internal-retry":
            assert len(fingerprints) == 1
        elif case in {"chunks", "markdown-fallback"}:
            assert len(fingerprints) == len(physical_calls)
        assert response not in dump
        assert all(request["text"] not in dump for request in physical_calls)
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None
    finally:
        await _cleanup(adapter, db)


def test_telegram_ack_and_fingerprint_are_bounded_and_exact():
    class _LinkPreview:
        def to_dict(self):
            return {"is_disabled": True}

    request = {
        "chat_id": 12345,
        "text": "sensitive final response",
        "parse_mode": "MarkdownV2",
        "reply_to_message_id": 42,
        "message_thread_id": None,
        "link_preview_options": _LinkPreview(),
        "disable_notification": True,
    }
    result = SimpleNamespace(
        message_id=7001,
        chat=SimpleNamespace(id=12345),
    )
    reference = _telegram_send_message_acknowledgement_ref(
        bot_id=999,
        request=request,
        result=result,
    )
    assert reference is not None
    assert len(reference.encode("utf-8")) < 128
    assert all(value not in reference for value in ("999", "12345", "7001"))
    assert len(
        {
            reference,
            _telegram_send_message_acknowledgement_ref(
                bot_id=998,
                request=request,
                result=result,
            ),
            _telegram_send_message_acknowledgement_ref(
                bot_id=999,
                request={**request, "chat_id": 12346},
                result=SimpleNamespace(
                    message_id=7001,
                    chat=SimpleNamespace(id=12346),
                ),
            ),
            _telegram_send_message_acknowledgement_ref(
                bot_id=999,
                request=request,
                result=SimpleNamespace(
                    message_id=7002,
                    chat=SimpleNamespace(id=12345),
                ),
            ),
        }
    ) == 4

    fingerprint = _telegram_send_message_fingerprint(
        bot_id=999,
        request=request,
    )
    assert len(fingerprint) == 64
    assert "sensitive final response" not in fingerprint
    assert len(
        {
            fingerprint,
            _telegram_send_message_fingerprint(
                bot_id=998,
                request=request,
            ),
            _telegram_send_message_fingerprint(
                bot_id=999,
                request={**request, "parse_mode": None},
            ),
            _telegram_send_message_fingerprint(
                bot_id=999,
                request={
                    **request,
                    "link_preview_options": {"is_disabled": False},
                },
            ),
        }
    ) == 4
    with pytest.raises(TypeError, match="unsupported Telegram request value"):
        _telegram_send_message_fingerprint(
            bot_id=999,
            request={**request, "link_preview_options": object()},
        )


@pytest.mark.asyncio
async def test_telegram_rich_final_stays_outside_send_message_audit(tmp_path):
    adapter, db = _stack(tmp_path, configured_key=_session_key())
    runner = adapter._task_fence_ingress_handler.__self__
    adapter.gateway_runner = runner
    adapter._rich_messages_enabled = True
    response = "| Item | Status |\n|---|---|\n| Hermes | ready |"
    physical_entered = asyncio.Event()

    async def handler(event):
        generation = db.reserve_task_fence_generation(
            event.task_fence_acceptance
        )
        assert db.finish_task_fence_generation(generation, state="committed")
        capability = _task_fence_delivery_capability_for_event(
            runner,
            event,
            _session_key(),
            response,
            generation,
            queued_followup=False,
        )
        assert capability is not None
        setattr(event, TASK_FENCE_DELIVERY_CAPABILITY_ATTR, capability)
        return response

    async def do_api_request(method, **kwargs):
        assert method == "sendRichMessage"
        assert kwargs["api_kwargs"]["rich_message"]
        assert current_task_fence_policy() is None
        assert _CURRENT_DELIVERY_CAPABILITY.get() is None
        physical_entered.set()
        return {"ok": True}

    send_message = AsyncMock()
    adapter._bot = SimpleNamespace(
        id=999,
        username="test_bot",
        do_api_request=do_api_request,
        send_message=send_message,
    )
    adapter.set_message_handler(handler)
    try:
        with (
            patch("gateway.delivery_ledger.ledger_enabled", return_value=False),
            patch("hermes_cli.plugins.invoke_hook", return_value=[]),
            patch(
                "plugins.platforms.telegram.adapter.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            await adapter._handle_text_message(
                _update("request rich delivery", update_id=9002),
                SimpleNamespace(),
            )
            await _await_delivery_task(adapter, physical_entered)

        send_message.assert_not_awaited()
        assert _count(db, "task_fence_policy_decisions") == 0
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
    finally:
        await _cleanup(adapter, db)


@pytest.mark.asyncio
async def test_wrong_source_capability_cannot_authorize_telegram_send(tmp_path):
    adapter, db = _stack(tmp_path, configured_key=_session_key())
    runner = adapter._task_fence_ingress_handler.__self__
    adapter.gateway_runner = runner
    calls = []

    async def send_message(**kwargs):
        assert current_task_fence_policy() is None
        assert _CURRENT_DELIVERY_CAPABILITY.get() is None
        calls.append(dict(kwargs))
        return SimpleNamespace(
            message_id=7001,
            chat=SimpleNamespace(id=kwargs["chat_id"]),
        )

    adapter._bot = SimpleNamespace(
        id=999,
        username="test_bot",
        send_message=send_message,
    )
    capability = TaskFenceDeliveryCapability(
        parent=None,
        delivery_source="gateway:slack",
        conversation_id=_session_key(),
    )
    try:
        with (
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_task_fence_delivery_capability(capability),
        ):
            result = await adapter.send(
                "12345",
                "legacy direct send",
                metadata={"notify": True},
            )

        assert result.success is True
        assert len(calls) == 1
        assert calls[0]["chat_id"] == 12345
        assert calls[0]["text"]
        assert _count(db, "task_fence_policy_decisions") == 0
        assert _count(db, "task_fence_dispatch_permits") == 0
        assert _count(db, "task_fence_attempts") == 0
    finally:
        await _cleanup(adapter, db)
