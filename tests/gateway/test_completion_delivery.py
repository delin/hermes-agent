"""Lifecycle-scoped gateway delivery regressions for terminal completions.

The gateway contract here is deliberately narrower than exactly-once: one live
GatewayRunner suppresses concurrent/replayed copies after successful adapter
injection, failed injection remains retryable, and durable async-delegation
state (when available) is acknowledged through its authoritative SQLite API.
"""

import asyncio
import json
import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from task_fence import (
    TaskFenceCapabilityKind,
    TaskFenceLaunchRoute,
    TaskFenceLaunchRouteValidation,
)
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Any current/future durable compatibility path must stay in tmp state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _runner(adapter, *, origins=None):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries=origins or {},
    )
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    return runner


def _recording_launch_catalog(*, order=None):
    witnesses = []

    def classify_route(route):
        witnesses.append(route)
        if order is not None:
            order.append(("wake_route", route.route_id))
        return TaskFenceLaunchRouteValidation(
            verified=True,
            reason="verified",
            route_id=route.route_id,
        )

    return SimpleNamespace(classify_route=classify_route), witnesses


def _wake_route():
    return TaskFenceLaunchRoute(
        kind=TaskFenceCapabilityKind.RUNTIME,
        route_id="runtime:wake-continuation",
        capability_version="task-fence-capability-v4",
    )


def _async_event(delegation_id="deleg_duplicate"):
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "agent:main:telegram:dm:12345:678",
        "goal": "Investigate flaky test",
        "status": "completed",
        "summary": "Found it",
        "api_calls": 1,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
        # PR #62479 stamps these on gateway-owned events. They must not
        # change the producer identity used for queue replay.
        "origin_profile": "default",
        "origin_hermes_home": "/tmp/hermes-default",
    }


def _completion_event(*, started_at, session_id="proc_reused"):
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "started_at": started_at,
        "command": "echo done",
        "exit_code": 0,
        "completion_reason": "exited",
        "output": "done\n",
    }


def _stop_after_sleeps(monkeypatch, runner, count):
    sleep_calls = 0

    async def _bounded_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= count:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _bounded_sleep)


def test_duplicate_async_queue_replay_injects_once(monkeypatch, isolated_registry):
    """Byte-identical queue replays produce one turn in one gateway lifecycle."""
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(dict(_async_event()))
    isolated.put(dict(_async_event()))

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()


def test_unroutable_async_event_is_not_requeued_forever(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    event = _async_event("deleg_desktop_or_cli")
    event["session_key"] = "20260711_unparseable_ui_session"
    isolated.put(event)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_not_awaited()
    assert isolated.empty()


def test_concurrent_claims_share_the_same_narrow_delivery_seam():
    """Concurrent consumers in one runner cannot both enter the adapter."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_injection(_event):
        entered.set()
        await release.wait()

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_blocked_injection))
    runner = _runner(adapter)
    event = _async_event()
    text = "completion"

    async def _exercise():
        first = asyncio.create_task(runner._deliver_completion_notification(text, dict(event)))
        await entered.wait()
        second = asyncio.create_task(runner._deliver_completion_notification(text, dict(event)))
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(first, second)

    assert sorted(asyncio.run(_exercise()), key=str) == [None, True]
    adapter.handle_message.assert_awaited_once()


def test_failed_async_injection_is_retried_and_only_success_is_acked(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_async_event())

    adapter = SimpleNamespace(
        handle_message=AsyncMock(side_effect=[RuntimeError("temporary"), None])
    )
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=3)

    from tools import async_delegation

    acknowledgements = []
    monkeypatch.setattr(
        async_delegation,
        "complete_completion_delivery",
        lambda delegation_id, _claim_id: acknowledgements.append(delegation_id) or True,
        raising=False,
    )

    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert adapter.handle_message.await_count == 2
    assert acknowledgements == ["deleg_duplicate"]


def _persist_pending_completion(event):
    from tools import async_delegation

    async_delegation._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": event["session_key"],
        "origin_ui_session_id": "",
        "parent_session_id": event.get("parent_session_id"),
        "dispatched_at": event["dispatched_at"],
    })
    async_delegation._persist_completion(event, {
        "status": "completed",
        "summary": event["summary"],
    })


def test_process_evidence_is_observed_after_adapter_acceptance(monkeypatch):
    order = []
    delivered = []
    acceptance = object()

    async def _accepted(event):
        order.append("adapter")
        delivered.append(event)

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_accepted))
    runner = _runner(adapter)
    launch_catalog, witnesses = _recording_launch_catalog(order=order)
    runner._task_fence_launch_catalog = launch_catalog
    from tools import process_registry

    def observe_process_completion(event, *, launch_catalog=None):
        assert launch_catalog is runner._task_fence_launch_catalog
        order.append(("wake_acceptance", event["session_id"]))
        return acceptance

    monkeypatch.setattr(
        process_registry,
        "observe_task_fence_process_completion",
        observe_process_completion,
    )

    assert asyncio.run(
        runner._deliver_completion_notification(
            "completion",
            _completion_event(started_at=10.0),
        )
    ) is True
    assert order == [
        "adapter",
        ("wake_acceptance", "proc_reused"),
        ("wake_route", "runtime:wake-continuation"),
    ]
    assert delivered[0].task_fence_acceptance is acceptance
    assert witnesses == [_wake_route()]


def test_catalogless_process_delivery_keeps_legacy_replay(monkeypatch):
    order = []
    acceptance = object()

    async def accepted(_event):
        order.append("adapter")

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=accepted))
    runner = _runner(adapter)
    from tools import async_delegation
    from tools import process_registry

    def observe_process_completion(event, *, launch_catalog=None):
        assert launch_catalog is None
        order.append(("wake_acceptance", event["session_id"]))
        return acceptance

    monkeypatch.setattr(
        process_registry,
        "observe_task_fence_process_completion",
        observe_process_completion,
    )
    monkeypatch.setattr(
        async_delegation,
        "complete_event_delivery",
        lambda event, claim: order.append(
            ("legacy_replay", event["session_id"], claim)
        ),
    )

    assert asyncio.run(
        runner._deliver_completion_notification(
            "completion",
            _completion_event(started_at=15.0),
        )
    ) is True
    assert order == [
        "adapter",
        ("wake_acceptance", "proc_reused"),
        ("legacy_replay", "proc_reused", ""),
    ]
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.task_fence_acceptance is acceptance


def test_failed_process_push_never_resolves_acceptance_or_wake_route(monkeypatch):
    adapter = SimpleNamespace(
        handle_message=AsyncMock(side_effect=RuntimeError("temporary"))
    )
    runner = _runner(adapter)
    launch_catalog, witnesses = _recording_launch_catalog()
    runner._task_fence_launch_catalog = launch_catalog
    process_observations = []

    from tools import process_registry

    monkeypatch.setattr(
        process_registry,
        "observe_task_fence_process_completion",
        lambda *_args, **_kwargs: process_observations.append(True),
    )

    assert asyncio.run(
        runner._deliver_completion_notification(
            "completion",
            _completion_event(started_at=20.0),
        )
    ) is False
    adapter.handle_message.assert_awaited_once()
    assert process_observations == []
    assert witnesses == []


def test_process_route_failure_has_no_catalogless_replay(monkeypatch):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    launch_catalog, witnesses = _recording_launch_catalog()
    runner._task_fence_launch_catalog = launch_catalog
    observed_catalogs = []

    from tools import process_registry

    def reject_process_evidence(_event, *, launch_catalog=None):
        observed_catalogs.append(launch_catalog)
        return None

    monkeypatch.setattr(
        process_registry,
        "observe_task_fence_process_completion",
        reject_process_evidence,
    )

    assert asyncio.run(
        runner._deliver_completion_notification(
            "completion",
            _completion_event(started_at=30.0),
        )
    ) is True
    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.task_fence_acceptance is None
    assert observed_catalogs == [launch_catalog]
    assert witnesses == []


def test_async_claim_acceptance_reaches_only_the_winning_wake(monkeypatch):
    acceptance = object()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    launch_catalog, witnesses = _recording_launch_catalog()
    runner._task_fence_launch_catalog = launch_catalog
    from tools import async_delegation

    def claim_completion(
        _delegation_id,
        _claim_id,
        *,
        launch_catalog=None,
    ):
        assert launch_catalog is runner._task_fence_launch_catalog
        return True, acceptance

    monkeypatch.setattr(
        async_delegation,
        "claim_completion_delivery_with_acceptance",
        claim_completion,
    )
    monkeypatch.setattr(
        async_delegation,
        "complete_completion_delivery",
        lambda _delegation_id, _claim_id: True,
    )

    assert asyncio.run(
        runner._deliver_completion_notification(
            "completion",
            _async_event("deleg_with_acceptance"),
        )
    ) is True

    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.internal is True
    assert delivered.task_fence_ingress is None
    assert delivered.task_fence_acceptance is acceptance
    assert witnesses == [_wake_route()]
def test_explicit_kill_returns_output_before_consuming_notification(monkeypatch):
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_kill_consumed",
        command="sleep 999",
        task_id="task",
        started_at=1.0,
        output_buffer="important terminal output\n",
        notify_on_complete=True,
    )
    session.process = MagicMock()
    session.process.pid = 4242
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "_terminate_host_pid", lambda *_a, **_kw: None)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(pr_module, "process_registry", registry)

    result = registry.kill_process(session.id)
    assert result["status"] == "killed"
    assert result["output"] == "important terminal output\n"
    assert registry.is_completion_consumed(session.id)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": session.id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    adapter.handle_message.assert_not_awaited()


def test_process_tool_redacts_explicit_kill_output(monkeypatch):
    from tools import process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_kill_redacted",
        command="printenv",
        task_id="task",
        started_at=1.0,
        output_buffer="PRIVATE_TOKEN=opaque-value\n",
        exited=True,
        exit_code=0,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)

    def _redact(result):
        assert result["output"] == "PRIVATE_TOKEN=opaque-value\n"
        result["output"] = "PRIVATE_TOKEN=<redacted>\n"
        return result

    monkeypatch.setattr(pr_module, "_redact_process_result", _redact)

    result = json.loads(pr_module._handle_process({
        "action": "kill",
        "session_id": session.id,
    }))
    assert result["output"] == "PRIVATE_TOKEN=<redacted>\n"


def test_autonomous_completion_redacts_real_command_and_output_secrets(monkeypatch):
    import agent.redact as redact_module
    import tools.process_registry as pr_module

    secret = "abc123randomopaquetokenvalue999"
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_autonomous_redaction",
        command=f"printenv MY_SERVICE_TOKEN={secret}",
        task_id="task",
        started_at=1234.5,
        output_buffer=f"MY_SERVICE_TOKEN={secret}\nHOME=/home/user\n",
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)
    monkeypatch.setattr(redact_module, "_REDACT_ENABLED", True)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": session.id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    delivered = adapter.handle_message.await_args.args[0]
    assert secret not in delivered.text
    assert "HOME=/home/user" in delivered.text
