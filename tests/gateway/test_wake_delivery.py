"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_async_delivery=False) self-POSTs
  /v1/chat/completions with the RAW session id in X-Hermes-Session-Id, so the
  wake turn resumes the REAL session instead of a parallel invisible one
  keyed by build_session_key().
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import (
    MAX_NORMALIZED_TEXT_LENGTH,
    APIServerAdapter,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.wake import (
    TASK_FENCE_WAKE_TOKEN_HEADER,
    adapter_supports_push,
    deliver_wake,
)
from task_fence import (
    IngressAcceptance,
    TaskFenceCapabilityKind,
    TaskFenceLaunchRoute,
    TaskFenceLaunchRouteValidation,
)


class PushAdapter:
    """Default adapter shape — no supports_async_delivery attribute."""

    def __init__(self):
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self, host="0.0.0.0", port=0, key="test-key", model="hermes"):
        self._host = host
        self._port = port
        self._api_key = key
        self._model_name = model

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
    )


def _acceptance() -> IngressAcceptance:
    return IngressAcceptance(
        event_id="wake-acceptance",
        accepted_order=1,
        task_id=None,
        task_projection=None,
        pending_input_ids=(),
        replayed=False,
        opened_run_id=None,
        closed_run_id=None,
        accepted_at=1.0,
    )


def _wake_route() -> TaskFenceLaunchRoute:
    return TaskFenceLaunchRoute(
        kind=TaskFenceCapabilityKind.RUNTIME,
        route_id="runtime:wake-continuation",
        capability_version="task-fence-capability-v4",
    )


def _recording_launch_catalog(classification="verified", *, order=None):
    witnesses = []

    def classify_route(route):
        witnesses.append(route)
        if order is not None:
            order.append("classify")
        if classification == "fault":
            raise RuntimeError("raw-wake-launch-classifier-secret")
        if classification == "malformed":
            return object()
        return TaskFenceLaunchRouteValidation(
            verified=classification == "verified",
            route_id=route.route_id,
            reason=(
                "verified"
                if classification == "verified"
                else "changed_reachable_route"
            ),
        )

    return SimpleNamespace(classify_route=classify_route), witnesses


def test_adapter_supports_push_default_true():
    assert adapter_supports_push(PushAdapter()) is True
    assert adapter_supports_push(ApiServerLikeAdapter()) is False


def test_deliver_wake_push_preserves_only_process_local_acceptance():
    adapter = PushAdapter()
    acceptance = _acceptance()
    launch_catalog, witnesses = _recording_launch_catalog()

    returned = asyncio.run(
        deliver_wake(
            adapter,
            text="wake up",
            source=_source(),
            task_fence_acceptance=acceptance,
            launch_catalog=launch_catalog,
        )
    )

    assert returned is adapter.handled[0]
    assert returned.internal is True
    assert returned.task_fence_ingress is None
    assert returned.task_fence_acceptance is acceptance
    assert witnesses == [_wake_route()]


@pytest.mark.parametrize("classification", ["changed", "fault", "malformed"])
def test_deliver_wake_launch_failure_keeps_push_but_drops_acceptance(
    classification,
    caplog,
):
    adapter = PushAdapter()
    launch_catalog, witnesses = _recording_launch_catalog(classification)

    returned = asyncio.run(
        deliver_wake(
            adapter,
            text="wake despite launch drift",
            source=_source(),
            task_fence_acceptance=_acceptance(),
            launch_catalog=launch_catalog,
        )
    )

    assert adapter.handled == [returned]
    assert returned.task_fence_acceptance is None
    assert witnesses == [_wake_route()]
    if classification == "fault":
        assert "RuntimeError" in caplog.text
        assert "raw-wake-launch-classifier-secret" not in caplog.text


async def _serve(handler):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_deliver_wake_non_push_self_posts_raw_session_id(monkeypatch):
    """The self-post carries the RAW session id header + bearer auth and a
    single user message with stream=false — the exact entry point real
    gateway turns use."""
    from aiohttp import web

    seen = {}
    launch_calls = []

    def classify_route(route):
        launch_calls.append(route)
        raise AssertionError("bare wake must not classify")

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(host="0.0.0.0", port=port, key="sekrit")
            await deliver_wake(
                adapter,
                text="task done — wake",
                session_id="raw-sid-42",
                launch_catalog=SimpleNamespace(classify_route=classify_route),
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-42"
    assert seen["auth"] == "Bearer sekrit"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [
        {"role": "user", "content": "task done — wake"}
    ]
    assert launch_calls == []


@pytest.mark.parametrize(
    "wake_text",
    ["task done — wake", "x" * (MAX_NORMALIZED_TEXT_LENGTH + 1)],
    ids=["short", "normalized-limit"],
)
def test_real_api_self_post_resolves_one_use_task_fence_acceptance(
    wake_text,
):
    acceptance = _acceptance()
    seen = []
    launch_catalog, witnesses = _recording_launch_catalog()

    async def run():
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": "sekrit"})
        )

        async def no_session_db():
            return None

        async def fake_run_agent(**kwargs):
            seen.append(
                (
                    kwargs.get("task_fence_acceptance"),
                    kwargs["user_message"],
                )
            )
            return (
                {
                    "final_response": "ok",
                    "completed": True,
                    "session_id": kwargs["session_id"],
                },
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        adapter._ensure_session_db_async = no_session_db
        adapter._run_agent = fake_run_agent
        runner, port = await _serve(adapter._handle_chat_completions)
        adapter._host = "127.0.0.1"
        adapter._port = port
        try:
            await deliver_wake(
                adapter,
                text=wake_text,
                session_id="raw-sid-42",
                task_fence_acceptance=acceptance,
                launch_catalog=launch_catalog,
            )
            assert adapter._task_fence_wake_tokens == {}
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen == [
        (acceptance, wake_text[:MAX_NORMALIZED_TEXT_LENGTH])
    ]
    assert witnesses == [_wake_route()]


def test_process_completion_api_self_post_resolves_after_admission(
    monkeypatch,
):
    from aiohttp import web

    import gateway.wake as wake_mod
    import tools.process_registry as process_registry_mod

    acceptance = _acceptance()
    event = {
        "type": "completion",
        "session_id": "proc-wake",
        "origin_session_id": "raw-sid-42",
    }
    order = []
    seen = []
    launch_catalog, witnesses = _recording_launch_catalog(order=order)
    expected_launch_catalog = launch_catalog

    def observe(candidate, *, launch_catalog=None):
        assert candidate is event
        assert launch_catalog is expected_launch_catalog
        order.append("observe")
        return acceptance

    monkeypatch.setattr(
        process_registry_mod,
        "observe_task_fence_process_completion",
        observe,
    )
    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01,))

    async def run():
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": "sekrit"})
        )

        async def no_session_db():
            return None

        async def fake_run_agent(**kwargs):
            order.append("run")
            seen.append(kwargs.get("task_fence_acceptance"))
            return (
                {
                    "final_response": "ok",
                    "completed": True,
                    "session_id": kwargs["session_id"],
                },
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        adapter._ensure_session_db_async = no_session_db
        adapter._run_agent = fake_run_agent
        original_register = adapter._register_task_fence_wake

        def register(**kwargs):
            order.append("register")
            return original_register(**kwargs)

        adapter._register_task_fence_wake = register
        admission_calls = 0

        def concurrency_limited_response():
            nonlocal admission_calls
            admission_calls += 1
            order.append("admission")
            if admission_calls == 1:
                return web.json_response({"error": "busy"}, status=429)
            return None

        adapter._concurrency_limited_response = concurrency_limited_response
        http_runner, port = await _serve(adapter._handle_chat_completions)
        adapter._host = "127.0.0.1"
        adapter._port = port
        gateway = object.__new__(GatewayRunner)
        gateway.adapters = {Platform.API_SERVER: adapter}
        gateway._task_fence_launch_catalog = launch_catalog
        try:
            assert await gateway._inject_watch_notification(
                "process done — wake",
                event,
            ) is True
            assert adapter._task_fence_wake_tokens == {}
        finally:
            await http_runner.cleanup()

    asyncio.run(run())
    assert order == [
        "register",
        "admission",
        "register",
        "admission",
        "observe",
        "classify",
        "run",
    ]
    assert seen == [acceptance]
    assert witnesses == [_wake_route()]


def test_wake_launch_failure_keeps_non_push_delivery_but_drops_acceptance():
    seen = []
    factory_calls = []
    launch_catalog, witnesses = _recording_launch_catalog("changed")

    def acceptance_factory():
        factory_calls.append(True)
        return _acceptance()

    async def run():
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": "sekrit"})
        )

        async def no_session_db():
            return None

        async def fake_run_agent(**kwargs):
            seen.append(kwargs.get("task_fence_acceptance"))
            return (
                {
                    "final_response": "legacy wake delivered",
                    "completed": True,
                    "session_id": kwargs["session_id"],
                },
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        adapter._ensure_session_db_async = no_session_db
        adapter._run_agent = fake_run_agent
        runner, port = await _serve(adapter._handle_chat_completions)
        adapter._host = "127.0.0.1"
        adapter._port = port
        try:
            await deliver_wake(
                adapter,
                text="wake without Task Fence carrier",
                session_id="raw-sid-fail-open",
                task_fence_acceptance_factory=acceptance_factory,
                launch_catalog=launch_catalog,
            )
            assert adapter._task_fence_wake_tokens == {}
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert factory_calls == [True]
    assert seen == [None]
    assert witnesses == [_wake_route()]


def test_real_api_rejects_mismatched_or_replayed_wake_nonce():
    from aiohttp import ClientSession

    acceptance = _acceptance()
    seen = []
    factory_calls = []

    def acceptance_factory():
        factory_calls.append(True)
        return acceptance

    async def run():
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": "sekrit"})
        )

        async def no_session_db():
            return None

        async def fake_run_agent(**kwargs):
            seen.append(kwargs.get("task_fence_acceptance"))
            return (
                {
                    "final_response": "ok",
                    "completed": True,
                    "session_id": kwargs["session_id"],
                },
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        adapter._ensure_session_db_async = no_session_db
        adapter._run_agent = fake_run_agent
        runner, port = await _serve(adapter._handle_chat_completions)
        token = adapter._register_task_fence_wake(
            session_id="expected-sid",
            text="wake",
            acceptance_factory=acceptance_factory,
        )
        headers = {
            "Authorization": "Bearer sekrit",
            TASK_FENCE_WAKE_TOKEN_HEADER: token,
        }
        body = {
            "model": "hermes-agent",
            "messages": [{"role": "user", "content": "wake"}],
            "stream": False,
        }
        try:
            async with ClientSession() as client:
                wrong_headers = {
                    **headers,
                    "X-Hermes-Session-Id": "wrong-sid",
                }
                async with client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=body,
                    headers=wrong_headers,
                ) as response:
                    assert response.status == 200
                    await response.read()
                replay_headers = {
                    **headers,
                    "X-Hermes-Session-Id": "expected-sid",
                }
                async with client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=body,
                    headers=replay_headers,
                ) as response:
                    assert response.status == 200
                    await response.read()
            assert adapter._task_fence_wake_tokens == {}
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen == [None, None]
    assert factory_calls == []


def test_deliver_wake_retries_429_then_succeeds(monkeypatch):
    """HTTP 429 (max_concurrent_runs cap) is transient — retried with backoff."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2
