from contextlib import contextmanager
from contextvars import copy_context
import hashlib
import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from task_fence import (
    DecisionOutcome,
    DecisionReason,
    IngressEnvelope,
    TASK_FENCE_ACTIONS,
    TASK_FENCE_SELECTED_COHORT_CAPABILITIES,
    TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
    TaskFenceCapabilityKind,
    TaskFenceLaunchCatalog,
    TaskFenceLaunchRoute,
    TaskFencePolicy,
    bind_causal_envelope,
    bind_task_fence_policy,
    current_causal_envelope,
    current_task_fence_policy,
    task_fence_launch_manifest_fingerprint,
)
from tools.registry import _task_fence_tool_fingerprint, registry


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _context_carries_task_fence_authority() -> bool:
    return any(
        isinstance(value, (SessionDB, TaskFencePolicy))
        for _variable, value in copy_context().items()
    )


def _allowing_tool_guardrails():
    return SimpleNamespace(
        before_call=lambda _name, _args: SimpleNamespace(allows_execution=True)
    )


def _ingress(
    action: str,
    source_event_id: str,
    *,
    task_id: str | None = None,
) -> IngressEnvelope:
    return IngressEnvelope(
        source="gateway:test:membrane",
        source_event_id=source_event_id,
        conversation_id="membrane-conversation",
        action=TASK_FENCE_ACTIONS[action],
        payload_hash=_hash(source_event_id),
        task_id=task_id,
    )


def _live_lane(path):
    db = SessionDB(path)
    acceptance = db.accept_task_fence_ingress(
        _ingress("initial_submit", "membrane-initial")
    )
    generation = db.reserve_task_fence_generation(acceptance)
    assert db.finish_task_fence_generation(generation, state="committed")
    return db, acceptance, generation


def _live_model_lane(path):
    db = SessionDB(path)
    acceptance = db.accept_task_fence_ingress(
        _ingress("initial_submit", "membrane-model-initial")
    )
    generation = db.reserve_task_fence_generation(acceptance)
    return db, acceptance, generation


def _recording_launch_policy(db):
    witnesses = []
    catalog = TaskFenceLaunchCatalog(
        conversation_fingerprint="a" * 64,
        manifest_fingerprint=task_fence_launch_manifest_fingerprint(
            TASK_FENCE_SELECTED_LAUNCH_MANIFEST
        ),
        runtime_epoch=1,
        mode_generation=0,
        manifest=TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
        declarations=TASK_FENCE_SELECTED_COHORT_CAPABILITIES,
    )

    def classify_route(route):
        witnesses.append(route)
        return catalog.classify_route(route)

    return (
        TaskFencePolicy(
            db,
            launch_catalog=SimpleNamespace(classify_route=classify_route),
        ),
        witnesses,
    )


def _assert_launch_witnesses(
    witnesses,
    *routes: str | tuple[TaskFenceCapabilityKind, str],
) -> None:
    assert witnesses == [
        TaskFenceLaunchRoute(
            kind=(
                TaskFenceCapabilityKind.ADAPTER
                if isinstance(route, str)
                else route[0]
            ),
            route_id=route if isinstance(route, str) else route[1],
            capability_version="task-fence-capability-v4",
        )
        for route in routes
    ]


def _tool_schema(name: str) -> dict:
    return {
        "name": name,
        "description": "Task Fence membrane probe",
        "parameters": {"type": "object", "properties": {}},
    }


@contextmanager
def _registered_tool(name, handler, *, is_async=False):
    registry.register(
        name=name,
        toolset="mcp-task-fence-membrane",
        schema=_tool_schema(name),
        handler=handler,
        is_async=is_async,
    )
    try:
        yield
    finally:
        registry.deregister(name)


def test_registered_handler_starts_after_durable_authorization(tmp_path):
    import model_tools

    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    policy, launch_witnesses = _recording_launch_policy(db)
    name = "mcp_task_fence_membrane_started"
    secret = "raw-secret-must-not-be-durable"
    observed = []

    def handler(args, **_kwargs):
        envelope = current_causal_envelope()
        row = db._conn.execute(
            "SELECT a.state, d.invocation_fingerprint, p.executor "
            "FROM task_fence_attempts AS a "
            "JOIN task_fence_dispatch_permits AS p ON p.permit_id = a.permit_id "
            "JOIN task_fence_policy_decisions AS d ON d.attempt_id = a.attempt_id "
            "WHERE p.invocation_envelope_id = ?",
            (envelope.invocation_id,),
        ).fetchone()
        observed.append((dict(args), envelope, tuple(row) if row else None))
        return json.dumps({"legacy": "unchanged"})

    try:
        with (
            _registered_tool(name, handler),
            bind_task_fence_policy(policy),
            bind_causal_envelope(generation),
        ):
            result = model_tools.handle_function_call(
                name,
                {"token": secret},
                task_id="route-a",
                session_id="session-a",
                user_task="task-a",
            )

        assert json.loads(result) == {"legacy": "unchanged"}
        _assert_launch_witnesses(
            launch_witnesses,
            (
                TaskFenceCapabilityKind.RUNTIME,
                "runtime:registered-tool-handoff",
            ),
        )
        assert len(observed) == 1
        args, envelope, attempt = observed[0]
        assert args == {"token": secret}
        assert envelope.invocation_id is not None
        assert attempt == (
            "STARTED",
            _task_fence_tool_fingerprint(
                name,
                args,
                {
                    "task_id": "route-a",
                    "session_id": "session-a",
                    "user_task": "task-a",
                },
            ),
            f"registry:{name}",
        )
        assert attempt[1] != _task_fence_tool_fingerprint(
            name,
            args,
            {
                "task_id": "route-b",
                "session_id": "session-a",
                "user_task": "task-a",
            },
        )
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT decision_point, outcome, reason_code "
                "FROM task_fence_policy_decisions ORDER BY decision_order"
            )
        ] == [
            (
                "admission",
                DecisionOutcome.WOULD_RESERVE.value,
                DecisionReason.CURRENT_AUTHORITY.value,
            ),
            (
                "authorization",
                DecisionOutcome.WOULD_ALLOW.value,
                DecisionReason.CURRENT_AUTHORITY.value,
            ),
        ]
        assert secret not in "\n".join(db._conn.iterdump())
    finally:
        db.close()


def test_newer_ingress_blocks_shadow_authorization_but_not_legacy_handler(
    tmp_path,
):
    db, acceptance, generation = _live_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    name = "mcp_task_fence_membrane_race"
    calls = []
    original_admit = TaskFencePolicy.admit_operation
    advanced = False

    def admit_then_advance(self, envelope, operation):
        nonlocal advanced
        decision = original_admit(self, envelope, operation)
        if self is policy and not advanced:
            advanced = True
            db.accept_task_fence_ingress(
                _ingress(
                    "comment_hold",
                    "membrane-newer-input",
                    task_id=acceptance.task_id,
                )
            )
        return decision

    def handler(args, **_kwargs):
        calls.append(dict(args))
        return json.dumps({"legacy": "still-ran"})

    try:
        with (
            _registered_tool(name, handler),
            bind_task_fence_policy(policy),
            bind_causal_envelope(generation.for_invocation("tfiv_membrane_race")),
            patch.object(TaskFencePolicy, "admit_operation", new=admit_then_advance),
        ):
            result = registry.dispatch(name, {"value": 1})

        assert json.loads(result) == {"legacy": "still-ran"}
        assert calls == [{"value": 1}]
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT decision_point, outcome, reason_code "
                "FROM task_fence_policy_decisions ORDER BY decision_order"
            )
        ] == [
            (
                "admission",
                DecisionOutcome.WOULD_RESERVE.value,
                DecisionReason.CURRENT_AUTHORITY.value,
            ),
            (
                "authorization",
                DecisionOutcome.WOULD_BLOCK.value,
                DecisionReason.PERMIT_REVOKED.value,
            ),
        ]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_missing_provenance_is_journaled_without_changing_legacy_result(tmp_path):
    db, _acceptance, _generation = _live_lane(tmp_path / "state.db")
    name = "mcp_task_fence_membrane_missing"
    calls = []

    def handler(args, **_kwargs):
        calls.append(dict(args))
        assert current_causal_envelope() is None
        return "legacy-result"

    try:
        with (
            _registered_tool(name, handler),
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(None),
        ):
            result = registry.dispatch(name, {"value": 2})

        assert result == "legacy-result"
        assert calls == [{"value": 2}]
        decision = db._conn.execute(
            "SELECT decision_point, outcome, reason_code, "
            "candidate_task_id, envelope_invocation_id "
            "FROM task_fence_policy_decisions"
        ).fetchone()
        assert tuple(decision) == (
            "admission",
            DecisionOutcome.WOULD_BLOCK.value,
            DecisionReason.MISSING_PROVENANCE.value,
            None,
            None,
        )
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_deferred_bridge_is_local_until_resolved_registry_handoff(tmp_path):
    import model_tools

    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    name = "mcp_task_fence_membrane_deferred"
    observed = []

    def handler(args, **_kwargs):
        envelope = current_causal_envelope()
        state = db._conn.execute(
            "SELECT a.state FROM task_fence_attempts AS a "
            "JOIN task_fence_dispatch_permits AS p ON p.permit_id = a.permit_id "
            "WHERE p.invocation_envelope_id = ?",
            (envelope.invocation_id,),
        ).fetchone()[0]
        observed.append((dict(args), envelope, state))
        return "deferred-result"

    try:
        with (
            _registered_tool(name, handler),
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation),
        ):
            described = model_tools.handle_function_call(
                "tool_describe",
                {"name": name},
                enabled_toolsets=["mcp-task-fence-membrane"],
            )
            assert json.loads(described)["name"] == name
            assert db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_policy_decisions"
            ).fetchone()[0] == 0

            result = model_tools.handle_function_call(
                "tool_call",
                {"name": name, "arguments": {"value": 2}},
                enabled_toolsets=["mcp-task-fence-membrane"],
            )

        assert result == "deferred-result"
        assert len(observed) == 1
        args, envelope, state = observed[0]
        assert args == {"value": 2}
        assert envelope.invocation_id is not None
        assert envelope.parent_invocation_id is not None
        assert state == "STARTED"
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 2
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_explicit_handler_provenance_tracks_final_handoff_child(tmp_path):
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    name = "mcp_task_fence_membrane_explicit"
    dispatcher = generation.for_invocation("tfiv_membrane_dispatcher")
    observed = []

    def handler(args, **kwargs):
        observed.append(
            (
                dict(args),
                current_causal_envelope(),
                kwargs["task_fence_envelope"],
            )
        )
        return "legacy-result"

    try:
        with (
            _registered_tool(name, handler),
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(dispatcher),
        ):
            result = registry.dispatch(
                name,
                {"value": 3},
                task_fence_envelope=dispatcher,
            )

        assert result == "legacy-result"
        assert len(observed) == 1
        _args, ambient, explicit = observed[0]
        assert ambient == explicit
        assert ambient.invocation_id != dispatcher.invocation_id
        assert ambient.parent_invocation_id == dispatcher.invocation_id

        with (
            _registered_tool(name, handler),
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(None),
        ):
            result = registry.dispatch(
                name,
                {"value": 4},
                task_fence_envelope=dispatcher,
            )

        assert result == "legacy-result"
        _args, ambient, explicit = observed[1]
        assert ambient is None
        assert explicit is None
        assert tuple(
            db._conn.execute(
                "SELECT outcome, reason_code, candidate_task_id "
                "FROM task_fence_policy_decisions ORDER BY decision_order DESC "
                "LIMIT 1"
            ).fetchone()
        ) == (
            DecisionOutcome.WOULD_BLOCK.value,
            DecisionReason.MISSING_PROVENANCE.value,
            None,
        )
    finally:
        db.close()


def test_async_registry_handler_starts_after_durable_authorization(tmp_path):
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    name = "mcp_task_fence_membrane_async"
    observed = []

    async def handler(args, **_kwargs):
        envelope = current_causal_envelope()
        state = db._conn.execute(
            "SELECT a.state FROM task_fence_attempts AS a "
            "JOIN task_fence_dispatch_permits AS p ON p.permit_id = a.permit_id "
            "WHERE p.invocation_envelope_id = ?",
            (envelope.invocation_id,),
        ).fetchone()[0]
        observed.append((dict(args), state))
        return "async-result"

    try:
        with (
            _registered_tool(name, handler, is_async=True),
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation),
        ):
            result = registry.dispatch(name, {"value": 5})

        assert result == "async-result"
        assert observed == [({"value": 5}, "STARTED")]
    finally:
        db.close()


def test_registry_audit_failure_is_secret_free_and_fail_open(
    tmp_path,
    caplog,
):
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    name = "mcp_task_fence_membrane_fault"
    secret = "audit-error-secret"
    calls = []

    def handler(args, **_kwargs):
        calls.append(dict(args))
        return "legacy-result"

    try:
        caplog.set_level(logging.WARNING, logger="tools.registry")
        with (
            _registered_tool(name, handler),
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation.for_invocation("tfiv_membrane_fault")),
            patch.object(
                TaskFencePolicy,
                "admit_operation",
                side_effect=RuntimeError(secret),
            ),
        ):
            result = registry.dispatch(name, {"value": 3})

        assert result == "legacy-result"
        assert calls == [{"value": 3}]
        assert secret not in caplog.text
        assert "RuntimeError" in caplog.text
    finally:
        db.close()


def test_openai_model_wire_race_is_shadow_only(tmp_path):
    from agent.chat_completion_helpers import _create_openai_chat_completion

    db, acceptance, generation = _live_model_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    original_admit = TaskFencePolicy.admit_operation
    calls = []
    advanced = False
    response = object()

    def admit_then_advance(self, envelope, operation):
        nonlocal advanced
        decision = original_admit(self, envelope, operation)
        if self is policy and not advanced:
            advanced = True
            db.accept_task_fence_ingress(
                _ingress(
                    "comment_hold",
                    "membrane-model-newer-input",
                    task_id=acceptance.task_id,
                )
            )
        return decision

    def create(**kwargs):
        calls.append(
            (
                dict(kwargs),
                current_causal_envelope(),
                current_task_fence_policy(),
            )
        )
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        model="test/model",
        base_url="https://openrouter.ai/api/v1",
    )
    request = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "shadow race"}],
    }
    try:
        with (
            bind_task_fence_policy(policy),
            bind_causal_envelope(generation),
            patch.object(
                TaskFencePolicy,
                "admit_operation",
                new=admit_then_advance,
            ),
        ):
            result = _create_openai_chat_completion(
                agent,
                client,
                request,
                task_fence_model_policy=policy,
            )

        assert result is response
        assert len(calls) == 1
        assert calls[0][0] == request
        assert calls[0][1].generation_id == generation.generation_id
        assert calls[0][1].invocation_id is not None
        assert calls[0][2] is None
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT decision_point, outcome, reason_code "
                "FROM task_fence_policy_decisions ORDER BY decision_order"
            )
        ] == [
            (
                "admission",
                DecisionOutcome.WOULD_RESERVE.value,
                DecisionReason.CURRENT_AUTHORITY.value,
            ),
            (
                "authorization",
                DecisionOutcome.WOULD_BLOCK.value,
                DecisionReason.PERMIT_REVOKED.value,
            ),
        ]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_openai_model_wire_audit_fault_is_secret_free_and_fail_open(
    tmp_path,
    caplog,
):
    from agent.chat_completion_helpers import _create_openai_chat_completion

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    secret = "provider-audit-error-secret"
    calls = []
    response = object()

    def create(**kwargs):
        calls.append((dict(kwargs), current_task_fence_policy()))
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        model="test/model",
        base_url="https://openrouter.ai/api/v1",
    )
    request = {"model": "test/model", "messages": []}
    policy = TaskFencePolicy(db)
    try:
        caplog.set_level(logging.WARNING, logger="agent.task_fence_provider")
        with (
            bind_task_fence_policy(policy),
            bind_causal_envelope(generation),
            patch.object(
                TaskFencePolicy,
                "admit_operation",
                side_effect=RuntimeError(secret),
            ),
        ):
            result = _create_openai_chat_completion(
                agent,
                client,
                request,
                task_fence_model_policy=policy,
            )

        assert result is response
        assert calls == [(request, None)]
        assert secret not in caplog.text
        assert "RuntimeError" in caplog.text
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_openai_model_wire_context_fault_does_not_lend_policy_to_sdk(
    tmp_path,
    caplog,
):
    from agent.chat_completion_helpers import _create_openai_chat_completion

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    secret = "provider-context-error-secret"
    calls = []
    response = object()

    def create(**kwargs):
        calls.append(
            (
                dict(kwargs),
                current_causal_envelope(),
                current_task_fence_policy(),
            )
        )
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        model="test/model",
        base_url="https://openrouter.ai/api/v1",
    )
    request = {"model": "test/model", "messages": []}
    policy = TaskFencePolicy(db)
    try:
        caplog.set_level(logging.WARNING, logger="agent.task_fence_provider")
        with (
            bind_task_fence_policy(policy),
            bind_causal_envelope(generation),
            patch(
                "task_fence.CausalEnvelope.for_invocation",
                side_effect=RuntimeError(secret),
            ),
        ):
            result = _create_openai_chat_completion(
                agent,
                client,
                request,
                task_fence_model_policy=policy,
            )

        assert result is response
        assert calls == [(request, generation, None)]
        assert secret not in caplog.text
        assert "RuntimeError" in caplog.text
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_openai_model_wire_ignores_public_tool_policy_without_explicit_capability(
    tmp_path,
):
    from agent.chat_completion_helpers import _create_openai_chat_completion

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    calls = []
    response = object()

    def create(**kwargs):
        calls.append((dict(kwargs), current_task_fence_policy()))
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        model="test/model",
        base_url="https://openrouter.ai/api/v1",
    )
    request = {"model": "test/model", "messages": []}
    try:
        with (
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation),
        ):
            result = _create_openai_chat_completion(agent, client, request)

        assert result is response
        assert calls == [(request, None)]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.parametrize(
    ("api_mode", "provider", "base_url", "openai_owned", "supported"),
    [
        (
            "chat_completions",
            "openrouter",
            "https://openrouter.ai/api/v1",
            True,
            True,
        ),
        (
            "chat_completions",
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            False,
            True,
        ),
        (
            "anthropic_messages",
            "anthropic",
            "https://api.anthropic.com",
            False,
            True,
        ),
        (
            "anthropic_messages",
            "bedrock",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            False,
            False,
        ),
        (
            "codex_responses",
            "openai-codex",
            "https://chatgpt.com/backend-api/codex",
            False,
            True,
        ),
        (
            "codex_responses",
            "copilot",
            "https://models.github.ai/inference",
            False,
            True,
        ),
        (
            "codex_responses",
            "xai-oauth",
            "https://api.x.ai/v1",
            False,
            True,
        ),
        ("codex_app_server", "openai-codex", "", False, False),
        ("bedrock_converse", "bedrock", "", False, True),
        (
            "bedrock_converse",
            "",
            "https://bedrock-runtime.eu-west-1.amazonaws.com",
            False,
            True,
        ),
        ("chat_completions", "moa", "https://virtual.invalid/v1", False, False),
        ("chat_completions", "copilot-acp", "acp://copilot", False, False),
        (
            "codex_responses",
            "custom",
            "acp+tcp://127.0.0.1:7777",
            False,
            False,
        ),
    ],
)
def test_model_wire_ownership_excludes_non_sdk_facades(
    api_mode,
    provider,
    base_url,
    openai_owned,
    supported,
):
    from agent.chat_completion_helpers import (
        _is_task_fence_openai_chat_wire,
        _is_task_fence_supported_model_wire,
    )

    agent = SimpleNamespace(
        api_mode=api_mode,
        provider=provider,
        base_url=base_url,
    )

    assert _is_task_fence_openai_chat_wire(agent) is openai_owned
    assert _is_task_fence_supported_model_wire(agent) is supported


@pytest.mark.parametrize(
    ("api_mode", "provider", "base_url", "supported"),
    [
        (
            "chat_completions",
            "openrouter",
            "https://openrouter.ai/api/v1",
            True,
        ),
        (
            "chat_completions",
            "google",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            True,
        ),
        ("anthropic_messages", "anthropic", "https://api.anthropic.com", True),
        ("codex_responses", "openai-codex", "", True),
        ("bedrock_converse", "bedrock", "", False),
        ("anthropic_messages", "bedrock", "", False),
        ("chat_completions", "moa", "moa://local", False),
        ("chat_completions", "copilot-acp", "acp://copilot", False),
    ],
)
def test_iteration_summary_owns_only_existing_provider_wires(
    api_mode,
    provider,
    base_url,
    supported,
):
    from agent.chat_completion_helpers import (
        _is_task_fence_supported_iteration_summary_wire,
    )

    agent = SimpleNamespace(
        api_mode=api_mode,
        provider=provider,
        base_url=base_url,
    )

    assert _is_task_fence_supported_iteration_summary_wire(agent) is supported


def test_persistent_moa_model_capability_is_per_call(tmp_path):
    from agent.auxiliary_client import _task_fence_sync_model_create

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    policy, launch_witnesses = _recording_launch_policy(db)
    prompt_secret = "raw-moa-aggregator-prompt-secret"
    request = {
        "model": "aggregator-model",
        "messages": [{"role": "user", "content": prompt_secret}],
    }
    observed = []

    def create(**kwargs):
        envelope = current_causal_envelope()
        attempt = None
        if envelope is not None and envelope.invocation_id is not None:
            attempt = db._conn.execute(
                "SELECT a.state FROM task_fence_attempts AS a "
                "JOIN task_fence_dispatch_permits AS p "
                "ON p.permit_id = a.permit_id "
                "WHERE p.invocation_envelope_id = ?",
                (envelope.invocation_id,),
            ).fetchone()
        observed.append(
            (
                dict(kwargs),
                envelope,
                None if attempt is None else attempt["state"],
                current_task_fence_policy(),
            )
        )
        return "legacy-result"

    client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )
    try:
        with (
            bind_causal_envelope(generation),
            bind_task_fence_policy(None),
        ):
            first = _task_fence_sync_model_create(
                client,
                request,
                route_provider="openrouter",
                task_fence_model_policy=policy,
            )
            second = _task_fence_sync_model_create(
                client,
                request,
                route_provider="openrouter",
            )

        assert first == second == "legacy-result"
        _assert_launch_witnesses(
            launch_witnesses,
            "provider:openai.chat.completions.create",
        )
        assert [entry[0] for entry in observed] == [request, request]
        assert observed[0][1].generation_id == generation.generation_id
        assert observed[0][1].invocation_id is not None
        assert observed[0][2] == "STARTED"
        assert observed[0][3] is None
        assert observed[1] == (request, generation, None, None)
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 1
        assert prompt_secret not in "\n".join(db._conn.iterdump())
    finally:
        db.close()


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("changed", "changed_reachable_route"),
        ("fault", "RuntimeError"),
        ("malformed", "AttributeError"),
    ],
)
def test_provider_launch_witness_failure_keeps_physical_handoff(
    tmp_path,
    caplog,
    case,
    expected_reason,
):
    from agent.auxiliary_client import _task_fence_sync_model_create

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    fault_secret = "raw-provider-classifier-fault-secret"
    if case == "changed":
        policy, _witnesses = _recording_launch_policy(db)
        launch_route = TaskFenceLaunchRoute(
            kind=TaskFenceCapabilityKind.ADAPTER,
            route_id="provider:openai.chat.completions.create",
            capability_version="task-fence-capability-v5",
        )
    elif case == "fault":
        def classify_route(_route):
            raise RuntimeError(fault_secret)

        policy = TaskFencePolicy(
            db,
            launch_catalog=SimpleNamespace(classify_route=classify_route),
        )
        launch_route = TaskFenceLaunchRoute(
            kind=TaskFenceCapabilityKind.ADAPTER,
            route_id="provider:openai.chat.completions.create",
            capability_version="task-fence-capability-v4",
        )
    else:
        policy = TaskFencePolicy(
            db,
            launch_catalog=SimpleNamespace(classify_route=lambda _route: object()),
        )
        launch_route = TaskFenceLaunchRoute(
            kind=TaskFenceCapabilityKind.ADAPTER,
            route_id="provider:openai.chat.completions.create",
            capability_version="task-fence-capability-v4",
        )

    request = {
        "model": "provider-launch-test",
        "messages": [{"role": "user", "content": "legacy call survives"}],
    }
    response = object()
    physical_calls = []

    def create(**kwargs):
        physical_calls.append(dict(kwargs))
        assert current_task_fence_policy() is None
        return response

    client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    try:
        caplog.set_level(logging.WARNING, logger="agent.task_fence_provider")
        with (
            bind_causal_envelope(generation),
            bind_task_fence_policy(None),
            patch(
                "agent.auxiliary_client."
                "_TASK_FENCE_OPENAI_CHAT_COMPLETIONS_LAUNCH_ROUTE",
                launch_route,
            ),
        ):
            result = _task_fence_sync_model_create(
                client,
                request,
                route_provider="openrouter",
                task_fence_model_policy=policy,
            )

        assert result is response
        assert physical_calls == [request]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_dispatch_permits"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
        assert any(
            expected_reason in record.message for record in caplog.records
        )
        assert fault_secret not in caplog.text
    finally:
        db.close()


def test_persistent_moa_facade_does_not_retain_capability():
    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("task-fence-test")
    policy = TaskFencePolicy(object())
    prepared = {"messages": [{"role": "user", "content": "hello"}]}
    observed = []

    def call_prepared(
        prepared_request,
        api_kwargs,
        *,
        task_fence_model_policy=None,
    ):
        observed.append(
            (
                prepared_request,
                dict(api_kwargs),
                task_fence_model_policy,
            )
        )
        return f"legacy-result-{len(observed)}"

    with patch.object(
        facade,
        "_call_prepared_aggregator",
        side_effect=call_prepared,
    ):
        first = facade.create(
            messages=prepared["messages"],
            _moa_prepared_request=prepared,
            _task_fence_model_policy=policy,
        )
        second = facade.create(
            messages=prepared["messages"],
            _moa_prepared_request=prepared,
        )

    assert (first, second) == ("legacy-result-1", "legacy-result-2")
    assert [entry[0] for entry in observed] == [prepared, prepared]
    assert [entry[1] for entry in observed] == [
        {"messages": prepared["messages"]},
        {"messages": prepared["messages"]},
    ]
    assert [entry[2] for entry in observed] == [policy, None]
    assert not any("task_fence" in key for key in facade.__dict__)
    assert prepared == {
        "messages": [{"role": "user", "content": "hello"}]
    }


@pytest.mark.parametrize(
    "base_url",
    [
        "acp://copilot",
        "acp+tcp://127.0.0.1:7777",
        "moa://local",
        "https://bedrock-runtime.us-east-1.amazonaws.com",
    ],
)
def test_persistent_moa_capability_excludes_unowned_auxiliary_facades(
    tmp_path,
    base_url,
):
    from agent.auxiliary_client import _task_fence_sync_model_create

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    calls = []
    request = {
        "model": "unsupported-aggregator",
        "messages": [{"role": "user", "content": "legacy input"}],
    }

    def create(**kwargs):
        calls.append(
            (
                dict(kwargs),
                current_causal_envelope(),
                current_task_fence_policy(),
            )
        )
        return "legacy-result"

    client = SimpleNamespace(
        base_url=base_url,
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )
    try:
        with (
            bind_causal_envelope(generation),
            bind_task_fence_policy(None),
        ):
            result = _task_fence_sync_model_create(
                client,
                request,
                route_provider="unsupported",
                task_fence_model_policy=TaskFencePolicy(db),
            )

        assert result == "legacy-result"
        assert calls == [(request, generation, None)]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.parametrize(
    ("client_type", "expected_private_fields"),
    [
        ("codex", {"_task_fence_model_policy", "_task_fence_model_route"}),
        (
            "anthropic",
            {"_task_fence_model_policy", "_task_fence_model_route"},
        ),
        ("gemini", {"_task_fence_model_policy"}),
    ],
)
def test_persistent_moa_capability_forwards_only_to_owned_auxiliary_adapter(
    tmp_path,
    client_type,
    expected_private_fields,
):
    from agent.auxiliary_client import (
        AnthropicAuxiliaryClient,
        CodexAuxiliaryClient,
        _task_fence_sync_model_create,
    )
    from agent.gemini_native_adapter import GeminiNativeClient

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    observed = []

    def create(**kwargs):
        observed.append(dict(kwargs))
        return "legacy-result"

    wrapper_type = {
        "codex": CodexAuxiliaryClient,
        "anthropic": AnthropicAuxiliaryClient,
        "gemini": GeminiNativeClient,
    }[client_type]
    client = object.__new__(wrapper_type)
    client.base_url = "https://owned-provider.invalid/v1"
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=create),
    )
    request = {
        "model": "owned-aggregator",
        "messages": [{"role": "user", "content": "hello"}],
    }
    try:
        with (
            bind_causal_envelope(generation),
            bind_task_fence_policy(None),
        ):
            result = _task_fence_sync_model_create(
                client,
                request,
                route_provider="owned-provider",
                task_fence_model_policy=policy,
            )

        assert result == "legacy-result"
        assert len(observed) == 1
        assert all(
            observed[0].get(key) == value
            for key, value in request.items()
        )
        assert {
            key for key in observed[0]
            if key.startswith("_task_fence_")
        } == expected_private_fields
        assert request == {
            "model": "owned-aggregator",
            "messages": [{"role": "user", "content": "hello"}],
        }
    finally:
        db.close()


def test_persistent_moa_codex_adapter_starts_before_responses_create(
    tmp_path,
):
    from agent.auxiliary_client import (
        CodexAuxiliaryClient,
        _task_fence_sync_model_create,
    )
    from agent.task_fence_provider import model_wire_fingerprint

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    policy, launch_witnesses = _recording_launch_policy(db)
    prompt_secret = "raw-moa-codex-prompt-secret"
    creates = []

    message_item = SimpleNamespace(
        type="message",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="codex done")],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_item.done", item=message_item),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                status="completed",
                id="resp-task-fence-moa",
                usage=SimpleNamespace(
                    input_tokens=1,
                    output_tokens=1,
                    total_tokens=2,
                ),
            ),
        ),
    ]

    class EventStream:
        def __iter__(self):
            return iter(events)

        def close(self):
            return None

    def create(**kwargs):
        envelope = current_causal_envelope()
        attempt = db._conn.execute(
            "SELECT a.state FROM task_fence_attempts AS a "
            "JOIN task_fence_dispatch_permits AS p "
            "ON p.permit_id = a.permit_id "
            "WHERE p.invocation_envelope_id = ?",
            (envelope.invocation_id,),
        ).fetchone()
        creates.append(
            (
                dict(kwargs),
                envelope,
                attempt["state"],
                current_task_fence_policy(),
            )
        )
        return EventStream()

    real_client = SimpleNamespace(
        api_key="raw-moa-codex-api-key",
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=create),
        close=lambda: None,
    )
    client = CodexAuxiliaryClient(real_client, "gpt-test-codex")
    request = {
        "model": "gpt-test-codex",
        "messages": [{"role": "user", "content": prompt_secret}],
    }
    try:
        with (
            bind_causal_envelope(generation),
            bind_task_fence_policy(None),
        ):
            response = _task_fence_sync_model_create(
                client,
                request,
                route_provider="openai-codex",
                task_fence_model_policy=policy,
            )

        assert response.choices[0].message.content == "codex done"
        assert len(creates) == 1
        _assert_launch_witnesses(
            launch_witnesses,
            "provider:openai.responses.create",
        )
        wire_request, envelope, state, ambient_policy = creates[0]
        assert wire_request["stream"] is True
        assert envelope.generation_id == generation.generation_id
        assert envelope.invocation_id is not None
        assert state == "STARTED"
        assert ambient_policy is None
        route = {
            "api_mode": "codex_responses",
            "provider": "openai-codex",
            "model": "gpt-test-codex",
            "endpoint": real_client.base_url,
        }
        expected = model_wire_fingerprint(
            adapter="provider:openai.responses.create",
            request=wire_request,
            route=route,
        )
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT adapter, invocation_fingerprint "
                "FROM task_fence_policy_decisions ORDER BY decision_order"
            )
        ] == [
            ("provider:openai.responses.create", expected),
            ("provider:openai.responses.create", expected),
        ]
        audit_dump = repr(
            [
                tuple(row)
                for table in (
                    "task_fence_policy_decisions",
                    "task_fence_dispatch_permits",
                    "task_fence_attempts",
                )
                for row in db._conn.execute(f"SELECT * FROM {table}")
            ]
        )
        assert prompt_secret not in audit_dump
        assert real_client.api_key not in audit_dump
    finally:
        db.close()


def test_persistent_moa_anthropic_adapter_consumes_private_capability(
    tmp_path,
):
    from agent.auxiliary_client import (
        AnthropicAuxiliaryClient,
        _task_fence_sync_model_create,
    )

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    observed = {}
    final_message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="anthropic done")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )

    def create_anthropic_message(_client, kwargs, **private):
        observed.update(request=dict(kwargs), private=dict(private))
        return final_message

    real_client = SimpleNamespace(messages=SimpleNamespace())
    client = AnthropicAuxiliaryClient(
        real_client,
        "claude-test",
        "raw-anthropic-key",
        "https://api.anthropic.com",
    )
    try:
        with (
            bind_causal_envelope(generation),
            bind_task_fence_policy(None),
            patch(
                "agent.anthropic_adapter.create_anthropic_message",
                side_effect=create_anthropic_message,
            ),
        ):
            response = _task_fence_sync_model_create(
                client,
                {
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                route_provider="anthropic",
                task_fence_model_policy=policy,
            )

        assert response.choices[0].message.content == "anthropic done"
        assert "_task_fence_model_policy" not in observed["request"]
        assert observed["private"]["task_fence_model_policy"] is policy
        assert observed["private"]["task_fence_model_route"] == {
            "api_mode": "anthropic_messages",
            "provider": "anthropic",
            "model": "claude-test",
            "endpoint": "https://api.anthropic.com",
        }
    finally:
        db.close()


def test_persistent_moa_bedrock_auxiliary_owns_exact_converse_leaf(
    tmp_path,
):
    from agent.auxiliary_client import (
        BedrockAuxiliaryClient,
        _task_fence_sync_model_create,
    )
    from agent.bedrock_adapter import build_converse_kwargs
    from agent.task_fence_provider import model_wire_fingerprint

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    policy, launch_witnesses = _recording_launch_policy(db)
    model = "openai.gpt-oss-20b-1:0"
    prompt_secret = "raw-moa-bedrock-converse-prompt"
    observed = []
    raw_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "bedrock done"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }

    def converse(**kwargs):
        envelope = current_causal_envelope()
        attempt = None
        if envelope is not None and envelope.invocation_id is not None:
            attempt = db._conn.execute(
                "SELECT a.state FROM task_fence_attempts AS a "
                "JOIN task_fence_dispatch_permits AS p "
                "ON p.permit_id = a.permit_id "
                "WHERE p.invocation_envelope_id = ?",
                (envelope.invocation_id,),
            ).fetchone()
        observed.append(
            (
                dict(kwargs),
                envelope,
                None if attempt is None else attempt["state"],
                current_task_fence_policy(),
                _context_carries_task_fence_authority(),
            )
        )
        return raw_response

    boto_client = SimpleNamespace(converse=converse)
    client = BedrockAuxiliaryClient("us-east-1", model)
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_secret}],
        "max_tokens": 32,
        "temperature": 0.2,
        "stop": "END",
        "stream": True,
    }
    route = {
        "api_mode": "bedrock_converse",
        "provider": "bedrock",
        "model": model,
        "endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "region": "us-east-1",
    }
    try:
        with (
            bind_causal_envelope(generation),
            bind_task_fence_policy(None),
            patch(
                "agent.bedrock_adapter._get_bedrock_runtime_client",
                return_value=boto_client,
            ),
        ):
            audited = _task_fence_sync_model_create(
                client,
                request,
                route_provider="bedrock",
                task_fence_model_policy=policy,
            )
            legacy = _task_fence_sync_model_create(
                client,
                request,
                route_provider="bedrock",
            )

        assert audited.choices[0].message.content == "bedrock done"
        assert legacy.choices[0].message.content == "bedrock done"
        _assert_launch_witnesses(
            launch_witnesses,
            "provider:bedrock.converse",
        )
        assert len(observed) == 2
        native_request = observed[0][0]
        assert native_request == build_converse_kwargs(
            model=model,
            messages=request["messages"],
            max_tokens=32,
            temperature=0.2,
            stop_sequences=["END"],
        )
        assert observed[0][1].generation_id == generation.generation_id
        assert observed[0][1].invocation_id is not None
        assert observed[0][2:] == ("STARTED", None, False)
        assert observed[1] == (native_request, generation, None, None, False)
        assert not any(key.startswith("_task_fence_") for key in native_request)

        expected_fingerprint = model_wire_fingerprint(
            adapter="provider:bedrock.converse",
            request=native_request,
            route=route,
        )
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT adapter, invocation_fingerprint "
                "FROM task_fence_policy_decisions ORDER BY decision_order"
            )
        ] == [
            ("provider:bedrock.converse", expected_fingerprint),
            ("provider:bedrock.converse", expected_fingerprint),
        ]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 1
        audit_dump = "\n".join(db._conn.iterdump())
        assert prompt_secret not in audit_dump
        assert client.api_key not in audit_dump
    finally:
        db.close()


def test_persistent_moa_anthropic_bedrock_owns_stream_fallback_leaves(
    tmp_path,
):
    from agent.auxiliary_client import (
        AnthropicAuxiliaryClient,
        _task_fence_sync_model_create,
    )
    from agent.task_fence_provider import model_wire_fingerprint

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    prompt_secret = "raw-moa-anthropic-bedrock-prompt"
    observed = []
    final_message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="anthropic bedrock done")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )

    def inspect(stage, kwargs):
        envelope = current_causal_envelope()
        attempt = None
        if envelope is not None and envelope.invocation_id is not None:
            attempt = db._conn.execute(
                "SELECT a.state FROM task_fence_attempts AS a "
                "JOIN task_fence_dispatch_permits AS p "
                "ON p.permit_id = a.permit_id "
                "WHERE p.invocation_envelope_id = ?",
                (envelope.invocation_id,),
            ).fetchone()
        observed.append(
            (
                stage,
                dict(kwargs),
                envelope,
                None if attempt is None else attempt["state"],
                current_task_fence_policy(),
                _context_carries_task_fence_authority(),
            )
        )

    class UnavailableStream:
        def __init__(self, kwargs):
            self._kwargs = kwargs

        def __enter__(self):
            inspect("stream_enter", self._kwargs)
            raise RuntimeError(
                "not authorized to perform: "
                "bedrock:InvokeModelWithResponseStream"
            )

        def __exit__(self, *_args):
            return False

    class Messages:
        @staticmethod
        def stream(**kwargs):
            inspect("stream_factory", kwargs)
            return UnavailableStream(kwargs)

        @staticmethod
        def create(**kwargs):
            inspect("create", kwargs)
            return final_message

    real_client = SimpleNamespace(
        messages=Messages(),
        close=lambda: None,
    )
    client = AnthropicAuxiliaryClient(
        real_client,
        model,
        "aws-sdk",
        "https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_secret}],
        "max_tokens": 64,
    }
    route = {
        "api_mode": "anthropic_messages",
        "provider": "bedrock",
        "model": model,
        "endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
    }
    try:
        with (
            bind_causal_envelope(generation),
            bind_task_fence_policy(None),
        ):
            audited = _task_fence_sync_model_create(
                client,
                request,
                route_provider="bedrock",
                task_fence_model_policy=policy,
            )
            legacy = _task_fence_sync_model_create(
                client,
                request,
                route_provider="bedrock",
            )

        assert audited.choices[0].message.content == "anthropic bedrock done"
        assert legacy.choices[0].message.content == "anthropic bedrock done"
        assert [entry[0] for entry in observed] == [
            "stream_factory",
            "stream_enter",
            "create",
            "stream_factory",
            "stream_enter",
            "create",
        ]
        assert observed[0][1] == observed[3][1]
        assert observed[2][1] == observed[5][1]
        assert {entry[3] for entry in observed[:3]} == {"STARTED"}
        assert {entry[3] for entry in observed[3:]} == {None}
        assert all(entry[4] is None for entry in observed)
        assert not any(entry[5] for entry in observed)
        stream_envelope = observed[0][2]
        assert observed[1][2] == stream_envelope
        create_envelope = observed[2][2]
        assert stream_envelope.invocation_id != create_envelope.invocation_id
        assert {
            stream_envelope.generation_id,
            create_envelope.generation_id,
        } == {generation.generation_id}
        assert all(
            not any(key.startswith("_task_fence_") for key in entry[1])
            for entry in observed
        )

        stream_fingerprint = model_wire_fingerprint(
            adapter="provider:anthropic.messages.stream",
            request=observed[0][1],
            route=route,
        )
        create_fingerprint = model_wire_fingerprint(
            adapter="provider:anthropic.messages.create",
            request=observed[2][1],
            route=route,
        )
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT adapter, invocation_fingerprint "
                "FROM task_fence_policy_decisions ORDER BY decision_order"
            )
        ] == [
            ("provider:anthropic.messages.stream", stream_fingerprint),
            ("provider:anthropic.messages.stream", stream_fingerprint),
            ("provider:anthropic.messages.create", create_fingerprint),
            ("provider:anthropic.messages.create", create_fingerprint),
        ]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 2
        audit_dump = "\n".join(db._conn.iterdump())
        assert prompt_secret not in audit_dump
        assert client.api_key not in audit_dump
    finally:
        db.close()


@pytest.mark.parametrize(
    ("wrapper", "base_url", "model"),
    [
        (
            "anthropic",
            "https://bedrock-runtime.us-east-1.amazonaws.com.attacker.test",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ),
        (
            "anthropic",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            "openai.gpt-oss-20b-1:0",
        ),
        (
            "converse",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        ),
        (
            "converse",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            "",
        ),
    ],
)
def test_persistent_moa_bedrock_capability_rejects_wrapper_mismatch(
    tmp_path,
    wrapper,
    base_url,
    model,
):
    from agent.auxiliary_client import (
        AnthropicAuxiliaryClient,
        BedrockAuxiliaryClient,
        _task_fence_sync_model_create,
    )

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    observed = []

    def create(**kwargs):
        observed.append(dict(kwargs))
        return "legacy-result"

    if wrapper == "anthropic":
        client = AnthropicAuxiliaryClient(
            SimpleNamespace(messages=SimpleNamespace()),
            model,
            "aws-sdk",
            base_url,
        )
    else:
        client = BedrockAuxiliaryClient("us-east-1", model)
    client.base_url = base_url
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=create),
    )
    request = {
        "model": model,
        "messages": [{"role": "user", "content": "legacy input"}],
    }
    try:
        with (
            bind_causal_envelope(generation),
            bind_task_fence_policy(None),
        ):
            result = _task_fence_sync_model_create(
                client,
                request,
                route_provider="bedrock",
                task_fence_model_policy=TaskFencePolicy(db),
            )

        assert result == "legacy-result"
        assert observed == [request]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_model_wire_fingerprint_commits_binary_payloads():
    from agent.task_fence_provider import model_wire_fingerprint

    route = {
        "api_mode": "bedrock_converse",
        "provider": "bedrock",
        "model": "amazon.nova-test-v1:0",
        "endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "region": "us-east-1",
    }

    def fingerprint(payload):
        return model_wire_fingerprint(
            adapter="provider:bedrock.converse",
            request={
                "modelId": route["model"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": payload},
                                }
                            }
                        ],
                    }
                ],
            },
            route=route,
        )

    payload = b"binary-image-a"
    assert fingerprint(payload) == fingerprint(bytearray(payload))
    assert fingerprint(payload) != fingerprint(b"binary-image-b")
    assert fingerprint(payload) != fingerprint(
        {
            "__task_fence_binary__": {
                "length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
    )


def test_bedrock_converse_requires_explicit_model_capability(tmp_path):
    from agent.chat_completion_helpers import (
        _dispatch_nonstreaming_api_request,
    )

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    observed = []
    raw_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "legacy bedrock response"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }

    def converse(**kwargs):
        observed.append(
            (
                dict(kwargs),
                current_causal_envelope(),
                current_task_fence_policy(),
            )
        )
        return raw_response

    client = SimpleNamespace(converse=converse)
    agent = SimpleNamespace(
        api_mode="bedrock_converse",
        provider="bedrock",
        model="amazon.nova-test-v1:0",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    request = {
        "modelId": agent.model,
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "__bedrock_converse__": True,
        "__bedrock_region__": "us-east-1",
    }
    try:
        with (
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation),
            patch(
                "agent.bedrock_adapter._get_bedrock_runtime_client",
                return_value=client,
            ),
        ):
            response = _dispatch_nonstreaming_api_request(
                agent,
                request,
                make_client=lambda *_args, **_kwargs: pytest.fail(
                    "Bedrock must not build an OpenAI client"
                ),
            )

        assert response.choices[0].message.content == "legacy bedrock response"
        assert observed == [
            (
                {
                    "modelId": agent.model,
                    "messages": [
                        {"role": "user", "content": [{"text": "hello"}]}
                    ],
                },
                generation,
                None,
            )
        ]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_bedrock_nonstream_stale_error_preserves_legacy_failure(tmp_path):
    from agent.chat_completion_helpers import (
        _dispatch_nonstreaming_api_request,
    )

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    stale_error = RuntimeError("sentinel stale Bedrock connection")
    observed = []

    def converse(**kwargs):
        observed.append(dict(kwargs))
        raise stale_error

    client = SimpleNamespace(converse=converse)
    agent = SimpleNamespace(
        api_mode="bedrock_converse",
        provider="bedrock",
        model="amazon.nova-test-v1:0",
        base_url="https://bedrock-runtime.ap-south-1.amazonaws.com",
    )
    request = {
        "modelId": agent.model,
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "__bedrock_converse__": True,
        "__bedrock_region__": "ap-south-1",
    }
    try:
        with (
            bind_causal_envelope(generation),
            patch(
                "agent.bedrock_adapter._get_bedrock_runtime_client",
                return_value=client,
            ),
            patch(
                "agent.bedrock_adapter.is_stale_connection_error",
                return_value=True,
            ) as classify,
            patch(
                "agent.bedrock_adapter.invalidate_runtime_client",
            ) as invalidate,
        ):
            with pytest.raises(RuntimeError) as raised:
                _dispatch_nonstreaming_api_request(
                    agent,
                    request,
                    make_client=lambda *_args, **_kwargs: pytest.fail(
                        "Bedrock must not build an OpenAI client"
                    ),
                    task_fence_model_policy=policy,
                )

        assert raised.value is stale_error
        assert observed == [
            {
                "modelId": agent.model,
                "messages": [
                    {"role": "user", "content": [{"text": "hello"}]}
                ],
            }
        ]
        classify.assert_called_once_with(stale_error)
        invalidate.assert_called_once_with("ap-south-1")
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_gemini_native_nonstream_starts_before_http_handoff(tmp_path):
    from agent.gemini_native_adapter import GeminiNativeClient
    from agent.task_fence_provider import model_wire_fingerprint

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    prompt_secret = "raw-gemini-prompt-secret"
    api_secret = "raw-gemini-api-key-secret"
    observed = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "gemini done"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
            }

    class HTTP:
        def post(self, url, *, json, headers, timeout):
            envelope = current_causal_envelope()
            attempt = db._conn.execute(
                "SELECT a.state FROM task_fence_attempts AS a "
                "JOIN task_fence_dispatch_permits AS p "
                "ON p.permit_id = a.permit_id "
                "WHERE p.invocation_envelope_id = ?",
                (envelope.invocation_id,),
            ).fetchone()
            observed.update(
                url=url,
                request=json,
                headers=headers,
                envelope=envelope,
                policy=current_task_fence_policy(),
                attempt=attempt["state"],
            )
            return Response()

        def close(self):
            return None

    endpoint = "https://generativelanguage.googleapis.com/v1beta"
    client = GeminiNativeClient(
        api_key=api_secret,
        base_url=endpoint,
        http_client=HTTP(),
    )
    policy, launch_witnesses = _recording_launch_policy(db)
    try:
        with (
            bind_task_fence_policy(policy),
            bind_causal_envelope(generation),
        ):
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt_secret}],
                _task_fence_model_policy=policy,
            )

        assert response.choices[0].message.content == "gemini done"
        _assert_launch_witnesses(
            launch_witnesses,
            "provider:gemini.generateContent",
        )
        assert observed["attempt"] == "STARTED"
        assert observed["envelope"].generation_id == generation.generation_id
        assert observed["envelope"].invocation_id is not None
        assert observed["policy"] is None
        assert observed["headers"]["x-goog-api-key"] == api_secret
        expected = model_wire_fingerprint(
            adapter="provider:gemini.generateContent",
            request={
                "model": "gemini-2.5-flash",
                "request": observed["request"],
            },
            route={
                "api_mode": "chat_completions",
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "endpoint": endpoint,
            },
        )
        decisions = db._conn.execute(
            "SELECT adapter, invocation_fingerprint "
            "FROM task_fence_policy_decisions ORDER BY decision_order"
        ).fetchall()
        assert [tuple(row) for row in decisions] == [
            ("provider:gemini.generateContent", expected),
            ("provider:gemini.generateContent", expected),
        ]
        audit_dump = repr(
            [
                tuple(row)
                for table in (
                    "task_fence_policy_decisions",
                    "task_fence_dispatch_permits",
                    "task_fence_attempts",
                )
                for row in db._conn.execute(f"SELECT * FROM {table}")
            ]
        )
        assert prompt_secret not in audit_dump
        assert api_secret not in audit_dump
    finally:
        client.close()
        db.close()


def test_gemini_native_stream_starts_at_lazy_http_handoff(tmp_path):
    from agent.gemini_native_adapter import GeminiNativeClient

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    observed = []

    class StreamResponse:
        status_code = 200

        @staticmethod
        def iter_text():
            payload = {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "streamed"}]},
                        "finishReason": "STOP",
                    }
                ]
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

    class StreamContext:
        def __init__(self, method, url, request):
            self.method = method
            self.url = url
            self.request = request

        def __enter__(self):
            envelope = current_causal_envelope()
            attempt = db._conn.execute(
                "SELECT a.state FROM task_fence_attempts AS a "
                "JOIN task_fence_dispatch_permits AS p "
                "ON p.permit_id = a.permit_id "
                "WHERE p.invocation_envelope_id = ?",
                (envelope.invocation_id,),
            ).fetchone()
            observed.append(
                (
                    self.method,
                    self.url,
                    self.request,
                    envelope,
                    attempt["state"],
                    current_task_fence_policy(),
                    len(launch_witnesses),
                )
            )
            return StreamResponse()

        def __exit__(self, *_args):
            return False

    class HTTP:
        def stream(self, method, url, *, json, headers, timeout):
            return StreamContext(method, url, dict(json))

        def close(self):
            return None

    client = GeminiNativeClient(
        api_key="test-key",
        http_client=HTTP(),
    )
    policy, launch_witnesses = _recording_launch_policy(db)
    try:
        with (
            bind_task_fence_policy(policy),
            bind_causal_envelope(generation),
        ):
            stream = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "stream please"}],
                stream=True,
                _task_fence_model_policy=policy,
            )
            assert observed == []
            assert launch_witnesses == []
            chunks = list(stream)

        assert len(observed) == 1
        _assert_launch_witnesses(
            launch_witnesses,
            "provider:gemini.streamGenerateContent",
        )
        assert observed[0][0] == "POST"
        assert observed[0][4] == "STARTED"
        assert observed[0][3].generation_id == generation.generation_id
        assert observed[0][3].invocation_id is not None
        assert observed[0][5] is None
        assert observed[0][6] == 1
        assert chunks[0].choices[0].delta.content == "streamed"
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 1
        assert [
            row[0]
            for row in db._conn.execute(
                "SELECT adapter FROM task_fence_policy_decisions "
                "ORDER BY decision_order"
            )
        ] == [
            "provider:gemini.streamGenerateContent",
            "provider:gemini.streamGenerateContent",
        ]
    finally:
        client.close()
        db.close()


@pytest.mark.asyncio
async def test_async_gemini_stream_does_not_carry_policy_across_threaded_yields(
    tmp_path,
):
    from agent.gemini_native_adapter import (
        AsyncGeminiNativeClient,
        GeminiNativeClient,
    )

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    observed = []

    class StreamResponse:
        status_code = 200

        @staticmethod
        def iter_text():
            for text in ("first", "second"):
                payload = {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": text}]},
                            "finishReason": "STOP",
                        }
                    ]
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

    class StreamContext:
        def __enter__(self):
            envelope = current_causal_envelope()
            attempt = db._conn.execute(
                "SELECT a.state FROM task_fence_attempts AS a "
                "JOIN task_fence_dispatch_permits AS p "
                "ON p.permit_id = a.permit_id "
                "WHERE p.invocation_envelope_id = ?",
                (envelope.invocation_id,),
            ).fetchone()
            observed.append(
                ("enter", current_task_fence_policy(), envelope, attempt["state"])
            )
            return StreamResponse()

        def __exit__(self, *_args):
            observed.append(
                (
                    "exit",
                    current_task_fence_policy(),
                    current_causal_envelope(),
                    None,
                )
            )
            return False

    class HTTP:
        def stream(self, method, url, *, json, headers, timeout):
            observed.append(
                (
                    "stream",
                    current_task_fence_policy(),
                    current_causal_envelope(),
                    method,
                )
            )
            return StreamContext()

        def close(self):
            return None

    sync_client = GeminiNativeClient(
        api_key="test-key",
        http_client=HTTP(),
    )
    client = AsyncGeminiNativeClient(sync_client)
    try:
        with (
            bind_task_fence_policy(policy),
            bind_causal_envelope(generation),
        ):
            stream = await client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "stream twice"}],
                stream=True,
                _task_fence_model_policy=policy,
            )
            chunks = [chunk async for chunk in stream]
            assert current_task_fence_policy() is policy

        assert [
            chunk.choices[0].delta.content
            for chunk in chunks
            if chunk.choices[0].delta.content is not None
        ] == [
            "first",
            "second",
        ]
        assert [entry[0] for entry in observed] == [
            "stream",
            "enter",
            "exit",
        ]
        assert all(entry[1] is None for entry in observed)
        assert observed[0][2].invocation_id is not None
        assert observed[0][2] == observed[1][2]
        assert observed[1][3] == "STARTED"
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 1
    finally:
        await client.close()
        db.close()


def test_anthropic_stream_fallback_starts_each_physical_handoff(tmp_path):
    from agent.anthropic_adapter import create_anthropic_message

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    policy, launch_witnesses = _recording_launch_policy(db)
    prompt_secret = "raw-anthropic-fallback-secret"
    observed = []
    response = SimpleNamespace(content=[], stop_reason="end_turn")

    def inspect_handoff(stage, request):
        envelope = current_causal_envelope()
        attempt = db._conn.execute(
            "SELECT a.state FROM task_fence_attempts AS a "
            "JOIN task_fence_dispatch_permits AS p "
            "ON p.permit_id = a.permit_id "
            "WHERE p.invocation_envelope_id = ?",
            (envelope.invocation_id,),
        ).fetchone()
        observed.append(
            (
                stage,
                dict(request),
                envelope,
                attempt["state"],
                current_task_fence_policy(),
            )
        )

    class UnavailableStream:
        def __enter__(self):
            inspect_handoff("stream_enter", request)
            raise RuntimeError("stream is not supported by this provider")

        def __exit__(self, *_args):
            return False

    class Messages:
        @staticmethod
        def stream(**kwargs):
            inspect_handoff("stream_factory", kwargs)
            return UnavailableStream()

        @staticmethod
        def create(**kwargs):
            inspect_handoff("create", kwargs)
            return response

    request = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": prompt_secret}],
    }
    route = {
        "api_mode": "anthropic_messages",
        "provider": "anthropic",
        "model": "claude-test",
        "endpoint": "https://api.anthropic.com",
    }
    try:
        with (
            bind_task_fence_policy(policy),
            bind_causal_envelope(generation),
        ):
            result = create_anthropic_message(
                SimpleNamespace(messages=Messages()),
                request,
                task_fence_model_policy=policy,
                task_fence_model_route=route,
            )

        assert result is response
        _assert_launch_witnesses(
            launch_witnesses,
            "provider:anthropic.messages.stream",
            "provider:anthropic.messages.create",
        )
        assert [entry[0] for entry in observed] == [
            "stream_factory",
            "stream_enter",
            "create",
        ]
        assert {entry[3] for entry in observed} == {"STARTED"}
        assert {entry[4] for entry in observed} == {None}
        stream_envelope = observed[0][2]
        assert observed[1][2] == stream_envelope
        create_envelope = observed[2][2]
        assert stream_envelope.invocation_id != create_envelope.invocation_id
        assert {
            stream_envelope.generation_id,
            create_envelope.generation_id,
        } == {generation.generation_id}
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT decision_point, adapter "
                "FROM task_fence_policy_decisions ORDER BY decision_order"
            )
        ] == [
            ("admission", "provider:anthropic.messages.stream"),
            ("authorization", "provider:anthropic.messages.stream"),
            ("admission", "provider:anthropic.messages.create"),
            ("authorization", "provider:anthropic.messages.create"),
        ]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 2
        audit_dump = repr(
            [
                tuple(row)
                for table in (
                    "task_fence_policy_decisions",
                    "task_fence_dispatch_permits",
                    "task_fence_attempts",
                )
                for row in db._conn.execute(f"SELECT * FROM {table}")
            ]
        )
        assert prompt_secret not in audit_dump
    finally:
        db.close()


def test_anthropic_create_requires_explicit_model_capability(tmp_path):
    from agent.anthropic_adapter import create_anthropic_message

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    observed = []
    response = SimpleNamespace(content=[], stop_reason="end_turn")

    class Messages:
        @staticmethod
        def create(**kwargs):
            observed.append(
                (
                    dict(kwargs),
                    current_causal_envelope(),
                    current_task_fence_policy(),
                )
            )
            return response

    try:
        with (
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation),
        ):
            result = create_anthropic_message(
                SimpleNamespace(messages=Messages()),
                {"model": "claude-test", "messages": []},
                prefer_stream=False,
            )

        assert result is response
        assert len(observed) == 1
        assert observed[0][1] == generation
        assert observed[0][2] is None
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_codex_responses_requires_explicit_model_capability(tmp_path):
    from agent.codex_runtime import run_codex_stream

    db, _acceptance, generation = _live_model_lane(tmp_path / "state.db")
    observed = []
    response = SimpleNamespace(output=[object()])

    def create(**kwargs):
        observed.append(
            (
                dict(kwargs),
                current_causal_envelope(),
                current_task_fence_policy(),
            )
        )
        return response

    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        model="gpt-test-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        _interrupt_requested=False,
        _fire_stream_delta=lambda _text: None,
        _fire_reasoning_delta=lambda _text: None,
        _fire_streamed_codex_commentary=lambda _text: None,
        _touch_activity=lambda _description: None,
        _client_log_context=lambda: "",
        interim_assistant_callback=None,
        show_commentary=True,
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(create=create),
    )
    try:
        with (
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation),
        ):
            result = run_codex_stream(
                agent,
                {"model": "gpt-test-codex", "input": []},
                client=client,
            )

        assert result is response
        assert len(observed) == 1
        assert observed[0][0]["stream"] is True
        assert observed[0][1] == generation
        assert observed[0][2] is None
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_non_json_handler_extension_does_not_create_unstable_attempt(
    tmp_path,
    caplog,
):
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    name = "mcp_task_fence_membrane_non_json"
    extension = object()
    observed = []

    def handler(args, **kwargs):
        observed.append((dict(args), kwargs["extension"]))
        return "legacy-result"

    try:
        caplog.set_level(logging.WARNING, logger="tools.registry")
        with (
            _registered_tool(name, handler),
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation),
        ):
            result = registry.dispatch(name, {"value": 3}, extension=extension)

        assert result == "legacy-result"
        assert observed == [({"value": 3}, extension)]
        assert "TypeError" in caplog.text
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_execution_middleware_direct_registry_dispatch_is_observed(tmp_path):
    import model_tools

    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    original_name = "mcp_task_fence_membrane_original"
    bypass_name = "mcp_task_fence_membrane_bypass"
    original_calls = []
    bypass_calls = []

    def started_state():
        envelope = current_causal_envelope()
        state = db._conn.execute(
            "SELECT a.state FROM task_fence_attempts AS a "
            "JOIN task_fence_dispatch_permits AS p ON p.permit_id = a.permit_id "
            "WHERE p.invocation_envelope_id = ?",
            (envelope.invocation_id,),
        ).fetchone()[0]
        return envelope, state

    def original_handler(args, **_kwargs):
        envelope, state = started_state()
        original_calls.append((dict(args), envelope, state))
        return "original"

    def bypass_handler(args, **_kwargs):
        envelope, state = started_state()
        bypass_calls.append((dict(args), envelope, state))
        return "bypass"

    def redirect_to_registry(_name, args, next_call, **_kwargs):
        assert registry.dispatch(bypass_name, {"route": "plugin"}) == "bypass"
        return next_call(args)

    try:
        with (
            _registered_tool(original_name, original_handler),
            _registered_tool(bypass_name, bypass_handler),
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation),
            patch(
                "hermes_cli.middleware.run_tool_execution_middleware",
                side_effect=redirect_to_registry,
            ),
        ):
            result = model_tools.handle_function_call(original_name, {"value": 4})

        assert result == "original"
        assert len(original_calls) == 1
        assert len(bypass_calls) == 1
        original_args, original_envelope, original_state = original_calls[0]
        bypass_args, bypass_envelope, bypass_state = bypass_calls[0]
        assert original_args == {"value": 4}
        assert bypass_args == {"route": "plugin"}
        assert original_state == bypass_state == "STARTED"
        assert original_envelope.invocation_id != bypass_envelope.invocation_id
        assert (
            original_envelope.parent_invocation_id
            == bypass_envelope.parent_invocation_id
        )
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT adapter, invocation_fingerprint "
                "FROM task_fence_policy_decisions "
                "ORDER BY decision_order"
            )
        ] == [
            (
                f"registry:{bypass_name}",
                _task_fence_tool_fingerprint(
                    bypass_name,
                    {"route": "plugin"},
                ),
            ),
        ] * 2 + [
            (
                f"registry:{original_name}",
                _task_fence_tool_fingerprint(
                    original_name,
                    {"value": 4},
                    {
                        "task_id": None,
                        "session_id": None,
                        "user_task": None,
                    },
                ),
            ),
        ] * 2
    finally:
        db.close()


def test_execution_middleware_short_circuit_creates_no_native_attempt(tmp_path):
    import model_tools

    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    name = "mcp_task_fence_membrane_short_circuit"
    calls = []

    def handler(args, **_kwargs):
        calls.append(dict(args))
        return "native"

    try:
        with (
            _registered_tool(name, handler),
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation),
            patch(
                "hermes_cli.middleware.run_tool_execution_middleware",
                return_value="managed",
            ),
        ):
            result = model_tools.handle_function_call(name, {"value": 5})

        assert result == "managed"
        assert calls == []
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_unknown_registry_tool_creates_no_phantom_attempt(tmp_path):
    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    try:
        with (
            bind_task_fence_policy(TaskFencePolicy(db)),
            bind_causal_envelope(generation),
        ):
            result = registry.dispatch("mcp_task_fence_unknown", {})

        assert json.loads(result) == {"error": "Unknown tool: mcp_task_fence_unknown"}
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_inline_handoff_barrier_race_is_shadow_only(tmp_path):
    from agent.tool_executor import _run_agent_tool_execution_middleware

    db, acceptance, generation = _live_lane(tmp_path / "state.db")
    policy, launch_witnesses = _recording_launch_policy(db)
    dispatcher = generation.for_invocation("tfiv_inline_race_dispatcher")
    original_admit = TaskFencePolicy.admit_operation
    calls = []
    advanced = False

    def admit_then_advance(self, envelope, operation):
        nonlocal advanced
        decision = original_admit(self, envelope, operation)
        if self is policy and not advanced:
            advanced = True
            db.accept_task_fence_ingress(
                _ingress(
                    "comment_hold",
                    "inline-race-newer-input",
                    task_id=acceptance.task_id,
                )
            )
        return decision

    def execute(args):
        calls.append((dict(args), current_causal_envelope()))
        return "legacy-inline-result"

    def call_next(_name, args, next_call, **_kwargs):
        return next_call(args)

    agent = SimpleNamespace(
        session_id="inline-session",
        _current_turn_id="turn-inline",
        _current_api_request_id="request-inline",
        _tool_guardrails=_allowing_tool_guardrails(),
    )
    try:
        with (
            bind_task_fence_policy(policy),
            patch.object(
                TaskFencePolicy,
                "admit_operation",
                new=admit_then_advance,
            ),
            patch(
                "hermes_cli.middleware.run_tool_execution_middleware",
                side_effect=call_next,
            ),
            patch("agent.tool_executor._begin_tool_execution") as begin_execution,
        ):
            outcome = _run_agent_tool_execution_middleware(
                agent,
                function_name="todo",
                function_args={"value": 1},
                effective_task_id="sandbox-task",
                tool_call_id="call-inline",
                causal_envelope=dispatcher,
                execute=execute,
            )

        assert outcome.result == "legacy-inline-result"
        _assert_launch_witnesses(
            launch_witnesses,
            (TaskFenceCapabilityKind.RUNTIME, "runtime:inline-tool-handoff"),
        )
        assert outcome.args == {"value": 1}
        begin_execution.assert_called_once()
        assert len(calls) == 1
        args, envelope = calls[0]
        assert args == {"value": 1}
        assert envelope.parent_invocation_id == dispatcher.invocation_id
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT decision_point, outcome, reason_code, adapter "
                "FROM task_fence_policy_decisions ORDER BY decision_order"
            )
        ] == [
            (
                "admission",
                DecisionOutcome.WOULD_RESERVE.value,
                DecisionReason.CURRENT_AUTHORITY.value,
                "agent-runtime:todo",
            ),
            (
                "authorization",
                DecisionOutcome.WOULD_BLOCK.value,
                DecisionReason.PERMIT_REVOKED.value,
                "agent-runtime:todo",
            ),
        ]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("changed", "changed_reachable_route"),
        ("fault", "RuntimeError"),
        ("malformed", "AttributeError"),
    ],
)
def test_registered_launch_witness_failure_keeps_handler_call(
    tmp_path,
    caplog,
    case,
    expected_reason,
):
    import tools.registry as registry_module

    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    fault_secret = "raw-tool-classifier-fault-secret"
    if case == "changed":
        policy, _witnesses = _recording_launch_policy(db)
        launch_route = TaskFenceLaunchRoute(
            kind=TaskFenceCapabilityKind.RUNTIME,
            route_id="runtime:registered-tool-handoff",
            capability_version="task-fence-capability-v5",
        )
    elif case == "fault":
        def classify_route(_route):
            raise RuntimeError(fault_secret)

        policy = TaskFencePolicy(
            db,
            launch_catalog=SimpleNamespace(classify_route=classify_route),
        )
        launch_route = registry_module._TASK_FENCE_REGISTERED_TOOL_LAUNCH_ROUTE
    else:
        policy = TaskFencePolicy(
            db,
            launch_catalog=SimpleNamespace(classify_route=lambda _route: object()),
        )
        launch_route = registry_module._TASK_FENCE_REGISTERED_TOOL_LAUNCH_ROUTE

    name = f"mcp_task_fence_launch_failure_{case}"
    calls = []

    def handler(args, **kwargs):
        calls.append((dict(args), dict(kwargs)))
        assert current_task_fence_policy() is None
        return "legacy-handler-result"

    try:
        caplog.set_level(logging.WARNING, logger="tools.registry")
        with (
            _registered_tool(name, handler),
            bind_task_fence_policy(policy),
            bind_causal_envelope(generation),
            patch.object(
                registry_module,
                "_TASK_FENCE_REGISTERED_TOOL_LAUNCH_ROUTE",
                launch_route,
            ),
        ):
            result = registry.dispatch(name, {"value": 1})

        assert result == "legacy-handler-result"
        assert calls == [({"value": 1}, {})]
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_dispatch_permits"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
        assert any(
            expected_reason in record.message for record in caplog.records
        )
        assert fault_secret not in caplog.text
    finally:
        db.close()


def test_inline_execution_middleware_short_circuit_creates_no_attempt(tmp_path):
    from agent.tool_executor import _run_agent_tool_execution_middleware

    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    calls = []
    agent = SimpleNamespace(
        session_id="inline-session",
        _current_turn_id="turn-inline",
        _current_api_request_id="request-inline",
        _tool_guardrails=_allowing_tool_guardrails(),
    )
    try:
        with (
            bind_task_fence_policy(TaskFencePolicy(db)),
            patch(
                "hermes_cli.middleware.run_tool_execution_middleware",
                return_value="managed-inline-result",
            ),
            patch("agent.tool_executor._begin_tool_execution") as begin_execution,
        ):
            outcome = _run_agent_tool_execution_middleware(
                agent,
                function_name="todo",
                function_args={"value": 2},
                effective_task_id="sandbox-task",
                tool_call_id="call-inline",
                causal_envelope=generation.for_invocation(
                    "tfiv_inline_short_circuit"
                ),
                execute=lambda args: calls.append(dict(args)),
            )

        assert outcome.result == "managed-inline-result"
        assert outcome.args == {"value": 2}
        begin_execution.assert_not_called()
        assert calls == []
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_middleware_registry_detour_and_inline_next_are_siblings(tmp_path):
    from agent.tool_executor import _run_agent_tool_execution_middleware

    db, _acceptance, generation = _live_lane(tmp_path / "state.db")
    dispatcher = generation.for_invocation("tfiv_inline_detour_dispatcher")
    bypass_name = "mcp_task_fence_inline_detour"
    envelopes = []

    def capture(label):
        envelopes.append((label, current_causal_envelope()))
        return label

    def middleware_detour(_name, args, next_call, **_kwargs):
        assert registry.dispatch(bypass_name, {}) == "registry"
        return next_call(args)

    agent = SimpleNamespace(
        session_id="inline-session",
        _current_turn_id="turn-inline",
        _current_api_request_id="request-inline",
        _tool_guardrails=_allowing_tool_guardrails(),
    )
    try:
        with (
            _registered_tool(
                bypass_name,
                lambda _args, **_kwargs: capture("registry"),
            ),
            bind_task_fence_policy(TaskFencePolicy(db)),
            patch(
                "hermes_cli.middleware.run_tool_execution_middleware",
                side_effect=middleware_detour,
            ),
            patch("agent.tool_executor._begin_tool_execution") as begin_execution,
        ):
            outcome = _run_agent_tool_execution_middleware(
                agent,
                function_name="todo",
                function_args={},
                effective_task_id="sandbox-task",
                tool_call_id="call-inline",
                causal_envelope=dispatcher,
                execute=lambda _args: capture("inline"),
            )

        assert outcome.result == "inline"
        begin_execution.assert_called_once()
        assert {label for label, _envelope in envelopes} == {
            "registry",
            "inline",
        }
        invocation_ids = {envelope.invocation_id for _, envelope in envelopes}
        assert len(invocation_ids) == 2
        assert {envelope.parent_invocation_id for _, envelope in envelopes} == {
            dispatcher.invocation_id
        }
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 2
    finally:
        db.close()
