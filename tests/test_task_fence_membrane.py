from contextlib import contextmanager
import hashlib
import json
import logging
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from task_fence import (
    DecisionOutcome,
    DecisionReason,
    IngressEnvelope,
    TASK_FENCE_ACTIONS,
    TaskFencePolicy,
    bind_causal_envelope,
    bind_task_fence_policy,
    current_causal_envelope,
)
from tools.registry import _task_fence_tool_fingerprint, registry


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
            bind_task_fence_policy(TaskFencePolicy(db)),
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
