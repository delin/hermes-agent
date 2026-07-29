from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import socket
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from task_fence import (
    CONTROL_PROTOCOL_VERSION,
    CausalEnvelope,
    DecisionOutcome,
    IngressEnvelope,
    TASK_FENCE_ACTIONS,
    TASK_FENCE_STORE_SCHEMA_VERSION,
    TaskFenceIngressUnavailable,
    TaskFencePolicy,
    TaskFenceProtocolRejected,
    TaskFenceProvenanceRejected,
    TaskFenceProvenanceUnavailable,
    bind_causal_envelope,
    bind_task_fence_policy,
    current_causal_envelope,
    current_task_fence_policy,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ingress(
    action: str,
    source_event_id: str,
    *,
    task_id: str | None = None,
) -> IngressEnvelope:
    return IngressEnvelope(
        source="gateway:test:conversation-1",
        source_event_id=source_event_id,
        conversation_id="conversation-1",
        action=TASK_FENCE_ACTIONS[action],
        payload_hash=_hash(source_event_id),
        task_id=task_id,
    )


def _generation() -> CausalEnvelope:
    return CausalEnvelope(
        task_id="tft_test",
        authority_event_id="tfi_authority",
        run_id="tfr_run",
        generation_id="tfg_generation",
        snapshot_event_id="tfi_snapshot",
        input_manifest_hash=_hash("manifest"),
        store_schema_version=TASK_FENCE_STORE_SCHEMA_VERSION,
        control_protocol_version=CONTROL_PROTOCOL_VERSION,
        intent_epoch=1,
        control_revision=2,
        runtime_epoch=0,
        accepted_order=3,
    )


def _tool_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Task Fence provenance probe",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


def _tool_call(
    name: str,
    call_id: str,
    arguments: str = "{}",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _model_response(
    *,
    content: str | None,
    tool_calls: list[SimpleNamespace] | None,
    finish_reason: str,
) -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        reasoning_content=None,
        reasoning=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_causal_envelope_is_frozen_canonical_and_context_scoped():
    generation = _generation()

    with pytest.raises(FrozenInstanceError):
        generation.task_id = "other"
    assert CausalEnvelope.from_json(generation.to_json()) == generation
    assert current_causal_envelope() is None
    with bind_causal_envelope(generation):
        assert current_causal_envelope() is generation
        with bind_causal_envelope(None):
            assert current_causal_envelope() is None
        assert current_causal_envelope() is generation
    assert current_causal_envelope() is None


def test_task_fence_policy_context_is_nested_and_reset():
    outer = TaskFencePolicy(object())
    inner = TaskFencePolicy(object())

    assert current_task_fence_policy() is None
    with bind_task_fence_policy(outer):
        assert current_task_fence_policy() is outer
        with bind_task_fence_policy(inner):
            assert current_task_fence_policy() is inner
        assert current_task_fence_policy() is outer
    assert current_task_fence_policy() is None


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("task_id", "", "invalid_task_id"),
        ("intent_epoch", True, "invalid_intent_epoch"),
        ("control_revision", -1, "invalid_control_revision"),
        ("input_manifest_hash", "0" * 63, "invalid_input_manifest_hash"),
        (
            "store_schema_version",
            TASK_FENCE_STORE_SCHEMA_VERSION + 1,
            "invalid_store_schema_version",
        ),
        ("accepted_order", 0, "invalid_accepted_order"),
    ),
)
def test_causal_envelope_rejects_missing_or_malformed_identity(
    field,
    value,
    reason,
):
    payload = _generation().to_dict()
    payload[field] = value

    with pytest.raises(TaskFenceProtocolRejected, match=reason):
        CausalEnvelope.from_dict(payload)


def test_causal_envelope_rejects_unknown_noncanonical_and_oversized_wire_data():
    generation = _generation()
    unknown = generation.to_dict()
    unknown["prompt"] = "must-not-cross"

    with pytest.raises(
        TaskFenceProtocolRejected,
        match="invalid_causal_envelope_fields",
    ):
        CausalEnvelope.from_dict(unknown)
    with pytest.raises(
        TaskFenceProtocolRejected,
        match="noncanonical_causal_envelope_json",
    ):
        CausalEnvelope.from_json(json.dumps(generation.to_dict()))
    with pytest.raises(
        TaskFenceProtocolRejected,
        match="causal_envelope_json_too_large",
    ):
        CausalEnvelope.from_json("{" + (" " * 16_384) + "}")


def test_invocation_derivation_preserves_authority_and_rejects_mixed_parentage():
    generation = _generation()
    parent = generation.for_invocation("tfiv_parent")
    child = parent.for_invocation("tfiv_child")

    assert child.invocation_id == "tfiv_child"
    assert child.parent_invocation_id == "tfiv_parent"
    assert CausalEnvelope.invocation_from_dict(
        child.to_dict(),
        parent=parent,
    ) == child

    mixed = child.to_dict()
    mixed["task_id"] = "tft_other"
    with pytest.raises(TaskFenceProtocolRejected, match="mixed_causal_parentage"):
        CausalEnvelope.invocation_from_dict(mixed, parent=parent)
    with pytest.raises(TaskFenceProtocolRejected, match="invalid_invocation_id"):
        generation.for_invocation("")


def test_generation_reservation_is_durable_and_exact(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-1")
        )
        generation = db.reserve_task_fence_generation(acceptance)

        row = db._conn.execute(
            "SELECT task_id, run_id, input_manifest_hash, snapshot_event_id, "
            "state FROM task_fence_model_generations WHERE generation_id = ?",
            (generation.generation_id,),
        ).fetchone()
        assert tuple(row) == (
            generation.task_id,
            generation.run_id,
            generation.input_manifest_hash,
            generation.snapshot_event_id,
            "started",
        )
        assert db.finish_task_fence_generation(
            generation,
            state="committed",
        )
        committed = db._conn.execute(
            "SELECT state, closed_at FROM task_fence_model_generations "
            "WHERE generation_id = ?",
            (generation.generation_id,),
        ).fetchone()
        assert tuple(committed) == ("committed", None)
        assert db._conn.execute(
            "SELECT state FROM task_fence_task_inputs WHERE task_id = ?",
            (generation.task_id,),
        ).fetchone()[0] == "presented"
    finally:
        db.close()

    reopened = SessionDB(path)
    try:
        task = reopened.inspect_task_fence_task(generation.task_id)
        assert task.task is not None
        assert task.task.current_generation_id == generation.generation_id
        state = reopened._conn.execute(
            "SELECT state, closed_at FROM task_fence_model_generations "
            "WHERE generation_id = ?",
            (generation.generation_id,),
        ).fetchone()
        assert tuple(state) == ("committed", None)

        held = reopened.accept_task_fence_ingress(
            _ingress("comment_hold", "event-2", task_id=generation.task_id)
        )
        assert held.task_projection is not None
        assert held.task_projection.current_generation_id is None
        closed = reopened._conn.execute(
            "SELECT state, closed_at FROM task_fence_model_generations "
            "WHERE generation_id = ?",
            (generation.generation_id,),
        ).fetchone()
        assert closed["state"] == "committed"
        assert closed["closed_at"] is not None
    finally:
        reopened.close()


def test_committed_generation_is_atomically_superseded(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-1")
        )
        first = db.reserve_task_fence_generation(acceptance)
        assert db.finish_task_fence_generation(first, state="committed")

        second = db.reserve_task_fence_generation(acceptance)
        rows = db._conn.execute(
            "SELECT generation_id, state, closed_at "
            "FROM task_fence_model_generations ORDER BY opened_at"
        ).fetchall()
        assert rows[0]["generation_id"] == first.generation_id
        assert rows[0]["state"] == "committed"
        assert rows[0]["closed_at"] is not None
        assert tuple(rows[1]) == (second.generation_id, "started", None)
        task = db.inspect_task_fence_task(first.task_id).task
        assert task is not None
        assert task.current_generation_id == second.generation_id
    finally:
        db.close()


def test_generation_supersede_rolls_back_if_close_transaction_fails(
    tmp_path,
    monkeypatch,
):
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-1")
        )
        first = db.reserve_task_fence_generation(acceptance)
        assert db.finish_task_fence_generation(first, state="committed")
        close_generation = db._close_task_fence_generation_unlocked

        def close_then_fail(conn, **kwargs):
            assert close_generation(conn, **kwargs)
            raise sqlite3.IntegrityError("injected supersede failure")

        monkeypatch.setattr(
            db,
            "_close_task_fence_generation_unlocked",
            close_then_fail,
        )
        with pytest.raises(TaskFenceProvenanceUnavailable):
            db.reserve_task_fence_generation(acceptance)

        row = db._conn.execute(
            "SELECT state, closed_at FROM task_fence_model_generations "
            "WHERE generation_id = ?",
            (first.generation_id,),
        ).fetchone()
        assert tuple(row) == ("committed", None)
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_model_generations"
        ).fetchone()[0] == 1
        task = db.inspect_task_fence_task(first.task_id).task
        assert task is not None
        assert task.current_generation_id == first.generation_id
    finally:
        db.close()


def test_generation_reservation_rejects_unpointed_committed_open_without_mutation(
    tmp_path,
):
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-orphan")
        )
        generation = db.reserve_task_fence_generation(acceptance)
        assert db.finish_task_fence_generation(generation, state="committed")
        db._conn.execute(
            "UPDATE task_fence_tasks SET current_generation_id = NULL "
            "WHERE task_id = ?",
            (generation.task_id,),
        )
        db._conn.commit()
        before = tuple(
            db._conn.execute(
                "SELECT state, closed_at FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (generation.generation_id,),
            ).fetchone()
        )

        with pytest.raises(
            TaskFenceProvenanceUnavailable,
            match="incompatible_current_generation",
        ):
            db.reserve_task_fence_generation(acceptance)

        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_model_generations"
        ).fetchone()[0] == 1
        assert tuple(
            db._conn.execute(
                "SELECT state, closed_at FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (generation.generation_id,),
            ).fetchone()
        ) == before
    finally:
        db.close()


def test_no_generation_run_requeues_unpresented_inputs_on_hold(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        initial = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-no-generation")
        )
        held = db.accept_task_fence_ingress(
            _ingress("comment_hold", "event-no-generation-hold", task_id=initial.task_id)
        )

        assert held.pending_input_ids == (initial.event_id, held.event_id)
        assert db._conn.execute(
            "SELECT close_reason FROM task_fence_execution_runs "
            "WHERE run_id = ?",
            (initial.opened_run_id,),
        ).fetchone()[0] == "accepted_ingress_unpresented"
    finally:
        db.close()


@pytest.mark.parametrize("terminal_state", ("failed", "cancelled"))
def test_failed_generation_clears_pointer_and_keeps_inputs_retryable(
    tmp_path,
    terminal_state,
):
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-1")
        )
        generation = db.reserve_task_fence_generation(acceptance)
        assert db.finish_task_fence_generation(
            generation,
            state=terminal_state,
        )

        task = db.inspect_task_fence_task(generation.task_id).task
        assert task is not None
        assert task.current_generation_id is None
        row = db._conn.execute(
            "SELECT state, closed_at FROM task_fence_model_generations "
            "WHERE generation_id = ?",
            (generation.generation_id,),
        ).fetchone()
        assert row["state"] == terminal_state
        assert row["closed_at"] is not None
        assert db._conn.execute(
            "SELECT state FROM task_fence_task_inputs WHERE task_id = ?",
            (generation.task_id,),
        ).fetchone()[0] == "bound"

        held = db.accept_task_fence_ingress(
            _ingress("comment_hold", "event-2", task_id=generation.task_id)
        )
        assert held.task_projection is not None
        assert held.task_projection.status == "paused"
        assert held.pending_input_ids == (acceptance.event_id, held.event_id)
        assert tuple(
            row[0]
            for row in db._conn.execute(
                "SELECT state FROM task_fence_task_inputs "
                "WHERE task_id = ? ORDER BY event_id",
                (generation.task_id,),
            )
        ) == ("pending", "pending")

        resumed = db.accept_task_fence_ingress(
            _ingress("change_and_run", "event-3", task_id=generation.task_id)
        )
        assert resumed.task_projection is not None
        assert resumed.task_projection.status == "running"
        retry = db.reserve_task_fence_generation(resumed)
        expected_ids = (
            acceptance.event_id,
            held.event_id,
            resumed.event_id,
        )
        assert retry.input_manifest_hash == SessionDB._task_fence_input_manifest_hash(
            expected_ids
        )
        assert tuple(
            row[0]
            for row in db._conn.execute(
                "SELECT ti.event_id FROM task_fence_task_inputs AS ti "
                "JOIN task_fence_ingress AS i ON i.event_id = ti.event_id "
                "WHERE ti.bound_run_id = ? ORDER BY i.accepted_order",
                (retry.run_id,),
            )
        ) == expected_ids
    finally:
        db.close()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("task_id", "tft_other"),
        ("authority_event_id", "tfi_other"),
        ("run_id", "tfr_other"),
        ("generation_id", "tfg_other"),
        ("snapshot_event_id", "tfi_other_snapshot"),
        ("input_manifest_hash", _hash("other-manifest")),
        ("intent_epoch", 99),
        ("control_revision", 99),
        ("runtime_epoch", 99),
        ("accepted_order", 99),
    ),
)
def test_generation_finish_rejects_every_mutated_lineage_field(
    tmp_path,
    field,
    value,
):
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-1")
        )
        generation = db.reserve_task_fence_generation(acceptance)
        with pytest.raises(
            TaskFenceProvenanceUnavailable,
            match="generation_projection_mismatch",
        ):
            db.finish_task_fence_generation(
                replace(generation, **{field: value}),
                state="committed",
            )
        assert db._conn.execute(
            "SELECT state FROM task_fence_model_generations "
            "WHERE generation_id = ?",
            (generation.generation_id,),
        ).fetchone()[0] == "started"
    finally:
        db.close()


@pytest.mark.parametrize("corruption", ("missing", "extra", "misbound"))
def test_generation_commit_rejects_corrupt_input_membership_without_mutation(
    tmp_path,
    corruption,
):
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-corrupt-input")
        )
        generation = db.reserve_task_fence_generation(acceptance)
        if corruption == "missing":
            db._conn.execute(
                "UPDATE task_fence_task_inputs "
                "SET state = 'discarded', bound_run_id = NULL "
                "WHERE event_id = ?",
                (acceptance.event_id,),
            )
        elif corruption == "extra":
            note = db.accept_task_fence_ingress(
                _ingress(
                    "explicit_note",
                    "event-extra-input",
                    task_id=generation.task_id,
                )
            )
            db._conn.execute(
                "INSERT INTO task_fence_task_inputs ("
                "event_id, task_id, state, bound_run_id, state_changed_at"
                ") VALUES (?, ?, 'bound', ?, 1.0)",
                (note.event_id, generation.task_id, generation.run_id),
            )
        else:
            other = db.accept_task_fence_ingress(
                replace(
                    _ingress("initial_submit", "event-other-task"),
                    source="gateway:test:conversation-2",
                    conversation_id="conversation-2",
                )
            )
            assert other.opened_run_id is not None
            db._conn.execute(
                "UPDATE task_fence_task_inputs SET bound_run_id = ? "
                "WHERE event_id = ?",
                (other.opened_run_id, acceptance.event_id),
            )
        db._conn.commit()
        before = tuple(
            db._conn.execute(
                "SELECT state, closed_at FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (generation.generation_id,),
            ).fetchone()
        )
        inputs_before = tuple(
            tuple(row)
            for row in db._conn.execute(
                "SELECT event_id, task_id, state, bound_run_id "
                "FROM task_fence_task_inputs ORDER BY event_id"
            )
        )

        with pytest.raises(
            TaskFenceProvenanceUnavailable,
            match="incompatible_generation_input_manifest",
        ):
            db.finish_task_fence_generation(generation, state="committed")

        assert tuple(
            db._conn.execute(
                "SELECT state, closed_at FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (generation.generation_id,),
            ).fetchone()
        ) == before
        task = db.inspect_task_fence_task(generation.task_id).task
        assert task is not None
        assert task.current_generation_id == generation.generation_id
        assert tuple(
            tuple(row)
            for row in db._conn.execute(
                "SELECT event_id, task_id, state, bound_run_id "
                "FROM task_fence_task_inputs ORDER BY event_id"
            )
        ) == inputs_before
    finally:
        db.close()


def test_generation_commit_input_fault_rolls_back_generation_and_inputs(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-input-fault")
        )
        generation = db.reserve_task_fence_generation(acceptance)
        db._conn.execute(
            "CREATE TEMP TRIGGER fail_generation_input_present "
            "BEFORE UPDATE OF state ON task_fence_task_inputs "
            "WHEN NEW.state = 'presented' BEGIN "
            "SELECT RAISE(ABORT, 'injected input presentation fault'); END"
        )
        db._conn.commit()

        with pytest.raises(
            TaskFenceProvenanceUnavailable,
            match="generation_finish_database_error",
        ):
            db.finish_task_fence_generation(generation, state="committed")

        assert tuple(
            db._conn.execute(
                "SELECT state, closed_at FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (generation.generation_id,),
            ).fetchone()
        ) == ("started", None)
        assert db._conn.execute(
            "SELECT state FROM task_fence_task_inputs WHERE event_id = ?",
            (acceptance.event_id,),
        ).fetchone()[0] == "bound"
        task = db.inspect_task_fence_task(generation.task_id).task
        assert task is not None
        assert task.current_generation_id == generation.generation_id
    finally:
        db.close()


@pytest.mark.parametrize("corruption", ("unpointed_open", "pointed_closed"))
def test_committed_generation_pointer_invariant_fails_closed(
    tmp_path,
    corruption,
):
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-1")
        )
        generation = db.reserve_task_fence_generation(acceptance)
        assert db.finish_task_fence_generation(generation, state="committed")
        if corruption == "unpointed_open":
            db._conn.execute(
                "UPDATE task_fence_tasks SET current_generation_id = NULL "
                "WHERE task_id = ?",
                (generation.task_id,),
            )
        else:
            db._conn.execute(
                "UPDATE task_fence_model_generations SET closed_at = 1 "
                "WHERE generation_id = ?",
                (generation.generation_id,),
            )
        db._conn.commit()

        with pytest.raises(
            TaskFenceIngressUnavailable,
            match="incompatible_current_generation",
        ):
            db.accept_task_fence_ingress(
                _ingress("comment_hold", "event-2", task_id=generation.task_id)
            )
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_ingress"
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_stale_acceptance_cannot_mint_current_generation(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        first = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-1")
        )
        generation = db.reserve_task_fence_generation(first)
        held = db.accept_task_fence_ingress(
            _ingress(
                "comment_hold",
                "event-2",
                task_id=first.task_id,
            )
        )

        assert held.task_projection is not None
        assert held.task_projection.active_execution_run_id is None
        with pytest.raises(TaskFenceProvenanceRejected):
            db.reserve_task_fence_generation(first)
        assert generation == replace(generation)
        assert db.finish_task_fence_generation(
            generation,
            state="committed",
        ) is False
    finally:
        db.close()


@pytest.fixture
def registered_probe_tools():
    from tools.registry import registry

    names = ("mcp_task_fence_probe_a", "mcp_task_fence_probe_b")
    toolset = "mcp-task-fence-probe"
    observed = []
    probe = {"inspect": None}

    def handler(args, **kwargs):
        envelope = current_causal_envelope()
        inspect = probe["inspect"]
        observed.append(
            (
                kwargs.get("task_fence_envelope"),
                envelope,
                threading.current_thread().ident,
                inspect(envelope) if callable(inspect) else None,
            )
        )
        return json.dumps({"ok": True})

    for name in names:
        registry.register(
            name=name,
            toolset=toolset,
            schema=_tool_schema(name),
            handler=handler,
        )
    try:
        yield names, toolset, observed, probe
    finally:
        for name in names:
            registry.deregister(name)


@pytest.fixture
def provenance_agent(registered_probe_tools):
    from run_agent import AIAgent

    names, _, _, _ = registered_probe_tools
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[_tool_schema(name) for name in names],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        yield agent


def _prepare_real_conversation(
    agent,
    responses,
    *,
    provider_observed=None,
    provider_inspect=None,
) -> None:
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = False
    response_iter = iter(responses)

    def _create(**_kwargs):
        envelope = current_causal_envelope()
        if provider_observed is not None:
            provider_observed.append(envelope)
        if callable(provider_inspect):
            provider_inspect(envelope)
        return next(response_iter)

    agent.client.chat.completions.create.side_effect = _create


def test_real_conversation_records_generation_and_binds_emitted_tool(
    provenance_agent,
    registered_probe_tools,
    tmp_path,
):
    names, _, observed, probe = registered_probe_tools
    provider_observed = []
    provider_states = []
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-real")
        )
        provenance_agent._session_db = db

        def inspect_generation(envelope):
            assert envelope is not None
            with db._lock:
                row = db._conn.execute(
                    "SELECT g.state, g.closed_at, t.current_generation_id "
                    "FROM task_fence_model_generations AS g "
                    "JOIN task_fence_tasks AS t ON t.task_id = g.task_id "
                    "WHERE g.generation_id = ?",
                    (envelope.generation_id,),
                ).fetchone()
                attempt = db._conn.execute(
                    "SELECT a.state FROM task_fence_attempts AS a "
                    "JOIN task_fence_dispatch_permits AS p "
                    "ON p.permit_id = a.permit_id "
                    "WHERE p.invocation_envelope_id = ?",
                    (envelope.invocation_id,),
                ).fetchone()
            assert row is not None
            return (*tuple(row), None if attempt is None else attempt["state"])

        probe["inspect"] = inspect_generation
        _prepare_real_conversation(
            provenance_agent,
            [
                _model_response(
                    content=None,
                    tool_calls=[_tool_call(names[0], "call-real")],
                    finish_reason="tool_calls",
                ),
                _model_response(
                    content="done",
                    tool_calls=None,
                    finish_reason="stop",
                ),
            ],
            provider_observed=provider_observed,
            provider_inspect=lambda envelope: provider_states.append(
                inspect_generation(envelope)
            ),
        )

        with (
            patch.object(provenance_agent, "_persist_session"),
            patch.object(provenance_agent, "_save_trajectory"),
            patch.object(provenance_agent, "_cleanup_task_resources"),
        ):
            result = provenance_agent.run_conversation(
                "do the recorded task",
                task_fence_acceptance=acceptance,
            )

        assert result["completed"] is True
        assert result["final_response"] == "done"
        assert len(observed) == 1
        explicit, invocation, _, tool_generation_state = observed[0]
        assert explicit is None
        assert invocation is not None
        assert invocation.invocation_id is not None
        assert invocation.task_id == acceptance.task_id
        assert invocation.snapshot_event_id == acceptance.event_id
        assert invocation.authority_event_id == (
            acceptance.task_projection.active_authority_event_id
        )
        assert tool_generation_state == (
            "committed",
            None,
            invocation.generation_id,
            "STARTED",
        )

        rows = db._conn.execute(
            "SELECT generation_id, task_id, run_id, snapshot_event_id, "
            "state, closed_at "
            "FROM task_fence_model_generations ORDER BY opened_at, generation_id"
        ).fetchall()
        assert len(rows) == 2
        assert {row["state"] for row in rows} == {"committed"}
        assert {row["task_id"] for row in rows} == {acceptance.task_id}
        assert {row["run_id"] for row in rows} == {invocation.run_id}
        assert {row["snapshot_event_id"] for row in rows} == {
            acceptance.event_id
        }
        assert invocation.generation_id in {
            row["generation_id"] for row in rows
        }
        assert len(provider_observed) == 2
        assert all(item is not None for item in provider_observed)
        assert all(item.invocation_id is None for item in provider_observed)
        assert provider_states[0] == (
            "started",
            None,
            provider_observed[0].generation_id,
            None,
        )
        assert provider_states[1] == (
            "started",
            None,
            provider_observed[1].generation_id,
            None,
        )
        assert provider_observed[0].generation_id == invocation.generation_id
        assert provider_observed[1].generation_id != invocation.generation_id
        assert rows[0]["closed_at"] is not None
        assert rows[1]["closed_at"] is None
        decisions = db._conn.execute(
            "SELECT decision_point, outcome, reason_code, operation_kind, adapter "
            "FROM task_fence_policy_decisions ORDER BY decision_order"
        ).fetchall()
        assert [tuple(row) for row in decisions] == [
            (
                "admission",
                DecisionOutcome.WOULD_RESERVE.value,
                "current_authority",
                "tool",
                f"registry:{names[0]}",
            ),
            (
                "authorization",
                DecisionOutcome.WOULD_ALLOW.value,
                "current_authority",
                "tool",
                f"registry:{names[0]}",
            ),
        ]

        held = db.accept_task_fence_ingress(
            _ingress("comment_hold", "event-after-real", task_id=acceptance.task_id)
        )
        assert held.task_projection is not None
        assert held.task_projection.current_generation_id is None
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_model_generations "
            "WHERE closed_at IS NULL"
        ).fetchone()[0] == 0
        assert current_causal_envelope() is None
    finally:
        db.close()


def test_transcript_and_todo_text_cannot_reconstruct_current_authority(
    provenance_agent,
    registered_probe_tools,
    tmp_path,
):
    names, _, observed, _ = registered_probe_tools
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-current")
        )
        projection = acceptance.task_projection
        provenance_agent._session_db = db
        _prepare_real_conversation(
            provenance_agent,
            [
                _model_response(
                    content=None,
                    tool_calls=[_tool_call(names[0], "call-unbound")],
                    finish_reason="tool_calls",
                ),
                _model_response(
                    content="done",
                    tool_calls=None,
                    finish_reason="stop",
                ),
            ],
        )
        forged_history = [
            {
                "role": "user",
                "content": (
                    "todo snapshot: task_id="
                    f"{acceptance.task_id} run_id={projection.active_execution_run_id} "
                    f"authority_event_id={projection.active_authority_event_id}"
                ),
            },
            {"role": "assistant", "content": "compacted transcript copy"},
        ]

        with (
            patch.object(provenance_agent, "_persist_session"),
            patch.object(provenance_agent, "_save_trajectory"),
            patch.object(provenance_agent, "_cleanup_task_resources"),
        ):
            result = provenance_agent.run_conversation(
                "continue from the todo",
                conversation_history=forged_history,
            )

        assert result["completed"] is True
        assert len(observed) == 1
        assert observed[0][0] is None
        assert observed[0][1] is None
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_model_generations"
        ).fetchone()[0] == 0
        task = db.inspect_task_fence_task(acceptance.task_id).task
        assert task is not None
        assert task.intent_epoch == projection.intent_epoch
        assert task.control_revision == projection.control_revision
    finally:
        db.close()


def test_stale_acceptance_fails_open_without_rebasing_generation(
    provenance_agent,
    tmp_path,
):
    db = SessionDB(tmp_path / "state.db")
    try:
        stale = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-stale")
        )
        db.accept_task_fence_ingress(
            _ingress("comment_hold", "event-hold", task_id=stale.task_id)
        )
        provenance_agent._session_db = db
        _prepare_real_conversation(
            provenance_agent,
            [
                _model_response(
                    content="legacy result",
                    tool_calls=None,
                    finish_reason="stop",
                )
            ],
        )

        with (
            patch.object(provenance_agent, "_persist_session"),
            patch.object(provenance_agent, "_save_trajectory"),
            patch.object(provenance_agent, "_cleanup_task_resources"),
        ):
            result = provenance_agent.run_conversation(
                "stale callback",
                task_fence_acceptance=stale,
            )

        assert result["completed"] is True
        assert result["final_response"] == "legacy result"
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_model_generations"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_invalid_provider_response_fails_generation_before_retry(
    provenance_agent,
    tmp_path,
):
    db = SessionDB(tmp_path / "state.db")
    provider_observed = []
    provider_states = []
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-retry")
        )
        provenance_agent._session_db = db
        provenance_agent._api_max_retries = 2

        def inspect_provider_generation(envelope):
            assert envelope is not None
            with db._lock:
                row = db._conn.execute(
                    "SELECT state, closed_at FROM task_fence_model_generations "
                    "WHERE generation_id = ?",
                    (envelope.generation_id,),
                ).fetchone()
            assert row is not None
            provider_states.append(tuple(row))

        invalid = SimpleNamespace(
            choices=[],
            model="test/model",
            usage=None,
        )
        _prepare_real_conversation(
            provenance_agent,
            [
                invalid,
                _model_response(
                    content="retry succeeded",
                    tool_calls=None,
                    finish_reason="stop",
                ),
            ],
            provider_observed=provider_observed,
            provider_inspect=inspect_provider_generation,
        )

        with (
            patch("agent.conversation_loop.jittered_backoff", return_value=0),
            patch.object(provenance_agent, "_persist_session"),
            patch.object(provenance_agent, "_save_trajectory"),
            patch.object(provenance_agent, "_cleanup_task_resources"),
        ):
            result = provenance_agent.run_conversation(
                "retry the recorded task",
                task_fence_acceptance=acceptance,
            )

        assert result["completed"] is True
        assert result["final_response"] == "retry succeeded"
        assert provider_states == [("started", None), ("started", None)]
        assert len(provider_observed) == 2
        assert provider_observed[0].generation_id != (
            provider_observed[1].generation_id
        )
        rows = db._conn.execute(
            "SELECT generation_id, state, closed_at "
            "FROM task_fence_model_generations ORDER BY opened_at"
        ).fetchall()
        assert rows[0]["generation_id"] == provider_observed[0].generation_id
        assert rows[0]["state"] == "failed"
        assert rows[0]["closed_at"] is not None
        assert rows[1]["generation_id"] == provider_observed[1].generation_id
        assert rows[1]["state"] == "committed"
        assert rows[1]["closed_at"] is None
        task = db.inspect_task_fence_task(acceptance.task_id).task
        assert task is not None
        assert task.current_generation_id == rows[1]["generation_id"]
    finally:
        db.close()


def test_provider_exception_records_one_failure_without_missing_provenance_warning(
    provenance_agent,
    tmp_path,
    caplog,
):
    db = SessionDB(tmp_path / "state.db")
    provider_observed = []
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-provider-exception")
        )
        provenance_agent._session_db = db
        provenance_agent._api_max_retries = 2
        _prepare_real_conversation(provenance_agent, [])
        outcomes = iter(
            [
                RuntimeError("provider failed"),
                _model_response(
                    content="retry succeeded",
                    tool_calls=None,
                    finish_reason="stop",
                ),
            ]
        )

        def create(**_kwargs):
            provider_observed.append(current_causal_envelope())
            outcome = next(outcomes)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        provenance_agent.client.chat.completions.create.side_effect = create

        with (
            patch("agent.conversation_loop.jittered_backoff", return_value=0),
            patch.object(provenance_agent, "_persist_session"),
            patch.object(provenance_agent, "_save_trajectory"),
            patch.object(provenance_agent, "_cleanup_task_resources"),
        ):
            result = provenance_agent.run_conversation(
                "retry after provider failure",
                task_fence_acceptance=acceptance,
            )

        assert result["completed"] is True
        assert result["final_response"] == "retry succeeded"
        assert len(provider_observed) == 2
        assert all(envelope is not None for envelope in provider_observed)
        rows = db._conn.execute(
            "SELECT generation_id, state, closed_at "
            "FROM task_fence_model_generations ORDER BY opened_at"
        ).fetchall()
        assert [row["generation_id"] for row in rows] == [
            provider_observed[0].generation_id,
            provider_observed[1].generation_id,
        ]
        assert sum(row["state"] == "failed" for row in rows) == 1
        assert rows[0]["closed_at"] is not None
        assert rows[1]["state"] == "committed"
        assert rows[1]["closed_at"] is None
        assert not any(
            "Task Fence shadow provenance missing" in record.getMessage()
            for record in caplog.records
        )
    finally:
        db.close()


def test_post_provider_middleware_failure_closes_generation_before_retry(
    provenance_agent,
    tmp_path,
):
    db = SessionDB(tmp_path / "state.db")
    provider_observed = []
    middleware_calls = 0
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-middleware-retry")
        )
        provenance_agent._session_db = db
        provenance_agent._api_max_retries = 2
        _prepare_real_conversation(
            provenance_agent,
            [
                _model_response(
                    content="discarded",
                    tool_calls=None,
                    finish_reason="stop",
                ),
                _model_response(
                    content="retry succeeded",
                    tool_calls=None,
                    finish_reason="stop",
                ),
            ],
            provider_observed=provider_observed,
        )

        def fail_first_response(request, next_call, **_kwargs):
            nonlocal middleware_calls
            middleware_calls += 1
            response = next_call(request)
            if middleware_calls == 1:
                raise RuntimeError("post-provider middleware failed")
            return response

        with (
            patch(
                "hermes_cli.middleware.run_llm_execution_middleware",
                side_effect=fail_first_response,
            ),
            patch("agent.conversation_loop.jittered_backoff", return_value=0),
            patch.object(provenance_agent, "_persist_session"),
            patch.object(provenance_agent, "_save_trajectory"),
            patch.object(provenance_agent, "_cleanup_task_resources"),
        ):
            result = provenance_agent.run_conversation(
                "retry after middleware failure",
                task_fence_acceptance=acceptance,
            )

        assert result["completed"] is True
        assert result["final_response"] == "retry succeeded"
        assert len(provider_observed) == 2
        rows = db._conn.execute(
            "SELECT generation_id, state, closed_at "
            "FROM task_fence_model_generations ORDER BY opened_at"
        ).fetchall()
        assert [row["generation_id"] for row in rows] == [
            provider_observed[0].generation_id,
            provider_observed[1].generation_id,
        ]
        assert rows[0]["state"] == "failed"
        assert rows[0]["closed_at"] is not None
        assert rows[1]["state"] == "committed"
        assert rows[1]["closed_at"] is None
    finally:
        db.close()


def test_redirect_crossing_provider_response_cancels_discarded_generation(
    provenance_agent,
    tmp_path,
):
    db = SessionDB(tmp_path / "state.db")
    provider_observed = []
    middleware_calls = 0
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-redirect")
        )
        provenance_agent._session_db = db
        _prepare_real_conversation(
            provenance_agent,
            [
                _model_response(
                    content="discarded",
                    tool_calls=None,
                    finish_reason="stop",
                ),
                _model_response(
                    content="redirected result",
                    tool_calls=None,
                    finish_reason="stop",
                ),
            ],
            provider_observed=provider_observed,
        )

        def redirect_first_response(request, next_call, **_kwargs):
            nonlocal middleware_calls
            middleware_calls += 1
            response = next_call(request)
            if middleware_calls == 1:
                assert provenance_agent.redirect("use the corrected path") is True
            return response

        with (
            patch(
                "hermes_cli.middleware.run_llm_execution_middleware",
                side_effect=redirect_first_response,
            ),
            patch.object(provenance_agent, "_persist_session"),
            patch.object(provenance_agent, "_save_trajectory"),
            patch.object(provenance_agent, "_cleanup_task_resources"),
        ):
            result = provenance_agent.run_conversation(
                "start the recorded task",
                task_fence_acceptance=acceptance,
            )

        assert result["completed"] is True
        assert result["final_response"] == "redirected result"
        assert len(provider_observed) == 2
        rows = db._conn.execute(
            "SELECT generation_id, state, closed_at "
            "FROM task_fence_model_generations ORDER BY opened_at"
        ).fetchall()
        assert [row["generation_id"] for row in rows] == [
            provider_observed[0].generation_id,
            provider_observed[1].generation_id,
        ]
        assert rows[0]["state"] == "cancelled"
        assert rows[0]["closed_at"] is not None
        assert rows[1]["state"] == "committed"
        assert rows[1]["closed_at"] is None
    finally:
        db.close()


def test_sequential_and_concurrent_tools_copy_generation_with_unique_invocations(
    provenance_agent,
    registered_probe_tools,
    tmp_path,
):
    names, _, observed, _ = registered_probe_tools
    db = SessionDB(tmp_path / "state.db")
    acceptance = db.accept_task_fence_ingress(
        _ingress("initial_submit", "event-tool-concurrency")
    )
    generation = db.reserve_task_fence_generation(acceptance)
    assert db.finish_task_fence_generation(generation, state="committed")
    policy = TaskFencePolicy(db)

    sequential = SimpleNamespace(
        content="",
        tool_calls=[_tool_call(names[0], "call-sequential")],
    )
    with bind_causal_envelope(generation), bind_task_fence_policy(policy):
        provenance_agent._execute_tool_calls_sequential(
            sequential,
            [],
            "sandbox-task",
        )

    concurrent = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call(names[0], "call-concurrent-a"),
            _tool_call(names[1], "call-concurrent-b"),
        ],
    )
    with bind_causal_envelope(generation), bind_task_fence_policy(policy):
        provenance_agent._execute_tool_calls_concurrent(
            concurrent,
            [],
            "sandbox-task",
        )

    assert len(observed) == 3
    assert all(item[0] is None for item in observed)
    envelopes = [item[1] for item in observed]
    assert all(envelope is not None for envelope in envelopes)
    assert {envelope.generation_id for envelope in envelopes} == {
        generation.generation_id
    }
    assert len({envelope.invocation_id for envelope in envelopes}) == 3
    attempts = db._conn.execute(
        "SELECT p.invocation_envelope_id, a.state "
        "FROM task_fence_attempts AS a "
        "JOIN task_fence_dispatch_permits AS p ON p.permit_id = a.permit_id"
    ).fetchall()
    assert len(attempts) == 3
    assert {row["invocation_envelope_id"] for row in attempts} == {
        envelope.invocation_id for envelope in envelopes
    }
    assert {row["state"] for row in attempts} == {"STARTED"}
    assert db._conn.execute(
        "SELECT COUNT(*) FROM task_fence_policy_decisions"
    ).fetchone()[0] == 6
    db.close()


def test_regular_registry_handler_keeps_legacy_signature_and_scoped_context():
    import model_tools
    from tools.registry import registry

    name = "mcp_task_fence_strict_handler"
    toolset = "mcp-task-fence-strict"
    observed = []

    def strict_handler(args, task_id=None, session_id=None, user_task=None):
        observed.append(current_causal_envelope())
        return json.dumps({"ok": True})

    registry.register(
        name=name,
        toolset=toolset,
        schema=_tool_schema(name),
        handler=strict_handler,
    )
    try:
        generation = _generation()
        assert json.loads(
            model_tools.handle_function_call(
                name,
                {},
                causal_envelope=generation,
            )
        ) == {"ok": True}
        with bind_causal_envelope(generation):
            assert json.loads(
                model_tools.handle_function_call(
                    name,
                    {},
                    causal_envelope=None,
                )
            ) == {"ok": True}

        assert observed[0] is not None
        assert observed[0].generation_id == generation.generation_id
        assert observed[0].invocation_id is not None
        assert observed[1] is None
    finally:
        registry.deregister(name)


def test_sequential_inline_tool_gets_its_own_invocation(
    provenance_agent,
    monkeypatch,
):
    observed = []

    def fake_todo_tool(*, todos, merge, store):
        observed.append(current_causal_envelope())
        return json.dumps({"ok": True})

    monkeypatch.setattr("tools.todo_tool.todo_tool", fake_todo_tool)
    provenance_agent.valid_tool_names = set(
        provenance_agent.valid_tool_names
    ) | {"todo"}
    response = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call(
                "todo",
                "call-inline",
                json.dumps({"todos": []}),
            )
        ],
    )
    generation = _generation()

    with bind_causal_envelope(generation):
        provenance_agent._execute_tool_calls_sequential(
            response,
            [],
            "sandbox-task",
        )

    assert len(observed) == 1
    assert observed[0] is not None
    assert observed[0].generation_id == generation.generation_id
    assert observed[0].invocation_id is not None


def test_deferred_tool_gets_distinct_child_invocation(
    registered_probe_tools,
):
    import model_tools

    names, toolset, observed, _ = registered_probe_tools
    bridge = _generation().for_invocation("tfiv_bridge")
    result = json.loads(
        model_tools.handle_function_call(
            function_name="tool_call",
            function_args={"name": names[0], "arguments": {}},
            enabled_toolsets=[toolset],
            causal_envelope=bridge,
        )
    )

    assert result == {"ok": True}
    explicit, child, _, _ = observed[-1]
    assert explicit is None
    assert child.generation_id == bridge.generation_id
    assert child.invocation_id != bridge.invocation_id
    assert child.parent_invocation_id == bridge.invocation_id


def test_stale_worker_keeps_captured_authority_instead_of_rebasing(
    registered_probe_tools,
    tmp_path,
):
    import model_tools

    names, _, observed, _ = registered_probe_tools
    db = SessionDB(tmp_path / "state.db")
    try:
        acceptance = db.accept_task_fence_ingress(
            _ingress("initial_submit", "event-worker")
        )
        generation = db.reserve_task_fence_generation(acceptance)
        assert db.finish_task_fence_generation(generation, state="committed")
        captured = generation.for_invocation("tfiv_captured_worker")
        db.accept_task_fence_ingress(
            _ingress(
                "comment_hold",
                "event-worker-hold",
                task_id=generation.task_id,
            )
        )

        with bind_causal_envelope(captured):
            assert json.loads(
                model_tools.handle_function_call(names[0], {})
            ) == {"ok": True}

        assert observed[-1][1] == captured
        task = db.inspect_task_fence_task(generation.task_id).task
        assert task is not None
        assert task.current_generation_id is None
        assert captured.control_revision == generation.control_revision
    finally:
        db.close()


def test_execute_code_rpc_parent_mints_child_without_trusting_request(monkeypatch):
    from tools.code_execution_tool import _rpc_server_loop

    parent = _generation().for_invocation("tfiv_execute_code")
    captured = []

    def fake_handle(name, args, **kwargs):
        captured.append((name, args, kwargs, current_causal_envelope()))
        return json.dumps({"ok": True})

    monkeypatch.setattr("model_tools.handle_function_call", fake_handle)
    server_side, client_side = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_STREAM,
    )

    class Listener:
        def settimeout(self, _timeout):
            return None

        def accept(self):
            return server_side, ("peer", 0)

    stop = threading.Event()
    worker = threading.Thread(
        target=_rpc_server_loop,
        args=(
            Listener(),
            "sandbox-task",
            [],
            [0],
            1,
            frozenset({"terminal"}),
            stop,
            "secret",
            parent,
        ),
        daemon=True,
    )
    worker.start()
    try:
        client_side.sendall(
            (
                json.dumps(
                    {
                        "tool": "terminal",
                        "args": {"command": "pwd"},
                        "token": "secret",
                        "causal_envelope": _generation().to_dict(),
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        client_side.settimeout(5)
        assert json.loads(client_side.recv(65_536).decode("utf-8")) == {
            "ok": True
        }
    finally:
        stop.set()
        client_side.close()
        worker.join(timeout=5)
        server_side.close()

    assert "causal_envelope" not in captured[0][2]
    child = captured[0][3]
    assert child is not None
    assert child.generation_id == parent.generation_id
    assert child.parent_invocation_id == parent.invocation_id
    assert child.invocation_id != parent.invocation_id


def test_execute_code_remote_rpc_uses_trusted_parent_and_legacy_handler_shape(
    monkeypatch,
):
    from tools.code_execution_tool import _rpc_poll_loop

    parent = _generation().for_invocation("tfiv_execute_code_remote")
    forged = replace(
        _generation(),
        task_id="tft_forged_remote",
        generation_id="tfg_forged_remote",
    ).for_invocation("tfiv_forged_remote")
    captured = []
    stop = threading.Event()

    def strict_handle(name, args, task_id=None):
        captured.append((name, args, task_id, current_causal_envelope()))
        return json.dumps({"ok": True})

    class RemoteEnv:
        def __init__(self):
            self.listed = False

        def execute(self, command, **_kwargs):
            if command.startswith("ls -1 "):
                if self.listed:
                    return {"output": ""}
                self.listed = True
                return {"output": "/rpc/req_000001\n"}
            if command.startswith("cat "):
                return {
                    "output": json.dumps(
                        {
                            "tool": "terminal",
                            "args": {"command": "pwd"},
                            "seq": 1,
                            "token": "secret",
                            "causal_envelope": forged.to_dict(),
                        }
                    )
                }
            if command.startswith("rm -f "):
                stop.set()
            return {"output": ""}

    monkeypatch.setattr("model_tools.handle_function_call", strict_handle)
    counter = [0]
    _rpc_poll_loop(
        RemoteEnv(),
        "/rpc",
        "sandbox-task",
        [],
        counter,
        1,
        frozenset({"terminal"}),
        stop,
        "secret",
        parent,
    )

    assert counter == [1]
    assert len(captured) == 1
    name, args, task_id, child = captured[0]
    assert (name, args, task_id) == (
        "terminal",
        {"command": "pwd"},
        "sandbox-task",
    )
    assert child is not None
    assert child.task_id == parent.task_id
    assert child.generation_id == parent.generation_id
    assert child.parent_invocation_id == parent.invocation_id
    assert child.invocation_id not in {
        parent.invocation_id,
        forged.invocation_id,
    }


def test_execute_code_real_child_process_preserves_trusted_parent():
    from tools.code_execution_tool import execute_code

    parent = _generation().for_invocation("tfiv_execute_code_real")
    observed = []

    def fake_handle(name, args, **kwargs):
        observed.append(current_causal_envelope())
        return json.dumps({"output": "child-ok", "exit_code": 0})

    code = (
        "from hermes_tools import terminal\n"
        "result = terminal('echo ignored')\n"
        "print(result.get('output', ''))\n"
    )
    with patch("model_tools.handle_function_call", side_effect=fake_handle):
        result = json.loads(
            execute_code(
                code=code,
                task_id="task-fence-real-process",
                enabled_tools=["terminal"],
                task_fence_envelope=parent,
            )
        )

    assert result["status"] == "success"
    assert "child-ok" in result["output"]
    assert len(observed) == 1
    child = observed[0]
    assert child is not None
    assert child.generation_id == parent.generation_id
    assert child.parent_invocation_id == parent.invocation_id
    assert child.invocation_id != parent.invocation_id
