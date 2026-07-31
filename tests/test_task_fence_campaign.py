from dataclasses import FrozenInstanceError, asdict, dataclass
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import _create_openai_chat_completion
from agent.task_fence_provider import (
    _finish_task_fence_openai_chat_completion,
)
from hermes_state import SessionDB
from task_fence import (
    CausalEnvelope,
    DecisionOutcome,
    DecisionReason,
    DispatchDecision,
    IngressEnvelope,
    OperationDescriptor,
    OperationKind,
    TASK_FENCE_ACTIONS,
    TASK_FENCE_SELECTED_COHORT_CAPABILITIES,
    TaskFenceCapabilityState,
    TaskFencePolicy,
    bind_causal_envelope,
    current_causal_envelope,
    current_task_fence_policy,
)


_MAX_CAMPAIGN_DECISION_RECORDS = 64
_ANTHROPIC_CREATE_ROUTE = "provider:anthropic.messages.create"
_ANTHROPIC_STREAM_ROUTE = "provider:anthropic.messages.stream"
_BEDROCK_ANTHROPIC_ROUTE = "provider:bedrock.anthropic_messages"
_CODEX_APP_SERVER_ROUTE = "provider:codex.app_server"
_CODEX_ROUTE = "provider:openai.responses.create"
_COPILOT_ACP_ROUTE = "provider:copilot.acp"
_GEMINI_ROUTE = "provider:gemini.generateContent"
_GENERIC_AUXILIARY_ROUTE = "runtime:generic-auxiliary"
_MOA_ONE_SHOT_ROUTE = "runtime:moa-one-shot"
_OPENAI_ROUTE = "provider:openai.chat.completions.create"


@dataclass(frozen=True)
class _CampaignDecisionRecord:
    route_id: str
    decision_point: str
    outcome: str
    reason_code: str
    invocation_id: str
    task_id: str | None
    generation_id: str | None
    decision_id: str | None


class _CampaignRecordOverflow(RuntimeError):
    pass


class _CampaignPolicyProbe:
    """Test-only bounded projection of real policy facade returns."""

    def __init__(
        self,
        policy: TaskFencePolicy,
        *,
        route_id: str,
    ):
        if not isinstance(policy, TaskFencePolicy):
            raise TypeError("invalid_campaign_policy")
        declaration = next(
            (
                candidate
                for candidate in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
                if candidate.capability_id == route_id
            ),
            None,
        )
        if (
            declaration is None
            or declaration.state is not TaskFenceCapabilityState.SUPPORTED
        ):
            raise ValueError("unsupported_campaign_route")
        self._policy = policy
        self._route_id = route_id
        self._records: list[_CampaignDecisionRecord] = []
        self._failure_reason: str | None = None

    @property
    def records(self) -> tuple[_CampaignDecisionRecord, ...]:
        return tuple(self._records)

    def assert_complete(self) -> None:
        if self._failure_reason == "campaign_record_limit":
            raise _CampaignRecordOverflow(self._failure_reason)
        if self._failure_reason is not None:
            raise ValueError(self._failure_reason)

    def _append(
        self,
        *,
        decision_point: str,
        envelope: CausalEnvelope | None,
        operation: OperationDescriptor,
        decision: DispatchDecision,
    ) -> None:
        if self._failure_reason is not None:
            return
        if operation.adapter != self._route_id:
            self._failure_reason = "campaign_route_adapter_mismatch"
            return
        if len(self._records) >= _MAX_CAMPAIGN_DECISION_RECORDS:
            self._failure_reason = "campaign_record_limit"
            return
        self._records.append(
            _CampaignDecisionRecord(
                route_id=self._route_id,
                decision_point=decision_point,
                outcome=decision.outcome.value,
                reason_code=decision.reason.value,
                invocation_id=operation.invocation_id,
                task_id=None if envelope is None else envelope.task_id,
                generation_id=(
                    None if envelope is None else envelope.generation_id
                ),
                decision_id=decision.decision_id,
            )
        )

    def admit_operation(
        self,
        envelope: CausalEnvelope | None,
        operation: OperationDescriptor,
    ) -> DispatchDecision:
        decision = self._policy.admit_operation(envelope, operation)
        self._append(
            decision_point="admission",
            envelope=envelope,
            operation=operation,
            decision=decision,
        )
        return decision

    def authorize_and_start(
        self,
        envelope: CausalEnvelope | None,
        operation: OperationDescriptor,
        permit_id: str,
    ) -> DispatchDecision:
        decision = self._policy.authorize_and_start(
            envelope,
            operation,
            permit_id,
        )
        self._append(
            decision_point="authorization",
            envelope=envelope,
            operation=operation,
            decision=decision,
        )
        return decision

    def finish_attempt(self, *args, **kwargs) -> None:
        self._policy.finish_attempt(*args, **kwargs)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _live_summary_lane(path):
    db = SessionDB(path)
    acceptance = db.accept_task_fence_ingress(
        IngressEnvelope(
            source="gateway:test:campaign",
            source_event_id="campaign-summary",
            conversation_id="campaign-conversation",
            action=TASK_FENCE_ACTIONS["initial_submit"],
            payload_hash=_hash("campaign-summary"),
        )
    )
    return db, acceptance


def _live_model_lane(path):
    db = SessionDB(path)
    acceptance = db.accept_task_fence_ingress(
        IngressEnvelope(
            source="gateway:test:campaign",
            source_event_id="campaign-initial",
            conversation_id="campaign-conversation",
            action=TASK_FENCE_ACTIONS["initial_submit"],
            payload_hash=_hash("campaign-initial"),
        )
    )
    generation = db.reserve_task_fence_generation(acceptance)
    return db, generation


def _openai_agent():
    return SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="campaign-model",
    )


def _openai_client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


@pytest.fixture
def campaign_summary_agent():
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="campaign-test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.max_iterations = 1
    return agent


def _campaign_probe(
    policy: TaskFencePolicy,
    *,
    route_id: str = _OPENAI_ROUTE,
) -> _CampaignPolicyProbe:
    return _CampaignPolicyProbe(
        policy,
        route_id=route_id,
    )


def _campaign_probe_factory(
    *,
    route_id: str,
    store: SessionDB,
    probes: list[_CampaignPolicyProbe],
):
    def factory(candidate_store):
        assert candidate_store is store
        probe = _campaign_probe(
            TaskFencePolicy(candidate_store),
            route_id=route_id,
        )
        probes.append(probe)
        return probe

    return factory


def _assert_summary_physical_entry(
    *,
    db: SessionDB,
    probes: list[_CampaignPolicyProbe],
    route_id: str,
) -> CausalEnvelope:
    assert len(probes) == 1
    records = probes[0].records
    assert [record.route_id for record in records] == [route_id, route_id]
    assert [record.decision_point for record in records] == [
        "admission",
        "authorization",
    ]
    assert [record.outcome for record in records] == [
        DecisionOutcome.WOULD_RESERVE.value,
        DecisionOutcome.WOULD_ALLOW.value,
    ]
    assert {record.reason_code for record in records} == {
        DecisionReason.CURRENT_AUTHORITY.value
    }
    assert all(record.decision_id is not None for record in records)
    envelope = current_causal_envelope()
    assert envelope is not None
    assert {record.invocation_id for record in records} == {
        envelope.invocation_id
    }
    assert {record.task_id for record in records} == {envelope.task_id}
    assert {record.generation_id for record in records} == {
        envelope.generation_id
    }
    attempt = db._conn.execute(
        "SELECT a.state FROM task_fence_attempts AS a "
        "JOIN task_fence_dispatch_permits AS p ON p.permit_id = a.permit_id "
        "WHERE p.invocation_envelope_id = ?",
        (envelope.invocation_id,),
    ).fetchone()
    assert attempt["state"] == "STARTED"
    assert current_task_fence_policy() is None
    return envelope


def _run_campaign_summary(
    *,
    agent,
    acceptance,
    db: SessionDB,
    route_id: str,
    probes: list[_CampaignPolicyProbe],
):
    agent._session_db = db
    generations = []
    with patch(
        "task_fence.TaskFencePolicy",
        new=_campaign_probe_factory(
            route_id=route_id,
            store=db,
            probes=probes,
        ),
    ):
        result = agent._handle_max_iterations(
            [{"role": "user", "content": "summarize campaign work"}],
            1,
            _task_fence_acceptance=acceptance,
            _task_fence_generation_out=generations,
        )
    assert current_causal_envelope() is None
    assert current_task_fence_policy() is None
    return result, generations


def test_campaign_probe_records_gemini_iteration_summary_at_http_entry(
    campaign_summary_agent,
    tmp_path,
):
    from agent.gemini_native_adapter import GeminiNativeClient

    db, acceptance = _live_summary_lane(tmp_path / "state.db")
    probes = []
    physical = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "gemini campaign summary"}]
                        },
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
            envelope = _assert_summary_physical_entry(
                db=db,
                probes=probes,
                route_id=_GEMINI_ROUTE,
            )
            physical.append((url, json, headers, timeout, envelope))
            return Response()

        def close(self):
            return None

    endpoint = "https://generativelanguage.googleapis.com/v1beta"
    client = GeminiNativeClient(
        api_key="campaign-gemini-key",
        base_url=endpoint,
        http_client=HTTP(),
    )
    agent = campaign_summary_agent
    agent.api_mode = "chat_completions"
    agent.provider = "gemini"
    agent.model = "gemini-2.5-flash"
    agent.base_url = endpoint
    agent._base_url_lower = endpoint.lower()
    agent._base_url_hostname = "generativelanguage.googleapis.com"
    try:
        with patch.object(
            agent,
            "_ensure_primary_openai_client",
            return_value=client,
        ) as ensure_client:
            result, generations = _run_campaign_summary(
                agent=agent,
                acceptance=acceptance,
                db=db,
                route_id=_GEMINI_ROUTE,
                probes=probes,
            )

        assert result == "gemini campaign summary"
        ensure_client.assert_called_once_with(reason="iteration_limit_summary")
        assert len(physical) == 1
        assert physical[0][0].endswith(
            "/models/gemini-2.5-flash:generateContent"
        )
        assert len(generations) == 1
        assert (
            generations[0].generation_id
            == physical[0][4].generation_id
        )
        assert len(probes) == 1
        probes[0].assert_complete()
    finally:
        client.close()
        db.close()


def test_campaign_probe_records_anthropic_iteration_summary_at_stream_open(
    campaign_summary_agent,
    tmp_path,
):
    db, acceptance = _live_summary_lane(tmp_path / "state.db")
    probes = []
    physical = []
    calls = []
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text="anthropic stream campaign summary",
            )
        ],
        stop_reason="end_turn",
        usage=None,
    )

    class Stream:
        response = SimpleNamespace(headers={})

        def __enter__(self):
            physical.append(
                _assert_summary_physical_entry(
                    db=db,
                    probes=probes,
                    route_id=_ANTHROPIC_STREAM_ROUTE,
                )
            )
            return self

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            return response

        def __exit__(self, *_args):
            return False

    class Messages:
        def stream(self, **kwargs):
            assert "_task_fence_model_policy" not in kwargs
            calls.append(("stream", dict(kwargs)))
            return Stream()

        def create(self, **_kwargs):
            pytest.fail("stream summary must not use messages.create")

    agent = campaign_summary_agent
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent.model = "claude-test"
    agent.base_url = "https://api.anthropic.com"
    agent._base_url_lower = agent.base_url.lower()
    agent._base_url_hostname = "api.anthropic.com"
    agent._anthropic_base_url = agent.base_url
    agent._anthropic_api_key = "campaign-anthropic-key"
    agent._anthropic_client = SimpleNamespace(messages=Messages())
    agent._is_anthropic_oauth = False
    agent._disable_streaming = False
    try:
        result, generations = _run_campaign_summary(
            agent=agent,
            acceptance=acceptance,
            db=db,
            route_id=_ANTHROPIC_STREAM_ROUTE,
            probes=probes,
        )

        assert result == "anthropic stream campaign summary"
        assert [call[0] for call in calls] == ["stream"]
        assert len(physical) == 1
        assert len(generations) == 1
        assert generations[0].generation_id == physical[0].generation_id
        assert len(probes) == 1
        probes[0].assert_complete()
    finally:
        db.close()


def test_campaign_probe_records_anthropic_iteration_summary_at_messages_create(
    campaign_summary_agent,
    tmp_path,
):
    db, acceptance = _live_summary_lane(tmp_path / "state.db")
    probes = []
    physical = []
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text="anthropic create campaign summary",
            )
        ],
        stop_reason="end_turn",
        usage=None,
    )

    class Messages:
        def stream(self, **_kwargs):
            pytest.fail("non-stream summary must not use messages.stream")

        def create(self, **kwargs):
            assert "_task_fence_model_policy" not in kwargs
            envelope = _assert_summary_physical_entry(
                db=db,
                probes=probes,
                route_id=_ANTHROPIC_CREATE_ROUTE,
            )
            physical.append((dict(kwargs), envelope))
            return response

    agent = campaign_summary_agent
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent.model = "claude-test"
    agent.base_url = "https://api.anthropic.com"
    agent._base_url_lower = agent.base_url.lower()
    agent._base_url_hostname = "api.anthropic.com"
    agent._anthropic_base_url = agent.base_url
    agent._anthropic_api_key = "campaign-anthropic-key"
    agent._anthropic_client = SimpleNamespace(messages=Messages())
    agent._is_anthropic_oauth = False
    agent._disable_streaming = True
    try:
        result, generations = _run_campaign_summary(
            agent=agent,
            acceptance=acceptance,
            db=db,
            route_id=_ANTHROPIC_CREATE_ROUTE,
            probes=probes,
        )

        assert result == "anthropic create campaign summary"
        assert len(physical) == 1
        assert len(generations) == 1
        assert generations[0].generation_id == physical[0][1].generation_id
        assert len(probes) == 1
        probes[0].assert_complete()
    finally:
        db.close()


def test_campaign_probe_records_codex_iteration_summary_at_responses_create(
    campaign_summary_agent,
    tmp_path,
):
    db, acceptance = _live_summary_lane(tmp_path / "state.db")
    probes = []
    physical = []
    lifecycle = []
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            delta="codex campaign summary",
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                status="completed",
                usage=None,
                id="resp-campaign-summary",
            ),
        ),
    ]

    class EventStream:
        def __iter__(self):
            lifecycle.append("iterate")
            return iter(events)

        def close(self):
            lifecycle.append("close")

    class Responses:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            assert "tools" not in kwargs
            envelope = _assert_summary_physical_entry(
                db=db,
                probes=probes,
                route_id=_CODEX_ROUTE,
            )
            physical.append((dict(kwargs), envelope))
            return EventStream()

    agent = campaign_summary_agent
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    agent.model = "gpt-test-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent._base_url_lower = agent.base_url.lower()
    agent._base_url_hostname = "chatgpt.com"
    agent._disable_streaming = False
    client = SimpleNamespace(responses=Responses())
    try:
        with patch.object(
            agent,
            "_ensure_primary_openai_client",
            return_value=client,
        ) as ensure_client:
            result, generations = _run_campaign_summary(
                agent=agent,
                acceptance=acceptance,
                db=db,
                route_id=_CODEX_ROUTE,
                probes=probes,
            )

        assert result == "codex campaign summary"
        ensure_client.assert_called_once_with(reason="codex_stream_direct")
        assert len(physical) == 1
        assert lifecycle == ["iterate", "close"]
        assert len(generations) == 1
        assert generations[0].generation_id == physical[0][1].generation_id
        assert len(probes) == 1
        probes[0].assert_complete()
    finally:
        db.close()


def test_campaign_excludes_bedrock_anthropic_main_route_at_real_fallback_edges(
    campaign_summary_agent,
    tmp_path,
):
    db, acceptance = _live_summary_lane(tmp_path / "state.db")
    policy_constructions = []
    calls = []
    physical = []
    request_secret = "raw-bedrock-anthropic-campaign-secret"
    model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    endpoint = "https://bedrock-runtime.us-east-1.amazonaws.com"
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text="bedrock anthropic legacy response",
            )
        ],
        stop_reason="end_turn",
        usage=None,
    )

    assert next(
        declaration.state
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        if declaration.capability_id == _BEDROCK_ANTHROPIC_ROUTE
    ) is TaskFenceCapabilityState.UNSUPPORTED

    def inspect_physical_entry(stage, kwargs):
        envelope = current_causal_envelope()
        assert envelope is not None
        assert envelope.task_id == acceptance.task_id
        assert envelope.invocation_id is None
        assert current_task_fence_policy() is None
        assert not any(key.startswith("_task_fence_") for key in kwargs)
        with db._lock:
            counts = tuple(
                db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "task_fence_policy_decisions",
                    "task_fence_dispatch_permits",
                    "task_fence_attempts",
                )
            )
        assert counts == (0, 0, 0)
        physical.append((stage, dict(kwargs), envelope))

    class UnavailableStream:
        def __init__(self, kwargs):
            self._kwargs = kwargs

        def __enter__(self):
            calls.append("stream_enter")
            inspect_physical_entry("stream_enter", self._kwargs)
            raise RuntimeError(
                "not authorized to perform: "
                "bedrock:InvokeModelWithResponseStream"
            )

        def __exit__(self, *_args):
            return False

    class Messages:
        @staticmethod
        def stream(**kwargs):
            calls.append("stream_factory")
            assert not any(key.startswith("_task_fence_") for key in kwargs)
            return UnavailableStream(dict(kwargs))

        @staticmethod
        def create(**kwargs):
            calls.append("create")
            inspect_physical_entry("create", kwargs)
            return response

    def tracked_policy(*args, **kwargs):
        policy_constructions.append((args, kwargs))
        return TaskFencePolicy(*args, **kwargs)

    agent = campaign_summary_agent
    agent._session_db = db
    agent.api_mode = "anthropic_messages"
    agent.provider = "bedrock"
    agent.model = model
    agent.base_url = endpoint
    agent._base_url_lower = endpoint.lower()
    agent._base_url_hostname = "bedrock-runtime.us-east-1.amazonaws.com"
    agent._anthropic_base_url = endpoint
    agent._anthropic_api_key = "aws-sdk"
    agent._is_anthropic_oauth = False
    agent._disable_streaming = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = False
    request_client = SimpleNamespace(messages=Messages())
    try:
        with (
            patch(
                "task_fence.TaskFencePolicy",
                side_effect=tracked_policy,
            ),
            patch.object(
                agent,
                "_create_request_anthropic_client",
                return_value=request_client,
            ) as make_client,
            patch.object(agent, "_close_request_anthropic_client"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                request_secret,
                task_fence_acceptance=acceptance,
            )

        assert result["completed"] is True
        assert result["final_response"] == "bedrock anthropic legacy response"
        assert calls == ["stream_factory", "stream_enter", "create"]
        assert [entry[0] for entry in physical] == ["stream_enter", "create"]
        assert len({entry[2].generation_id for entry in physical}) == 1
        make_client.assert_called_once_with(reason="anthropic_messages_request")
        assert policy_constructions == []
        generation = db._conn.execute(
            "SELECT generation_id, task_id, state, closed_at "
            "FROM task_fence_model_generations"
        ).fetchone()
        assert generation is not None
        assert tuple(generation) == (
            physical[0][2].generation_id,
            acceptance.task_id,
            "committed",
            None,
        )
        for table in (
            "task_fence_policy_decisions",
            "task_fence_dispatch_permits",
            "task_fence_attempts",
        ):
            assert db._conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == 0
        assert request_secret not in "\n".join(db._conn.iterdump())
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None
    finally:
        db.close()


def test_campaign_excludes_codex_app_server_at_real_session_handoff(
    campaign_summary_agent,
    tmp_path,
):
    from agent.transports.codex_app_server_session import (
        CodexAppServerSession,
        TurnResult,
    )

    db, acceptance = _live_summary_lane(tmp_path / "state.db")
    policy_constructions = []
    physical = []
    request_secret = "raw-codex-app-server-campaign-secret"
    response_text = "codex app-server legacy response"

    assert next(
        declaration.state
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        if declaration.capability_id == _CODEX_APP_SERVER_ROUTE
    ) is TaskFenceCapabilityState.UNSUPPORTED

    def run_turn(*, user_input):
        assert user_input == request_secret
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None
        with db._lock:
            assert db._conn.execute(
                "SELECT COUNT(*) FROM task_fence_ingress"
            ).fetchone()[0] == 1
            counts = tuple(
                db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "task_fence_model_generations",
                    "task_fence_policy_decisions",
                    "task_fence_dispatch_permits",
                    "task_fence_attempts",
                )
            )
        assert counts == (0, 0, 0, 0)
        physical.append(user_input)
        return TurnResult(
            final_text=response_text,
            projected_messages=[],
            tool_iterations=0,
            turn_id="turn-task-fence-campaign",
            thread_id="thread-task-fence-campaign",
        )

    def tracked_policy(*args, **kwargs):
        policy_constructions.append((args, kwargs))
        return TaskFencePolicy(*args, **kwargs)

    agent = campaign_summary_agent
    agent._session_db = db
    agent.api_mode = "codex_app_server"
    agent.provider = "openai-codex"
    session = CodexAppServerSession(cwd=str(tmp_path))
    agent._codex_session = session
    try:
        with (
            patch.object(session, "run_turn", side_effect=run_turn),
            patch(
                "task_fence.TaskFencePolicy",
                side_effect=tracked_policy,
            ),
            patch.object(agent, "_persist_session"),
        ):
            result = agent.run_conversation(
                request_secret,
                task_fence_acceptance=acceptance,
            )

        assert physical == [request_secret]
        assert result["completed"] is True
        assert result["final_response"] == response_text
        assert policy_constructions == []
        for table in (
            "task_fence_model_generations",
            "task_fence_policy_decisions",
            "task_fence_dispatch_permits",
            "task_fence_attempts",
        ):
            assert db._conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == 0
        assert request_secret not in "\n".join(db._conn.iterdump())
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None
    finally:
        session.close()
        db.close()


def test_campaign_excludes_copilot_acp_at_real_session_prompt_write(tmp_path):
    import io
    import json
    import queue
    import threading

    from run_agent import AIAgent

    db, acceptance = _live_summary_lane(tmp_path / "state.db")
    policy_constructions = []
    methods = []
    physical = []
    processes = []
    request_secret = "raw-copilot-acp-campaign-secret"
    response_text = "copilot ACP legacy response"
    session_id = "session-task-fence-campaign"
    caller_thread_id = threading.get_ident()

    assert next(
        declaration.state
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        if declaration.capability_id == _COPILOT_ACP_ROUTE
    ) is TaskFenceCapabilityState.UNSUPPORTED

    class Output:
        def __init__(self):
            self._lines = queue.Queue()
            self.closed = False

        def put(self, message):
            self._lines.put(json.dumps(message) + "\n")

        def close(self):
            if not self.closed:
                self.closed = True
                self._lines.put(None)

        def __iter__(self):
            return self

        def __next__(self):
            line = self._lines.get()
            if line is None:
                raise StopIteration
            return line

    class Input:
        def __init__(self, process):
            self._process = process
            self._buffer = ""

        def write(self, data):
            self._buffer += data
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line:
                    self._process.handle(json.loads(line))
            return len(data)

        def flush(self):
            return None

    class Process:
        def __init__(self):
            self.stdout = Output()
            self.stderr = io.StringIO("")
            self.stdin = Input(self)
            self.returncode = None

        def handle(self, payload):
            method = payload["method"]
            methods.append(method)
            if method == "initialize":
                result = {}
            elif method == "session/new":
                result = {"sessionId": session_id}
            elif method == "session/prompt":
                envelope = current_causal_envelope()
                assert threading.get_ident() != caller_thread_id
                assert envelope is not None
                assert envelope.task_id == acceptance.task_id
                assert envelope.invocation_id is None
                assert current_task_fence_policy() is None
                assert policy_constructions == []
                assert payload["jsonrpc"] == "2.0"
                assert payload["params"]["sessionId"] == session_id
                prompt = payload["params"]["prompt"]
                assert request_secret in prompt[0]["text"]
                assert "_task_fence_" not in json.dumps(payload)
                with db._lock:
                    generation = db._conn.execute(
                        "SELECT generation_id, task_id, state "
                        "FROM task_fence_model_generations"
                    ).fetchone()
                    counts = tuple(
                        db._conn.execute(
                            f"SELECT COUNT(*) FROM {table}"
                        ).fetchone()[0]
                        for table in (
                            "task_fence_policy_decisions",
                            "task_fence_dispatch_permits",
                            "task_fence_attempts",
                        )
                    )
                assert tuple(generation) == (
                    envelope.generation_id,
                    acceptance.task_id,
                    "started",
                )
                assert counts == (0, 0, 0)
                physical.append((payload, envelope, threading.get_ident()))
                self.stdout.put(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {
                                    "type": "text",
                                    "text": response_text,
                                },
                            },
                        },
                    }
                )
                result = {"stopReason": "end_turn"}
            else:
                raise AssertionError(f"unexpected ACP method: {method}")
            self.stdout.put(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": result,
                }
            )

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0
            self.stdout.close()

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.terminate()

    def popen(*_args, **_kwargs):
        process = Process()
        processes.append(process)
        return process

    def tracked_policy(*args, **kwargs):
        policy_constructions.append((args, kwargs))
        return TaskFencePolicy(*args, **kwargs)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(
            api_key="copilot-acp",
            base_url="acp://copilot",
            provider="copilot-acp",
            api_mode="chat_completions",
            model="copilot-campaign-model",
            acp_command="copilot-campaign",
            acp_args=["--acp", "--stdio"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=1,
        )
    agent._session_db = db
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = False
    try:
        with (
            patch(
                "agent.copilot_acp_client.subprocess.Popen",
                side_effect=popen,
            ),
            patch(
                "task_fence.TaskFencePolicy",
                side_effect=tracked_policy,
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                request_secret,
                task_fence_acceptance=acceptance,
            )

        assert result["completed"] is True
        assert result["final_response"] == response_text
        assert methods == ["initialize", "session/new", "session/prompt"]
        assert len(processes) == 1
        assert len(physical) == 1
        assert processes[0].returncode == 0
        assert processes[0].stdout.closed is True
        assert policy_constructions == []
        generation = db._conn.execute(
            "SELECT generation_id, task_id, state "
            "FROM task_fence_model_generations"
        ).fetchone()
        assert tuple(generation) == (
            physical[0][1].generation_id,
            acceptance.task_id,
            "committed",
        )
        for table in (
            "task_fence_policy_decisions",
            "task_fence_dispatch_permits",
            "task_fence_attempts",
        ):
            assert db._conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == 0
        assert request_secret not in "\n".join(db._conn.iterdump())
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None
    finally:
        agent.close()
        db.close()


def test_campaign_excludes_moa_one_shot_at_real_aggregator_create(
    campaign_summary_agent,
    tmp_path,
):
    db, acceptance = _live_summary_lane(tmp_path / "state.db")
    probes = []
    factories = []
    physical = []
    main_calls = []
    request_secret = "raw-moa-one-shot-campaign-secret"
    reference_advice = "bounded campaign reference advice"
    aggregate_guidance = "bounded campaign aggregate guidance"
    main_text = "one-shot legacy main response"

    assert next(
        declaration.state
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        if declaration.capability_id == _MOA_ONE_SHOT_ROUTE
    ) is TaskFenceCapabilityState.UNSUPPORTED

    def response(content, *, model, response_id):
        return SimpleNamespace(
            id=response_id,
            model=model,
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content,
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        reasoning_details=None,
                    ),
                    finish_reason="stop",
                )
            ],
        )

    def aggregator_create(**kwargs):
        with db._lock:
            ingress_task_id = db._conn.execute(
                "SELECT task_id FROM task_fence_ingress"
            ).fetchone()["task_id"]
            counts = tuple(
                db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "task_fence_model_generations",
                    "task_fence_policy_decisions",
                    "task_fence_dispatch_permits",
                    "task_fence_attempts",
                )
            )
        physical.append(
            (
                dict(kwargs),
                current_causal_envelope(),
                current_task_fence_policy(),
                ingress_task_id,
                counts,
                len(probes),
            )
        )
        return response(
            aggregate_guidance,
            model="campaign-aggregator-model",
            response_id="chatcmpl-moa-one-shot-aggregate",
        )

    auxiliary_client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        api_key="campaign-aux-key",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=aggregator_create),
        ),
    )

    def get_cached_client(provider, model, **kwargs):
        factories.append(
            (
                provider,
                model,
                dict(kwargs),
                current_causal_envelope(),
                current_task_fence_policy(),
            )
        )
        return auxiliary_client, model

    def main_create(**kwargs):
        envelope = _assert_summary_physical_entry(
            db=db,
            probes=probes,
            route_id=_OPENAI_ROUTE,
        )
        with db._lock:
            state = db._conn.execute(
                "SELECT state FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (envelope.generation_id,),
            ).fetchone()["state"]
        main_calls.append((dict(kwargs), envelope, state))
        return response(
            main_text,
            model="campaign-main-model",
            response_id="chatcmpl-moa-one-shot-main",
        )

    moa_config = {
        "reference_models": [
            {
                "provider": "openrouter",
                "model": "campaign-reference-model",
                "enabled": True,
            }
        ],
        "aggregator": {
            "provider": "openrouter",
            "model": "campaign-aggregator-model",
        },
        "reference_temperature": None,
        "aggregator_temperature": None,
        "degraded_reference_policy": "loud",
    }

    agent = campaign_summary_agent
    agent._session_db = db
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.model = "campaign-main-model"
    agent.client.base_url = agent.base_url
    agent.client.api_key = "campaign-main-key"
    agent.client.chat.completions.create.side_effect = main_create
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = False
    try:
        with (
            patch(
                "agent.moa_loop._run_references_parallel",
                return_value=[
                    (
                        "openrouter:campaign-reference-model",
                        reference_advice,
                        None,
                    )
                ],
            ) as fanout,
            patch(
                "agent.auxiliary_client._get_cached_client",
                side_effect=get_cached_client,
            ),
            patch(
                "task_fence.TaskFencePolicy",
                new=_campaign_probe_factory(
                    route_id=_OPENAI_ROUTE,
                    store=db,
                    probes=probes,
                ),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                request_secret,
                moa_config=moa_config,
                task_fence_acceptance=acceptance,
            )

        fanout.assert_called_once()
        assert len(factories) == 1
        assert factories[0][0:2] == (
            "openrouter",
            "campaign-aggregator-model",
        )
        assert factories[0][3:] == (None, None)

        assert len(physical) == 1
        (
            aggregator_kwargs,
            aggregator_envelope,
            aggregator_policy,
            ingress_task_id,
            aggregator_counts,
            probe_count,
        ) = physical[0]
        assert aggregator_envelope is None
        assert aggregator_policy is None
        assert ingress_task_id == acceptance.task_id
        assert aggregator_counts == (0, 0, 0, 0)
        assert probe_count == 0
        assert not any(
            key.startswith("_task_fence_") for key in aggregator_kwargs
        )
        synthesis_prompt = str(aggregator_kwargs["messages"])
        assert request_secret in synthesis_prompt
        assert reference_advice in synthesis_prompt

        assert len(main_calls) == 1
        main_kwargs, main_envelope, main_state = main_calls[0]
        assert main_envelope.task_id == acceptance.task_id
        assert main_envelope.invocation_id is not None
        assert main_envelope.parent_invocation_id is None
        assert main_state == "started"
        assert aggregate_guidance in str(main_kwargs["messages"])

        assert result["completed"] is True
        assert result["final_response"] == main_text
        assert len(probes) == 1
        probes[0].assert_complete()
        generation = db._conn.execute(
            "SELECT generation_id, task_id, state, closed_at "
            "FROM task_fence_model_generations"
        ).fetchone()
        assert tuple(generation) == (
            main_envelope.generation_id,
            acceptance.task_id,
            "committed",
            None,
        )
        assert request_secret not in "\n".join(db._conn.iterdump())
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_campaign_excludes_sync_goal_judge_at_real_auxiliary_create(
    tmp_path,
    monkeypatch,
):
    from typing import Any

    from agent import auxiliary_client
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, build_session_key
    from hermes_cli import goals
    from hermes_state import AsyncSessionDB

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    source = SessionSource(
        platform=Platform.SLACK,
        user_id="U-CAMPAIGN",
        chat_id="C-CAMPAIGN",
        user_name="campaign",
        chat_type="channel",
        thread_id="1718600000.000100",
    )
    conversation_id = build_session_key(source)
    session_id = "campaign-goal-session"
    goal = "finish the bounded generic auxiliary campaign"
    final_response = "campaign work is incomplete"
    judge_reason = "one concrete campaign step remains"

    db = SessionDB(home / "state.db")
    conn = db._conn
    assert conn is not None
    acceptance = db.accept_task_fence_ingress(
        IngressEnvelope(
            source="gateway:slack",
            source_event_id="campaign-goal-parent",
            conversation_id=conversation_id,
            action=TASK_FENCE_ACTIONS["initial_submit"],
            payload_hash=_hash("campaign-goal-parent"),
            opaque_payload_ref="slack:campaign-goal-parent",
        )
    )
    parent_generation = db.reserve_task_fence_generation(acceptance)
    assert db.finish_task_fence_generation(
        parent_generation,
        state="committed",
    )

    assert next(
        declaration.state
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        if declaration.capability_id == _GENERIC_AUXILIARY_ROUTE
    ) is TaskFenceCapabilityState.UNSUPPORTED

    notices = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        notices.append((chat_id, content, metadata))
        return SimpleNamespace(success=True)

    adapter = SimpleNamespace(_pending_messages={}, send=send)
    runner: Any = object.__new__(GatewayRunner)
    runner.config = {"goals": {"max_turns": 2}}
    runner._queued_events = {}
    runner.adapters = {Platform.SLACK: adapter}
    runner.session_store = SimpleNamespace(
        _generate_session_key=lambda _source: conversation_id,
    )
    runner._session_db = AsyncSessionDB(db)
    session_entry = SimpleNamespace(session_id=session_id)

    physical = []

    def counts():
        with db._lock:
            return tuple(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "task_fence_model_generations",
                    "task_fence_policy_decisions",
                    "task_fence_dispatch_permits",
                    "task_fence_attempts",
                )
            )

    def create(**kwargs):
        with db._lock:
            generation = conn.execute(
                "SELECT generation_id, task_id, state "
                "FROM task_fence_model_generations"
            ).fetchone()
        physical.append(
            (
                dict(kwargs),
                current_causal_envelope(),
                current_task_fence_policy(),
                tuple(generation),
                counts(),
            )
        )
        return SimpleNamespace(
            id="chatcmpl-goal-judge-campaign",
            model="campaign-goal-judge-model",
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"verdict":"continue","reason":'
                            f'"{judge_reason}"}}'
                        )
                    ),
                    finish_reason="stop",
                )
            ],
        )

    client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )

    try:
        goals.GoalManager(session_id).set(goal, max_turns=2)
        baseline_counts = counts()
        with (
            patch.object(
                auxiliary_client,
                "_get_cached_client",
                return_value=(client, "campaign-goal-judge-model"),
            ) as get_client,
            patch(
                "task_fence.TaskFencePolicy",
                side_effect=TaskFencePolicy,
            ) as policy_factory,
        ):
            await runner._post_turn_goal_continuation(
                session_entry=session_entry,
                source=source,
                final_response=final_response,
                task_fence_parent_generation=parent_generation,
            )

        assert baseline_counts == (1, 0, 0, 0)
        get_client.assert_called_once()
        assert get_client.call_args.kwargs.get("async_mode", False) is False
        policy_factory.assert_not_called()

        assert len(physical) == 1
        (
            model_kwargs,
            envelope,
            policy,
            generation,
            physical_counts,
        ) = physical[0]
        assert envelope is None
        assert policy is None
        assert generation == (
            parent_generation.generation_id,
            acceptance.task_id,
            "committed",
        )
        assert physical_counts == baseline_counts
        assert not any(key.startswith("_task_fence_") for key in model_kwargs)
        judge_prompt = str(model_kwargs["messages"])
        assert goal in judge_prompt
        assert final_response in judge_prompt

        goal_state = goals.load_goal(session_id)
        assert goal_state is not None
        assert goal_state.last_verdict == "continue"
        assert goal_state.last_reason == judge_reason
        assert goal_state.consecutive_parse_failures == 0
        assert goal_state.consecutive_transport_failures == 0

        assert len(notices) == 1
        assert notices[0][0] == source.chat_id
        assert judge_reason in notices[0][1]
        continuation = adapter._pending_messages[conversation_id]
        assert goal in continuation.text
        continuation_acceptance = continuation.task_fence_acceptance
        assert continuation_acceptance is not None
        assert continuation_acceptance.task_id == acceptance.task_id
        assert continuation_acceptance.event_id != acceptance.event_id
        recorded = conn.execute(
            "SELECT source, causal_parent_generation_id "
            "FROM task_fence_ingress WHERE event_id = ?",
            (continuation_acceptance.event_id,),
        ).fetchone()
        assert tuple(recorded) == (
            "runtime:goal_continuation",
            parent_generation.generation_id,
        )
        assert counts() == baseline_counts
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None
    finally:
        goal_db = goals._DB_CACHE.pop(str(home), None)
        if goal_db is not None:
            goal_db.close()
        db.close()


def test_campaign_excludes_sync_compression_at_real_progress_stream_create(
    campaign_summary_agent,
    tmp_path,
):
    from agent.context_compressor import ContextCompressor

    session_id = "campaign-conversation"
    db, acceptance = _live_summary_lane(tmp_path / "state.db")
    conn = db._conn
    assert conn is not None
    db.create_session(
        session_id,
        source="gateway",
        system_prompt="You are helpful.",
    )
    assert next(
        declaration.state
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        if declaration.capability_id == _GENERIC_AUXILIARY_ROUTE
    ) is TaskFenceCapabilityState.UNSUPPORTED

    probes = []
    physical = []
    main_calls = []
    stream_closed = []
    middle_marker = "bounded-compression-middle-marker"
    compression_summary = "bounded compression retained the campaign state"
    request_secret = "raw-compression-campaign-request"
    main_text = "compression campaign legacy main response"
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": (
                f"historical turn {index} "
                f"{middle_marker if index == 10 else ''} "
                + ("x" * 1400)
            ),
        }
        for index in range(24)
    ]

    def counts():
        with db._lock:
            return tuple(
                conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                for table in (
                    "task_fence_ingress",
                    "task_fence_model_generations",
                    "task_fence_policy_decisions",
                    "task_fence_dispatch_permits",
                    "task_fence_attempts",
                )
            )

    class Chunks:
        def __iter__(self):
            yield SimpleNamespace(
                id="chatcmpl-compression-campaign",
                model="campaign-compression-model",
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=compression_summary,
                            reasoning=None,
                            reasoning_content=None,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
            )

        def close(self):
            stream_closed.append(True)

    def compression_create(**kwargs):
        physical.append(
            (
                dict(kwargs),
                current_causal_envelope(),
                current_task_fence_policy(),
                counts(),
                len(probes),
            )
        )
        return Chunks()

    auxiliary_client = _openai_client(compression_create)
    auxiliary_client.base_url = "https://openrouter.ai/api/v1"

    def main_create(**kwargs):
        envelope = _assert_summary_physical_entry(
            db=db,
            probes=probes,
            route_id=_OPENAI_ROUTE,
        )
        with db._lock:
            state = conn.execute(
                "SELECT state FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (envelope.generation_id,),
            ).fetchone()["state"]
        main_calls.append((dict(kwargs), envelope, state))
        return SimpleNamespace(
            id="chatcmpl-compression-main",
            model="campaign-main-model",
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=main_text,
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        reasoning_details=None,
                    ),
                    finish_reason="stop",
                )
            ],
        )

    agent = campaign_summary_agent
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.model = "campaign-main-model"
    agent.client.base_url = agent.base_url
    agent.client.api_key = "campaign-main-key"
    agent.client.chat.completions.create.side_effect = main_create
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = True
    agent.compression_in_place = True
    agent.max_compression_attempts = 1
    agent._compression_feasibility_checked = True
    agent.context_compressor = ContextCompressor(
        model=agent.model,
        threshold_percent=0.5,
        protect_first_n=1,
        protect_last_n=2,
        summary_target_ratio=0.1,
        quiet_mode=True,
        base_url=agent.base_url,
        api_key=agent.client.api_key,
        config_context_length=8_000,
        provider=agent.provider,
        api_mode=agent.api_mode,
    )

    try:
        with (
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(
                    auxiliary_client,
                    "campaign-compression-model",
                ),
            ) as get_client,
            patch(
                "task_fence.TaskFencePolicy",
                new=_campaign_probe_factory(
                    route_id=_OPENAI_ROUTE,
                    store=db,
                    probes=probes,
                ),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                request_secret,
                conversation_history=history,
                task_fence_acceptance=acceptance,
            )

        get_client.assert_called_once()
        assert get_client.call_args.kwargs.get("async_mode", False) is False
        assert len(physical) == 1
        (
            compression_kwargs,
            compression_envelope,
            compression_policy,
            compression_counts,
            compression_probe_count,
        ) = physical[0]
        assert compression_envelope is None
        assert compression_policy is None
        assert compression_counts == (1, 0, 0, 0, 0)
        assert compression_probe_count == 0
        assert compression_kwargs["stream"] is True
        assert compression_kwargs["stream_options"] == {
            "include_usage": True,
        }
        assert not any(
            key.startswith("_task_fence_") for key in compression_kwargs
        )
        assert middle_marker in str(compression_kwargs["messages"])
        assert stream_closed == [True]

        assert len(main_calls) == 1
        main_kwargs, main_envelope, main_state = main_calls[0]
        assert main_envelope.task_id == acceptance.task_id
        assert main_envelope.invocation_id is not None
        assert main_envelope.parent_invocation_id is None
        assert main_state == "started"
        assert compression_summary in str(main_kwargs["messages"])
        assert request_secret in str(main_kwargs["messages"])

        assert result["completed"] is True
        assert result["final_response"] == main_text
        assert len(probes) == 1
        probes[0].assert_complete()
        generation = conn.execute(
            "SELECT generation_id, task_id, state "
            "FROM task_fence_model_generations"
        ).fetchone()
        assert tuple(generation) == (
            main_envelope.generation_id,
            acceptance.task_id,
            "committed",
        )
        attempt = conn.execute(
            "SELECT state FROM task_fence_attempts"
        ).fetchone()
        assert attempt["state"] == "SUCCEEDED"
        assert counts() == (1, 1, 2, 1, 1)
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None
    finally:
        db.close()


def test_campaign_excludes_sync_title_generation_at_real_background_create(
    campaign_summary_agent,
    tmp_path,
    monkeypatch,
):
    import threading

    from agent import auxiliary_client, title_generator

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    session_id = "campaign-conversation"
    db, acceptance = _live_summary_lane(tmp_path / "state.db")
    conn = db._conn
    assert conn is not None
    db.create_session(
        session_id,
        source="gateway",
        system_prompt="You are helpful.",
    )
    assert next(
        declaration.state
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
        if declaration.capability_id == _GENERIC_AUXILIARY_ROUTE
    ) is TaskFenceCapabilityState.UNSUPPORTED

    probes = []
    main_calls = []
    title_physical = []
    worker_threads = []
    request_text = "prove bounded background title exclusion"
    main_text = "the accepted campaign turn completed successfully"
    expected_title = "Bounded Background Title Exclusion"

    def counts():
        with db._lock:
            return tuple(
                conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                for table in (
                    "task_fence_ingress",
                    "task_fence_model_generations",
                    "task_fence_policy_decisions",
                    "task_fence_dispatch_permits",
                    "task_fence_attempts",
                )
            )

    def main_create(**kwargs):
        envelope = _assert_summary_physical_entry(
            db=db,
            probes=probes,
            route_id=_OPENAI_ROUTE,
        )
        with db._lock:
            state = conn.execute(
                "SELECT state FROM task_fence_model_generations "
                "WHERE generation_id = ?",
                (envelope.generation_id,),
            ).fetchone()["state"]
        main_calls.append((dict(kwargs), envelope, state))
        return SimpleNamespace(
            id="chatcmpl-title-main",
            model="campaign-main-model",
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=main_text,
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        reasoning_details=None,
                    ),
                    finish_reason="stop",
                )
            ],
        )

    agent = campaign_summary_agent
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.model = "campaign-main-model"
    agent.client.base_url = agent.base_url
    agent.client.api_key = "campaign-main-key"
    agent.client.chat.completions.create.side_effect = main_create
    agent.compression_enabled = False

    try:
        with (
            patch(
                "task_fence.TaskFencePolicy",
                new=_campaign_probe_factory(
                    route_id=_OPENAI_ROUTE,
                    store=db,
                    probes=probes,
                ),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                request_text,
                task_fence_acceptance=acceptance,
            )

        assert result["completed"] is True
        assert result["final_response"] == main_text
        assert len(main_calls) == 1
        main_kwargs, main_envelope, main_state = main_calls[0]
        assert main_state == "started"
        assert request_text in str(main_kwargs["messages"])
        assert len(probes) == 1
        probes[0].assert_complete()

        with db._lock:
            generation = conn.execute(
                "SELECT generation_id, task_id, state "
                "FROM task_fence_model_generations"
            ).fetchone()
            attempt = conn.execute(
                "SELECT state FROM task_fence_attempts"
            ).fetchone()
        assert tuple(generation) == (
            main_envelope.generation_id,
            acceptance.task_id,
            "committed",
        )
        assert attempt["state"] == "SUCCEEDED"
        baseline_counts = counts()
        assert baseline_counts == (1, 1, 2, 1, 1)
        assert db.get_session_title(session_id) is None

        caller_thread = threading.current_thread()
        physical_called = threading.Event()

        def title_create(**kwargs):
            worker_threads.append(threading.current_thread())
            physical_counts = counts()
            with db._lock:
                physical_generation = conn.execute(
                    "SELECT generation_id, task_id, state "
                    "FROM task_fence_model_generations"
                ).fetchone()
                physical_attempt = conn.execute(
                    "SELECT state FROM task_fence_attempts"
                ).fetchone()
            title_physical.append(
                (
                    dict(kwargs),
                    current_causal_envelope(),
                    current_task_fence_policy(),
                    physical_counts,
                    tuple(physical_generation),
                    physical_attempt["state"],
                )
            )
            physical_called.set()
            return SimpleNamespace(
                id="chatcmpl-title-background",
                model="campaign-title-model",
                usage=None,
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=expected_title),
                        finish_reason="stop",
                    )
                ],
            )

        title_client = _openai_client(title_create)
        title_client.base_url = "https://openrouter.ai/api/v1"

        with (
            patch.object(
                auxiliary_client,
                "_get_cached_client",
                return_value=(title_client, "campaign-title-model"),
            ) as get_client,
            patch(
                "task_fence.TaskFencePolicy",
                side_effect=TaskFencePolicy,
            ) as title_policy_factory,
        ):
            title_generator.maybe_auto_title(
                db,
                session_id,
                request_text,
                main_text,
                result["messages"],
                main_runtime={
                    "model": agent.model,
                    "provider": agent.provider,
                    "base_url": agent.base_url,
                    "api_key": agent.client.api_key,
                    "api_mode": agent.api_mode,
                },
                runtime_validator=lambda: True,
            )
            assert physical_called.wait(timeout=10), "auto-title create never ran"
            assert len(worker_threads) == 1
            for thread in worker_threads:
                thread.join(timeout=10)

        get_client.assert_called_once()
        title_policy_factory.assert_not_called()
        assert len(worker_threads) == 1
        title_thread = worker_threads[0]
        assert title_thread is not caller_thread
        assert title_thread.daemon is True
        assert title_thread.name == "auto-title"
        assert not title_thread.is_alive()

        assert len(title_physical) == 1
        (
            title_kwargs,
            title_envelope,
            title_policy,
            physical_counts,
            physical_generation,
            physical_attempt_state,
        ) = title_physical[0]
        assert title_envelope is None
        assert title_policy is None
        assert physical_counts == baseline_counts
        assert physical_generation == tuple(generation)
        assert physical_attempt_state == "SUCCEEDED"
        assert title_kwargs["model"] == "campaign-title-model"
        assert not title_kwargs.get("stream", False)
        assert not any(
            key.startswith("_task_fence_") for key in title_kwargs
        )
        title_prompt = str(title_kwargs["messages"])
        assert request_text in title_prompt
        assert main_text in title_prompt

        assert db.get_session_title(session_id) == expected_title
        assert counts() == baseline_counts
    finally:
        for thread in worker_threads:
            thread.join(timeout=10)
        db.close()


def test_campaign_probe_records_real_openai_handoff_before_sdk_entry(tmp_path):
    db, generation = _live_model_lane(tmp_path / "state.db")
    probe = _campaign_probe(TaskFencePolicy(db))
    request_secret = "raw-campaign-request-secret"
    request = {
        "model": "campaign-model",
        "messages": [{"role": "user", "content": request_secret}],
    }
    response = SimpleNamespace(
        id="chatcmpl-campaign",
        choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
        error=None,
    )
    attempt_holder = {}
    physical_calls = []

    def create(**kwargs):
        records = probe.records
        assert [record.decision_point for record in records] == [
            "admission",
            "authorization",
        ]
        assert [record.outcome for record in records] == [
            DecisionOutcome.WOULD_RESERVE.value,
            DecisionOutcome.WOULD_ALLOW.value,
        ]
        assert {record.reason_code for record in records} == {
            DecisionReason.CURRENT_AUTHORITY.value
        }
        assert len({record.invocation_id for record in records}) == 1
        assert {record.task_id for record in records} == {
            generation.task_id
        }
        assert {record.generation_id for record in records} == {
            generation.generation_id
        }
        assert all(record.decision_id is not None for record in records)
        envelope = current_causal_envelope()
        row = db._conn.execute(
            "SELECT a.state FROM task_fence_attempts AS a "
            "JOIN task_fence_dispatch_permits AS p "
            "ON p.permit_id = a.permit_id "
            "WHERE p.invocation_envelope_id = ?",
            (records[-1].invocation_id,),
        ).fetchone()
        physical_calls.append(
            (
                dict(kwargs),
                envelope,
                current_task_fence_policy(),
                None if row is None else row["state"],
            )
        )
        return response

    try:
        with bind_causal_envelope(generation):
            result = _create_openai_chat_completion(
                _openai_agent(),
                _openai_client(create),
                request,
                task_fence_model_policy=probe,
                task_fence_attempt_holder=attempt_holder,
            )

        assert result is response
        assert len(physical_calls) == 1
        physical_request, physical_envelope, physical_policy, state = (
            physical_calls[0]
        )
        assert physical_request == request
        assert physical_envelope is not None
        assert (
            physical_envelope.invocation_id
            == probe.records[-1].invocation_id
        )
        assert physical_policy is None
        assert state == "STARTED"
        assert tuple(asdict(probe.records[0]).items()) == (
            ("route_id", _OPENAI_ROUTE),
            ("decision_point", "admission"),
            ("outcome", DecisionOutcome.WOULD_RESERVE.value),
            ("reason_code", DecisionReason.CURRENT_AUTHORITY.value),
            ("invocation_id", probe.records[0].invocation_id),
            ("task_id", generation.task_id),
            ("generation_id", generation.generation_id),
            ("decision_id", probe.records[0].decision_id),
        )
        with pytest.raises(FrozenInstanceError):
            probe.records[0].route_id = "changed"
        assert request_secret not in repr(probe.records)
        assert request_secret not in "\n".join(db._conn.iterdump())
        assert attempt_holder["attempt_id"] not in repr(probe.records)

        _finish_task_fence_openai_chat_completion(
            policy=probe,
            attempt_holder=attempt_holder,
            response=response,
        )
        terminal = db._conn.execute(
            "SELECT state FROM task_fence_attempts WHERE attempt_id = ?",
            (attempt_holder["attempt_id"],),
        ).fetchone()
        assert terminal["state"] == "SUCCEEDED"
        assert len(probe.records) == 2
        probe.assert_complete()
    finally:
        db.close()


def test_campaign_probe_keeps_taskless_missing_provenance_at_real_handoff(
    tmp_path,
):
    db, _generation = _live_model_lane(tmp_path / "state.db")
    probe = _campaign_probe(TaskFencePolicy(db))
    response = object()
    physical_records = []

    def create(**_kwargs):
        assert len(probe.records) == 1
        physical_records.append(probe.records)
        assert current_causal_envelope() is None
        assert current_task_fence_policy() is None
        return response

    try:
        result = _create_openai_chat_completion(
            _openai_agent(),
            _openai_client(create),
            {"model": "campaign-model", "messages": []},
            task_fence_model_policy=probe,
        )

        assert result is response
        assert len(physical_records) == 1
        assert len(probe.records) == 1
        record = probe.records[0]
        assert record.decision_point == "admission"
        assert record.outcome == DecisionOutcome.WOULD_BLOCK.value
        assert record.reason_code == DecisionReason.MISSING_PROVENANCE.value
        assert record.task_id is None
        assert record.generation_id is None
        assert record.decision_id is not None
        probe.assert_complete()
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_campaign_probe_keeps_no_decision_id_from_read_only_store(tmp_path):
    path = tmp_path / "state.db"
    owner, generation = _live_model_lane(path)
    owner.close()
    read_only = SessionDB(path, read_only=True)
    probe = _campaign_probe(TaskFencePolicy(read_only))
    response = object()
    physical_records = []

    def create(**_kwargs):
        assert len(probe.records) == 1
        physical_records.append(probe.records)
        assert current_task_fence_policy() is None
        return response

    try:
        with bind_causal_envelope(generation):
            result = _create_openai_chat_completion(
                _openai_agent(),
                _openai_client(create),
                {"model": "campaign-model", "messages": []},
                task_fence_model_policy=probe,
            )

        assert result is response
        assert len(physical_records) == 1
        assert len(probe.records) == 1
        record = probe.records[0]
        assert record.decision_point == "admission"
        assert record.outcome == DecisionOutcome.WOULD_BLOCK.value
        assert record.reason_code == DecisionReason.STORE_UNAVAILABLE.value
        assert record.task_id == generation.task_id
        assert record.generation_id == generation.generation_id
        assert record.decision_id is None
        probe.assert_complete()
        assert read_only._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 0
        assert read_only._conn.execute(
            "SELECT COUNT(*) FROM task_fence_dispatch_permits"
        ).fetchone()[0] == 0
        assert read_only._conn.execute(
            "SELECT COUNT(*) FROM task_fence_attempts"
        ).fetchone()[0] == 0
    finally:
        read_only.close()


def test_campaign_probe_accepts_only_selected_supported_routes(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    policy = TaskFencePolicy(db)
    try:
        probe = _campaign_probe(policy)
        assert probe.records == ()
        probe.assert_complete()
        with pytest.raises(ValueError, match="unsupported_campaign_route"):
            _CampaignPolicyProbe(
                policy,
                route_id="provider:codex.app_server",
            )
        with pytest.raises(ValueError, match="unsupported_campaign_route"):
            _CampaignPolicyProbe(
                policy,
                route_id="provider:unknown",
            )
    finally:
        db.close()


def test_campaign_probe_marks_swallowed_overflow_after_real_handoff(tmp_path):
    db, generation = _live_model_lane(tmp_path / "state.db")
    probe = _campaign_probe(TaskFencePolicy(db))
    for index in range(_MAX_CAMPAIGN_DECISION_RECORDS - 1):
        decision = probe.admit_operation(
            None,
            OperationDescriptor(
                invocation_id=f"tfiv_{index:032x}",
                kind=OperationKind.MODEL,
                adapter=_OPENAI_ROUTE,
                invocation_fingerprint=f"{index:064x}",
            ),
        )
        assert decision.reason is DecisionReason.MISSING_PROVENANCE

    response = object()
    physical_states = []

    def create(**_kwargs):
        invocation_id = current_causal_envelope().invocation_id
        row = db._conn.execute(
            "SELECT a.state FROM task_fence_attempts AS a "
            "JOIN task_fence_dispatch_permits AS p "
            "ON p.permit_id = a.permit_id "
            "WHERE p.invocation_envelope_id = ?",
            (invocation_id,),
        ).fetchone()
        physical_states.append((len(probe.records), row["state"]))
        return response

    try:
        with bind_causal_envelope(generation):
            result = _create_openai_chat_completion(
                _openai_agent(),
                _openai_client(create),
                {"model": "campaign-model", "messages": []},
                task_fence_model_policy=probe,
            )

        assert result is response
        assert physical_states == [(_MAX_CAMPAIGN_DECISION_RECORDS, "STARTED")]
        assert len(probe.records) == _MAX_CAMPAIGN_DECISION_RECORDS
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == _MAX_CAMPAIGN_DECISION_RECORDS + 1
        with pytest.raises(
            _CampaignRecordOverflow,
            match="campaign_record_limit",
        ):
            probe.assert_complete()
    finally:
        db.close()


def test_campaign_probe_marks_route_mismatch_after_facade_call(tmp_path):
    db, _generation = _live_model_lane(tmp_path / "state.db")
    probe = _campaign_probe(TaskFencePolicy(db))
    operation = OperationDescriptor(
        invocation_id="tfiv_route_mismatch",
        kind=OperationKind.MODEL,
        adapter="provider:anthropic.messages.create",
        invocation_fingerprint="a" * 64,
    )

    try:
        decision = probe.admit_operation(None, operation)
        assert decision.reason is DecisionReason.MISSING_PROVENANCE
        assert probe.records == ()
        assert db._conn.execute(
            "SELECT COUNT(*) FROM task_fence_policy_decisions"
        ).fetchone()[0] == 1
        with pytest.raises(
            ValueError,
            match="campaign_route_adapter_mismatch",
        ):
            probe.assert_complete()
    finally:
        db.close()
