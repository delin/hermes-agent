#!/usr/bin/env python3
"""Run one receipt-bound Task Fence shadow campaign inside a tested image.

The runner is an isolated CI owner, not runtime telemetry. It exercises one
configured Slack ingress lane and one inert OpenAI-compatible physical handoff
without changing legacy dispatch outcomes or contacting an external service.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import platform
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from scripts.task_fence_launcher import (
    RECEIPT_SCHEMA,
    LauncherError,
    TestedArtifactReceipt,
    parse_receipt,
)
from task_fence import (
    CausalEnvelope,
    DecisionOutcome,
    DecisionReason,
    DispatchDecision,
    OperationDescriptor,
    TASK_FENCE_CAPABILITY_VERSION,
    TASK_FENCE_SELECTED_COHORT_CAPABILITIES,
    TaskFenceArtifactIdentity,
    TaskFenceCapabilityState,
    TaskFencePolicy,
    bind_causal_envelope,
    current_causal_envelope,
    current_task_fence_policy,
)


REPORT_SCHEMA = "hermes.task-fence.campaign-report/v1"
DECISION_RECORD_SCHEMA = "task-fence-campaign-decision/v1"
SCENARIO_SET_VERSION = "task-fence-slack-openai-shadow-v1"
MAX_CAMPAIGN_DECISION_RECORDS = 64
MAX_REPORT_BYTES = 65_536

DEFAULT_BUILD_COMMIT_PATH = Path("/opt/hermes/.hermes_build_sha")
DEFAULT_LOCK_PATH = Path("/opt/hermes/uv.lock")
_OPENAI_ROUTE = "provider:openai.chat.completions.create"
_CAMPAIGN_PAYLOAD_MARKER = "task-fence-campaign-private-payload"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_BLOCKING_OUTCOMES = frozenset({
    DecisionOutcome.WOULD_BLOCK.value,
    DecisionOutcome.HALT_DISPATCH.value,
})


class CampaignError(RuntimeError):
    """A stable, secret-free campaign contract failure."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CampaignDecisionRecord:
    route_id: str
    decision_point: str
    outcome: str
    reason_code: str
    invocation_id: str
    task_id: str | None
    generation_id: str | None
    decision_id: str | None


class CampaignRecordOverflow(RuntimeError):
    pass


class CampaignPolicyProbe:
    """Bounded projection of real audit-only policy facade returns."""

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
        self._records: list[CampaignDecisionRecord] = []
        self._failure_reason: str | None = None

    @property
    def records(self) -> tuple[CampaignDecisionRecord, ...]:
        return tuple(self._records)

    def assert_complete(self) -> None:
        if self._failure_reason == "campaign_record_limit":
            raise CampaignRecordOverflow(self._failure_reason)
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
        if len(self._records) >= MAX_CAMPAIGN_DECISION_RECORDS:
            self._failure_reason = "campaign_record_limit"
            return
        self._records.append(
            CampaignDecisionRecord(
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

    def finish_attempt(self, *args: Any, **kwargs: Any) -> None:
        self._policy.finish_attempt(*args, **kwargs)


@dataclass(frozen=True)
class _ScenarioEvidence:
    scenario_id: str
    oracle: str
    records: tuple[CampaignDecisionRecord, ...]
    physical_handoff_count: int
    contract_match: bool
    record_handoff_binding_verified: bool


def _runtime_platform() -> str:
    if platform.system().lower() != "linux":
        raise CampaignError("unsupported_runtime_platform")
    machine = platform.machine().lower()
    architecture = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "x86_64": "amd64",
        "amd64": "amd64",
    }.get(machine)
    if architecture is None:
        raise CampaignError("unsupported_runtime_platform")
    return f"linux/{architecture}"


def verify_artifact_receipt(
    receipt_path: Path,
    *,
    build_commit_path: Path = DEFAULT_BUILD_COMMIT_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
    platform_id: str | None = None,
) -> TestedArtifactReceipt:
    """Bind a closed receipt to this image's commit, lock, and platform."""

    try:
        raw_commit = build_commit_path.read_bytes()
    except OSError as exc:
        raise CampaignError("build_identity_unavailable") from exc
    if len(raw_commit) > 128:
        raise CampaignError("build_identity_invalid")
    try:
        build_commit = raw_commit.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise CampaignError("build_identity_invalid") from exc
    if (
        _COMMIT_RE.fullmatch(build_commit) is None
        or set(build_commit) == {"0"}
    ):
        raise CampaignError("build_identity_invalid")

    try:
        receipt_raw = receipt_path.read_bytes()
    except OSError as exc:
        raise CampaignError("receipt_unavailable") from exc
    try:
        receipt = parse_receipt(
            receipt_raw,
            expected_platform=(platform_id or _runtime_platform()),
            expected_commit=build_commit,
        )
    except LauncherError as exc:
        raise CampaignError("receipt_invalid") from exc

    try:
        with lock_path.open("rb") as lock_file:
            lock_hex = hashlib.file_digest(lock_file, "sha256").hexdigest()
    except OSError as exc:
        raise CampaignError("dependency_lock_unavailable") from exc
    if receipt.dependency_lock_fingerprint != f"sha256:{lock_hex}":
        raise CampaignError("dependency_lock_mismatch")
    return receipt


async def _accept_configured_slack_ingress(db: Any) -> tuple[Any, int]:
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import (
        MessageEvent,
        MessageType,
        task_fence_sidecar_for_human_message,
    )
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, build_session_key
    from hermes_state import AsyncSessionDB

    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="D_TASK_FENCE_CAMPAIGN",
        chat_type="dm",
        user_id="U_TASK_FENCE_CAMPAIGN",
        thread_id="1700000000.000001",
        scope_id="T_TASK_FENCE_CAMPAIGN",
    )
    session_key = build_session_key(source)
    event = MessageEvent(
        text=_CAMPAIGN_PAYLOAD_MARKER,
        message_type=MessageType.TEXT,
        source=source,
        message_id="1700000000.000001",
    )
    event.task_fence_ingress = task_fence_sidecar_for_human_message(
        event,
        source_event_id=(
            "event:T_TASK_FENCE_CAMPAIGN:D_TASK_FENCE_CAMPAIGN:"
            "1700000000.000001"
        ),
        payload_text=_CAMPAIGN_PAYLOAD_MARKER,
    )
    if event.task_fence_ingress is None:
        raise CampaignError("configured_ingress_not_classified")

    runner: Any = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True)},
        task_fence_shadow_session_key=session_key,
    )
    runner._session_db = AsyncSessionDB(db)
    runner._update_prompt_pending = {}
    runner._is_user_authorized = lambda source: source is not None

    started = time.perf_counter_ns()
    await runner._accept_task_fence_gateway_ingress(event, session_key)
    elapsed = time.perf_counter_ns() - started
    acceptance = event.task_fence_acceptance
    if (
        event.task_fence_acceptance_attempted is not True
        or acceptance is None
        or acceptance.replayed is not False
        or acceptance.opened_run_id is None
    ):
        raise CampaignError("configured_ingress_not_accepted")
    return acceptance, elapsed


def _run_openai_scenario(
    *,
    scenario_id: str,
    oracle: str,
    policy: TaskFencePolicy,
    envelope: CausalEnvelope | None,
    expected: tuple[tuple[str, str, str], ...],
    decision_id_missing: bool,
) -> _ScenarioEvidence:
    from agent.chat_completion_helpers import _create_openai_chat_completion
    from agent.task_fence_provider import (
        _finish_task_fence_openai_chat_completion,
    )

    probe = CampaignPolicyProbe(policy, route_id=_OPENAI_ROUTE)
    physical_handoffs = 0
    handoff_records: tuple[CampaignDecisionRecord, ...] = ()
    handoff_envelope: CausalEnvelope | None = None
    handoff_policy: TaskFencePolicy | None = None
    response = SimpleNamespace(
        id=f"task-fence-campaign-{scenario_id}",
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        error=None,
    )

    def create(**_kwargs: Any) -> Any:
        nonlocal handoff_envelope, handoff_policy, handoff_records
        nonlocal physical_handoffs
        handoff_records = probe.records
        handoff_envelope = current_causal_envelope()
        handoff_policy = current_task_fence_policy()
        physical_handoffs += 1
        return response

    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="task-fence-campaign-model",
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    request = {
        "model": agent.model,
        "messages": [
            {
                "role": "user",
                "content": f"{_CAMPAIGN_PAYLOAD_MARKER}:{scenario_id}",
            }
        ],
    }
    attempt_holder: dict[str, Any] = {}
    context = (
        bind_causal_envelope(envelope)
        if envelope is not None
        else nullcontext()
    )
    with context:
        result = _create_openai_chat_completion(
            agent,
            client,
            request,
            task_fence_model_policy=probe,
            task_fence_attempt_holder=attempt_holder,
        )
    if result is not response or physical_handoffs != 1:
        raise CampaignError("physical_handoff_not_reached")
    _finish_task_fence_openai_chat_completion(
        policy=probe,
        attempt_holder=attempt_holder,
        response=response,
    )
    probe.assert_complete()
    final_records = probe.records
    if envelope is None:
        binding_verified = (
            bool(handoff_records)
            and handoff_envelope is None
            and all(
                record.task_id is None and record.generation_id is None
                for record in handoff_records
            )
        )
    else:
        binding_verified = (
            bool(handoff_records)
            and handoff_envelope is not None
            and all(
                record.invocation_id == handoff_envelope.invocation_id
                and record.task_id == handoff_envelope.task_id
                and record.generation_id == handoff_envelope.generation_id
                for record in handoff_records
            )
        )
    contract_match = (
        handoff_policy is None
        and final_records == handoff_records
        and binding_verified
        and _records_match(
            handoff_records,
            expected,
            decision_id_missing=decision_id_missing,
        )
    )
    return _ScenarioEvidence(
        scenario_id=scenario_id,
        oracle=oracle,
        records=final_records,
        physical_handoff_count=physical_handoffs,
        contract_match=contract_match,
        record_handoff_binding_verified=binding_verified,
    )


def _inventory_projection(declarations: Sequence[Any]) -> list[dict[str, str]]:
    return [
        {
            "capability_id": declaration.capability_id,
            "capability_kind": declaration.kind.value,
            "capability_version": declaration.capability_version,
            "declaration_state": declaration.state.value,
        }
        for declaration in declarations
    ]


def _inventory_fingerprint(inventory: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        inventory,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _counter_projection(
    records: Sequence[CampaignDecisionRecord],
) -> list[dict[str, Any]]:
    counts = Counter(
        (
            record.route_id,
            record.decision_point,
            record.outcome,
            record.reason_code,
        )
        for record in records
    )
    return [
        {
            "count": count,
            "decision_point": decision_point,
            "outcome": outcome,
            "reason_code": reason_code,
            "route_id": route_id,
        }
        for (route_id, decision_point, outcome, reason_code), count in sorted(
            counts.items()
        )
    ]


def _has_blocking_outcome(
    records: Sequence[CampaignDecisionRecord],
    *,
    reason_code: str | None = None,
) -> bool:
    return any(
        record.outcome in _BLOCKING_OUTCOMES
        and (reason_code is None or record.reason_code == reason_code)
        for record in records
    )


def _records_match(
    records: Sequence[CampaignDecisionRecord],
    expected: tuple[tuple[str, str, str], ...],
    *,
    decision_id_missing: bool,
) -> bool:
    actual = tuple(
        (record.decision_point, record.outcome, record.reason_code)
        for record in records
    )
    if actual != expected:
        return False
    missing = tuple(record.decision_id is None for record in records)
    return all(missing) if decision_id_missing else not any(missing)


def _state_storage_bytes(db_path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (
            db_path,
            Path(f"{db_path}-wal"),
            Path(f"{db_path}-shm"),
        )
        if candidate.exists()
    )


def build_campaign_report(
    receipt: TestedArtifactReceipt,
    *,
    in_image_identity_verified: bool = False,
) -> dict[str, Any]:
    """Run the fixed bounded scenario set and return a closed report value."""

    from hermes_state import SessionDB

    total_started = time.perf_counter_ns()
    setup_ns = 0
    ingress_ns = 0
    handoff_ns = 0
    inventory: list[dict[str, str]] = []
    scenarios: tuple[_ScenarioEvidence, ...] = ()

    with tempfile.TemporaryDirectory(prefix="task-fence-campaign-") as raw_dir:
        db_path = Path(raw_dir) / "state.db"
        db = SessionDB(db_path)
        try:
            setup_started = time.perf_counter_ns()
            store = db.inspect_task_fence_store(include_counts=False)
            if (
                not store.compatible
                or store.runtime_epoch is None
                or store.mode_generation is None
            ):
                raise CampaignError("campaign_store_incompatible")
            identity = TaskFenceArtifactIdentity(
                tested_artifact_commit=receipt.tested_artifact_commit,
                tested_artifact_checksum=receipt.tested_artifact_checksum,
                dependency_lock_fingerprint=(
                    receipt.dependency_lock_fingerprint
                ),
            )
            db.pin_task_fence_tested_artifact(
                identity,
                expected_runtime_epoch=store.runtime_epoch,
                expected_mode_generation=store.mode_generation,
            )
            verified_identity = db.verify_task_fence_tested_artifact(identity)
            if not verified_identity.verified:
                raise CampaignError("campaign_artifact_pin_unverified")
            materialized = (
                db.materialize_task_fence_selected_cohort_capabilities(
                    expected_runtime_epoch=store.runtime_epoch,
                    expected_mode_generation=store.mode_generation,
                )
            )
            inspected = db.inspect_task_fence_selected_cohort_capabilities()
            if (
                not inspected.verified
                or inspected.declarations != materialized.declarations
                or inspected.declarations
                != TASK_FENCE_SELECTED_COHORT_CAPABILITIES
            ):
                raise CampaignError("campaign_inventory_unverified")
            inventory = _inventory_projection(inspected.declarations)
            setup_ns = time.perf_counter_ns() - setup_started

            acceptance, ingress_ns = asyncio.run(
                _accept_configured_slack_ingress(db)
            )
            generation = db.reserve_task_fence_generation(acceptance)

            handoff_started = time.perf_counter_ns()
            current = _run_openai_scenario(
                scenario_id="current_authority",
                oracle="allow",
                policy=TaskFencePolicy(db),
                envelope=generation,
                expected=(
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
                ),
                decision_id_missing=False,
            )
            taskless = _run_openai_scenario(
                scenario_id="missing_provenance",
                oracle="would_block_shadow_dispatch",
                policy=TaskFencePolicy(db),
                envelope=None,
                expected=((
                    "admission",
                    DecisionOutcome.WOULD_BLOCK.value,
                    DecisionReason.MISSING_PROVENANCE.value,
                ),),
                decision_id_missing=False,
            )
            read_only = SessionDB(db_path, read_only=True)
            try:
                unavailable = _run_openai_scenario(
                    scenario_id="store_unavailable",
                    oracle="would_block_shadow_dispatch",
                    policy=TaskFencePolicy(read_only),
                    envelope=generation,
                    expected=((
                        "admission",
                        DecisionOutcome.WOULD_BLOCK.value,
                        DecisionReason.STORE_UNAVAILABLE.value,
                    ),),
                    decision_id_missing=True,
                )
            finally:
                read_only.close()
            handoff_ns = time.perf_counter_ns() - handoff_started
            scenarios = (current, taskless, unavailable)

            if not db.finish_task_fence_generation(
                generation,
                state="committed",
            ):
                raise CampaignError("campaign_generation_not_committed")
        finally:
            db.close()

        storage_bytes = _state_storage_bytes(db_path)

    all_records = tuple(
        record for scenario in scenarios for record in scenario.records
    )
    current, taskless, unavailable = scenarios
    supported_count = sum(
        declaration["declaration_state"]
        == TaskFenceCapabilityState.SUPPORTED.value
        for declaration in inventory
    )
    unsupported_count = len(inventory) - supported_count

    return {
        "artifact_binding": {
            "descriptor_observation": "external_ci_required",
            "ephemeral_campaign_store_identity_pin_verified": True,
            "in_image_commit_lock_platform_verified": (
                in_image_identity_verified
            ),
        },
        "cohort_complete": False,
        "execution_complete": True,
        "counters": _counter_projection(all_records),
        "decision_record_schema": DECISION_RECORD_SCHEMA,
        "inventory": {
            "capability_version": TASK_FENCE_CAPABILITY_VERSION,
            "declarations": inventory,
            "fingerprint": _inventory_fingerprint(inventory),
            "supported_count": supported_count,
            "total_count": len(inventory),
            "unsupported_count": unsupported_count,
        },
        "legacy_behavior": {
            "physical_handoff_count": sum(
                scenario.physical_handoff_count for scenario in scenarios
            ),
            "suppressed_handoff_count": 0,
        },
        "limits": {
            "decision_records_per_scenario": (
                MAX_CAMPAIGN_DECISION_RECORDS
            ),
            "report_bytes": MAX_REPORT_BYTES,
        },
        "measurements": {
            "campaign_total_ns": time.perf_counter_ns() - total_started,
            "configured_ingress_ns": ingress_ns,
            "model_handoffs_ns": handoff_ns,
            "setup_ns": setup_ns,
            "state_storage_bytes": storage_bytes,
        },
        "observed_route_ids": [_OPENAI_ROUTE],
        "receipt": {
            "dependency_lock_fingerprint": (
                receipt.dependency_lock_fingerprint
            ),
            "schema": RECEIPT_SCHEMA,
            "target_platform": receipt.target_platform,
            "tested_artifact_checksum": (
                receipt.tested_artifact_checksum
            ),
            "tested_artifact_commit": receipt.tested_artifact_commit,
        },
        "report_scope": "configured_slack_openai_shadow_slice",
        "reviews": {
            "decision_id_missing_count": sum(
                record.decision_id is None for record in all_records
            ),
            "false_block": {
                "denominator": 1,
                "numerator": int(_has_blocking_outcome(current.records)),
            },
            "missing_provenance": {
                "denominator": 1,
                "numerator": int(
                    _has_blocking_outcome(
                        taskless.records,
                        reason_code=(
                            DecisionReason.MISSING_PROVENANCE.value
                        ),
                    )
                ),
            },
            "store_unavailable": {
                "denominator": 1,
                "numerator": int(
                    _has_blocking_outcome(
                        unavailable.records,
                        reason_code=DecisionReason.STORE_UNAVAILABLE.value,
                    )
                ),
            },
            "taskless_record_count": sum(
                record.task_id is None for record in all_records
            ),
        },
        "scenario_contracts_match": all(
            scenario.contract_match for scenario in scenarios
        ),
        "scenario_set_version": SCENARIO_SET_VERSION,
        "scenarios": [
            {
                "contract_match": scenario.contract_match,
                "oracle": scenario.oracle,
                "physical_handoff_count": scenario.physical_handoff_count,
                "record_handoff_binding_verified": (
                    scenario.record_handoff_binding_verified
                ),
                "records": [asdict(record) for record in scenario.records],
                "scenario_id": scenario.scenario_id,
            }
            for scenario in scenarios
        ],
        "schema": REPORT_SCHEMA,
    }


def serialize_campaign_report(report: dict[str, Any]) -> bytes:
    encoded = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_REPORT_BYTES:
        raise CampaignError("campaign_report_too_large")
    return encoded


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one receipt-bound Task Fence shadow campaign.",
    )
    parser.add_argument(
        "--receipt",
        required=True,
        type=Path,
        help="Closed tested-artifact receipt mounted read-only by CI.",
    )
    args = parser.parse_args(argv)

    try:
        receipt = verify_artifact_receipt(args.receipt)
        encoded = serialize_campaign_report(
            build_campaign_report(
                receipt,
                in_image_identity_verified=True,
            )
        )
    except CampaignError as exc:
        print(f"task-fence campaign failed: {exc.reason}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            "task-fence campaign failed: "
            f"campaign_internal_error:{type(exc).__name__}",
            file=sys.stderr,
        )
        return 2

    sys.stdout.buffer.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
