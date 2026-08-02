from collections import Counter
import hashlib
import json

import pytest

from hermes_state import SessionDB
import scripts.task_fence_campaign as campaign
from scripts.task_fence_campaign import (
    DECISION_RECORD_SCHEMA,
    MAX_CAMPAIGN_DECISION_RECORDS,
    MAX_REPORT_BYTES,
    REPORT_SCHEMA,
    SCENARIO_SET_VERSION,
    CampaignError,
    CampaignDecisionRecord,
    CampaignPolicyProbe,
    CampaignRecordOverflow,
    build_campaign_report,
    serialize_campaign_report,
    verify_artifact_receipt,
)
from scripts.task_fence_launcher import (
    RECEIPT_SCHEMA,
    TestedArtifactReceipt,
)
from task_fence import (
    DecisionOutcome,
    DecisionReason,
    DispatchDecision,
    OperationDescriptor,
    OperationKind,
    TASK_FENCE_SELECTED_COHORT_CAPABILITIES,
    TASK_FENCE_SELECTED_LAUNCH_MANIFEST,
    TaskFenceCapabilityState,
    TaskFenceLaunchRouteValidation,
    TaskFencePolicy,
)


_COMMIT = "a" * 40
_ARTIFACT = "sha256:" + "b" * 64


def _write_receipt_subject(tmp_path):
    build_commit = tmp_path / "build-commit"
    lock = tmp_path / "uv.lock"
    receipt = tmp_path / "receipt.json"
    build_commit.write_text(_COMMIT + "\n", encoding="ascii")
    lock.write_bytes(b"bounded-campaign-lock\n")
    lock_fingerprint = "sha256:" + hashlib.sha256(lock.read_bytes()).hexdigest()
    payload = {
        "dependency_lock_fingerprint": lock_fingerprint,
        "schema": RECEIPT_SCHEMA,
        "target_platform": "linux/amd64",
        "tested_artifact_checksum": _ARTIFACT,
        "tested_artifact_commit": _COMMIT,
    }
    receipt.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt, build_commit, lock, payload


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_receipt_verification_binds_closed_identity_to_image_files(tmp_path):
    receipt_path, build_commit, lock, payload = _write_receipt_subject(tmp_path)

    receipt = verify_artifact_receipt(
        receipt_path,
        build_commit_path=build_commit,
        lock_path=lock,
        platform_id="linux/amd64",
    )

    assert receipt == TestedArtifactReceipt(
        target_platform=payload["target_platform"],
        tested_artifact_commit=payload["tested_artifact_commit"],
        tested_artifact_checksum=payload["tested_artifact_checksum"],
        dependency_lock_fingerprint=payload["dependency_lock_fingerprint"],
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("duplicate", "receipt_invalid"),
        ("platform", "receipt_invalid"),
        ("commit", "receipt_invalid"),
        ("lock", "dependency_lock_mismatch"),
    ),
)
def test_receipt_verification_rejects_mismatched_image_identity(
    tmp_path,
    mutation,
    reason,
):
    receipt_path, build_commit, lock, payload = _write_receipt_subject(tmp_path)
    if mutation == "duplicate":
        receipt_path.write_text(
            '{"schema":"hermes.task-fence.tested-artifact-receipt/v1",'
            '"schema":"hermes.task-fence.tested-artifact-receipt/v1"}',
            encoding="utf-8",
        )
    elif mutation == "platform":
        payload["target_platform"] = "linux/arm64"
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "commit":
        build_commit.write_text("c" * 40 + "\n", encoding="ascii")
    else:
        lock.write_bytes(b"changed-lock\n")

    with pytest.raises(CampaignError, match=reason) as exc:
        verify_artifact_receipt(
            receipt_path,
            build_commit_path=build_commit,
            lock_path=lock,
            platform_id="linux/amd64",
        )

    assert exc.value.reason == reason


def test_campaign_report_runs_real_configured_ingress_and_physical_handoffs(
    monkeypatch,
):
    receipt = TestedArtifactReceipt(
        target_platform="linux/amd64",
        tested_artifact_commit=_COMMIT,
        tested_artifact_checksum=_ARTIFACT,
        dependency_lock_fingerprint="sha256:" + "c" * 64,
    )

    catalog_loads = []
    real_load_catalog = SessionDB.load_task_fence_selected_launch_catalog

    def load_catalog(database, **kwargs):
        catalog_loads.append((database.read_only, kwargs))
        return real_load_catalog(database, **kwargs)

    monkeypatch.setattr(
        SessionDB,
        "load_task_fence_selected_launch_catalog",
        load_catalog,
    )

    report = build_campaign_report(receipt)
    encoded = serialize_campaign_report(report)

    assert len(catalog_loads) == 1
    read_only, catalog_kwargs = catalog_loads[0]
    assert read_only is True
    assert catalog_kwargs["manifest"] is TASK_FENCE_SELECTED_LAUNCH_MANIFEST
    assert catalog_kwargs["shadow_conversation_key"] == (
        campaign._configured_slack_session_key()
    )

    assert set(report) == {
        "artifact_binding",
        "cohort_complete",
        "counters",
        "decision_record_schema",
        "execution_complete",
        "inventory",
        "legacy_behavior",
        "limits",
        "measurements",
        "observed_route_ids",
        "receipt",
        "report_scope",
        "reviews",
        "scenario_contracts_match",
        "scenario_set_version",
        "scenarios",
        "schema",
    }
    assert report["schema"] == REPORT_SCHEMA
    assert report["scenario_set_version"] == SCENARIO_SET_VERSION
    assert report["decision_record_schema"] == DECISION_RECORD_SCHEMA
    assert report["execution_complete"] is True
    assert report["scenario_contracts_match"] is True
    assert report["cohort_complete"] is False
    assert report["report_scope"] == (
        "configured_slack_openai_registered_tool_shadow_slice"
    )
    assert report["receipt"] == {
        "dependency_lock_fingerprint": receipt.dependency_lock_fingerprint,
        "schema": RECEIPT_SCHEMA,
        "target_platform": receipt.target_platform,
        "tested_artifact_checksum": receipt.tested_artifact_checksum,
        "tested_artifact_commit": receipt.tested_artifact_commit,
    }
    assert report["artifact_binding"] == {
        "descriptor_observation": "external_ci_required",
        "ephemeral_campaign_store_identity_pin_verified": True,
        "in_image_commit_lock_platform_verified": False,
    }

    inventory = report["inventory"]
    assert set(inventory) == {
        "capability_version",
        "declarations",
        "fingerprint",
        "supported_count",
        "total_count",
        "unsupported_count",
    }
    expected_supported = sum(
        declaration.state is TaskFenceCapabilityState.SUPPORTED
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    )
    assert inventory["total_count"] == len(
        TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    )
    assert inventory["supported_count"] == expected_supported
    assert inventory["unsupported_count"] == (
        len(TASK_FENCE_SELECTED_COHORT_CAPABILITIES) - expected_supported
    )
    assert len(inventory["declarations"]) == inventory["total_count"]
    assert inventory["declarations"] == [
        {
            "capability_id": declaration.capability_id,
            "capability_kind": declaration.kind.value,
            "capability_version": declaration.capability_version,
            "declaration_state": declaration.state.value,
        }
        for declaration in TASK_FENCE_SELECTED_COHORT_CAPABILITIES
    ]
    assert all(
        set(declaration)
        == {
            "capability_id",
            "capability_kind",
            "capability_version",
            "declaration_state",
        }
        for declaration in inventory["declarations"]
    )
    assert inventory["fingerprint"].startswith("sha256:")
    inventory_bytes = json.dumps(
        inventory["declarations"],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert inventory["fingerprint"] == (
        "sha256:" + hashlib.sha256(inventory_bytes).hexdigest()
    )
    assert report["observed_route_ids"] == [
        "provider:openai.chat.completions.create",
        "runtime:registered-tool-handoff",
    ]

    scenarios = report["scenarios"]
    assert all(
        set(scenario)
        == {
            "contract_match",
            "oracle",
            "physical_handoff_count",
            "record_handoff_binding_verified",
            "records",
            "scenario_id",
        }
        for scenario in scenarios
    )
    assert [scenario["scenario_id"] for scenario in scenarios] == [
        "current_authority",
        "missing_provenance",
        "store_unavailable",
        "registered_tool_current_authority",
    ]
    assert [scenario["physical_handoff_count"] for scenario in scenarios] == [
        1,
        1,
        1,
        1,
    ]
    assert all(scenario["contract_match"] is True for scenario in scenarios)
    assert all(
        scenario["record_handoff_binding_verified"] is True
        for scenario in scenarios
    )
    records = [
        record for scenario in scenarios for record in scenario["records"]
    ]
    assert len(records) == 6
    assert all(
        set(record)
        == {
            "decision_id",
            "decision_point",
            "generation_id",
            "invocation_id",
            "outcome",
            "reason_code",
            "route_id",
            "task_id",
        }
        for record in records
    )
    assert [record["decision_point"] for record in scenarios[0]["records"]] == [
        "admission",
        "authorization",
    ]
    assert [record["reason_code"] for record in scenarios[0]["records"]] == [
        DecisionReason.CURRENT_AUTHORITY.value,
        DecisionReason.CURRENT_AUTHORITY.value,
    ]
    assert scenarios[1]["records"][0]["reason_code"] == (
        DecisionReason.MISSING_PROVENANCE.value
    )
    assert scenarios[2]["records"][0]["reason_code"] == (
        DecisionReason.STORE_UNAVAILABLE.value
    )
    assert scenarios[2]["records"][0]["decision_id"] is None
    assert report["reviews"] == {
        "decision_id_missing_count": 1,
        "false_block": {"denominator": 2, "numerator": 0},
        "missing_provenance": {"denominator": 1, "numerator": 1},
        "store_unavailable": {"denominator": 1, "numerator": 1},
        "taskless_record_count": 1,
    }
    assert report["legacy_behavior"] == {
        "physical_handoff_count": 4,
        "suppressed_handoff_count": 0,
    }
    assert report["counters"] == sorted(
        report["counters"],
        key=lambda item: (
            item["route_id"],
            item["decision_point"],
            item["outcome"],
            item["reason_code"],
        ),
    )
    expected_counts = Counter(
        (
            record["route_id"],
            record["decision_point"],
            record["outcome"],
            record["reason_code"],
        )
        for record in records
    )
    assert report["counters"] == [
        {
            "count": count,
            "decision_point": decision_point,
            "outcome": outcome,
            "reason_code": reason_code,
            "route_id": route_id,
        }
        for (route_id, decision_point, outcome, reason_code), count in sorted(
            expected_counts.items()
        )
    ]
    assert all(
        set(counter)
        == {
            "count",
            "decision_point",
            "outcome",
            "reason_code",
            "route_id",
        }
        for counter in report["counters"]
    )
    assert report["limits"] == {
        "decision_records_per_scenario": MAX_CAMPAIGN_DECISION_RECORDS,
        "report_bytes": MAX_REPORT_BYTES,
    }
    assert set(report["measurements"]) == {
        "campaign_total_ns",
        "configured_ingress_ns",
        "model_handoffs_ns",
        "setup_ns",
        "state_storage_bytes",
    }
    assert all(value >= 0 for value in report["measurements"].values())
    assert report["measurements"]["state_storage_bytes"] > 0

    forbidden_keys = {
        "attempt_id",
        "credential",
        "exception",
        "invocation_fingerprint",
        "payload",
        "permit_id",
        "request",
        "response",
        "timestamp",
    }
    assert not forbidden_keys.intersection(_all_keys(report))
    assert b"task-fence-campaign-private-payload" not in encoded
    assert len(encoded) <= MAX_REPORT_BYTES
    assert encoded.endswith(b"\n")
    assert serialize_campaign_report(report) == encoded
    assert json.loads(encoded) == report


@pytest.mark.parametrize(
    "classification",
    ("missing", "changed", "fault", "malformed"),
)
def test_campaign_launch_route_failure_blocks_conformance_not_legacy_handoff(
    monkeypatch,
    caplog,
    classification,
):
    receipt = TestedArtifactReceipt(
        target_platform="linux/amd64",
        tested_artifact_commit=_COMMIT,
        tested_artifact_checksum=_ARTIFACT,
        dependency_lock_fingerprint="sha256:" + "c" * 64,
    )
    secret = "campaign-classifier-private-detail"

    def classify_launch_route(_policy, route):
        if classification == "missing":
            return None
        if classification == "changed":
            return TaskFenceLaunchRouteValidation(
                verified=False,
                reason="changed_reachable_route",
                route_id=route.route_id,
            )
        if classification == "fault":
            raise RuntimeError(secret)
        return object()

    monkeypatch.setattr(
        TaskFencePolicy,
        "classify_launch_route",
        classify_launch_route,
    )

    report = build_campaign_report(receipt)

    assert report["execution_complete"] is True
    assert report["scenario_contracts_match"] is False
    assert report["cohort_complete"] is False
    assert report["observed_route_ids"] == []
    assert report["legacy_behavior"] == {
        "physical_handoff_count": 4,
        "suppressed_handoff_count": 0,
    }
    assert all(
        scenario["physical_handoff_count"] == 1
        and scenario["contract_match"] is False
        for scenario in report["scenarios"]
    )
    assert secret not in caplog.text


@pytest.mark.parametrize(
    "classification",
    ("missing", "changed", "fault", "malformed"),
)
def test_registered_route_failure_preserves_other_route_and_all_handoffs(
    monkeypatch,
    caplog,
    classification,
):
    receipt = TestedArtifactReceipt(
        target_platform="linux/amd64",
        tested_artifact_commit=_COMMIT,
        tested_artifact_checksum=_ARTIFACT,
        dependency_lock_fingerprint="sha256:" + "c" * 64,
    )
    secret = "registered-classifier-private-detail"
    real_classify = TaskFencePolicy.classify_launch_route

    def classify_launch_route(policy, route):
        if route.route_id != campaign._REGISTERED_TOOL_ROUTE:
            return real_classify(policy, route)
        if classification == "missing":
            return None
        if classification == "changed":
            return TaskFenceLaunchRouteValidation(
                verified=False,
                reason="changed_reachable_route",
                route_id=route.route_id,
            )
        if classification == "fault":
            raise RuntimeError(secret)
        return object()

    monkeypatch.setattr(
        TaskFencePolicy,
        "classify_launch_route",
        classify_launch_route,
    )

    report = build_campaign_report(receipt)

    assert report["execution_complete"] is True
    assert report["scenario_contracts_match"] is False
    assert report["observed_route_ids"] == [
        "provider:openai.chat.completions.create"
    ]
    assert report["legacy_behavior"] == {
        "physical_handoff_count": 4,
        "suppressed_handoff_count": 0,
    }
    assert all(
        scenario["contract_match"] is True
        for scenario in report["scenarios"][:3]
    )
    registered = report["scenarios"][3]
    assert registered["scenario_id"] == "registered_tool_current_authority"
    assert registered["physical_handoff_count"] == 1
    assert registered["contract_match"] is False
    assert len(registered["records"]) == (2 if classification == "missing" else 0)
    assert secret not in caplog.text


def test_campaign_probe_latches_overflow_before_unbounded_projection(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    probe = CampaignPolicyProbe(
        TaskFencePolicy(db),
        route_id="provider:openai.chat.completions.create",
    )
    decision = DispatchDecision(
        outcome=DecisionOutcome.WOULD_BLOCK,
        reason=DecisionReason.MISSING_PROVENANCE,
    )
    try:
        for index in range(MAX_CAMPAIGN_DECISION_RECORDS + 1):
            operation = OperationDescriptor(
                invocation_id=f"tfiv_campaign_{index}",
                kind=OperationKind.MODEL,
                adapter="provider:openai.chat.completions.create",
                invocation_fingerprint=f"{index:064x}",
            )
            probe._append(
                decision_point="admission",
                envelope=None,
                operation=operation,
                decision=decision,
            )

        assert len(probe.records) == MAX_CAMPAIGN_DECISION_RECORDS
        with pytest.raises(
            CampaignRecordOverflow,
            match="campaign_record_limit",
        ):
            probe.assert_complete()
    finally:
        db.close()


def test_campaign_probe_bounds_launch_route_observations(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    probe = CampaignPolicyProbe(
        TaskFencePolicy(db),
        route_id="provider:openai.chat.completions.create",
    )
    route = next(
        route
        for route in TASK_FENCE_SELECTED_LAUNCH_MANIFEST.routes
        if route.route_id == "provider:openai.chat.completions.create"
    )
    try:
        for _ in range(MAX_CAMPAIGN_DECISION_RECORDS + 1):
            probe.classify_launch_route(route)

        assert len(probe.launch_route_observations) == (
            MAX_CAMPAIGN_DECISION_RECORDS
        )
        with pytest.raises(
            CampaignRecordOverflow,
            match="campaign_record_limit",
        ):
            probe.assert_complete()
    finally:
        db.close()


def test_campaign_contract_mismatch_does_not_invent_false_block(monkeypatch):
    receipt = TestedArtifactReceipt(
        target_platform="linux/amd64",
        tested_artifact_commit=_COMMIT,
        tested_artifact_checksum=_ARTIFACT,
        dependency_lock_fingerprint="sha256:" + "c" * 64,
    )
    monkeypatch.setattr(
        campaign,
        "_records_match",
        lambda *_args, **_kwargs: False,
    )

    report = campaign.build_campaign_report(receipt)

    assert report["execution_complete"] is True
    assert report["scenario_contracts_match"] is False
    assert report["observed_route_ids"] == []
    assert report["reviews"]["false_block"] == {
        "denominator": 2,
        "numerator": 0,
    }
    assert report["reviews"]["missing_provenance"]["numerator"] == 1
    assert report["reviews"]["store_unavailable"]["numerator"] == 1
    assert report["legacy_behavior"]["physical_handoff_count"] == 4


def test_campaign_post_handoff_record_invalidates_scenario_contract(monkeypatch):
    from agent import task_fence_provider

    receipt = TestedArtifactReceipt(
        target_platform="linux/amd64",
        tested_artifact_commit=_COMMIT,
        tested_artifact_checksum=_ARTIFACT,
        dependency_lock_fingerprint="sha256:" + "c" * 64,
    )
    original_finish = task_fence_provider._finish_task_fence_openai_chat_completion

    def finish_with_extra_record(*, policy, attempt_holder, response):
        original_finish(
            policy=policy,
            attempt_holder=attempt_holder,
            response=response,
        )
        policy._append(
            decision_point="authorization",
            envelope=None,
            operation=OperationDescriptor(
                invocation_id="tfiv_post_handoff_extra",
                kind=OperationKind.MODEL,
                adapter="provider:openai.chat.completions.create",
                invocation_fingerprint="d" * 64,
            ),
            decision=DispatchDecision(
                outcome=DecisionOutcome.WOULD_BLOCK,
                reason=DecisionReason.MISSING_PROVENANCE,
            ),
        )

    monkeypatch.setattr(
        task_fence_provider,
        "_finish_task_fence_openai_chat_completion",
        finish_with_extra_record,
    )

    report = campaign.build_campaign_report(receipt)

    assert report["execution_complete"] is True
    assert report["scenario_contracts_match"] is False
    assert report["observed_route_ids"] == [
        "runtime:registered-tool-handoff"
    ]
    assert all(
        not scenario["contract_match"] for scenario in report["scenarios"][:3]
    )
    assert report["scenarios"][3]["contract_match"] is True
    assert all(
        scenario["records"][-1]["invocation_id"]
        == "tfiv_post_handoff_extra"
        for scenario in report["scenarios"][:3]
    )


def test_campaign_registered_post_handoff_observation_invalidates_only_route(
    monkeypatch,
):
    from tools.registry import ToolRegistry

    receipt = TestedArtifactReceipt(
        target_platform="linux/amd64",
        tested_artifact_commit=_COMMIT,
        tested_artifact_checksum=_ARTIFACT,
        dependency_lock_fingerprint="sha256:" + "c" * 64,
    )
    real_normalize = ToolRegistry._normalize_handler_result
    registered_route = next(
        route
        for route in TASK_FENCE_SELECTED_LAUNCH_MANIFEST.routes
        if route.route_id == campaign._REGISTERED_TOOL_ROUTE
    )

    def normalize_with_extra_observation(registry, name, result):
        policy = campaign.current_task_fence_policy()
        assert isinstance(policy, CampaignPolicyProbe)
        policy.classify_launch_route(registered_route)
        return real_normalize(registry, name, result)

    monkeypatch.setattr(
        ToolRegistry,
        "_normalize_handler_result",
        normalize_with_extra_observation,
    )

    report = campaign.build_campaign_report(receipt)

    assert report["execution_complete"] is True
    assert report["scenario_contracts_match"] is False
    assert report["observed_route_ids"] == [
        "provider:openai.chat.completions.create"
    ]
    assert all(
        scenario["contract_match"] is True
        for scenario in report["scenarios"][:3]
    )
    assert report["scenarios"][3]["contract_match"] is False
    assert report["legacy_behavior"] == {
        "physical_handoff_count": 4,
        "suppressed_handoff_count": 0,
    }


@pytest.mark.parametrize(
    ("outcome", "expected"),
    (
        (DecisionOutcome.WOULD_BLOCK, True),
        (DecisionOutcome.HALT_DISPATCH, True),
        (DecisionOutcome.WOULD_ALLOW, False),
    ),
)
def test_campaign_block_review_counts_only_blocking_outcomes(outcome, expected):
    record = CampaignDecisionRecord(
        route_id="provider:openai.chat.completions.create",
        decision_point="admission",
        outcome=outcome.value,
        reason_code=DecisionReason.MISSING_PROVENANCE.value,
        invocation_id="tfiv_campaign",
        task_id=None,
        generation_id=None,
        decision_id=None,
    )

    assert campaign._has_blocking_outcome((record,)) is expected
    assert campaign._has_blocking_outcome(
        (record,),
        reason_code=DecisionReason.MISSING_PROVENANCE.value,
    ) is expected


def test_campaign_report_size_is_fail_closed():
    with pytest.raises(CampaignError, match="campaign_report_too_large") as exc:
        serialize_campaign_report({"oversized": "x" * MAX_REPORT_BYTES})

    assert exc.value.reason == "campaign_report_too_large"
