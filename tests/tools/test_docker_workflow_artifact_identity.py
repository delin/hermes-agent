"""Contracts for producing an exact tested OCI artifact identity in CI."""

import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "docker.yml"


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise AssertionError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _workflow() -> dict:
    return yaml.load(
        WORKFLOW_PATH.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )


def _publish_steps() -> list[dict]:
    return _workflow()["jobs"]["publish"]["steps"]


def _step_where(predicate) -> dict:
    return next(step for step in _publish_steps() if predicate(step))


def test_publish_tests_the_exact_pushed_descriptor_before_export() -> None:
    steps = _publish_steps()
    push = _step_where(lambda step: step.get("id") == "push")
    pull = _step_where(
        lambda step: (
            step.get("uses") == "./.github/actions/retry"
            and step.get("with", {}).get("command", "").startswith("docker pull ")
        )
    )
    logout = _step_where(lambda step: step.get("run") == "docker logout")
    exact_test = _step_where(lambda step: "HERMES_TEST_IMAGE" in step.get("env", {}))
    receipt = _step_where(lambda step: "ARTIFACT_DIGEST" in step.get("env", {}))

    assert steps.index(push) < steps.index(logout)
    assert steps.index(logout) < steps.index(pull)
    assert steps.index(pull) < steps.index(exact_test)
    assert steps.index(exact_test) < steps.index(receipt)

    build_steps = [
        step
        for step in steps
        if step.get("uses", "").startswith("docker/build-push-action@")
    ]
    assert build_steps == [push]
    assert "push-by-digest=true" in push["with"]["outputs"]
    assert "name-canonical=true" in push["with"]["outputs"]
    assert "push=true" in push["with"]["outputs"]

    immutable_ref = "${{ env.IMAGE_NAME }}@${{ steps.push.outputs.digest }}"
    assert pull["with"]["command"] == (
        f'docker pull --platform "${{{{ matrix.platform }}}}" "{immutable_ref}"'
    )

    assert exact_test["env"]["HERMES_TEST_IMAGE"] == immutable_ref
    assert exact_test["run"] == (
        "scripts/run_tests.sh tests/docker/ --file-timeout 600"
    )


def test_receipt_binds_commit_raw_lock_and_platform_to_pushed_digest() -> None:
    receipt = _step_where(lambda step: "ARTIFACT_DIGEST" in step.get("env", {}))
    env = receipt["env"]
    script = receipt["run"]

    assert env["ARTIFACT_DIGEST"] == "${{ steps.push.outputs.digest }}"
    assert env["TESTED_COMMIT"] == "${{ github.sha }}"
    assert env["TARGET_PLATFORM"] == "${{ matrix.platform }}"
    assert env["IMAGE_REF"] == (
        "${{ env.IMAGE_NAME }}@${{ steps.push.outputs.digest }}"
    )

    assert "git rev-parse --verify 'HEAD^{commit}'" in script
    assert 'git cat-file blob "${TESTED_COMMIT}:uv.lock"' in script
    assert 'cmp --silent "$committed_lock" uv.lock' in script
    assert 'sha256sum --binary "$committed_lock"' in script
    assert "/opt/hermes/.hermes_build_sha" in script
    assert "/opt/hermes/uv.lock" in script
    assert "docker image inspect --format '{{.Os}}/{{.Architecture}}'" in script
    assert '[[ "$checkout_commit" != "$TESTED_COMMIT" ]]' in script
    assert '[[ "$image_commit" != "$TESTED_COMMIT" ]]' in script
    assert '[[ "$image_lock_hex" != "$lock_hex" ]]' in script
    assert '[[ "$image_platform" != "$TARGET_PLATFORM" ]]' in script
    assert '"hermes.task-fence.tested-artifact-receipt/v1"' in script
    for field in (
        "tested_artifact_commit",
        "tested_artifact_checksum",
        "dependency_lock_fingerprint",
        "target_platform",
    ):
        assert field in script

    assert ":main" not in script
    assert ":latest" not in script

    syntax = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_only_tested_receipts_feed_the_existing_manifest_merge() -> None:
    steps = _publish_steps()
    exact_test = _step_where(lambda step: "HERMES_TEST_IMAGE" in step.get("env", {}))
    receipt_export = _step_where(lambda step: "ARTIFACT_DIGEST" in step.get("env", {}))
    receipt_upload = _step_where(
        lambda step: (
            step.get("with", {}).get("name") == "tested-artifact-${{ matrix.arch }}"
        )
    )

    assert steps.index(exact_test) < steps.index(receipt_upload)
    assert steps.index(receipt_export) < steps.index(receipt_upload)
    assert (
        len([
            step
            for step in steps
            if step.get("uses", "").startswith("actions/upload-artifact@")
        ])
        == 1
    )

    assert receipt_upload["with"]["path"] == (
        "/tmp/tested-artifact/${{ matrix.arch }}.json"
    )
    assert receipt_upload["with"]["if-no-files-found"] == "error"

    workflow = _workflow()
    merge = workflow["jobs"]["merge"]
    assert merge["needs"] == ["publish"]
    downloads = [
        step
        for step in merge["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert len(downloads) == 1
    download = downloads[0]
    assert download["with"]["pattern"] == "tested-artifact-*"
    merge_script = next(
        step["run"]
        for step in merge["steps"]
        if "imagetools create" in step.get("run", "")
    )
    assert ".tested_artifact_checksum" in merge_script
    assert "sha256:${digest_file}" not in merge_script
    assert "hermes.task-fence.tested-artifact-receipt/v1" in merge_script
    assert "receipt_files=(amd64.json arm64.json)" in merge_script
    assert "expected_platforms=(linux/amd64 linux/arm64)" in merge_script
    assert '"${#downloaded_receipts[@]}" -ne "${#receipt_files[@]}"' in merge_script
    assert "dependency_lock_fingerprint" in merge_script
    assert "target_platform" in merge_script
    syntax = subprocess.run(
        ["bash", "-n"],
        input=merge_script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "/tmp/digests" not in workflow_text
    assert "pattern: digest-*" not in workflow_text


def test_untrusted_build_lane_never_receives_publish_authority() -> None:
    build = _workflow()["jobs"]["build"]
    text = yaml.safe_dump(build)

    assert "github.event_name == 'pull_request'" in build["if"]
    assert "environment" not in build
    assert "docker/login-action" not in text
    assert "DOCKERHUB_TOKEN" not in text
    assert "secrets." not in text
    assert "push-by-digest=true" not in text
    assert "hermes.task-fence.tested-artifact-receipt/v1" not in text
    for step in build["steps"]:
        if not step.get("uses", "").startswith("docker/build-push-action@"):
            continue
        assert step.get("with", {}).get("push") not in (True, "true")
        assert "push=true" not in step.get("with", {}).get("outputs", "")

    publish = _workflow()["jobs"]["publish"]
    assert "needs" not in publish
    assert build["timeout-minutes"] == 45
    assert publish["timeout-minutes"] == 60
