"""Contracts for producing an exact tested OCI artifact identity in CI."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "docker.yml"
DEPLOY_SITE_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "deploy-site.yml"


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


def _deploy_site_workflow() -> dict:
    return yaml.load(
        DEPLOY_SITE_WORKFLOW_PATH.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )


def _publish_steps() -> list[dict]:
    return _workflow()["jobs"]["publish"]["steps"]


def _fork_publish_steps() -> list[dict]:
    return _workflow()["jobs"]["publish-maintained-fork"]["steps"]


def _archive_steps() -> list[dict]:
    return _workflow()["jobs"]["archive-release-receipts"]["steps"]


def _step_where(predicate) -> dict:
    return next(step for step in _publish_steps() if predicate(step))


def _fork_step_where(predicate) -> dict:
    return next(step for step in _fork_publish_steps() if predicate(step))


def _archive_script() -> str:
    return next(
        step["run"] for step in _archive_steps() if "RELEASE_ID" in step.get("env", {})
    )


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


def test_maintained_fork_release_has_closed_ghcr_publish_authority() -> None:
    workflow = _workflow()
    publish = workflow["jobs"]["publish"]
    fork_publish = workflow["jobs"]["publish-maintained-fork"]
    merge = workflow["jobs"]["merge"]
    archive = workflow["jobs"]["archive-release-receipts"]
    fork_publish_condition = " ".join(fork_publish["if"].split())
    archive_condition = " ".join(archive["if"].split())

    assert workflow["env"]["IMAGE_NAME"] == "nousresearch/hermes-agent"
    assert workflow["permissions"] == {"contents": "read"}
    assert publish["if"] == (
        "github.repository == 'NousResearch/hermes-agent' && "
        "(github.event_name == 'push' && github.ref == 'refs/heads/main' || "
        "github.event_name == 'release')"
    )
    assert fork_publish_condition == (
        "github.repository == 'delin/hermes-agent' && "
        "github.event_name == 'release' && "
        "github.event.action == 'published' && "
        "github.ref == format('refs/tags/{0}', github.event.release.tag_name)"
    )
    assert merge["if"] == (
        "github.repository == 'NousResearch/hermes-agent' && "
        "(github.event_name == 'push' && github.ref == 'refs/heads/main' || "
        "github.event_name == 'release')"
    )
    assert archive_condition == (
        "github.repository == 'delin/hermes-agent' && "
        "github.event_name == 'release' && "
        "github.event.action == 'published' && "
        "github.ref == format('refs/tags/{0}', github.event.release.tag_name)"
    )

    assert "permissions" not in publish
    assert fork_publish["permissions"] == {
        "contents": "read",
        "packages": "write",
    }
    assert archive["permissions"] == {
        "actions": "read",
        "contents": "write",
    }
    assert "packages" not in archive["permissions"]
    assert fork_publish["env"] == {"IMAGE_NAME": "ghcr.io/delin/hermes-agent"}
    assert fork_publish["strategy"] == publish["strategy"]
    assert fork_publish["runs-on"] == publish["runs-on"]
    assert fork_publish["timeout-minutes"] == publish["timeout-minutes"]

    publish_steps = _publish_steps()
    fork_steps = _fork_publish_steps()
    checkout = next(
        step
        for step in fork_steps
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["persist-credentials"] is False

    upstream_logins = [
        step
        for step in publish_steps
        if step.get("uses", "").startswith("docker/login-action@")
    ]
    fork_logins = [
        step
        for step in fork_steps
        if step.get("uses", "").startswith("docker/login-action@")
    ]
    assert len(upstream_logins) == 1
    assert len(fork_logins) == 1
    docker_hub_login = upstream_logins[0]
    ghcr_login = fork_logins[0]
    assert docker_hub_login["with"] == {
        "username": "${{ secrets.DOCKERHUB_USERNAME }}",
        "password": "${{ secrets.DOCKERHUB_TOKEN }}",
    }
    assert "if" not in docker_hub_login
    assert "if" not in ghcr_login
    assert ghcr_login["with"]["username"] == "${{ github.actor }}"
    assert ghcr_login["with"]["password"] == "${{ github.token }}"
    upstream_text = json.dumps(publish, sort_keys=True)
    fork_text = json.dumps(fork_publish, sort_keys=True)
    archive_text = json.dumps(archive, sort_keys=True)
    assert "ghcr.io" not in upstream_text
    assert "github.token" not in upstream_text
    assert "DOCKERHUB_" not in fork_text
    assert "secrets." not in fork_text
    assert "docker/login-action@" not in archive_text
    assert "DOCKERHUB_" not in archive_text
    assert "secrets." not in archive_text


def test_maintained_fork_release_cannot_trigger_upstream_site_deploy() -> None:
    deploy_vercel = _deploy_site_workflow()["jobs"]["deploy-vercel"]
    condition = " ".join(deploy_vercel["if"].split())

    assert condition == (
        "github.repository == 'NousResearch/hermes-agent' && "
        "(github.event_name == 'release' || "
        "github.event_name == 'workflow_dispatch')"
    )
    assert "VERCEL_DEPLOY_HOOK" in json.dumps(deploy_vercel, sort_keys=True)


def test_private_ghcr_credentials_end_before_exact_tests() -> None:
    steps = _fork_publish_steps()
    ghcr_login = next(
        step for step in steps if step.get("with", {}).get("registry") == "ghcr.io"
    )
    push = _fork_step_where(lambda step: step.get("id") == "push")
    pull = _fork_step_where(
        lambda step: (
            step.get("uses") == "./.github/actions/retry"
            and step.get("with", {}).get("command", "").startswith("docker pull ")
        )
    )
    ghcr_logout = _fork_step_where(
        lambda step: step.get("run") == "docker logout ghcr.io"
    )
    exact_test = _fork_step_where(
        lambda step: "HERMES_TEST_IMAGE" in step.get("env", {})
    )
    receipt = _fork_step_where(lambda step: "ARTIFACT_DIGEST" in step.get("env", {}))

    assert steps.index(ghcr_login) < steps.index(push)
    assert steps.index(push) < steps.index(pull)
    assert steps.index(pull) < steps.index(ghcr_logout)
    assert steps.index(ghcr_logout) < steps.index(exact_test)
    assert steps.index(exact_test) < steps.index(receipt)
    assert all("TOKEN" not in key for key in exact_test["env"])
    assert "github.token" not in json.dumps(exact_test, sort_keys=True)

    upstream_push = _step_where(lambda step: step.get("id") == "push")
    upstream_pull = _step_where(
        lambda step: (
            step.get("uses") == "./.github/actions/retry"
            and step.get("with", {}).get("command", "").startswith("docker pull ")
        )
    )
    upstream_test = _step_where(lambda step: "HERMES_TEST_IMAGE" in step.get("env", {}))
    upstream_receipt = _step_where(
        lambda step: "ARTIFACT_DIGEST" in step.get("env", {})
    )
    assert push["uses"] == upstream_push["uses"]
    assert push["with"] == upstream_push["with"]
    assert pull["uses"] == upstream_pull["uses"]
    assert pull["with"] == upstream_pull["with"]
    assert exact_test["env"] == upstream_test["env"]
    assert exact_test["run"] == upstream_test["run"]
    assert receipt["env"] == upstream_receipt["env"]
    assert receipt["run"] == upstream_receipt["run"]


def test_fork_campaign_is_receipt_bound_isolated_and_separately_retained() -> None:
    steps = _fork_publish_steps()
    exact_test = _fork_step_where(
        lambda step: "HERMES_TEST_IMAGE" in step.get("env", {})
    )
    receipt_export = _fork_step_where(
        lambda step: "ARTIFACT_DIGEST" in step.get("env", {})
    )
    campaign = _fork_step_where(
        lambda step: step.get("name")
        == "Run receipt-bound Task Fence shadow campaign"
    )
    receipt_upload = _fork_step_where(
        lambda step: step.get("with", {}).get("name")
        == "tested-artifact-${{ matrix.arch }}"
    )
    report_upload = _fork_step_where(
        lambda step: step.get("with", {}).get("name")
        == "task-fence-campaign-${{ matrix.arch }}"
    )
    conformance_gate = _fork_step_where(
        lambda step: step.get("name")
        == "Gate Task Fence campaign conformance"
    )

    assert steps.index(exact_test) < steps.index(receipt_export)
    assert steps.index(receipt_export) < steps.index(campaign)
    assert steps.index(campaign) < steps.index(receipt_upload)
    assert steps.index(receipt_upload) < steps.index(report_upload)
    assert steps.index(report_upload) < steps.index(conformance_gate)

    immutable_ref = "${{ env.IMAGE_NAME }}@${{ steps.push.outputs.digest }}"
    assert campaign["env"] == {
        "IMAGE_REF": immutable_ref,
        "RECEIPT_PATH": "/tmp/tested-artifact/${{ matrix.arch }}.json",
        "REPORT_PATH": "/tmp/task-fence-campaign/${{ matrix.arch }}.json",
    }
    script = campaign["run"]
    for fragment in (
        "set -Eeuo pipefail",
        "docker run --rm --pull never",
        "--network none",
        "--read-only",
        "--user 10000:10000",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m",
        (
            "--tmpfs /opt/data:rw,noexec,nosuid,nodev,size=32m,"
            "mode=0700,uid=10000,gid=10000"
        ),
        "source=${RECEIPT_PATH}",
        "target=/run/task-fence-receipt.json,readonly",
        "--entrypoint /opt/hermes/.venv/bin/python",
        '"$IMAGE_REF"',
        "-m scripts.task_fence_campaign",
        "--receipt /run/task-fence-receipt.json",
        '> "$REPORT_PATH"',
        'test -s "$REPORT_PATH"',
        "jq --exit-status --slurp",
        '--slurpfile receipt "$RECEIPT_PATH"',
        ".[0].schema == \"hermes.task-fence.campaign-report/v1\"",
        ".[0].execution_complete == true",
        ".[0].cohort_complete == false",
        (
            ".[0].artifact_binding."
            "in_image_commit_lock_platform_verified == true"
        ),
        ".[0].receipt == $receipt[0]",
    ):
        assert fragment in script

    campaign_text = json.dumps(campaign, sort_keys=True)
    assert "github.token" not in campaign_text
    assert "secrets." not in campaign_text
    assert "API_KEY" not in campaign_text
    assert "TOKEN" not in campaign_text

    uploads = [
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    upload_names = [step.get("with", {}).get("name") for step in uploads]
    assert upload_names.count("tested-artifact-${{ matrix.arch }}") == 1
    assert upload_names.count("task-fence-campaign-${{ matrix.arch }}") == 1
    assert report_upload["with"] == {
        "name": "task-fence-campaign-${{ matrix.arch }}",
        "path": "/tmp/task-fence-campaign/${{ matrix.arch }}.json",
        "if-no-files-found": "error",
        "retention-days": 90,
    }
    assert "tested-artifact" not in report_upload["with"]["name"]
    assert "tested-artifact" not in report_upload["with"]["path"]
    assert conformance_gate["env"] == {
        "REPORT_PATH": "/tmp/task-fence-campaign/${{ matrix.arch }}.json"
    }
    gate_script = conformance_gate["run"]
    assert "set -Eeuo pipefail" in gate_script
    assert "jq --exit-status --slurp" in gate_script
    assert "length == 1" in gate_script
    assert ".[0].scenario_contracts_match == true" in gate_script
    assert '"$REPORT_PATH"' in gate_script
    assert "github.token" not in json.dumps(conformance_gate, sort_keys=True)
    assert "secrets." not in json.dumps(conformance_gate, sort_keys=True)

    syntax = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    gate_syntax = subprocess.run(
        ["bash", "-n"],
        input=gate_script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert gate_syntax.returncode == 0, gate_syntax.stderr


def test_release_receipts_are_tag_bound_and_set_once() -> None:
    workflow = _workflow()
    archive = workflow["jobs"]["archive-release-receipts"]
    steps = _archive_steps()
    checkout = next(
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    )
    download = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/download-artifact@")
    )
    archive_step = next(step for step in steps if "RELEASE_ID" in step.get("env", {}))
    script = archive_step["run"]

    assert archive["needs"] == ["publish-maintained-fork"]
    assert checkout["with"] == {
        "ref": "${{ github.event.release.tag_name }}",
        "fetch-depth": 0,
        "persist-credentials": False,
    }
    assert download["with"] == {
        "path": "/tmp/tested-artifacts",
        "pattern": "tested-artifact-*",
        "merge-multiple": True,
    }
    assert archive_step["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "RELEASE_ID": "${{ github.event.release.id }}",
        "RELEASE_TAG": "${{ github.event.release.tag_name }}",
        "TESTED_COMMIT": "${{ github.sha }}",
    }
    assert "receipt_files=(amd64.json arm64.json)" in script
    assert "expected_platforms=(linux/amd64 linux/arm64)" in script
    assert "refs/tags/${RELEASE_TAG}^{commit}" in script
    assert '"repos/${GITHUB_REPOSITORY}/releases/${RELEASE_ID}"' in script
    assert ".tested_artifact_commit" in script
    assert "hermes.task-fence.tested-artifact-receipt/v1" in script
    assert "cmp --silent" in script
    assert "missing_receipts=()" in script
    assert script.index("missing_receipts=()") < script.index(
        'gh release upload "$RELEASE_TAG"'
    )
    assert "--clobber" not in script
    assert "docker " not in script

    syntax = subprocess.run(
        ["bash", "-n"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def _write_valid_receipts(receipt_dir: Path, commit: str) -> None:
    lock = f"sha256:{'c' * 64}"
    for arch, platform, digest_char in (
        ("amd64", "linux/amd64", "a"),
        ("arm64", "linux/arm64", "b"),
    ):
        receipt = {
            "schema": "hermes.task-fence.tested-artifact-receipt/v1",
            "target_platform": platform,
            "tested_artifact_commit": commit,
            "tested_artifact_checksum": f"sha256:{digest_char * 64}",
            "dependency_lock_fingerprint": lock,
        }
        (receipt_dir / f"{arch}.json").write_text(
            json.dumps(receipt, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_fake_gh(fake_bin: Path) -> None:
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        f"""#!{sys.executable}
import json
import os
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
asset_dir = Path(os.environ["FAKE_GH_ASSET_DIR"])
release_id = int(os.environ["RELEASE_ID"])
release_tag = os.environ["RELEASE_TAG"]
asset_dir.mkdir(parents=True, exist_ok=True)

if args[0] == "api":
    endpoint = args[-1]
    assets = sorted(asset_dir.iterdir())
    rows = [
        {{"id": index, "name": path.name}}
        for index, path in enumerate(assets, start=1)
    ]
    if endpoint.endswith(f"/releases/{{release_id}}"):
        print(json.dumps({{"id": release_id, "tag_name": release_tag, "draft": False}}))
    elif endpoint.endswith(f"/releases/{{release_id}}/assets?per_page=100"):
        print(json.dumps([rows]))
    elif "/releases/assets/" in endpoint:
        asset_id = int(endpoint.rsplit("/", 1)[1])
        try:
            path = assets[asset_id - 1]
        except IndexError:
            raise SystemExit(1)
        sys.stdout.buffer.write(path.read_bytes())
    else:
        raise SystemExit(f"unexpected gh api endpoint: {{endpoint}}")
elif args[:2] == ["release", "upload"]:
    repo_index = args.index("--repo")
    for source_name in args[3:repo_index]:
        source = Path(source_name)
        target = asset_dir / source.name
        if target.exists():
            raise SystemExit(1)
        shutil.copyfile(source, target)
else:
    raise SystemExit(f"unexpected gh invocation: {{args}}")
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)


def _run_archive_script(
    script: str,
    *,
    workspace: Path,
    receipt_dir: Path,
    fake_bin: Path,
    asset_dir: Path,
    runner_temp: Path,
    release_tag: str,
    tested_commit: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({
        "FAKE_GH_ASSET_DIR": str(asset_dir),
        "GH_TOKEN": "test-token",
        "GITHUB_REPOSITORY": "delin/hermes-agent",
        "GITHUB_WORKSPACE": str(workspace),
        "PATH": f"{fake_bin}:{env['PATH']}",
        "RELEASE_ID": "42",
        "RELEASE_TAG": release_tag,
        "RUNNER_TEMP": str(runner_temp),
        "TESTED_COMMIT": tested_commit,
    })
    return subprocess.run(
        ["bash"],
        input=script,
        cwd=receipt_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_release_archive_real_path_is_set_once(tmp_path: Path) -> None:
    assert shutil.which("jq") is not None, "workflow archive contract requires jq"
    workspace = tmp_path / "workspace"
    receipt_dir = tmp_path / "receipts"
    fake_bin = tmp_path / "bin"
    asset_dir = tmp_path / "assets"
    runner_temp = tmp_path / "runner"
    for path in (workspace, receipt_dir, fake_bin, asset_dir, runner_temp):
        path.mkdir()

    subprocess.run(["git", "init", "--quiet"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "task-fence@example.invalid"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Task Fence Test"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "tag.gpgsign", "false"],
        cwd=workspace,
        check=True,
    )
    (workspace / "tracked").write_text("tested\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "--message", "tested"],
        cwd=workspace,
        check=True,
    )
    tested_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    release_tag = "v2026.07.30"
    subprocess.run(
        ["git", "tag", "--annotate", release_tag, "--message", "release"],
        cwd=workspace,
        check=True,
    )
    _write_valid_receipts(receipt_dir, tested_commit)
    _write_fake_gh(fake_bin)
    script = _archive_script()

    mismatch_assets = tmp_path / "mismatch-assets"
    mismatch_assets.mkdir()
    mismatch = _run_archive_script(
        script,
        workspace=workspace,
        receipt_dir=receipt_dir,
        fake_bin=fake_bin,
        asset_dir=mismatch_assets,
        runner_temp=runner_temp,
        release_tag=release_tag,
        tested_commit="d" * 40,
    )
    assert mismatch.returncode != 0
    assert "release tag does not resolve to the tested commit" in mismatch.stdout
    assert not any(mismatch_assets.iterdir())

    first = _run_archive_script(
        script,
        workspace=workspace,
        receipt_dir=receipt_dir,
        fake_bin=fake_bin,
        asset_dir=asset_dir,
        runner_temp=runner_temp,
        release_tag=release_tag,
        tested_commit=tested_commit,
    )
    assert first.returncode == 0, first.stderr
    assert sorted(path.name for path in asset_dir.iterdir()) == [
        "amd64.json",
        "arm64.json",
    ]

    replay = _run_archive_script(
        script,
        workspace=workspace,
        receipt_dir=receipt_dir,
        fake_bin=fake_bin,
        asset_dir=asset_dir,
        runner_temp=runner_temp,
        release_tag=release_tag,
        tested_commit=tested_commit,
    )
    assert replay.returncode == 0, replay.stderr

    (asset_dir / "arm64.json").write_text("different\n", encoding="utf-8")
    collision = _run_archive_script(
        script,
        workspace=workspace,
        receipt_dir=receipt_dir,
        fake_bin=fake_bin,
        asset_dir=asset_dir,
        runner_temp=runner_temp,
        release_tag=release_tag,
        tested_commit=tested_commit,
    )
    assert collision.returncode != 0
    assert "already exists with different bytes: arm64.json" in collision.stdout
