"""Exercise the receipt-bound Task Fence campaign inside the runtime image."""

from __future__ import annotations

import json
import re
import stat
import subprocess
from pathlib import Path

from scripts.task_fence_campaign import REPORT_SCHEMA
from scripts.task_fence_launcher import RECEIPT_SCHEMA


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _image_probe(
    image: str,
    entrypoint: str,
    path: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--entrypoint",
            entrypoint,
            image,
            path,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _campaign_command(
    image: str,
    receipt_path: Path,
    extra_mounts: list[str],
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--user",
        "10000:10000",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--tmpfs",
        (
            "/opt/data:rw,noexec,nosuid,nodev,size=32m,"
            "mode=0700,uid=10000,gid=10000"
        ),
        "--mount",
        (
            f"type=bind,source={receipt_path},"
            "target=/run/task-fence-receipt.json,readonly"
        ),
        *extra_mounts,
        "--entrypoint",
        "/opt/hermes/.venv/bin/python",
        image,
        "-m",
        "scripts.task_fence_campaign",
        "--receipt",
        "/run/task-fence-receipt.json",
    ]


def test_campaign_runs_in_exact_read_only_image(
    built_image: str,
    tmp_path: Path,
) -> None:
    metadata = json.loads(
        subprocess.run(
            ["docker", "image", "inspect", built_image],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    )[0]
    target_platform = f"{metadata['Os']}/{metadata['Architecture']}"
    artifact_checksum = (
        "sha256:" + built_image.rsplit("@sha256:", 1)[1]
        if "@sha256:" in built_image
        else metadata["Id"]
    )
    assert _DIGEST_RE.fullmatch(artifact_checksum)

    build_probe = _image_probe(
        built_image,
        "cat",
        "/opt/hermes/.hermes_build_sha",
    )
    extra_mounts: list[str] = []
    if build_probe.returncode == 0:
        tested_commit = build_probe.stdout.strip()
    else:
        tested_commit = "a" * 40
        build_commit_path = tmp_path / "build-commit"
        build_commit_path.write_text(tested_commit + "\n", encoding="ascii")
        build_commit_path.chmod(0o444)
        extra_mounts.extend([
            "--mount",
            (
                f"type=bind,source={build_commit_path},"
                "target=/opt/hermes/.hermes_build_sha,readonly"
            ),
        ])
    assert _COMMIT_RE.fullmatch(tested_commit)
    assert set(tested_commit) != {"0"}

    lock_probe = _image_probe(
        built_image,
        "sha256sum",
        "/opt/hermes/uv.lock",
    )
    assert lock_probe.returncode == 0, lock_probe.stderr
    lock_hex = lock_probe.stdout.split(maxsplit=1)[0]
    assert re.fullmatch(r"[0-9a-f]{64}", lock_hex)

    receipt = {
        "dependency_lock_fingerprint": f"sha256:{lock_hex}",
        "schema": RECEIPT_SCHEMA,
        "target_platform": target_platform,
        "tested_artifact_checksum": artifact_checksum,
        "tested_artifact_commit": tested_commit,
    }
    receipt_path = tmp_path / "tested-artifact.json"
    receipt_path.write_text(
        json.dumps(receipt, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)

    result = subprocess.run(
        _campaign_command(built_image, receipt_path, extra_mounts),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, (
        f"campaign failed: stderr={result.stderr[-2000:]!r} "
        f"stdout={result.stdout[-2000:]!r}"
    )
    report = json.loads(result.stdout)
    assert report["schema"] == REPORT_SCHEMA
    assert report["execution_complete"] is True
    assert report["scenario_contracts_match"] is True
    assert report["cohort_complete"] is False
    assert report["receipt"] == receipt
    assert report["artifact_binding"][
        "in_image_commit_lock_platform_verified"
    ] is True
    assert all(
        scenario["contract_match"] is True
        and scenario["record_handoff_binding_verified"] is True
        for scenario in report["scenarios"]
    )
    assert report["observed_route_ids"] == [
        "provider:openai.chat.completions.create",
        "runtime:registered-tool-handoff",
    ]
    assert report["legacy_behavior"] == {
        "physical_handoff_count": 4,
        "suppressed_handoff_count": 0,
    }
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert "task-fence-campaign-private-payload" not in result.stdout

    mismatched = dict(receipt)
    wrong_first = "0" if lock_hex[0] != "0" else "1"
    mismatched["dependency_lock_fingerprint"] = (
        f"sha256:{wrong_first}{lock_hex[1:]}"
    )
    mismatch_path = tmp_path / "mismatched-receipt.json"
    mismatch_path.write_text(
        json.dumps(mismatched, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mismatch_path.chmod(0o444)

    mismatch = subprocess.run(
        _campaign_command(built_image, mismatch_path, extra_mounts),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert mismatch.returncode == 2
    assert mismatch.stdout == ""
    assert mismatch.stderr == (
        "task-fence campaign failed: dependency_lock_mismatch\n"
    )
    assert stat.S_IMODE(mismatch_path.stat().st_mode) == 0o444
