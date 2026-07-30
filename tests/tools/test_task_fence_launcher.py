from __future__ import annotations

import copy
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "task_fence_launcher.py"
SPEC = importlib.util.spec_from_file_location("task_fence_launcher", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

COMMIT = "c" * 40
ARTIFACT_DIGEST = "sha256:" + "a" * 64
LOCK_DIGEST = "sha256:" + "b" * 64
IMAGE_ID = "sha256:" + "d" * 64
CONTAINER_ID = "e" * 64
PLATFORM = "linux/amd64"


def _receipt_payload() -> dict[str, str]:
    return {
        "schema": launcher.RECEIPT_SCHEMA,
        "target_platform": PLATFORM,
        "tested_artifact_commit": COMMIT,
        "tested_artifact_checksum": ARTIFACT_DIGEST,
        "dependency_lock_fingerprint": LOCK_DIGEST,
    }


def _receipt_bytes(payload: dict[str, str] | None = None) -> bytes:
    return (
        json.dumps(payload or _receipt_payload(), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _container_inspection(
    *,
    hermes_home: Path,
    receipt_path: Path,
    command: Sequence[str] = ("gateway", "run"),
) -> list[dict[str, Any]]:
    metadata = hermes_home.stat()
    image_ref = f"{launcher.IMAGE_NAME}@{ARTIFACT_DIGEST}"
    return [
        {
            "Id": CONTAINER_ID,
            "State": {"Status": "created", "Running": False},
            "Config": {
                "Image": image_ref,
                "Cmd": list(command),
                "Env": [
                    f"HERMES_UID={metadata.st_uid}",
                    f"HERMES_GID={metadata.st_gid}",
                ],
            },
            "Image": IMAGE_ID,
            "HostConfig": {
                "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                "NetworkMode": "host",
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": str(hermes_home),
                    "Destination": launcher.CONTAINER_HERMES_HOME,
                    "RW": True,
                },
                {
                    "Type": "bind",
                    "Source": str(receipt_path),
                    "Destination": launcher.CONTAINER_RECEIPT_PATH,
                    "RW": False,
                },
            ],
        }
    ]


class FakeTransport:
    def __init__(self, *, hermes_home: Path, receipt_path: Path) -> None:
        self.raw_receipt = _receipt_bytes()
        self.calls: list[tuple[list[str], Mapping[str, str] | None]] = []
        self.assets = [
            {
                "id": 7,
                "name": "amd64.json",
                "size": len(self.raw_receipt),
                "state": "uploaded",
            }
        ]
        self.image_inspection: Any = [
            {"Id": IMAGE_ID, "Os": "linux", "Architecture": "amd64"}
        ]
        self.container_inspection: Any = _container_inspection(
            hermes_home=hermes_home,
            receipt_path=receipt_path,
        )

    def __call__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout: int = 60,
    ) -> bytes:
        del timeout
        command = list(argv)
        self.calls.append((command, env))
        if command[0] == "gh":
            endpoint = command[-1]
            if endpoint.endswith("/releases/assets/7"):
                return self.raw_receipt
            if endpoint.endswith("/releases/42"):
                return json.dumps({
                    "id": 42,
                    "tag_name": "v1.2.3",
                    "draft": False,
                    "published_at": "2026-07-30T00:00:00Z",
                }).encode()
            if endpoint.endswith("/releases/42/assets?per_page=100"):
                return json.dumps([self.assets]).encode()
            if endpoint.endswith("/git/ref/tags/v1.2.3"):
                return json.dumps({
                    "ref": "refs/tags/v1.2.3",
                    "object": {"type": "commit", "sha": COMMIT},
                }).encode()
        if command[:3] == ["docker", "image", "pull"]:
            return b""
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps(self.image_inspection).encode()
        if command[:3] == ["docker", "container", "create"]:
            return f"{CONTAINER_ID}\n".encode()
        if command[:3] == ["docker", "container", "inspect"]:
            return json.dumps(self.container_inspection).encode()
        if command[:3] == ["docker", "container", "start"]:
            return f"{CONTAINER_ID}\n".encode()
        raise AssertionError(f"unexpected command: {command}")


def _run_fake_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[FakeTransport, Path, Path]:
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    receipt_path = tmp_path / "trusted" / "tested-artifact.json"
    fake = FakeTransport(
        hermes_home=hermes_home,
        receipt_path=receipt_path,
    )
    monkeypatch.setattr(launcher, "_run", fake)
    monkeypatch.setattr(launcher, "_require_root", lambda: None)
    monkeypatch.setattr(launcher, "HOST_RECEIPT_PATH", receipt_path)
    monkeypatch.setenv("GH_TOKEN", "host-only-test-token")
    monkeypatch.setenv("CR_PAT", "host-only-registry-token")
    result = launcher.launch(
        release_id=42,
        platform=PLATFORM,
        hermes_home=hermes_home,
    )
    assert result == CONTAINER_ID
    return fake, hermes_home, receipt_path


def test_launch_acquires_exact_receipt_then_inspects_before_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake, hermes_home, receipt_path = _run_fake_launch(monkeypatch, tmp_path)

    assert receipt_path.read_bytes() == fake.raw_receipt
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    commands = [call for call, _ in fake.calls]
    assert [command[0] for command in commands] == [
        "gh",
        "gh",
        "gh",
        "gh",
        "docker",
        "docker",
        "docker",
        "docker",
        "docker",
    ]
    image_ref = f"{launcher.IMAGE_NAME}@{ARTIFACT_DIGEST}"
    assert commands[4] == [
        "docker",
        "image",
        "pull",
        "--platform",
        PLATFORM,
        image_ref,
    ]
    create = commands[6]
    assert create[:3] == ["docker", "container", "create"]
    assert "--restart" in create
    assert create[create.index("--restart") + 1] == "no"
    assert create[create.index("--pull") + 1] == "never"
    assert create[create.index("--platform") + 1] == PLATFORM
    assert (
        f"type=bind,src={receipt_path},dst={launcher.CONTAINER_RECEIPT_PATH},readonly"
    ) in create
    assert f"type=bind,src={hermes_home},dst=/opt/data" in create
    assert image_ref in create
    assert commands[7] == ["docker", "container", "inspect", CONTAINER_ID]
    assert commands[8] == ["docker", "container", "start", CONTAINER_ID]
    assert not any(
        command[1:3] in (["container", "run"], ["container", "rm"])
        for command in commands
        if command[0] == "docker"
    )
    assert not any(
        "docker.sock" in argument for command in commands for argument in command
    )
    assert all(
        "--hostname" in command
        and command[command.index("--hostname") + 1] == "github.com"
        for command in commands
        if command[0] == "gh"
    )
    docker_envs = [env for command, env in fake.calls if command[0] == "docker"]
    assert all(
        env is not None and not launcher.HOST_ONLY_CREDENTIAL_ENV.intersection(env)
        for env in docker_envs
    )


@pytest.mark.parametrize(
    "case",
    [
        "container_id",
        "state",
        "running",
        "image_ref",
        "command",
        "image_id",
        "restart",
        "restart_count",
        "network",
        "extra_mount",
        "receipt_source",
        "receipt_writable",
        "home_writable",
        "host_identity",
        "credential_env",
    ],
)
def test_created_container_verification_is_closed(
    tmp_path: Path,
    case: str,
) -> None:
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    receipt_path = tmp_path / "receipt.json"
    payload = _container_inspection(
        hermes_home=hermes_home,
        receipt_path=receipt_path,
    )
    container = payload[0]
    if case == "container_id":
        container["Id"] = "f" * 64
    elif case == "state":
        container["State"]["Status"] = "running"
    elif case == "running":
        container["State"]["Running"] = True
    elif case == "image_ref":
        container["Config"]["Image"] = f"{launcher.IMAGE_NAME}:latest"
    elif case == "command":
        container["Config"]["Cmd"] = ["dashboard"]
    elif case == "image_id":
        container["Image"] = "sha256:" + "f" * 64
    elif case == "restart":
        container["HostConfig"]["RestartPolicy"]["Name"] = "always"
    elif case == "restart_count":
        container["HostConfig"]["RestartPolicy"]["MaximumRetryCount"] = 1
    elif case == "network":
        container["HostConfig"]["NetworkMode"] = "bridge"
    elif case == "extra_mount":
        container["Mounts"].append({
            "Type": "bind",
            "Source": "/var/run/docker.sock",
            "Destination": "/var/run/docker.sock",
            "RW": True,
        })
    elif case == "receipt_source":
        container["Mounts"][1]["Source"] = "/tmp/untrusted.json"
    elif case == "receipt_writable":
        container["Mounts"][1]["RW"] = True
    elif case == "home_writable":
        container["Mounts"][0]["RW"] = False
    elif case == "host_identity":
        container["Config"]["Env"] = []
    elif case == "credential_env":
        container["Config"]["Env"].append("DOCKER_AUTH_CONFIG=leaked")

    metadata = hermes_home.stat()
    with pytest.raises(launcher.LauncherError):
        launcher.verify_created_container(
            payload,
            container_id=CONTAINER_ID,
            image_ref=f"{launcher.IMAGE_NAME}@{ARTIFACT_DIGEST}",
            image_id=IMAGE_ID,
            hermes_home=hermes_home,
            receipt_path=receipt_path,
            host_uid=metadata.st_uid,
            host_gid=metadata.st_gid,
            command=("gateway", "run"),
        )


def test_failed_inspection_never_starts_or_removes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    receipt_path = tmp_path / "trusted" / "tested-artifact.json"
    fake = FakeTransport(
        hermes_home=hermes_home,
        receipt_path=receipt_path,
    )
    fake.container_inspection[0]["HostConfig"]["RestartPolicy"]["Name"] = "always"
    monkeypatch.setattr(launcher, "_run", fake)
    monkeypatch.setattr(launcher, "_require_root", lambda: None)
    monkeypatch.setattr(launcher, "HOST_RECEIPT_PATH", receipt_path)

    with pytest.raises(
        launcher.LauncherError,
        match="start was not requested.*host policy mismatch",
    ):
        launcher.launch(
            release_id=42,
            platform=PLATFORM,
            hermes_home=hermes_home,
        )

    docker_commands = [command for command, _ in fake.calls if command[0] == "docker"]
    assert ["docker", "container", "inspect", CONTAINER_ID] in docker_commands
    assert not any(
        command[1:3] == ["container", "start"] for command in docker_commands
    )
    assert not any(command[1:3] == ["container", "rm"] for command in docker_commands)


def test_wrong_local_image_platform_fails_before_create(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    receipt_path = tmp_path / "trusted" / "tested-artifact.json"
    fake = FakeTransport(
        hermes_home=hermes_home,
        receipt_path=receipt_path,
    )
    fake.image_inspection[0]["Architecture"] = "arm64"
    monkeypatch.setattr(launcher, "_run", fake)
    monkeypatch.setattr(launcher, "_require_root", lambda: None)
    monkeypatch.setattr(launcher, "HOST_RECEIPT_PATH", receipt_path)

    with pytest.raises(launcher.LauncherError, match="platform mismatch"):
        launcher.launch(
            release_id=42,
            platform=PLATFORM,
            hermes_home=hermes_home,
        )

    docker_commands = [command for command, _ in fake.calls if command[0] == "docker"]
    assert [command[1:3] for command in docker_commands] == [
        ["image", "pull"],
        ["image", "inspect"],
    ]
    assert not receipt_path.exists()


@pytest.mark.parametrize(
    "case",
    [
        "extra_field",
        "wrong_schema",
        "wrong_platform",
        "wrong_commit",
        "zero_artifact",
        "bad_lock",
        "duplicate_key",
    ],
)
def test_receipt_validation_rejects_open_or_mismatched_input(case: str) -> None:
    payload = _receipt_payload()
    if case == "extra_field":
        payload["extra"] = "not-closed"
    elif case == "wrong_schema":
        payload["schema"] = "hermes.task-fence.tested-artifact-receipt/v2"
    elif case == "wrong_platform":
        payload["target_platform"] = "linux/arm64"
    elif case == "wrong_commit":
        payload["tested_artifact_commit"] = "f" * 40
    elif case == "zero_artifact":
        payload["tested_artifact_checksum"] = "sha256:" + "0" * 64
    elif case == "bad_lock":
        payload["dependency_lock_fingerprint"] = "latest"

    if case == "duplicate_key":
        raw = (
            b'{"schema":"'
            + launcher.RECEIPT_SCHEMA.encode()
            + b'","schema":"duplicate"}'
        )
    else:
        raw = _receipt_bytes(payload)
    with pytest.raises(launcher.LauncherError):
        launcher.parse_receipt(
            raw,
            expected_platform=PLATFORM,
            expected_commit=COMMIT,
        )


def test_receipt_handoff_is_atomic_set_once_and_read_only(tmp_path: Path) -> None:
    raw = _receipt_bytes()
    path = tmp_path / "trusted" / "tested-artifact.json"
    assert launcher.install_receipt_set_once(path, raw) == path
    assert launcher.install_receipt_set_once(path, raw) == path
    assert path.read_bytes() == raw
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert path.stat().st_nlink == 1

    with pytest.raises(launcher.LauncherError, match="different bytes"):
        launcher.install_receipt_set_once(path, raw + b" ")
    path.chmod(0o644)
    with pytest.raises(launcher.LauncherError, match="ownership or mode"):
        launcher.install_receipt_set_once(path, raw)

    target = tmp_path / "target.json"
    target.write_bytes(raw)
    symlink = tmp_path / "trusted" / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(launcher.LauncherError, match="regular file"):
        launcher.install_receipt_set_once(symlink, raw)


def test_duplicate_release_asset_fails_before_docker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    fake = FakeTransport(
        hermes_home=hermes_home,
        receipt_path=tmp_path / "receipt.json",
    )
    fake.assets = [copy.deepcopy(fake.assets[0]), copy.deepcopy(fake.assets[0])]
    monkeypatch.setattr(launcher, "_run", fake)

    with pytest.raises(launcher.LauncherError, match="exactly one"):
        launcher.acquire_release_receipt(42, PLATFORM)
    assert not any(command[0] == "docker" for command, _ in fake.calls)


def test_launcher_requires_root_before_external_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 1000)

    def unexpected_call(*args: Any, **kwargs: Any) -> bytes:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(launcher, "_run", unexpected_call)
    with pytest.raises(launcher.LauncherError, match="run as root"):
        launcher.launch(
            release_id=42,
            platform=PLATFORM,
            hermes_home=hermes_home,
        )
