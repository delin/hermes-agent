#!/usr/bin/env python3
"""Launch the first Task Fence Docker cohort from one tested release receipt.

This is a host-side transport barrier. It deliberately does not parse the
receipt inside Hermes, mutate Task Fence storage, replace an existing
container, or implement startup recovery. The operator must quiesce every
other consumer and pre-authenticate root's ``gh`` and Docker clients with
host-only read credentials before invoking it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

REPOSITORY = "delin/hermes-agent"
IMAGE_NAME = "ghcr.io/delin/hermes-agent"
RECEIPT_SCHEMA = "hermes.task-fence.tested-artifact-receipt/v1"
HOST_RECEIPT_PATH = Path("/etc/hermes/task-fence/tested-artifact.json")
CONTAINER_RECEIPT_PATH = "/run/hermes/task-fence/tested-artifact.json"
CONTAINER_HERMES_HOME = "/opt/data"
CONTAINER_NAME = "hermes"
MAX_RECEIPT_BYTES = 4096
PLATFORM_ASSETS = {
    "linux/amd64": "amd64.json",
    "linux/arm64": "arm64.json",
}
HOST_ONLY_CREDENTIAL_ENV = frozenset({
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
    "CR_PAT",
})
CONTAINER_CREDENTIAL_ENV = HOST_ONLY_CREDENTIAL_ENV | {"DOCKER_AUTH_CONFIG"}

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RECEIPT_KEYS = frozenset({
    "schema",
    "target_platform",
    "tested_artifact_commit",
    "tested_artifact_checksum",
    "dependency_lock_fingerprint",
})


class LauncherError(RuntimeError):
    """A fail-closed launcher contract violation."""


@dataclass(frozen=True)
class TestedArtifactReceipt:
    target_platform: str
    tested_artifact_commit: str
    tested_artifact_checksum: str
    dependency_lock_fingerprint: str


@dataclass(frozen=True)
class AcquiredReceipt:
    raw: bytes
    receipt: TestedArtifactReceipt
    release_tag: str


def _run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    timeout: int = 60,
) -> bytes:
    try:
        result = subprocess.run(
            list(argv),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise LauncherError(f"required command not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise LauncherError(f"command timed out: {argv[0]}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise LauncherError(
            f"{argv[0]} failed with exit code {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    return result.stdout


def _load_json(raw: bytes, description: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise LauncherError(f"invalid {description} JSON") from exc


def _gh_json(endpoint: str, *, paginate: bool = False) -> Any:
    argv = ["gh", "api", "--hostname", "github.com", "--method", "GET"]
    if paginate:
        argv.extend(["--paginate", "--slurp"])
    argv.append(endpoint)
    return _load_json(_run(argv), f"GitHub response for {endpoint}")


def _require_nonzero_commit(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or _COMMIT_RE.fullmatch(value) is None
        or set(value) == {"0"}
    ):
        raise LauncherError(f"invalid {description}")
    return value


def _require_nonzero_digest(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or _DIGEST_RE.fullmatch(value) is None
        or set(value.removeprefix("sha256:")) == {"0"}
    ):
        raise LauncherError(f"invalid {description}")
    return value


def _release_tag_commit(tag_name: str) -> str:
    tag_ref = quote(tag_name, safe="")
    ref = _gh_json(f"repos/{REPOSITORY}/git/ref/tags/{tag_ref}")
    if not isinstance(ref, dict) or ref.get("ref") != f"refs/tags/{tag_name}":
        raise LauncherError("release tag reference mismatch")
    target = ref.get("object")
    for _ in range(8):
        if not isinstance(target, dict):
            raise LauncherError("invalid release tag target")
        target_type = target.get("type")
        target_sha = _require_nonzero_commit(
            target.get("sha"),
            "release tag object",
        )
        if target_type == "commit":
            return target_sha
        if target_type != "tag":
            raise LauncherError("release tag does not resolve to a commit")
        annotated = _gh_json(f"repos/{REPOSITORY}/git/tags/{target_sha}")
        if not isinstance(annotated, dict) or annotated.get("sha") != target_sha:
            raise LauncherError("annotated release tag identity mismatch")
        target = annotated.get("object")
    raise LauncherError("release tag indirection is too deep")


def parse_receipt(
    raw: bytes,
    *,
    expected_platform: str,
    expected_commit: str,
) -> TestedArtifactReceipt:
    if not raw or len(raw) > MAX_RECEIPT_BYTES:
        raise LauncherError("tested artifact receipt size is invalid")
    payload = _load_json(raw, "tested artifact receipt")
    if not isinstance(payload, dict) or frozenset(payload) != _RECEIPT_KEYS:
        raise LauncherError("tested artifact receipt fields are not closed")
    if payload.get("schema") != RECEIPT_SCHEMA:
        raise LauncherError("tested artifact receipt schema mismatch")
    if payload.get("target_platform") != expected_platform:
        raise LauncherError("tested artifact receipt platform mismatch")
    commit = _require_nonzero_commit(
        payload.get("tested_artifact_commit"),
        "tested artifact commit",
    )
    if commit != expected_commit:
        raise LauncherError("release tag and tested artifact commit disagree")
    artifact = _require_nonzero_digest(
        payload.get("tested_artifact_checksum"),
        "tested artifact checksum",
    )
    lock = _require_nonzero_digest(
        payload.get("dependency_lock_fingerprint"),
        "dependency lock fingerprint",
    )
    return TestedArtifactReceipt(
        target_platform=expected_platform,
        tested_artifact_commit=commit,
        tested_artifact_checksum=artifact,
        dependency_lock_fingerprint=lock,
    )


def acquire_release_receipt(
    release_id: int,
    platform: str,
) -> AcquiredReceipt:
    if release_id <= 0:
        raise LauncherError("release ID must be positive")
    asset_name = PLATFORM_ASSETS.get(platform)
    if asset_name is None:
        raise LauncherError("unsupported target platform")

    release = _gh_json(f"repos/{REPOSITORY}/releases/{release_id}")
    if not isinstance(release, dict) or type(release.get("id")) is not int:
        raise LauncherError("invalid GitHub release response")
    tag_name = release.get("tag_name")
    if (
        release["id"] != release_id
        or release.get("draft") is not False
        or not isinstance(release.get("published_at"), str)
        or not release["published_at"]
        or not isinstance(tag_name, str)
        or not tag_name
    ):
        raise LauncherError("GitHub release identity is not published and exact")

    pages = _gh_json(
        f"repos/{REPOSITORY}/releases/{release_id}/assets?per_page=100",
        paginate=True,
    )
    if not isinstance(pages, list) or not all(isinstance(page, list) for page in pages):
        raise LauncherError("invalid GitHub release assets response")
    matches = [
        asset
        for page in pages
        for asset in page
        if isinstance(asset, dict) and asset.get("name") == asset_name
    ]
    if len(matches) != 1:
        raise LauncherError("release must contain exactly one platform receipt")
    asset = matches[0]
    asset_id = asset.get("id")
    asset_size = asset.get("size")
    if (
        type(asset_id) is not int
        or asset_id <= 0
        or type(asset_size) is not int
        or not 0 < asset_size <= MAX_RECEIPT_BYTES
        or asset.get("state") != "uploaded"
    ):
        raise LauncherError("release receipt asset metadata is invalid")

    raw = _run([
        "gh",
        "api",
        "--hostname",
        "github.com",
        "--method",
        "GET",
        "--header",
        "Accept: application/octet-stream",
        f"repos/{REPOSITORY}/releases/assets/{asset_id}",
    ])
    if len(raw) != asset_size:
        raise LauncherError("release receipt asset size changed during download")
    commit = _release_tag_commit(tag_name)
    receipt = parse_receipt(
        raw,
        expected_platform=platform,
        expected_commit=commit,
    )
    return AcquiredReceipt(raw=raw, receipt=receipt, release_tag=tag_name)


def _validate_installed_receipt(path: Path, expected: bytes) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise LauncherError("receipt handoff is not a readable regular file") from exc
    try:
        metadata = os.fstat(fd)
        owner_uid = os.geteuid()
        owner_gid = os.getegid()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != owner_uid
            or metadata.st_gid != owner_gid
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_nlink != 1
        ):
            raise LauncherError("receipt handoff ownership or mode is unsafe")
        actual = b""
        while len(actual) <= MAX_RECEIPT_BYTES:
            chunk = os.read(fd, MAX_RECEIPT_BYTES + 1 - len(actual))
            if not chunk:
                break
            actual += chunk
    finally:
        os.close(fd)
    if actual != expected:
        raise LauncherError("receipt handoff already contains different bytes")


def install_receipt_set_once(path: Path, raw: bytes) -> Path:
    if not path.is_absolute():
        raise LauncherError("receipt handoff path must be absolute")
    parent = path.parent
    parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    try:
        resolved_parent = parent.resolve(strict=True)
        parent_metadata = parent.stat()
    except OSError as exc:
        raise LauncherError("receipt handoff directory is unavailable") from exc
    if (
        resolved_parent != parent
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise LauncherError("receipt handoff directory is unsafe")

    try:
        path.lstat()
    except FileNotFoundError:
        pass
    else:
        _validate_installed_receipt(path, raw)
        return path

    fd, temporary_name = tempfile.mkstemp(
        prefix=".tested-artifact.",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise LauncherError("failed to write receipt handoff")
            view = view[written:]
        os.fchmod(fd, 0o444)
        os.fchown(fd, os.geteuid(), os.getegid())
        os.fsync(fd)
        os.close(fd)
        fd = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            _validate_installed_receipt(path, raw)
        else:
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _validate_installed_receipt(path, raw)
    return path


def _docker_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key not in HOST_ONLY_CREDENTIAL_ENV
    }


def inspect_local_image(
    image_ref: str,
    platform: str,
    *,
    env: Mapping[str, str],
) -> str:
    payload = _load_json(
        _run(
            ["docker", "image", "inspect", image_ref],
            env=env,
            timeout=30,
        ),
        "Docker image inspection",
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise LauncherError("Docker image inspection is ambiguous")
    image = payload[0]
    if not isinstance(image, dict):
        raise LauncherError("Docker image inspection is invalid")
    image_id = _require_nonzero_digest(image.get("Id"), "local Docker image ID")
    actual_platform = f"{image.get('Os')}/{image.get('Architecture')}"
    if actual_platform != platform:
        raise LauncherError("local Docker image platform mismatch")
    return image_id


def pull_exact_image(
    receipt: TestedArtifactReceipt,
    *,
    env: Mapping[str, str],
) -> tuple[str, str]:
    image_ref = f"{IMAGE_NAME}@{receipt.tested_artifact_checksum}"
    _run(
        [
            "docker",
            "image",
            "pull",
            "--platform",
            receipt.target_platform,
            image_ref,
        ],
        env=env,
        timeout=900,
    )
    image_id = inspect_local_image(
        image_ref,
        receipt.target_platform,
        env=env,
    )
    return image_ref, image_id


def _verify_mount(
    mounts: Mapping[str, Any],
    *,
    source: Path,
    destination: str,
    writable: bool,
) -> None:
    mount = mounts.get(destination)
    if (
        not isinstance(mount, dict)
        or mount.get("Type") != "bind"
        or mount.get("Source") != str(source)
        or mount.get("Destination") != destination
        or mount.get("RW") is not writable
    ):
        raise LauncherError(f"container mount mismatch: {destination}")


def verify_created_container(
    payload: Any,
    *,
    container_id: str,
    image_ref: str,
    image_id: str,
    hermes_home: Path,
    receipt_path: Path,
    host_uid: int,
    host_gid: int,
    command: Sequence[str],
) -> None:
    if not isinstance(payload, list) or len(payload) != 1:
        raise LauncherError("Docker container inspection is ambiguous")
    container = payload[0]
    if not isinstance(container, dict) or container.get("Id") != container_id:
        raise LauncherError("Docker container identity mismatch")
    state = container.get("State")
    if (
        not isinstance(state, dict)
        or state.get("Status") != "created"
        or state.get("Running") is not False
    ):
        raise LauncherError("Docker container was not held in created state")
    config = container.get("Config")
    if (
        not isinstance(config, dict)
        or config.get("Image") != image_ref
        or config.get("Cmd") != list(command)
    ):
        raise LauncherError("Docker container image or command mismatch")
    if container.get("Image") != image_id:
        raise LauncherError("Docker container local image ID mismatch")

    host_config = container.get("HostConfig")
    if not isinstance(host_config, dict):
        raise LauncherError("Docker container host policy is invalid")
    restart = host_config.get("RestartPolicy")
    if (
        not isinstance(restart, dict)
        or restart.get("Name") != "no"
        or restart.get("MaximumRetryCount") != 0
        or host_config.get("NetworkMode") != "host"
    ):
        raise LauncherError("Docker container host policy mismatch")

    raw_mounts = container.get("Mounts")
    if not isinstance(raw_mounts, list) or len(raw_mounts) != 2:
        raise LauncherError("Docker container mount set is not closed")
    mounts: dict[str, dict[str, Any]] = {}
    for mount in raw_mounts:
        if not isinstance(mount, dict):
            raise LauncherError("Docker container mount entry is invalid")
        destination = mount.get("Destination")
        if not isinstance(destination, str) or destination in mounts:
            raise LauncherError("Docker container mount destinations are ambiguous")
        mounts[destination] = mount
    _verify_mount(
        mounts,
        source=hermes_home,
        destination=CONTAINER_HERMES_HOME,
        writable=True,
    )
    _verify_mount(
        mounts,
        source=receipt_path,
        destination=CONTAINER_RECEIPT_PATH,
        writable=False,
    )

    container_env = config.get("Env")
    if not isinstance(container_env, list) or not all(
        isinstance(item, str) for item in container_env
    ):
        raise LauncherError("Docker container environment is invalid")
    expected_env = {
        f"HERMES_UID={host_uid}",
        f"HERMES_GID={host_gid}",
    }
    if not expected_env.issubset(container_env):
        raise LauncherError("Docker container host identity environment mismatch")
    if any(item.split("=", 1)[0] in CONTAINER_CREDENTIAL_ENV for item in container_env):
        raise LauncherError("host credential entered the Docker container")


def create_inspect_start(
    *,
    image_ref: str,
    image_id: str,
    platform: str,
    hermes_home: Path,
    receipt_path: Path,
    container_name: str,
    command: Sequence[str] = ("gateway", "run"),
    env: Mapping[str, str],
) -> str:
    if _CONTAINER_NAME_RE.fullmatch(container_name) is None:
        raise LauncherError("invalid Docker container name")
    home_metadata = hermes_home.stat()
    host_uid = home_metadata.st_uid
    host_gid = home_metadata.st_gid
    create_argv = [
        "docker",
        "container",
        "create",
        "--name",
        container_name,
        "--network",
        "host",
        "--restart",
        "no",
        "--pull",
        "never",
        "--mount",
        f"type=bind,src={hermes_home},dst={CONTAINER_HERMES_HOME}",
        "--mount",
        (f"type=bind,src={receipt_path},dst={CONTAINER_RECEIPT_PATH},readonly"),
        "--env",
        f"HERMES_UID={host_uid}",
        "--env",
        f"HERMES_GID={host_gid}",
        "--platform",
        platform,
        image_ref,
        *command,
    ]
    container_id = _run(create_argv, env=env, timeout=60).decode("ascii").strip()
    if _CONTAINER_ID_RE.fullmatch(container_id) is None or set(container_id) == {"0"}:
        raise LauncherError("Docker create did not return one full container ID")

    try:
        inspection = _load_json(
            _run(
                ["docker", "container", "inspect", container_id],
                env=env,
                timeout=30,
            ),
            "Docker container inspection",
        )
        verify_created_container(
            inspection,
            container_id=container_id,
            image_ref=image_ref,
            image_id=image_id,
            hermes_home=hermes_home,
            receipt_path=receipt_path,
            host_uid=host_uid,
            host_gid=host_gid,
            command=command,
        )
    except Exception as exc:
        raise LauncherError(
            f"container {container_id} failed pre-start inspection; "
            f"start was not requested and the container was retained: {exc}"
        ) from exc

    _run(
        ["docker", "container", "start", container_id],
        env=env,
        timeout=60,
    )
    return container_id


def _require_root() -> None:
    if os.geteuid() != 0:
        raise LauncherError("launcher must run as root")


def _resolve_hermes_home(path: Path) -> Path:
    if not path.is_absolute():
        raise LauncherError("HERMES_HOME must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LauncherError("HERMES_HOME does not exist") from exc
    if not resolved.is_dir():
        raise LauncherError("HERMES_HOME is not a directory")
    receipt_path = HOST_RECEIPT_PATH
    if receipt_path == resolved or receipt_path.is_relative_to(resolved):
        raise LauncherError("receipt handoff must stay outside HERMES_HOME")
    return resolved


def launch(
    *,
    release_id: int,
    platform: str,
    hermes_home: Path,
) -> str:
    _require_root()
    resolved_home = _resolve_hermes_home(hermes_home)
    acquired = acquire_release_receipt(release_id, platform)
    docker_env = _docker_environment()
    image_ref, image_id = pull_exact_image(acquired.receipt, env=docker_env)
    receipt_path = install_receipt_set_once(HOST_RECEIPT_PATH, acquired.raw)
    return create_inspect_start(
        image_ref=image_ref,
        image_id=image_id,
        platform=platform,
        hermes_home=resolved_home,
        receipt_path=receipt_path,
        container_name=CONTAINER_NAME,
        env=docker_env,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the bounded Hermes gateway container from one exact "
            "maintained-fork release receipt."
        )
    )
    parser.add_argument("--release-id", required=True, type=_positive_int)
    parser.add_argument(
        "--platform",
        required=True,
        choices=tuple(PLATFORM_ASSETS),
    )
    parser.add_argument("--hermes-home", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        container_id = launch(
            release_id=args.release_id,
            platform=args.platform,
            hermes_home=args.hermes_home,
        )
    except (LauncherError, OSError) as exc:
        print(f"task-fence launcher: {exc}", file=sys.stderr)
        return 1
    print(container_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
