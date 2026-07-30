from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "task_fence_launcher.py"
SPEC = importlib.util.spec_from_file_location(
    "task_fence_launcher_docker_test",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def test_real_docker_create_inspect_start_barrier(
    built_image: str,
    container_name: str,
    tmp_path: Path,
) -> None:
    image_metadata = json.loads(
        subprocess.run(
            ["docker", "image", "inspect", built_image],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    )[0]
    image_id = image_metadata["Id"]
    platform = f"{image_metadata['Os']}/{image_metadata['Architecture']}"
    image_ref = built_image if "@sha256:" in built_image else image_id
    assert (
        launcher.inspect_local_image(
            image_ref,
            platform,
            env=launcher._docker_environment(),
        )
        == image_id
    )

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    receipt_path = tmp_path / "tested-artifact.json"
    receipt_path.write_text('{"fixture":"real-docker-transport"}\n')
    receipt_path.chmod(0o444)

    container_id = launcher.create_inspect_start(
        image_ref=image_ref,
        image_id=image_id,
        platform=platform,
        hermes_home=hermes_home,
        receipt_path=receipt_path,
        container_name=container_name,
        command=("sleep", "infinity"),
        env=launcher._docker_environment(),
    )
    assert len(container_id) == 64

    inspection = json.loads(
        subprocess.run(
            ["docker", "container", "inspect", container_id],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    )[0]
    assert inspection["State"]["Running"] is True
    assert inspection["Config"]["Image"] == image_ref
    assert inspection["Image"] == image_id
    assert inspection["HostConfig"]["RestartPolicy"] == {
        "Name": "no",
        "MaximumRetryCount": 0,
    }
    receipt_mount = next(
        mount
        for mount in inspection["Mounts"]
        if mount["Destination"] == launcher.CONTAINER_RECEIPT_PATH
    )
    assert receipt_mount["Source"] == str(receipt_path)
    assert receipt_mount["Type"] == "bind"
    assert receipt_mount["RW"] is False
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert not any(
        "docker.sock" in mount["Source"] or "docker.sock" in mount["Destination"]
        for mount in inspection["Mounts"]
    )
    assert not launcher.CONTAINER_CREDENTIAL_ENV.intersection(
        item.split("=", 1)[0] for item in inspection["Config"]["Env"]
    )

    write_probe = subprocess.run(
        [
            "docker",
            "container",
            "exec",
            "--user",
            "root",
            container_id,
            "sh",
            "-c",
            f"printf x >> {launcher.CONTAINER_RECEIPT_PATH}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert write_probe.returncode != 0
