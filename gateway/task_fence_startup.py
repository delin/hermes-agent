"""Startup-only Task Fence receipt and recovery barrier."""

from __future__ import annotations

import json
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Any

from task_fence import (
    TaskFenceArtifactIdentity,
    TaskFenceProtocolRejected,
    TaskFenceRecovery,
    TaskFenceRecoveryUnavailable,
)

_TESTED_ARTIFACT_RECEIPT_PATH = Path("/run/hermes/task-fence/tested-artifact.json")
_RECEIPT_SCHEMA = "hermes.task-fence.tested-artifact-receipt/v1"
_MAX_RECEIPT_BYTES = 4096
_RECEIPT_KEYS = frozenset({
    "schema",
    "target_platform",
    "tested_artifact_commit",
    "tested_artifact_checksum",
    "dependency_lock_fingerprint",
})


class TaskFenceStartupUnavailable(RuntimeError):
    """The selected shadow cohort could not complete its startup barrier."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _runtime_platform() -> str:
    machine = platform.machine().lower()
    architectures = {
        "amd64": "amd64",
        "x86_64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    architecture = architectures.get(machine)
    if not sys.platform.startswith("linux") or architecture is None:
        raise TaskFenceStartupUnavailable("unsupported_runtime_platform")
    return f"linux/{architecture}"


def _parse_receipt(raw: bytes, *, expected_platform: str) -> TaskFenceArtifactIdentity:
    if not raw or len(raw) > _MAX_RECEIPT_BYTES:
        raise TaskFenceStartupUnavailable("invalid_receipt_size")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, ValueError):
        raise TaskFenceStartupUnavailable("invalid_receipt_json") from None
    if not isinstance(payload, dict) or frozenset(payload) != _RECEIPT_KEYS:
        raise TaskFenceStartupUnavailable("invalid_receipt_fields")
    if payload["schema"] != _RECEIPT_SCHEMA:
        raise TaskFenceStartupUnavailable("unsupported_receipt_schema")
    if payload["target_platform"] != expected_platform:
        raise TaskFenceStartupUnavailable("receipt_platform_mismatch")
    try:
        return TaskFenceArtifactIdentity(
            tested_artifact_commit=payload["tested_artifact_commit"],
            tested_artifact_checksum=payload["tested_artifact_checksum"],
            dependency_lock_fingerprint=payload["dependency_lock_fingerprint"],
        )
    except TaskFenceProtocolRejected as exc:
        raise TaskFenceStartupUnavailable(exc.reason) from None


def _read_receipt() -> TaskFenceArtifactIdentity:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(_TESTED_ARTIFACT_RECEIPT_PATH, flags)
    except OSError:
        raise TaskFenceStartupUnavailable("receipt_unavailable") from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= _MAX_RECEIPT_BYTES
        ):
            raise TaskFenceStartupUnavailable("unsafe_receipt_file")
        raw = b""
        while len(raw) <= _MAX_RECEIPT_BYTES:
            chunk = os.read(
                descriptor,
                _MAX_RECEIPT_BYTES + 1 - len(raw),
            )
            if not chunk:
                break
            raw += chunk
    except OSError:
        raise TaskFenceStartupUnavailable("receipt_unavailable") from None
    finally:
        os.close(descriptor)
    if len(raw) != metadata.st_size:
        raise TaskFenceStartupUnavailable("receipt_changed_during_read")
    return _parse_receipt(raw, expected_platform=_runtime_platform())


def prepare_task_fence_shadow_startup(config: Any) -> TaskFenceRecovery | None:
    """Run the selected cohort's receipt-bound recovery before consumers open."""

    if not getattr(config, "task_fence_shadow_session_key", ""):
        return None
    if getattr(config, "multiplex_profiles", False):
        raise TaskFenceStartupUnavailable("multiplex_profiles_unsupported")

    identity = _read_receipt()
    database = None
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        database = SessionDB(get_hermes_home() / "state.db")
        store = database.inspect_task_fence_store()
        if not store.compatible:
            raise TaskFenceStartupUnavailable(store.reason)
        if (
            store.runtime_epoch is None
            or store.mode_generation != 0
            or store.ever_enforced is not False
        ):
            raise TaskFenceStartupUnavailable("unsupported_shadow_control_state")
        return database.recover_task_fence_state(
            expected_runtime_epoch=store.runtime_epoch,
            expected_mode_generation=store.mode_generation,
            tested_artifact_identity=identity,
        )
    except TaskFenceStartupUnavailable:
        raise
    except (TaskFenceProtocolRejected, TaskFenceRecoveryUnavailable) as exc:
        raise TaskFenceStartupUnavailable(exc.reason) from None
    except Exception as exc:
        raise TaskFenceStartupUnavailable("store_unavailable") from exc
    finally:
        if database is not None:
            database.close()
