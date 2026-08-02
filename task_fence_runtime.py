"""Startup-only Task Fence receipt and recovery barrier."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import os
import platform
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from task_fence import (
    OperationDescriptor,
    OperationKind,
    TASK_FENCE_PROCESS_CHECKPOINT_RECOVERY_ADAPTER,
    TaskFenceArtifactIdentity,
    TaskFenceCapabilityUnavailable,
    TaskFenceProtocolRejected,
    TaskFenceRecovery,
    TaskFenceRecoveryUnavailable,
)

_TESTED_ARTIFACT_RECEIPT_PATH = Path("/run/hermes/task-fence/tested-artifact.json")
_RECEIPT_SCHEMA = "hermes.task-fence.tested-artifact-receipt/v1"
_MAX_RECEIPT_BYTES = 4096
_MAX_PROCESS_CHECKPOINT_BYTES = 1_048_576
_MAX_PROCESS_CHECKPOINT_ENTRIES = 64
_MAX_PROCESS_SESSION_ID_BYTES = 512
_TASK_FENCE_OWNER_LOCK_FILENAME = "task-fence-owner.lock"
_WINDOWS_LOCK_OFFSET = 1024 * 1024
_task_fence_owner_lock_handle: Any = None
_task_fence_owner_lock_path: Path | None = None
_RECEIPT_KEYS = frozenset({
    "schema",
    "target_platform",
    "tested_artifact_commit",
    "tested_artifact_checksum",
    "dependency_lock_fingerprint",
})
_PROCESS_CHECKPOINT_KEYS = frozenset({
    "session_id",
    "command",
    "pid",
    "pid_scope",
    "host_start_time",
    "cwd",
    "started_at",
    "task_id",
    "session_key",
    "watcher_platform",
    "watcher_chat_id",
    "watcher_user_id",
    "watcher_user_name",
    "watcher_thread_id",
    "watcher_message_id",
    "watcher_interval",
    "notify_on_complete",
    "watch_patterns",
})

logger = logging.getLogger(__name__)


class TaskFenceStartupUnavailable(RuntimeError):
    """The selected shadow cohort could not complete its startup barrier."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _owner_lock_path(hermes_home: Path | None = None) -> Path:
    from hermes_constants import get_process_hermes_home

    home = Path(hermes_home) if hermes_home is not None else get_process_hermes_home()
    return (home / _TASK_FENCE_OWNER_LOCK_FILENAME).expanduser().resolve(strict=False)


def _try_acquire_owner_file_lock(handle: Any) -> bool:
    try:
        if sys.platform == "win32":
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("\n")
                handle.flush()
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _release_owner_file_lock(handle: Any) -> None:
    try:
        if sys.platform == "win32":
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def acquire_task_fence_owner_lock(hermes_home: Path | None = None) -> bool:
    """Claim the process-wide Task Fence owner for one HERMES_HOME."""
    global _task_fence_owner_lock_handle, _task_fence_owner_lock_path

    path = _owner_lock_path(hermes_home)
    if _task_fence_owner_lock_handle is not None:
        return _task_fence_owner_lock_path == path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+", encoding="utf-8")
    except OSError:
        return False
    if not _try_acquire_owner_file_lock(handle):
        handle.close()
        return False
    _task_fence_owner_lock_handle = handle
    _task_fence_owner_lock_path = path
    return True


def release_task_fence_owner_lock() -> None:
    """Release the Task Fence owner lock when held by this process."""
    global _task_fence_owner_lock_handle, _task_fence_owner_lock_path

    handle = _task_fence_owner_lock_handle
    _task_fence_owner_lock_handle = None
    _task_fence_owner_lock_path = None
    if handle is None:
        return
    _release_owner_file_lock(handle)
    try:
        handle.close()
    except OSError:
        pass



@dataclass(frozen=True, slots=True)
class _ProcessCheckpointObservation:
    invocation_id: str
    invocation_fingerprint: str


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


def _read_process_checkpoint_snapshot(
    path: Path,
    *,
    shadow_session_key: str,
) -> tuple[_ProcessCheckpointObservation, ...]:
    """Capture bounded pre-liveness process recovery candidates without side effects."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return ()
        raise TaskFenceStartupUnavailable("process_checkpoint_unavailable") from None

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TaskFenceStartupUnavailable("process_checkpoint_unavailable")
        if before.st_size > _MAX_PROCESS_CHECKPOINT_BYTES:
            raise TaskFenceStartupUnavailable("process_checkpoint_limit_exceeded")
        raw = b""
        while len(raw) <= _MAX_PROCESS_CHECKPOINT_BYTES:
            chunk = os.read(
                descriptor,
                _MAX_PROCESS_CHECKPOINT_BYTES + 1 - len(raw),
            )
            if not chunk:
                break
            raw += chunk
        after = os.fstat(descriptor)
    except TaskFenceStartupUnavailable:
        raise
    except OSError:
        raise TaskFenceStartupUnavailable("process_checkpoint_unavailable") from None
    finally:
        os.close(descriptor)

    if len(raw) > _MAX_PROCESS_CHECKPOINT_BYTES:
        raise TaskFenceStartupUnavailable("process_checkpoint_limit_exceeded")
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        getattr(before, "st_mtime_ns", before.st_mtime),
        getattr(before, "st_ctime_ns", before.st_ctime),
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        getattr(after, "st_mtime_ns", after.st_mtime),
        getattr(after, "st_ctime_ns", after.st_ctime),
    )
    if len(raw) != before.st_size or after_identity != before_identity:
        raise TaskFenceStartupUnavailable("process_checkpoint_unavailable")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite number")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite number")
        return parsed

    try:
        entries = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=parse_finite_float,
        )
    except (RecursionError, UnicodeDecodeError, ValueError):
        raise TaskFenceStartupUnavailable(
            "incompatible_process_checkpoint"
        ) from None
    if not isinstance(entries, list):
        raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")

    def require_text(value: Any) -> str:
        if not isinstance(value, str):
            raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise TaskFenceStartupUnavailable(
                "incompatible_process_checkpoint"
            ) from None
        return value

    observations: list[_ProcessCheckpointObservation] = []
    selected_session_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")
        if entry.get("session_key", "") != shadow_session_key:
            continue
        if not frozenset(entry).issubset(_PROCESS_CHECKPOINT_KEYS):
            raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")

        pid = entry.get("pid")
        if not pid:
            continue
        pid_scope = entry.get("pid_scope", "host")
        if not isinstance(pid_scope, str):
            raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")
        if pid_scope != "host":
            continue
        if type(pid) is not int or pid <= 0:
            raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")

        watcher_interval = entry.get("watcher_interval", 0)
        if type(watcher_interval) is not int or watcher_interval < 0:
            raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")

        session_id = require_text(entry.get("session_id"))
        if (
            not session_id
            or len(session_id.encode("utf-8")) > _MAX_PROCESS_SESSION_ID_BYTES
            or session_id in selected_session_ids
        ):
            raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")
        selected_session_ids.add(session_id)

        host_start_time = entry.get("host_start_time")
        if host_start_time is not None and (
            type(host_start_time) is not int or host_start_time < 0
        ):
            raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")

        if "started_at" in entry:
            started_at = entry["started_at"]
            if type(started_at) not in {int, float}:
                raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")
            try:
                started_at = float(started_at)
            except OverflowError:
                raise TaskFenceStartupUnavailable(
                    "incompatible_process_checkpoint"
                ) from None
            if not math.isfinite(started_at) or started_at < 0:
                raise TaskFenceStartupUnavailable(
                    "incompatible_process_checkpoint"
                )
            started_at_projection: tuple[str, float | None] = (
                "persisted",
                started_at,
            )
        else:
            started_at_projection = ("legacy_missing", None)

        command = require_text(entry.get("command", "unknown"))
        task_id = require_text(entry.get("task_id", ""))
        session_key = require_text(entry.get("session_key", ""))
        cwd = entry.get("cwd")
        if cwd is not None:
            cwd = require_text(cwd)

        watcher_fields = tuple(
            require_text(entry.get(key, ""))
            for key in (
                "watcher_platform",
                "watcher_chat_id",
                "watcher_user_id",
                "watcher_user_name",
                "watcher_thread_id",
                "watcher_message_id",
            )
        )
        notify_on_complete = entry.get("notify_on_complete", False)
        if type(notify_on_complete) is not bool:
            raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")
        watch_patterns = entry.get("watch_patterns", [])
        if not isinstance(watch_patterns, list):
            raise TaskFenceStartupUnavailable("incompatible_process_checkpoint")
        normalized_patterns = tuple(require_text(pattern) for pattern in watch_patterns)

        identity = (session_id,)
        projection = (
            session_id,
            command,
            pid,
            pid_scope,
            host_start_time,
            cwd,
            started_at_projection,
            task_id,
            session_key,
            *watcher_fields,
            watcher_interval,
            notify_on_complete,
            normalized_patterns,
        )
        try:
            identity_bytes = json.dumps(
                identity,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            projection_bytes = json.dumps(
                projection,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError):
            raise TaskFenceStartupUnavailable(
                "incompatible_process_checkpoint"
            ) from None
        if len(observations) >= _MAX_PROCESS_CHECKPOINT_ENTRIES:
            raise TaskFenceStartupUnavailable(
                "process_checkpoint_limit_exceeded"
            )
        observations.append(
            _ProcessCheckpointObservation(
                invocation_id="tfqp_" + hashlib.sha256(identity_bytes).hexdigest(),
                invocation_fingerprint=hashlib.sha256(projection_bytes).hexdigest(),
            )
        )

    observations.sort(
        key=lambda observation: (
            observation.invocation_id,
            observation.invocation_fingerprint,
        )
    )
    return tuple(observations)


def prepare_task_fence_shadow_startup(
    shadow_conversation_key: str,
    *,
    hermes_home: Path | None = None,
    multiplex_profiles: bool = False,
) -> TaskFenceRecovery | None:
    """Run the selected cohort's receipt-bound recovery before consumers open."""

    if not shadow_conversation_key:
        return None
    if multiplex_profiles:
        raise TaskFenceStartupUnavailable("multiplex_profiles_unsupported")

    identity = _read_receipt()
    database = None
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        resolved_home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
        process_checkpoint = _read_process_checkpoint_snapshot(
            resolved_home / "processes.json",
            shadow_session_key=shadow_conversation_key,
        )
        logger.info(
            "Task Fence shadow startup captured %d process checkpoint "
            "recovery candidate(s)",
            len(process_checkpoint),
        )
        process_checkpoint_operations = tuple(
            OperationDescriptor(
                invocation_id=observation.invocation_id,
                kind=OperationKind.TOOL,
                adapter=TASK_FENCE_PROCESS_CHECKPOINT_RECOVERY_ADAPTER,
                invocation_fingerprint=observation.invocation_fingerprint,
            )
            for observation in process_checkpoint
        )
        database = SessionDB(resolved_home / "state.db")
        store = database.inspect_task_fence_store()
        if not store.compatible:
            raise TaskFenceStartupUnavailable(store.reason)
        if (
            store.runtime_epoch is None
            or store.mode_generation != 0
            or store.ever_enforced is not False
        ):
            raise TaskFenceStartupUnavailable("unsupported_shadow_control_state")
        database.materialize_task_fence_selected_cohort_capabilities(
            expected_runtime_epoch=store.runtime_epoch,
            expected_mode_generation=store.mode_generation,
        )
        return database.recover_task_fence_state(
            expected_runtime_epoch=store.runtime_epoch,
            expected_mode_generation=store.mode_generation,
            tested_artifact_identity=identity,
            shadow_session_key=shadow_conversation_key,
            process_checkpoint_operations=process_checkpoint_operations,
        )
    except TaskFenceStartupUnavailable:
        raise
    except (
        TaskFenceCapabilityUnavailable,
        TaskFenceProtocolRejected,
        TaskFenceRecoveryUnavailable,
    ) as exc:
        raise TaskFenceStartupUnavailable(exc.reason) from None
    except Exception as exc:
        raise TaskFenceStartupUnavailable("store_unavailable") from exc
    finally:
        if database is not None:
            database.close()
