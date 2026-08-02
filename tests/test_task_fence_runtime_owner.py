import sys

import pytest

import task_fence_runtime as runtime


@pytest.fixture(autouse=True)
def _release_owner_lock():
    runtime.release_task_fence_owner_lock()
    try:
        yield
    finally:
        runtime.release_task_fence_owner_lock()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock contention")
def test_owner_lock_is_exclusive_idempotent_and_releasable(tmp_path) -> None:
    import fcntl

    first_home = tmp_path / "first"
    second_home = tmp_path / "second"
    first_home.mkdir()
    lock_path = first_home / "task-fence-owner.lock"

    with lock_path.open("a+", encoding="utf-8") as contender:
        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert runtime.acquire_task_fence_owner_lock(first_home) is False
        fcntl.flock(contender.fileno(), fcntl.LOCK_UN)

    assert runtime.acquire_task_fence_owner_lock(first_home) is True
    assert runtime.acquire_task_fence_owner_lock(first_home) is True
    assert runtime.acquire_task_fence_owner_lock(second_home) is False

    runtime.release_task_fence_owner_lock()

    assert runtime.acquire_task_fence_owner_lock(second_home) is True
