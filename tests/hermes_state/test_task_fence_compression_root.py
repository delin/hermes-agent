from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def test_task_fence_compression_root_walks_only_compression_chain(db: SessionDB) -> None:
    db.create_session("root", source="cli")
    db.end_session("root", "compression")
    db.create_session("middle", source="cli", parent_session_id="root")
    db.end_session("middle", "compression")
    db.create_session("tip", source="cli", parent_session_id="middle")

    assert db.get_task_fence_compression_root("tip") == "root"
    assert db.get_task_fence_compression_root("middle") == "root"
    assert db.get_task_fence_compression_root("root") == "root"


@pytest.mark.parametrize(
    ("source", "model_config"),
    [
        ("cli", {"_branched_from": "root"}),
        ("cli", {"_delegate_from": "root"}),
        ("tool", {}),
    ],
)
def test_task_fence_compression_root_rejects_non_compression_child_kinds(
    db: SessionDB,
    source: str,
    model_config: dict,
) -> None:
    db.create_session("root", source="cli")
    db.end_session("root", "compression")
    db.create_session(
        "child",
        source=source,
        parent_session_id="root",
        model_config=model_config,
    )

    assert db.get_task_fence_compression_root("child") == "child"


def test_task_fence_compression_root_stops_on_missing_or_malformed_rows(
    db: SessionDB,
) -> None:
    assert db.get_task_fence_compression_root("missing") == "missing"

    db.create_session("root", source="cli")
    db.end_session("root", "compression")
    db.create_session("child", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET model_config = ? WHERE id = ?",
        ("not-json", "child"),
    )
    db._conn.commit()

    assert db.get_task_fence_compression_root("child") == "child"


def test_task_fence_compression_root_requires_compression_ended_parent(
    db: SessionDB,
) -> None:
    db.create_session("root", source="cli")
    db.create_session("child", source="cli", parent_session_id="root")

    assert db.get_task_fence_compression_root("child") == "child"


def test_task_fence_compression_root_stops_before_cycle(db: SessionDB) -> None:
    db.create_session("a", source="cli")
    db.end_session("a", "compression")
    db.create_session("b", source="cli", parent_session_id="a")
    db.end_session("b", "compression")
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
        ("b", "a"),
    )
    db._conn.commit()

    assert db.get_task_fence_compression_root("a") == "b"
