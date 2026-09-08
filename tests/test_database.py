"""Tests for ``Database`` init/schema seams added in P14c (WP1).

Covers the ``PRAGMA user_version`` stamp (in-memory, pre-existing file DB,
idempotent re-run) and the ``Database.path`` accessor.
"""

from __future__ import annotations

import pytest

from nerdit.db.database import Database


async def _user_version(db: Database) -> int:
    cursor = await db.conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    return row[0]


@pytest.mark.asyncio
async def test_user_version_stamped_in_memory():
    database = Database(":memory:")
    await database.connect()
    await database.init_schema()
    try:
        assert await _user_version(database) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_user_version_stamped_on_file_db(tmp_path):
    db_path = str(tmp_path / "nerdit.db")
    database = Database(db_path)
    await database.connect()
    await database.init_schema()
    try:
        assert await _user_version(database) == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_user_version_stamp_idempotent_on_reopen(tmp_path):
    db_path = str(tmp_path / "nerdit.db")

    first = Database(db_path)
    await first.connect()
    await first.init_schema()
    assert await _user_version(first) == 1
    await first.close()

    # Re-run init_schema against the pre-existing file DB: the stamp is an
    # absolute set, so it stays 1 (idempotent).
    second = Database(db_path)
    await second.connect()
    await second.init_schema()
    try:
        assert await _user_version(second) == 1
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_path_property_returns_db_path(tmp_path):
    db_path = str(tmp_path / "nerdit.db")
    database = Database(db_path)
    assert database.path == db_path

    memory = Database(":memory:")
    assert memory.path == ":memory:"
