"""Tests for how /view_memory lists memories (``memory.get_user_memories``).

The regression these guard against: mem0's ``get_all`` defaults to
``top_k=20`` and returns the *oldest* rows first, so /view_memory used to show
exactly two pages of ten and silently hide everything stored more recently -
which is how a freshly saved memory can be in the store (search finds it) yet
never appear in the listing.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access

from datetime import datetime, timedelta, timezone

import pytest


class FakeMem0:
    """Mimics mem0's ``get_all``: keyword-only args, insertion order (oldest
    first), truncated to the first ``top_k`` rows - the behaviour behind the bug.
    """

    def __init__(self, stored: list[dict] | None = None):
        self.stored: list[dict] = stored or []
        self.calls: list[dict] = []

    async def get_all(self, *, filters=None, top_k=20, **kwargs) -> dict:
        self.calls.append({"filters": filters, "top_k": top_k})
        rows = [
            row
            for row in self.stored
            if row.get("user_id") == (filters or {}).get("user_id")
        ]
        return {"results": [dict(row) for row in rows[:top_k]]}


def make_memory(index: int, user_id: str = "user-1", **overrides) -> dict:
    """A memory as mem0 would return it, timestamped ``index`` minutes apart."""
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
    item = {
        "id": f"mem-{index:04d}",
        "memory": f"fact {index}",
        "user_id": user_id,
        "created_at": stamp.isoformat(),
        "updated_at": stamp.isoformat(),
    }
    item.update(overrides)
    return item


@pytest.fixture()
def memory():
    """The synthea.memory module, imported lazily.

    ``synthea.memory`` builds a ``Config()`` at import time, so it can only be
    imported once the session fixture has changed into the repo root.
    """
    from synthea import memory as memory_module

    return memory_module


@pytest.fixture()
def store(memory, monkeypatch):
    """An in-memory stand-in for the mem0 client, wired into synthea.memory."""
    fake = FakeMem0()
    monkeypatch.setattr(memory, "_get_memory", lambda model_name: fake)
    return fake


class TestGetUserMemories:
    async def test_lists_more_than_twenty_memories(self, memory, store):
        """The core regression: >20 memories must all come back.

        With mem0's default top_k left in place this returns 20 of 37, and the
        newest ones - the ones a user just watched get saved - are missing.
        """
        store.stored = [make_memory(i) for i in range(37)]

        memories = await memory.get_user_memories("user-1")

        assert len(memories) == 37
        assert {m["id"] for m in memories} == {
            f"mem-{i:04d}" for i in range(37)
        }

    async def test_explicitly_raises_the_top_k_cap(self, memory, store):
        """Guards the actual fix: if top_k stops being passed, this fails."""
        store.stored = [make_memory(i) for i in range(5)]

        await memory.get_user_memories("user-1")

        call = store.calls[-1]
        assert call["top_k"] == memory.MAX_LISTED_MEMORIES
        assert call["top_k"] > 100  # well above mem0's default of 20

    async def test_returns_newest_memories_first(self, memory, store):
        store.stored = [make_memory(i) for i in range(37)]

        memories = await memory.get_user_memories("user-1")

        assert memories[0]["id"] == "mem-0036"
        assert memories[-1]["id"] == "mem-0000"

    async def test_a_recently_updated_memory_counts_as_recent(
        self, memory, store,
    ):
        old = make_memory(0)
        updated = make_memory(
            1, updated_at="2026-12-31T23:59:00+00:00",
        )
        store.stored = [old, updated]

        memories = await memory.get_user_memories("user-1")

        assert [m["id"] for m in memories] == ["mem-0001", "mem-0000"]

    async def test_only_returns_the_requested_users_memories(
        self, memory, store,
    ):
        store.stored = [make_memory(i) for i in range(3)]
        store.stored += [make_memory(i, user_id="user-2") for i in range(4)]

        memories = await memory.get_user_memories("user-1")

        assert len(memories) == 3
        assert store.calls[-1]["filters"] == {"user_id": "user-1"}

    async def test_memories_without_timestamps_sort_last_and_do_not_crash(
        self, memory, store,
    ):
        stamped = make_memory(0)
        unstamped = make_memory(1, created_at=None, updated_at=None)
        store.stored = [unstamped, stamped]

        memories = await memory.get_user_memories("user-1")

        assert [m["id"] for m in memories] == ["mem-0000", "mem-0001"]

    async def test_returns_an_empty_list_when_nothing_is_stored(
        self, memory, store,
    ):
        assert await memory.get_user_memories("user-1") == []

    async def test_tolerates_a_result_without_the_results_key(
        self, memory, store,
    ):
        async def odd_get_all(*, filters=None, top_k=20, **kwargs):
            return {}

        store.get_all = odd_get_all

        assert await memory.get_user_memories("user-1") == []


class TestMemoryRecencyKey:
    def test_prefers_updated_at_over_created_at(self):
        from synthea.memory import _memory_recency

        key = _memory_recency(
            {"created_at": "2026-01-01T00:00:00+00:00",
             "updated_at": "2026-06-01T00:00:00+00:00"},
        )

        assert key == "2026-06-01T00:00:00+00:00"

    @pytest.mark.parametrize(
        "item",
        [{}, {"created_at": None, "updated_at": None}, {"created_at": "x"}],
    )
    def test_returns_a_comparable_string_for_odd_records(self, item):
        from synthea.memory import _memory_recency

        assert isinstance(_memory_recency(item), str)
