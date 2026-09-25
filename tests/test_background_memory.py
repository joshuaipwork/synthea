"""Tests that memories are saved in the background.

The reply path must never wait on mem0: ``schedule_memory_save`` hands the work
to a background task and returns immediately, failures are logged rather than
raised into the conversation, and callers that do need the writes to be done
can drain them with ``wait_for_pending_memory_saves``.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name

import asyncio
import logging
import types
from types import SimpleNamespace

import pytest

from synthea.utilities import inference_logger


@pytest.fixture()
async def memory():
    """The synthea.memory module, imported lazily.

    ``synthea.memory`` builds a ``Config()`` at import time, so it can only be
    imported once the session fixture has changed into the repo root. The
    fixture is async so that it is torn down inside the test's event loop,
    which is where the background tasks live.
    """
    from synthea import memory as memory_module

    memory_module._SAVE_LOCKS.clear()
    yield memory_module
    # don't leave a failing test's tasks dangling in the event loop
    await memory_module.wait_for_pending_memory_saves(timeout=5)
    memory_module._SAVE_LOCKS.clear()


@pytest.fixture()
def no_saves(memory, monkeypatch):
    """Records background saves instead of running them through mem0."""
    calls: list[tuple] = []

    async def fake_add_memories(messages, model_name):
        calls.append((messages, model_name))

    monkeypatch.setattr(memory, "add_memories", fake_add_memories)
    return calls


class TestScheduleMemorySave:
    async def test_returns_before_the_save_finishes(self, memory, monkeypatch):
        save_started = asyncio.Event()
        release_save = asyncio.Event()
        calls: list[tuple] = []

        async def slow_add_memories(messages, model_name):
            calls.append((messages, model_name))
            save_started.set()
            await release_save.wait()

        monkeypatch.setattr(memory, "add_memories", slow_add_memories)

        task = memory.schedule_memory_save(["hello"], "test-model")
        await asyncio.wait_for(save_started.wait(), timeout=5)

        # the caller already has control back while the save is still running
        assert not task.done()
        assert task in memory._PENDING_SAVES

        release_save.set()
        await asyncio.wait_for(task, timeout=5)
        assert calls == [(["hello"], "test-model")]

    async def test_snapshots_the_message_list(self, memory, no_saves):
        messages = ["first"]
        memory.schedule_memory_save(messages, "test-model")
        messages.append("appended later")
        await asyncio.wait_for(memory.wait_for_pending_memory_saves(), timeout=5)

        assert no_saves == [(["first"], "test-model")]

    async def test_forgets_finished_saves(self, memory, no_saves):
        task = memory.schedule_memory_save(["hello"], "test-model")
        await asyncio.wait_for(task, timeout=5)
        # the done-callback has to have run by now, since it is what asyncio.wait
        # internally wakes up on as well
        await asyncio.sleep(0)

        assert memory._PENDING_SAVES == set()
        assert await memory.wait_for_pending_memory_saves() is True

    async def test_saves_for_one_model_never_overlap(self, memory, monkeypatch):
        active = 0
        peak_concurrency = 0

        async def overlapping_add_memories(messages, model_name):
            nonlocal active, peak_concurrency
            active += 1
            peak_concurrency = max(peak_concurrency, active)
            await asyncio.sleep(0.01)
            active -= 1

        monkeypatch.setattr(memory, "add_memories", overlapping_add_memories)

        tasks = [
            memory.schedule_memory_save([f"message {i}"], "test-model")
            for i in range(3)
        ]
        await asyncio.wait_for(
            memory.wait_for_pending_memory_saves(), timeout=5,
        )

        assert all(task.done() for task in tasks)
        assert peak_concurrency == 1


class TestBackgroundSaveFailures:
    async def test_failures_are_logged_instead_of_raised(
        self, memory, monkeypatch, caplog,
    ):
        async def exploding_add_memories(messages, model_name):
            raise RuntimeError("mem0 unavailable")

        monkeypatch.setattr(memory, "add_memories", exploding_add_memories)

        with caplog.at_level(logging.ERROR, logger=inference_logger.name):
            memory.schedule_memory_save(["hello"], "test-model")
            assert (
                await memory.wait_for_pending_memory_saves(timeout=5) is True
            )
            # let any trailing done-callbacks run before asserting on the logs
            await asyncio.sleep(0)

        messages = [record.getMessage() for record in caplog.records]
        assert any("Background memory save failed" in m for m in messages)
        assert any(
            record.exc_info
            and isinstance(record.exc_info[1], RuntimeError)
            for record in caplog.records
        )


class TestWaitForPendingMemorySaves:
    async def test_waits_until_the_background_save_is_done(
        self, memory, monkeypatch,
    ):
        save_finished = asyncio.Event()

        async def slow_add_memories(messages, model_name):
            await asyncio.sleep(0.05)
            save_finished.set()

        monkeypatch.setattr(memory, "add_memories", slow_add_memories)
        memory.schedule_memory_save(["hello"], "test-model")

        assert await memory.wait_for_pending_memory_saves(timeout=5) is True
        assert save_finished.is_set()

    async def test_returns_immediately_when_nothing_is_pending(self, memory):
        assert await memory.wait_for_pending_memory_saves() is True

    async def test_reports_when_the_timeout_elapses(self, memory, monkeypatch):
        release_save = asyncio.Event()

        async def blocked_add_memories(messages, model_name):
            await release_save.wait()

        monkeypatch.setattr(memory, "add_memories", blocked_add_memories)
        memory.schedule_memory_save(["hello"], "test-model")

        assert await memory.wait_for_pending_memory_saves(timeout=0.05) is False

        release_save.set()
        await asyncio.wait_for(
            memory.wait_for_pending_memory_saves(), timeout=5,
        )


@pytest.fixture()
def memory_saver_node(memory):
    """The real ``memory_saver_node`` method, bound to a stand-in instance.

    ``AgenticModel.__init__`` constructs langfuse/openai/graph clients that these
    tests don't need, so only the method itself is borrowed from the class.
    """
    from synthea.agentic_model import AgenticModel

    model = SimpleNamespace(synthea_config=SimpleNamespace(enable_memory=True))
    model.memory_saver_node = types.MethodType(
        AgenticModel.memory_saver_node, model,
    )
    return model


def build_state(**overrides) -> dict:
    state = {
        "args": SimpleNamespace(use_as_system_prompt=False),
        "messages": ["user: hello", "bot: hi"],
        "model": "test-model",
    }
    state.update(overrides)
    return state


class TestMemorySaverNode:
    """The graph node must schedule the save and return without awaiting it."""

    async def test_schedules_a_background_save_and_returns(
        self, memory, memory_saver_node, monkeypatch,
    ):
        scheduled: list[tuple] = []

        def fake_schedule_memory_save(messages, model_name):
            scheduled.append((messages, model_name))
            return asyncio.create_task(asyncio.sleep(0))

        monkeypatch.setattr(
            memory, "schedule_memory_save", fake_schedule_memory_save,
        )

        result = await memory_saver_node.memory_saver_node(build_state())

        assert result == {}
        assert scheduled == [(["user: hello", "bot: hi"], "test-model")]

    async def test_skips_when_memory_is_disabled(
        self, memory, memory_saver_node, monkeypatch,
    ):
        memory_saver_node.synthea_config.enable_memory = False
        monkeypatch.setattr(
            memory, "schedule_memory_save",
            lambda *args, **kwargs: pytest.fail("should not schedule a save"),
        )

        result = await memory_saver_node.memory_saver_node(build_state())

        assert result is None

    async def test_graph_returns_while_the_save_is_still_running(
        self, memory, memory_saver_node, monkeypatch,
    ):
        """End to end: invoking a graph ending in the node must not block on mem0."""
        from langchain_core.messages import HumanMessage
        from langgraph.graph import END, START, StateGraph

        from synthea.agentic_model import AgentState

        save_started = asyncio.Event()
        release_save = asyncio.Event()

        async def blocked_add_memories(messages, model_name):
            save_started.set()
            await release_save.wait()

        monkeypatch.setattr(memory, "add_memories", blocked_add_memories)

        graph = StateGraph(AgentState)
        graph.add_node("memory_saver", memory_saver_node.memory_saver_node)
        graph.add_edge(START, "memory_saver")
        graph.add_edge("memory_saver", END)
        agent = graph.compile()

        state = {
            "messages": [HumanMessage(content="hello", name="user")],
            "memories": "",
            "system_prompt": "",
            "current_time": "now",
            "day_of_week": "Monday",
            "model": "test-model",
            "args": SimpleNamespace(use_as_system_prompt=False),
            "discord_metadata": None,
        }
        result = await asyncio.wait_for(
            agent.ainvoke(state, config={}), timeout=5,
        )

        # the graph finished... even though the memory write is still in flight
        await asyncio.wait_for(save_started.wait(), timeout=5)
        assert not release_save.is_set()
        assert result["messages"][-1].content == "hello"

        release_save.set()
        assert await memory.wait_for_pending_memory_saves(timeout=5) is True

    async def test_skips_for_custom_system_prompts(
        self, memory, memory_saver_node, monkeypatch,
    ):
        monkeypatch.setattr(
            memory, "schedule_memory_save",
            lambda *args, **kwargs: pytest.fail("should not schedule a save"),
        )

        result = await memory_saver_node.memory_saver_node(
            build_state(args=SimpleNamespace(use_as_system_prompt=True)),
        )

        assert result is None
