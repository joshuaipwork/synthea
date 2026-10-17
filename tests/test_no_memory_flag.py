"""Tests for requests that opt out of long-term memory with ``-nm``.

The opt-out rides in the arguments of a single request: when ``no_memory`` is
set on the parsed arguments, the graph must neither load memories about the
user for that reply nor write the conversation back to mem0 afterwards. Both
memory nodes decide from the same predicate, so a request can never read
memory and then store it anyway (or the other way around).
"""

# pylint: disable=missing-function-docstring, redefined-outer-name

import types
from types import SimpleNamespace

import pytest

# stand-ins for a ParsedArgs carrying / not carrying the flag
FLAGGED = SimpleNamespace(use_as_system_prompt=False, no_memory=True)
PLAIN = SimpleNamespace(use_as_system_prompt=False, no_memory=False)
SYSTEM_PROMPT = SimpleNamespace(use_as_system_prompt=True, no_memory=False)


@pytest.fixture()
async def memory():
    """``synthea.memory``, imported lazily.

    ``synthea.memory`` builds a ``Config()`` at import time, so it can only be
    imported once the session fixture has changed into the repo root. The
    fixture is async so that it is torn down inside the test's event loop,
    which is where background tasks live.
    """
    from synthea import memory as memory_module

    memory_module._SAVE_LOCKS.clear()
    yield memory_module
    # don't leave a failing test's tasks dangling in the event loop
    await memory_module.wait_for_pending_memory_saves(timeout=5)
    memory_module._SAVE_LOCKS.clear()


@pytest.fixture()
def memory_nodes(memory):
    """Both memory nodes, bound to a stand-in instance.

    ``AgenticModel.__init__`` constructs langfuse/openai/graph clients these
    tests don't need, so only the methods themselves are borrowed from the
    class.
    """
    from synthea.agentic_model import AgenticModel

    model = SimpleNamespace(synthea_config=SimpleNamespace(enable_memory=True))
    model.memory_retrieval_node = types.MethodType(
        AgenticModel.memory_retrieval_node, model,
    )
    model.memory_saver_node = types.MethodType(
        AgenticModel.memory_saver_node, model,
    )
    return model


def build_state(args) -> dict:
    return {
        "args": args,
        "messages": ["user: hello"],
        "model": "test-model",
    }


def fail_if_called(*args, **kwargs):
    pytest.fail("memory must not be invoked for this request")


class TestMemoryRetrievalNode:
    async def test_flagged_requests_load_no_memories(
        self, memory, memory_nodes, monkeypatch,
    ):
        monkeypatch.setattr(
            memory, "retrieve_relevant_memories", fail_if_called,
        )

        result = await memory_nodes.memory_retrieval_node(build_state(FLAGGED))

        assert result is None

    async def test_flagged_requests_have_no_memories_in_state(
        self, memory, memory_nodes, monkeypatch,
    ):
        # returning nothing leaves the supervisor with its "Nothing yet" fallback
        async def fake_retrieve(messages, model_name):
            return "the user likes cats"

        monkeypatch.setattr(memory, "retrieve_relevant_memories", fake_retrieve)

        result = await memory_nodes.memory_retrieval_node(build_state(FLAGGED))

        assert result is None

    async def test_plain_requests_still_load_memories(
        self, memory, memory_nodes, monkeypatch,
    ):
        async def fake_retrieve(messages, model_name):
            return "the user likes cats"

        monkeypatch.setattr(memory, "retrieve_relevant_memories", fake_retrieve)

        result = await memory_nodes.memory_retrieval_node(build_state(PLAIN))

        assert result == {"memories": "the user likes cats"}


class TestMemorySaverNode:
    async def test_flagged_requests_are_not_saved(
        self, memory, memory_nodes, monkeypatch,
    ):
        monkeypatch.setattr(memory, "schedule_memory_save", fail_if_called)

        result = await memory_nodes.memory_saver_node(build_state(FLAGGED))

        assert result is None

    async def test_plain_requests_are_still_saved(
        self, memory, memory_nodes, monkeypatch,
    ):
        scheduled: list[tuple] = []
        monkeypatch.setattr(
            memory,
            "schedule_memory_save",
            lambda messages, model_name: scheduled.append((messages, model_name)),
        )

        result = await memory_nodes.memory_saver_node(build_state(PLAIN))

        assert result == {}
        assert scheduled == [(["user: hello"], "test-model")]


class TestSkipMemory:
    """The predicate behind both nodes, including the flags that predate -nm."""

    async def _skip(self, state):
        from synthea.agentic_model import skip_memory

        return skip_memory(state)

    async def test_no_memory_flag_skips(self):
        assert await self._skip(build_state(FLAGGED)) is True

    async def test_plain_request_does_not_skip(self):
        assert await self._skip(build_state(PLAIN)) is False

    async def test_custom_system_prompt_still_skips(self):
        assert await self._skip(build_state(SYSTEM_PROMPT)) is True

    async def test_missing_arguments_do_not_skip(self):
        # a state without parsed arguments is a plain request, not an opt-out
        assert await self._skip(build_state(None)) is False