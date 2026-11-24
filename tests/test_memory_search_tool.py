"""Tests for the agent's memory search tool.

The memories injected into the prompt are only the handful that matched the
opening of the request. ``search_memory`` is the tool the model reaches for
when that is not enough: it must look up the memories of the user who sent the
current message (never anyone else's), run through the same mem0 store as the
prompt injection, report itself as tool use, and stay out of requests that
opted out of memory with ``-nm``.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access

import types
from types import SimpleNamespace

import pytest

# stand-ins for a ParsedArgs carrying / not carrying the -nm flag
FLAGGED = SimpleNamespace(use_as_system_prompt=False, no_memory=True)
PLAIN = SimpleNamespace(use_as_system_prompt=False, no_memory=False)


@pytest.fixture()
def memory_module():
    """``synthea.memory``, imported lazily.

    The module builds a ``Config()`` at import time, so the import has to
    happen after the session fixture has changed into the repository root.
    """
    from synthea import memory as memory_module

    return memory_module


@pytest.fixture()
def agentic_module():
    """``synthea.agentic_model``, imported lazily for the same reason."""
    from synthea import agentic_model

    return agentic_model


@pytest.fixture()
def fake_memory_client(memory_module, monkeypatch):
    """A stand-in mem0 client that records the searches it is asked for."""

    class FakeClient:
        def __init__(self):
            self.searches = []
            self.results = []

        async def search(self, *, query, filters, top_k):
            self.searches.append(
                {"query": query, "filters": filters, "top_k": top_k},
            )
            return {"results": self.results}

    client = FakeClient()
    monkeypatch.setattr(memory_module, "_get_memory", lambda model_name: client)
    return client


def make_tool(name: str, result: str = "tool result"):
    async def ainvoke(args, config=None):
        return result

    return SimpleNamespace(name=name, ainvoke=ainvoke)


def tool_call(name: str, call_id: str, **args) -> dict:
    return {"name": name, "args": args or {"query": "the user's cat"}, "id": call_id}


class TestSearchUserMemories:
    """The lookup behind the tool: same store, same shape as the injection."""

    async def test_searches_the_given_user_with_a_top_k_cap(
        self, memory_module, fake_memory_client,
    ):
        fake_memory_client.results = [
            {"id": "abc", "memory": "the user has a cat called Biscuit"},
        ]

        found = await memory_module.search_user_memories(
            query="what is the cat called", user_id="42", model_name="test-model",
        )

        assert fake_memory_client.searches == [
            {
                "query": "what is the cat called",
                "filters": {"user_id": "42"},
                "top_k": memory_module.SEARCH_TOP_K,
            },
        ]
        # ids come back with the facts: /delete_memory needs them
        assert found == "- the user has a cat called Biscuit (id: abc)"

    async def test_says_so_when_nothing_matched(
        self, memory_module, fake_memory_client,
    ):
        found = await memory_module.search_user_memories(
            query="anything", user_id="42", model_name="test-model",
        )

        assert "No stored memories matched" in found
        assert found == found.strip()


class TestPromptInjectionStaysPlain:
    """The prompt injection keeps its original, id-free format."""

    async def test_injection_has_no_ids(
        self, memory_module, fake_memory_client,
    ):
        from langchain_core.messages import HumanMessage

        fake_memory_client.results = [
            {"id": "abc", "memory": "the user likes cats"},
            {"id": "def", "memory": "the user works night shifts"},
        ]

        context = await memory_module.retrieve_relevant_memories(
            messages=[HumanMessage(content="hello", name="42")],
            model_name="test-model",
        )

        assert context == (
            "- the user likes cats\n- the user works night shifts"
        )

    def test_empty_listing_is_an_empty_string(self, memory_module):
        assert memory_module.format_memories([]) == ""


class TestSearchMemoryTool:
    """The tool the model calls: identity from the request, nothing else."""

    def test_reports_itself_as_a_memory_search(self, agentic_module):
        from synthea.tool_report import describe_tool_call

        call = describe_tool_call(
            agentic_module.search_memory, {"query": "the user's cat", "other": 1},
        )

        assert (call.name, call.label) == ("search_memory", "memory search")
        assert call.detail == 'query="the user\'s cat"'

    def test_the_model_only_sees_the_query_argument(self, agentic_module):
        """The request context is injected, never something the model invents."""
        properties = agentic_module.search_memory.tool_call_schema \
            .model_json_schema()["properties"]

        assert list(properties) == ["query"]

    async def test_looks_up_the_sender_of_the_current_message(
        self, agentic_module, memory_module, monkeypatch,
    ):
        seen = {}

        async def fake_search(**kwargs):
            seen.update(kwargs)
            return "- the user likes cats (id: abc)"

        monkeypatch.setattr(memory_module, "search_user_memories", fake_search)

        result = await agentic_module.search_memory.ainvoke(
            {"query": "pets"},
            config={
                "configurable": {
                    "discord_metadata": SimpleNamespace(user_id=123456, guild_id=1),
                    "model_name": "test-model",
                },
            },
        )

        assert result == "- the user likes cats (id: abc)"
        # the user id is stored as a string everywhere else in mem0
        assert seen == {
            "query": "pets",
            "user_id": "123456",
            "model_name": "test-model",
        }

    async def test_falls_back_to_the_default_models_client(
        self, agentic_module, memory_module, monkeypatch,
    ):
        seen = {}

        async def fake_search(**kwargs):
            seen.update(kwargs)
            return "found"

        monkeypatch.setattr(memory_module, "search_user_memories", fake_search)

        await agentic_module.search_memory.ainvoke(
            {"query": "pets"},
            config={
                "configurable": {
                    "discord_metadata": SimpleNamespace(user_id=7, guild_id=None),
                },
            },
        )

        assert seen["model_name"] == memory_module.bot_config.default_model_name

    async def test_without_a_sender_nobody_is_searched(
        self, agentic_module, memory_module, monkeypatch,
    ):
        def fail_if_called(**kwargs):
            pytest.fail("no memory may be read when the sender is unknown")

        monkeypatch.setattr(memory_module, "search_user_memories", fail_if_called)

        result = await agentic_module.search_memory.ainvoke(
            {"query": "pets"},
            config={"configurable": {"discord_metadata": None}},
        )

        assert "unavailable" in result


class TestToolRegistration:
    """The tool exists only where memory itself is enabled."""

    @pytest.fixture()
    def model_factory(self, agentic_module, monkeypatch):
        def build(enable_memory: bool):
            real_config = agentic_module.Config

            def with_flag():
                config = real_config()
                config.enable_memory = enable_memory
                return config

            monkeypatch.setattr(agentic_module, "Config", with_flag)
            return agentic_module.AgenticModel()

        return build

    def test_the_tool_is_offered_when_memory_is_enabled(self, model_factory):
        model = model_factory(enable_memory=True)

        assert "search_memory" in [tool.name for tool in model.tools]

    def test_the_tool_is_withheld_when_memory_is_disabled(self, model_factory):
        model = model_factory(enable_memory=False)

        assert "search_memory" not in [tool.name for tool in model.tools]


class TestToolNodeGuardsMemory:
    """-nm requests cannot reach stored memories through a tool call."""

    @pytest.fixture()
    def tool_node(self, agentic_module):
        from synthea.tool_report import reported_as

        model = SimpleNamespace(
            tools=[
                reported_as(
                    make_tool("search_memory"), label="memory search",
                    show=("query",),
                ),
            ],
        )
        model.tool_node = types.MethodType(agentic_module.AgenticModel.tool_node, model)
        return model

    def build_state(self, args, *calls) -> dict:
        from langchain.messages import AIMessage

        return {
            "messages": [AIMessage(content="", tool_calls=list(calls))],
            "images": [],
            "args": args,
        }

    async def test_a_no_memory_request_is_refused_and_not_reported(self, tool_node):
        update = await tool_node.tool_node(
            self.build_state(FLAGGED, tool_call("search_memory", "1")),
            config={},
        )

        assert update["tools_used"] == []
        refusal = update["messages"][0]
        assert "turned off" in refusal.content

    async def test_a_plain_request_searches_and_is_reported(self, tool_node):
        update = await tool_node.tool_node(
            self.build_state(PLAIN, tool_call("search_memory", "1")),
            config={},
        )

        assert [call.name for call in update["tools_used"]] == ["search_memory"]
        assert update["tools_used"][0].label == "memory search"
        assert update["messages"][0].content == "tool result"


class TestQueueForGenerationCarriesRequestContext:
    """The graph run is told which model (and sender) it is working for."""

    @pytest.fixture()
    def model(self, agentic_module):
        from synthea.config import Config
        from synthea.openers import OpeningPhraseTracker

        instance = SimpleNamespace(
            tools=[],
            langfuse_handler=None,
            opening_phrase_tracker=OpeningPhraseTracker(),
        )
        instance.queue_for_generation = types.MethodType(
            agentic_module.AgenticModel.queue_for_generation, instance,
        )
        return instance

    async def test_configurable_carries_model_name_and_metadata(
        self, model, agentic_module,
    ):
        captured = {}

        async def ainvoke(state, config=None):
            captured["config"] = config
            from langchain.messages import AIMessage
            return {"messages": [AIMessage(content="answer")], "images": []}

        metadata = SimpleNamespace(user_id=99, guild_id=1)
        model.agent = SimpleNamespace(ainvoke=ainvoke)

        await model.queue_for_generation(
            [], discord_metadata=metadata,
        )

        configurable = captured["config"]["configurable"]
        assert configurable["discord_metadata"] is metadata
        assert configurable["model_name"] == agentic_module.Config().default_model_name