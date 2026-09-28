"""Tests for the tool use report that is shown to users.

Every reply the agent produces tells the user which tools were called and with
what (the search query, the url, ...), one line per call, so it is obvious
whether the answer came from live search or from the model's own knowledge.
Tools annotate themselves with the label and arguments they want shown, and
memory loading/saving — graph plumbing rather than tool use — must never show
up in that report.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access

import types
from types import SimpleNamespace

import pytest

from synthea.dtos import GenerationResponse
from synthea.tool_report import (
    MAX_REPORT_CHARS,
    NO_TOOLS_USED_TEXT,
    TOOL_USE_FIELD_NAME,
    ToolCall,
    describe_tool_call,
    format_tool_use,
    reported_as,
)


@pytest.fixture()
def client_module():
    """``synthea.client``, imported lazily.

    ``synthea.memory`` builds a ``Config()`` at import time (it is pulled in by
    ``synthea.client``), so the import has to happen after the session fixture
    has changed into the repository root.
    """
    from synthea import client

    return client


@pytest.fixture()
def agentic_module():
    """``synthea.agentic_model``, imported lazily for the same reason."""
    from synthea import agentic_model

    return agentic_model


def make_tool(name: str, result: str = "tool result", error: Exception = None):
    """A stand-in for a langchain tool: only ``name`` and ``ainvoke`` matter."""

    async def ainvoke(args, config=None):
        if error:
            raise error
        return result

    return SimpleNamespace(name=name, ainvoke=ainvoke)


def tool_call(name: str, call_id: str, **args) -> dict:
    arguments = {"query": "time in tokyo"} if not args else args
    return {"name": name, "args": arguments, "id": call_id}


class TestGenerationResponseToolUse:
    def test_tool_use_is_untracked_by_default(self):
        assert GenerationResponse().tools_used is None

    def test_tool_use_can_be_reported(self):
        response = GenerationResponse(
            tools_used=[ToolCall(name="tavily_search", label="web search")],
        )

        assert response.tools_used[0].name == "tavily_search"
        assert response.final_output == ""


class TestToolAnnotations:
    """Tools declare their own label and the arguments worth showing."""

    def test_decorator_labels_a_tool_function_and_picks_its_arguments(self):
        from langchain_core.tools import tool

        @reported_as(label="read page", show=("url", "query"))
        @tool
        def fetch(url: str) -> str:
            """fetches a page"""
            return url

        call = describe_tool_call(fetch, {"url": "https://example.com", "junk": "x"})

        # annotating must not change how the tool identifies itself
        assert call.name == "fetch"
        assert call.label == "read page"
        # arguments outside the annotation stay out of the report
        assert call.detail == 'url="https://example.com"'

    def test_an_instance_can_be_annotated_directly(self):
        search_tool = reported_as(
            SimpleNamespace(name="tavily_search", metadata=None),
            label="web search",
            show=("query",),
        )

        call = describe_tool_call(
            search_tool, {"query": "time in tokyo", "search_depth": "advanced"},
        )

        assert call.label == "web search"
        assert call.detail == 'query="time in tokyo"'

    def test_the_real_tools_annotate_themselves(self, agentic_module):
        """read_url, generate_image and document_search carry their annotations."""
        from synthea.url_reader import read_url

        page = describe_tool_call(
            read_url, {"url": "https://example.com", "junk": "x"},
        )
        image = describe_tool_call(
            agentic_module.generate_image, {"prompt": "a cat", "width": 1024},
        )
        documents = describe_tool_call(
            agentic_module.document_search,
            {"query": "policy", "exclude": [], "regex": None},
        )

        assert (page.label, page.detail) == ("read page", 'url="https://example.com"')
        assert image.label == "image generation"
        assert image.detail == 'prompt="a cat", width=1024'
        assert documents.label == "document search"
        assert documents.detail == 'query="policy"'

    def test_the_search_tool_is_annotated_when_wired_up(self, agentic_module):
        model = agentic_module.AgenticModel()
        search_tool = next(t for t in model.tools if t.name == "tavily_search")

        call = describe_tool_call(
            search_tool, {"query": "time in tokyo", "search_depth": "advanced"},
        )

        assert call.label == "web search"
        assert call.detail == 'query="time in tokyo"'


class TestDescribeToolCall:
    def test_renders_the_label_and_the_shown_arguments(self):
        tool = reported_as(
            SimpleNamespace(name="tavily_search"), label="web search",
            show=("query", "topic"),
        )

        call = describe_tool_call(
            tool, {"query": "time in tokyo", "search_depth": "advanced"},
        )

        assert call.name == "tavily_search"
        assert call.label == "web search"
        # arguments outside the annotation stay out of the report
        assert call.detail == 'query="time in tokyo"'

    def test_skips_arguments_the_model_did_not_pass(self):
        tool = reported_as(
            SimpleNamespace(name="read_url"), label="read page",
            show=("url", "query"),
        )

        call = describe_tool_call(tool, {"url": "https://example.com"})

        assert call.detail == 'url="https://example.com"'

    def test_falls_back_to_the_tool_name_and_every_argument(self):
        call = describe_tool_call(
            SimpleNamespace(name="mystery_tool"), {"query": "q", "count": 3},
        )

        assert call.label == "mystery_tool"
        assert call.detail == 'query="q", count=3'

    def test_collapses_and_truncates_long_values(self):
        tool = reported_as(
            SimpleNamespace(name="generate_image"), label="image generation",
            show=("prompt",),
        )

        call = describe_tool_call(tool, {"prompt": "a\n  very " * 50})

        assert "\n" not in call.detail
        assert len(call.detail) < 200
        assert call.detail.endswith('…"')

    def test_skips_empty_optional_arguments(self):
        tool = reported_as(
            SimpleNamespace(name="document_search"), label="document search",
            show=("query", "require"),
        )

        call = describe_tool_call(
            tool, {"query": "policy", "require": [], "exclude": None},
        )

        assert call.detail == 'query="policy"'


class TestFormatToolUse:
    def test_untracked_tool_use_renders_nothing(self):
        assert format_tool_use(None) is None

    def test_no_tools_says_the_reply_is_the_models_own_knowledge(self):
        assert format_tool_use([]) == NO_TOOLS_USED_TEXT

    def test_one_line_per_tool_call_in_call_order(self):
        report = format_tool_use(
            [
                ToolCall(
                    name="tavily_search", label="web search",
                    detail='query="time in tokyo"',
                ),
                ToolCall(
                    name="read_url", label="read page",
                    detail='url="https://example.com"',
                ),
            ],
        )

        assert report.splitlines() == [
            '• web search: query="time in tokyo"',
            '• read page: url="https://example.com"',
        ]

    def test_a_call_without_detail_still_gets_its_line(self):
        assert format_tool_use([ToolCall(name="x", label="thing")]) == "• thing"

    def test_stays_within_the_embed_field_limit(self):
        calls = [
            ToolCall(
                name="read_url", label="read page",
                detail=f'url="https://example.com/{index}"',
            )
            for index in range(100)
        ]

        report = format_tool_use(calls)

        assert len(report) <= MAX_REPORT_CHARS
        # whole lines only, and the truncation is announced
        assert report.endswith("more)")
        assert "\n".join(report.splitlines()) == report

    def test_a_single_oversized_line_is_trimmed_to_the_limit(self):
        report = format_tool_use(
            [ToolCall(name="x", label="x" * 5000, detail="y" * 5000)],
        )

        assert len(report) <= MAX_REPORT_CHARS


class TestToolNodeRecordsToolUse:
    @pytest.fixture()
    def tool_node(self, agentic_module):
        model = SimpleNamespace(tools=[])
        model.tool_node = types.MethodType(agentic_module.AgenticModel.tool_node, model)
        return model

    def build_state(self, *calls) -> dict:
        from langchain.messages import AIMessage

        return {
            "messages": [
                AIMessage(content="", tool_calls=list(calls)),
            ],
            "images": [],
        }

    async def test_records_successful_calls_with_their_arguments(self, tool_node):
        tool_node.tools = [
            reported_as(
                make_tool("read_url", result="page text"), label="read page",
                show=("url", "query"),
            ),
            reported_as(
                make_tool("tavily_search", result="search results"),
                label="web search", show=("query",),
            ),
        ]

        update = await tool_node.tool_node(
            self.build_state(
                tool_call("read_url", "1", url="https://example.com"),
                tool_call("tavily_search", "2"),
            ),
            config={},
        )

        calls = update["tools_used"]
        assert [call.name for call in calls] == ["read_url", "tavily_search"]
        assert [call.label for call in calls] == ["read page", "web search"]
        assert calls[0].detail == 'url="https://example.com"'
        assert calls[1].detail == 'query="time in tokyo"'
        assert len(update["messages"]) == 2

    async def test_does_not_record_failed_calls(self, tool_node):
        """A tool that blew up contributed nothing to the answer."""
        tool_node.tools = [
            make_tool("tavily_search", error=RuntimeError("tavily down")),
            make_tool("read_url", result="page text"),
        ]

        update = await tool_node.tool_node(
            self.build_state(
                tool_call("tavily_search", "1"), tool_call("read_url", "2"),
            ),
            config={},
        )

        assert [call.name for call in update["tools_used"]] == ["read_url"]

    async def test_reports_an_empty_list_when_nothing_was_called(self, tool_node):
        from langchain.messages import AIMessage

        update = await tool_node.tool_node(
            {"messages": [AIMessage(content="hi")], "images": []}, config={},
        )

        assert update["tools_used"] == []


class TestMemoryNodesAreNotReported:
    """Memory loading/saving is not tool use and must never be reported."""

    @pytest.fixture()
    def memory_nodes(self, agentic_module):
        model = SimpleNamespace(synthea_config=SimpleNamespace(enable_memory=True))
        model.memory_retrieval_node = types.MethodType(
            agentic_module.AgenticModel.memory_retrieval_node, model,
        )
        model.memory_saver_node = types.MethodType(
            agentic_module.AgenticModel.memory_saver_node, model,
        )
        return model

    def build_state(self) -> dict:
        return {
            "args": SimpleNamespace(use_as_system_prompt=False),
            "messages": ["user: hello"],
            "model": "test-model",
        }

    async def test_memory_retrieval_is_not_reported(
        self, memory_nodes, monkeypatch,
    ):
        async def fake_retrieve(messages, model_name):
            return "the user likes cats"

        monkeypatch.setattr("synthea.memory.retrieve_relevant_memories", fake_retrieve)

        update = await memory_nodes.memory_retrieval_node(self.build_state())

        assert "tools_used" not in update

    async def test_memory_saving_is_not_reported(self, memory_nodes, monkeypatch):
        monkeypatch.setattr(
            "synthea.memory.schedule_memory_save", lambda **kwargs: None,
        )

        update = await memory_nodes.memory_saver_node(self.build_state())

        assert "tools_used" not in (update or {})


class TestGraphToolUseReport:
    """End to end through the graph: tool calls accumulate, memory does not."""

    async def test_accumulates_across_rounds_and_ignores_memory(
        self, agentic_module, monkeypatch,
    ):
        from langchain.messages import AIMessage, HumanMessage
        from langgraph.graph import END, START, StateGraph

        from synthea import memory as memory_module

        scheduled = []
        monkeypatch.setattr(
            memory_module,
            "schedule_memory_save",
            lambda **kwargs: scheduled.append(kwargs),
        )

        model = SimpleNamespace(
            synthea_config=SimpleNamespace(enable_memory=True), tools=[],
        )
        model.tool_node = types.MethodType(agentic_module.AgenticModel.tool_node, model)
        model.memory_saver_node = types.MethodType(
            agentic_module.AgenticModel.memory_saver_node, model,
        )
        model.tools = [
            reported_as(
                make_tool("tavily_search", result="search results"),
                label="web search", show=("query",),
            ),
            reported_as(
                make_tool("read_url", result="page text"), label="read page",
                show=("url",),
            ),
            make_tool("broken_tool", error=RuntimeError("boom")),
        ]

        async def second_request(state):
            """Makes the model ask for a second round of tools."""
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            tool_call("read_url", "2", url="https://example.com"),
                        ],
                    ),
                ],
            }

        rounds = {"requested_second": False}

        def after_tools(state):
            if not rounds["requested_second"]:
                rounds["requested_second"] = True
                return "second_request"
            return "memory_saver"

        graph = StateGraph(agentic_module.AgentState)
        graph.add_node("tools", model.tool_node)
        graph.add_node("second_request", second_request)
        graph.add_node("memory_saver", model.memory_saver_node)
        graph.add_edge(START, "tools")
        graph.add_conditional_edges(
            "tools",
            after_tools,
            {"second_request": "second_request", "memory_saver": "memory_saver"},
        )
        graph.add_edge("second_request", "tools")
        graph.add_edge("memory_saver", END)
        agent = graph.compile()

        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content="hello"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            tool_call("tavily_search", "1"),
                            tool_call("broken_tool", "3"),
                        ],
                    ),
                ],
                "memories": "the user likes cats",
                "system_prompt": "",
                "current_time": "now",
                "day_of_week": "Monday",
                "model": "test-model",
                "args": SimpleNamespace(use_as_system_prompt=False),
                "discord_metadata": None,
            },
        )

        # both rounds are reported, in order, and memory saving is not among them
        assert [call.name for call in result["tools_used"]] == [
            "tavily_search",
            "read_url",
        ]
        assert format_tool_use(result["tools_used"]).splitlines() == [
            '• web search: query="time in tokyo"',
            '• read page: url="https://example.com"',
        ]
        assert scheduled  # the memory save really did happen
        assert all(
            call.name != "memory_saver" for call in result["tools_used"]
        )


class TestQueueForGenerationReportsToolUse:
    """The response handed back to the client carries the tool report."""

    @pytest.fixture()
    def model(self, agentic_module):
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

    def stub_agent(self, result: dict):
        async def ainvoke(state, config=None):
            return result

        return SimpleNamespace(ainvoke=ainvoke)

    async def test_response_lists_the_tools_the_agent_called(self, model):
        from langchain.messages import AIMessage

        model.agent = self.stub_agent(
            {
                "messages": [AIMessage(content="answer")],
                "images": [],
                "tools_used": [
                    ToolCall(
                        name="tavily_search", label="web search",
                        detail='query="time in tokyo"',
                    ),
                ],
            },
        )

        response = await model.queue_for_generation([])

        assert [call.name for call in response.tools_used] == ["tavily_search"]
        assert response.final_output == "answer"

    async def test_response_reports_when_no_tools_were_called(self, model):
        from langchain.messages import AIMessage

        model.agent = self.stub_agent(
            {"messages": [AIMessage(content="answer")], "images": []},
        )

        response = await model.queue_for_generation([])

        assert response.tools_used == []


class TestClientRendersToolUse:
    """The tool report reaches the user as a field on the reply embed."""

    @pytest.fixture()
    def sender(self, client_module):
        """A stand-in with the send methods bound and sending captured."""
        sent = []

        async def send_response(**kwargs):
            sent.append(kwargs)

        instance = SimpleNamespace(sent=sent, send_response=send_response)
        instance.convert_generation_response_to_files = types.MethodType(
            client_module.SyntheaClient.convert_generation_response_to_files,
            instance,
        )
        instance.send_response_as_base = types.MethodType(
            client_module.SyntheaClient.send_response_as_base, instance,
        )
        instance.send_response_as_character = types.MethodType(
            client_module.SyntheaClient.send_response_as_character, instance,
        )
        return instance

    def field(self, embed) -> str | None:
        for field in embed.fields:
            if field.name == TOOL_USE_FIELD_NAME:
                return field.value
        return None

    def build_response(self, tools_used):
        return GenerationResponse(final_output="the answer", tools_used=tools_used)

    async def test_base_response_names_the_tools_called(self, sender):
        await sender.send_response_as_base(
            self.build_response(
                [
                    ToolCall(
                        name="tavily_search", label="web search",
                        detail='query="time in tokyo"',
                    ),
                ],
            ),
            message=None,
        )

        value = self.field(sender.sent[-1]["embed"])

        assert value == '• web search: query="time in tokyo"'

    async def test_base_response_shows_one_line_per_call(self, sender):
        await sender.send_response_as_base(
            self.build_response(
                [
                    ToolCall(
                        name="tavily_search", label="web search",
                        detail='query="time in tokyo"',
                    ),
                    ToolCall(
                        name="read_url", label="read page",
                        detail='url="https://example.com"',
                    ),
                ],
            ),
            message=None,
        )

        value = self.field(sender.sent[-1]["embed"])

        assert value.splitlines() == [
            '• web search: query="time in tokyo"',
            '• read page: url="https://example.com"',
        ]

    async def test_base_response_says_when_no_tools_were_used(self, sender):
        await sender.send_response_as_base(
            self.build_response([]), message=None,
        )

        assert self.field(sender.sent[-1]["embed"]) == NO_TOOLS_USED_TEXT

    async def test_base_response_has_no_field_when_tool_use_untracked(self, sender):
        await sender.send_response_as_base(
            GenerationResponse(final_output="the answer"), message=None,
        )

        assert self.field(sender.sent[-1]["embed"]) is None

    async def test_character_response_reports_tool_use_next_to_its_footer(
        self, sender,
    ):
        await sender.send_response_as_character(
            self.build_response(
                [
                    ToolCall(
                        name="document_search", label="document search",
                        detail='query="expense policy"',
                    ),
                ],
            ),
            {"id": "syn", "display_name": "Syn"},
            message=None,
        )

        embed = sender.sent[-1]["embed"]
        # the character id keeps living in the footer, chat history depends on it
        assert embed.footer.text == "syn"
        assert self.field(embed) == '• document search: query="expense policy"'

    async def test_replies_stay_within_the_embed_limits(self, sender):
        long_response = GenerationResponse(
            final_output="x" * 4000,
            tools_used=[
                ToolCall(
                    name="read_url", label="read page",
                    detail=f'url="https://example.com/{index}"',
                )
                for index in range(50)
            ],
        )

        await sender.send_response_as_base(long_response, message=None)

        embed = sender.sent[-1]["embed"]
        assert len(embed.description) <= 4000
        assert len(embed) <= 6000
        assert len(self.field(embed)) <= MAX_REPORT_CHARS
