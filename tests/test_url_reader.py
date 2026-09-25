"""Tests for the read_url tool (synthea.url_reader).

This covers the Tavily-backed reader on main. The local fetch fallback (direct
HTTP fetch, trafilatura, HTML -> text, and the private-address guard that
protects it) is tested on the `url-reader-local-fallback` branch.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access

from types import SimpleNamespace

import pytest

from synthea import url_reader


@pytest.fixture()
def fake_config(monkeypatch):
    """Replaces synthea.url_reader's Config with an in-memory stand-in."""

    def apply(**overrides) -> SimpleNamespace:
        settings = SimpleNamespace(
            tavily_api_key="test-key", url_reader_max_chars=20_000,
        )
        for name, value in overrides.items():
            setattr(settings, name, value)
        monkeypatch.setattr(url_reader, "Config", lambda: settings)
        return settings

    return apply


async def _fail_tavily_client(key):
    pytest.fail("no Tavily client should be built for an invalid URL")


def _async_value(value):
    """An awaitable that immediately yields ``value``."""

    async def coroutine():
        return value

    return coroutine()


class TestValidateUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/page",
            "http://example.com/page",
            "https://example.com/page?q=1&x=2#frag",
            "http://localhost:8000/",  # fetching is Tavily's problem, not ours
        ],
    )
    def test_accepts_readable_links(self, url):
        assert url_reader.validate_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "  <https://example.com/page>  ",  # markdown/discord wrapping
            "\nhttps://example.com/page\n",
        ],
    )
    def test_cleans_up_wrapped_links(self, url):
        assert url_reader.validate_url(url) == "https://example.com/page"

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/file",
            "javascript:alert(1)",
            "file:///etc/passwd",
            "example.com/no-scheme",
            "/relative/path",
            "https://",
            "http://",
            "",
        ],
    )
    def test_rejects_unreadable_links(self, url):
        with pytest.raises(ValueError):
            url_reader.validate_url(url)


class TestReadUrlContent:
    async def test_returns_the_extracted_text(self, fake_config, monkeypatch):
        fake_config()
        monkeypatch.setattr(
            url_reader, "_extract_with_tavily",
            lambda url, query: _async_value("from tavily " * 30),
        )

        assert (
            await url_reader.read_url_content("https://example.com")
            == "from tavily " * 30
        )

    async def test_forwards_the_query_to_tavily(self, fake_config, monkeypatch):
        fake_config()
        captured: dict = {}

        class FakeExtractTool:
            async def ainvoke(self, params):
                captured.update(params)
                return {"results": [{"raw_content": "the page " * 30}]}

        monkeypatch.setattr(
            url_reader, "_get_tavily_extract", lambda key: FakeExtractTool(),
        )

        await url_reader.read_url_content(
            "https://example.com/pricing", query="how much?",
        )

        assert captured["urls"] == ["https://example.com/pricing"]
        assert captured["query"] == "how much?"

    async def test_treats_a_tavily_error_message_as_a_failed_read(
        self, fake_config, monkeypatch,
    ):
        """handle_tool_error hands failures back as a string, not an exception."""
        fake_config()
        monkeypatch.setattr(
            url_reader, "_extract_with_tavily", lambda url, query: _async_value(None),
        )

        with pytest.raises(url_reader.UrlReadError):
            await url_reader.read_url_content("https://example.com/unreadable")

    async def test_raises_a_helpful_error_when_there_is_no_content(
        self, fake_config, monkeypatch,
    ):
        fake_config()
        monkeypatch.setattr(
            url_reader, "_extract_with_tavily", lambda url, query: _async_value(None),
        )

        with pytest.raises(url_reader.UrlReadError) as excinfo:
            await url_reader.read_url_content("https://example.com/unreadable")

        message = str(excinfo.value)
        assert "https://example.com/unreadable" in message
        assert "search instead" in message  # steer the agent away from retrying

    async def test_skips_tavily_entirely_without_an_api_key(
        self, fake_config, monkeypatch,
    ):
        fake_config(tavily_api_key="")
        monkeypatch.setattr(
            url_reader, "_get_tavily_extract", _fail_tavily_client,
        )

        with pytest.raises(url_reader.UrlReadError):
            await url_reader.read_url_content("https://example.com")

    async def test_truncates_over_long_pages(self, fake_config, monkeypatch):
        fake_config(url_reader_max_chars=100)
        monkeypatch.setattr(
            url_reader, "_extract_with_tavily",
            lambda url, query: _async_value("x" * 50_000),
        )

        result = await url_reader.read_url_content("https://example.com")

        assert len(result) < 300
        assert result.endswith("]")
        assert "50000 characters" in result
        assert result.startswith("x" * 100)

    async def test_rejects_bad_urls_before_calling_tavily(
        self, fake_config, monkeypatch,
    ):
        fake_config()
        monkeypatch.setattr(url_reader, "_get_tavily_extract", _fail_tavily_client)

        with pytest.raises(ValueError):
            await url_reader.read_url_content("file:///etc/passwd")


class TestReadUrlTool:
    async def test_delegates_to_read_url_content(self, monkeypatch):
        async def fake_read(url, query=None):
            return f"read {url}"

        monkeypatch.setattr(url_reader, "read_url_content", fake_read)

        result = await url_reader.read_url.ainvoke(
            {"url": "https://example.com"},
        )

        assert result == "read https://example.com"

    async def test_surfaces_validation_errors_to_the_agent(self):
        with pytest.raises(ValueError):
            await url_reader.read_url.ainvoke({"url": "file:///etc/passwd"})

    def test_exposes_url_and_optional_query(self):
        schema = url_reader.read_url.args_schema.model_json_schema()

        assert set(schema["properties"]) == {"url", "query"}
        assert "url" in schema["required"]


class TestToolRegistration:
    def test_read_url_is_registered(self):
        from synthea.agentic_model import AgenticModel

        model = AgenticModel()

        assert "read_url" in [tool.name for tool in model.tools]

    def test_read_url_can_be_disabled(self, monkeypatch):
        from synthea.agentic_model import AgenticModel
        from synthea.config import Config

        config = Config()
        config.enable_url_reader = False
        monkeypatch.setattr("synthea.agentic_model.Config", lambda: config)

        model = AgenticModel()

        assert "read_url" not in [tool.name for tool in model.tools]
