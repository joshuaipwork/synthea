"""Tests for how the bot namespaces its vector collections.

chromaDB stores vectors in collections and rejects mixed vector shapes, so
everything the bot embeds (memories, saved documents) is stored in collections
named after the embeddings provider. Changing the provider in config.yaml must
therefore produce new collection names: the new vectors go into a fresh
collection instead of failing to merge with the ones the old service produced.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access

import re

import pytest

# chromaDB's documented collection name rules: 3-63 characters of
# [a-z0-9._-], starting and ending with an alphanumeric
CHROMA_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")

GUILD_ID = 1085939230284460102
USER_ID = 123456789012345678


@pytest.fixture()
def memory():
    """``synthea.memory``, imported lazily: it builds a ``Config()`` at import
    time, which needs the session fixture to have changed into the repo root.
    """
    from synthea import memory as memory_module

    return memory_module


@pytest.fixture()
def rag():
    """``synthea.rag``, imported lazily for the same reason."""
    from synthea import rag as rag_module

    return rag_module


def memory_collection(memory, config) -> str:
    """The collection name mem0 will be told to use, with ``config`` as the bot's."""
    return memory.create_config("test-model")["vector_store"]["config"][
        "collection_name"
    ]


class TestEmbeddingsScope:
    def test_scope_identifies_the_provider_and_model(self, config):
        config.embeddings_base_url = "https://embeddings.example.com/v1"
        config.embeddings_model = "qwen3-8b"

        assert config.embeddings_scope  # it always says something
        assert "qwen3-8b" in config.embeddings_scope

    def test_changing_the_provider_changes_the_scope(self, config):
        before = config.embeddings_scope

        config.embeddings_base_url = "https://another-provider.example.com/v1"

        assert config.embeddings_scope != before

    def test_changing_the_model_changes_the_scope(self, config):
        before = config.embeddings_scope

        config.embeddings_model = "some-other-embedding-model"

        assert config.embeddings_scope != before

    def test_the_same_settings_always_give_the_same_scope(self, config):
        assert config.embeddings_scope == config.embeddings_scope

    def test_a_trailing_slash_is_not_a_new_provider(self, config):
        config.embeddings_base_url = "https://embeddings.example.com/v1"
        before = config.embeddings_scope

        config.embeddings_base_url = "https://embeddings.example.com/v1/"

        assert config.embeddings_scope == before

    @pytest.mark.parametrize(
        "model",
        ["text-embedding-3-small", "BAAI/bge-large-en v1.5", "Qwen 3 !!", "x"],
    )
    def test_scope_survives_model_names_chroma_would_reject(self, config, model):
        config.embeddings_model = model

        assert CHROMA_NAME.match(f"chatbot_memories-{config.embeddings_scope}")

    def test_scope_stays_short_enough_for_a_full_collection_name(self, config):
        config.embeddings_model = "a" * 200
        config.embeddings_base_url = "https://example.com/" + "x" * 500

        # memories and documents both append prefixes of their own
        assert len(f"rag_docs_user_{USER_ID}-{config.embeddings_scope}") <= 63


class TestMemoryCollections:
    def test_memories_are_stored_per_embedding_space(
        self, config, memory, monkeypatch,
    ):
        monkeypatch.setattr(memory, "bot_config", config)

        name = memory_collection(memory, config)

        assert CHROMA_NAME.match(name)
        assert name.startswith("chatbot_memories-")
        assert config.embeddings_scope in name

    def test_switching_provider_opens_a_new_collection(
        self, config, memory, monkeypatch,
    ):
        monkeypatch.setattr(memory, "bot_config", config)
        before = memory_collection(memory, config)

        config.embeddings_base_url = "https://another-provider.example.com/v1"

        after = memory_collection(memory, config)

        assert after != before
        assert CHROMA_NAME.match(after)

    def test_memories_keep_their_collection_across_unrelated_config_changes(
        self, config, memory, monkeypatch,
    ):
        monkeypatch.setattr(memory, "bot_config", config)
        before = memory_collection(memory, config)

        config.max_new_tokens = 4096
        config.bot_name = "Someone Else"

        assert memory_collection(memory, config) == before


class TestDocumentCollections:
    @pytest.fixture()
    def rag_with_config(self, rag, config, monkeypatch):
        """``rag`` reading its collections from this test's config."""
        monkeypatch.setattr(rag, "Config", lambda: config)
        return rag

    def test_documents_are_stored_per_server_and_embedding_space(
        self, rag_with_config, config,
    ):
        name = rag_with_config.collection_name(GUILD_ID, USER_ID)

        assert CHROMA_NAME.match(name)
        assert name.startswith(f"rag_docs_{GUILD_ID}-")
        assert config.embeddings_scope in name

    def test_direct_messages_get_their_own_collection(
        self, rag_with_config, config,
    ):
        name = rag_with_config.collection_name(0, USER_ID)

        assert CHROMA_NAME.match(name)
        assert name.startswith(f"rag_docs_user_{USER_ID}-")
        assert config.embeddings_scope in name

    def test_switching_provider_opens_new_collections(self, rag_with_config, config):
        before = rag_with_config.collection_name(GUILD_ID, USER_ID)

        config.embeddings_base_url = "https://another-provider.example.com/v1"

        after = rag_with_config.collection_name(GUILD_ID, USER_ID)

        assert after != before
        assert CHROMA_NAME.match(after)

    def test_the_vectorstore_is_opened_on_that_collection(
        self, rag_with_config, config, monkeypatch,
    ):
        opened = {}

        def fake_chroma(**kwargs):
            opened.update(kwargs)
            return object()

        monkeypatch.setattr(rag_with_config, "Chroma", fake_chroma)

        rag_with_config.get_vectorstore(GUILD_ID, USER_ID)

        assert (
            opened["collection_name"]
            == rag_with_config.collection_name(GUILD_ID, USER_ID)
        )
        assert opened["embedding_function"].model == config.embeddings_model
