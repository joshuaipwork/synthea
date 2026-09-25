import asyncio
import os
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from mem0 import AsyncMemory

from synthea.config import Config
from synthea.utilities import inference_logger

bot_config = Config()
os.environ["OPENAI_API_KEY"] = bot_config.api_key


def create_config(llm_model: str):
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "api_key": bot_config.api_key,
                "openai_base_url": bot_config.api_base_url,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": bot_config.embeddings_model,
                "api_key": bot_config.api_key,
                "openai_base_url": bot_config.embeddings_base_url,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": f"chatbot_memories-{bot_config.embeddings_model}",
                "path": "./chroma_db/memories",  # just a local folder
            },
        },
    }

    return config


# cache of AsyncMemory clients, keyed by the model used for memory extraction.
# constructing an AsyncMemory instantiates its LLM, embedder, vector store and
# analytics clients, so reuse one instance per model instead of rebuilding it on
# every call (also avoids repeated "multiple PostHog clients" warnings)
_MEMORY_CLIENTS: dict[str, AsyncMemory] = {}


def _get_memory(model_name: str) -> AsyncMemory:
    """Returns a cached AsyncMemory client for the given model, creating it if needed.
    """
    if model_name not in _MEMORY_CLIENTS:
        _MEMORY_CLIENTS[model_name] = AsyncMemory.from_config(create_config(model_name))
    return _MEMORY_CLIENTS[model_name]


# ---------------------------------------------------------------------------
# background memory writes
#
# Memories are saved *after* the reply has been handed back to the user, so the
# comparatively slow mem0 extraction/embedding/persistence work runs in
# parallel with (rather than before) the response. The in-flight saves are
# tracked here so callers that do need the writes to be complete (graceful
# shutdown, tests) can wait for them.
# ---------------------------------------------------------------------------

# one lock per extraction model: mem0/chroma clients are shared per model, so
# concurrent writes are serialized to keep them from interleaving
_SAVE_LOCKS: dict[str, asyncio.Lock] = {}

# every background save task that has been scheduled and not yet finished
_PENDING_SAVES: set[asyncio.Task] = set()


def _save_lock(model_name: str) -> asyncio.Lock:
    """Returns the lock serializing background saves for the given model."""
    if model_name not in _SAVE_LOCKS:
        _SAVE_LOCKS[model_name] = asyncio.Lock()
    return _SAVE_LOCKS[model_name]


def _on_save_finished(task: asyncio.Task) -> None:
    """Bookkeeping for a finished background save: forget it and log failures.

    Nobody awaits a background save, so its exception has to be retrieved and
    logged here, otherwise it would only surface as
    "Task exception was never retrieved" at shutdown.
    """
    _PENDING_SAVES.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        inference_logger.error(
            "Background memory save failed: %s", error, exc_info=error,
        )


async def _save_memories_in_background(
    model_name: str, messages: list[BaseMessage],
) -> None:
    async with _save_lock(model_name):
        await add_memories(messages=messages, model_name=model_name)


def schedule_memory_save(
    messages: list[BaseMessage], model_name: str,
) -> asyncio.Task:
    """Saves memories for ``messages`` in the background and returns immediately.

    The reply to the user is never delayed by this call: the returned task is
    already running but is not awaited. A copy of the message list is taken so
    later state updates cannot mutate what the save sees. Failures are logged
    by the done-callback instead of being raised, since nothing is waiting on
    the result.
    """
    task = asyncio.create_task(
        _save_memories_in_background(model_name, list(messages)),
    )
    _PENDING_SAVES.add(task)
    task.add_done_callback(_on_save_finished)
    return task


async def wait_for_pending_memory_saves(timeout: float | None = None) -> bool:
    """Waits for every scheduled background save to finish.

    Returns True if all of them finished before ``timeout`` seconds elapsed.
    """
    pending = [task for task in _PENDING_SAVES if not task.done()]
    if not pending:
        return True
    _, still_pending = await asyncio.wait(pending, timeout=timeout)
    return not still_pending


async def retrieve_relevant_memories(
    messages: list[BaseMessage], model_name: str,
) -> str:
    """From a list of messages, retrieves a list of relevant memories about the last user from mem0
    """
    memory = _get_memory(model_name)
    user_turns: list[HumanMessage] = [
        msg for msg in messages if isinstance(msg, HumanMessage)
    ]

    # get the user id from the last user
    user_id = user_turns[-1].name
    # user_content = [turn.content for turn in user_turns]
    user_content = extract_text(user_turns[-1].content)

    # retrieve the memories from the last user
    relevant_memories = await memory.search(
        query=user_content, filters={"user_id": user_id}, top_k=5,
    )

    memory_context = "\n".join(
        f"- {m['memory']}" for m in relevant_memories.get("results", [])
    )

    return memory_context


async def add_memories(messages: list[BaseMessage], model_name: str) -> str:
    """From a list of messages, save information to a list of memories about the last user
    from their own messages.
    """
    memory = _get_memory(model_name)

    # filter the messages down to only human messages to avoid stuffing the context
    user_turns = [msg for msg in messages if isinstance(msg, HumanMessage)]

    # get the user id from the last user
    user_id = user_turns[-1].name

    # filter the messages down to only the messages from the last user
    user_messages = [
        {"role": "user", "content": extract_text(turn.content)}
        for turn in user_turns
        if turn.name == user_id
    ]

    memories = await memory.add(
        user_messages,
        user_id=user_id,
        prompt="""Here is a message submitted by a user to a chatbot. Extract relevant long-term factual information about the user, which will be made available as context to future chat sessions.

Guidance:
- Disregard information the user provides about other AI assistants such as Syn or other system users.
- Focus on information which is likely to be stable about the user: the mere fact that the user asked a particular question is unlikely to be useful. If in doubt, make the call based on whether the information may be useful context even in unrelated topics of conversation.
- Information to remember may include the user's possessions, preferences, job/hobbies, personal context, aspirations, etc.""",
    )

    return memories


def extract_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content


# mem0's get_all() defaults to top_k=20 and truncates its answer to the first
# top_k rows the vector store hands back, in insertion order (oldest first).
# Left at the default, /view_memory could only ever show the 20 oldest memories
# - exactly two pages - and everything stored later stayed invisible no matter
# how far you paged. Ask for plenty (mem0 itself uses 10000 for its own
# listings); get_user_memories also sorts newest-first, so if the cap is ever
# reached it is the oldest entries that fall off rather than the newest.
MAX_LISTED_MEMORIES: int = 1_000


def _memory_recency(memory: dict[str, Any]) -> str:
    """Sort key ordering memories most-recently-written first.

    Timestamps are ISO strings (mem0's MemoryItem.created_at/updated_at);
    records carrying neither sort last instead of breaking the listing.
    """
    return str(memory.get("updated_at") or memory.get("created_at") or "")


async def get_user_memories(user_id) -> list[dict[str, Any]]:
    """Every stored memory about ``user_id``, newest first.

    The explicit top_k is load-bearing - see MAX_LISTED_MEMORIES.
    """
    memory = _get_memory(bot_config.default_model_name)

    result = await memory.get_all(
        filters={"user_id": user_id}, top_k=MAX_LISTED_MEMORIES,
    )
    memories: list[dict[str, Any]] = result.get("results", [])
    return sorted(memories, key=_memory_recency, reverse=True)


async def clear_user_memory(user_id: str, persona=None):
    memory = _get_memory(bot_config.default_model_name)

    if persona:
        await memory.delete_all(user_id=user_id, agent_id=persona)
    else:
        await memory.delete_all(user_id=user_id)


async def add_user_memory(
    new_memory: str, user_id: str, persona=None,
) -> dict[str, Any]:
    memory = _get_memory(bot_config.default_model_name)

    if persona:
        return await memory.add(new_memory, user_id=user_id, agent_id=persona)
    return await memory.add(new_memory, user_id=user_id)


async def delete_memory(memory_id: str):
    memory = _get_memory(bot_config.default_model_name)

    await memory.delete(memory_id)


async def get_memory(memory_id: str) -> dict[str, Any] | None:
    memory = _get_memory(bot_config.default_model_name)

    try:
        return await memory.get(memory_id)
    except IndexError:
        return None
