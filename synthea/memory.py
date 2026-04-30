from typing import Any

from mem0 import AsyncMemory

from synthea.config import Config

from langchain_core.messages import BaseMessage, HumanMessage
import os

bot_config = Config()
os.environ["OPENAI_API_KEY"] = bot_config.api_key

def create_config(llm_model: str, embedding_model: str):
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
                "model": embedding_model,
                "api_key": bot_config.api_key,
                "openai_base_url": bot_config.api_base_url,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": f"chatbot_memories-{embedding_model}",
                "path": "./chroma_db",  # just a local folder
            },
        },
    }

    return config


async def retrieve_relevant_memories(messages: list[BaseMessage], model_name: str) -> str:
    """
    From a list of messages, retrieves a list of relevant memories about the last user from mem0
    """
    memory = await AsyncMemory.from_config(create_config(model_name, model_name))
    user_turns: list[HumanMessage] = [msg for msg in messages if isinstance(msg, HumanMessage)]

    # get the user id from the last user
    user_id = user_turns[-1].name
    # user_content = [turn.content for turn in user_turns]
    user_content = extract_text(user_turns[-1].content)

    # retrieve the memories from the last user
    relevant_memories = await memory.search(query=user_content, user_id=user_id, limit=5)

    memory_context = "\n".join(
        f"- {m['memory']}" for m in relevant_memories.get("results", [])
    )

    return memory_context


async def add_memories(messages: list[BaseMessage], model_name: str) -> str:
    """
    From a list of messages, save information to a list of memories about the last user
    from their own messages.
    """
    memory = await AsyncMemory.from_config(create_config(model_name, model_name))

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
        prompt="Here are the last couple messages from a user. Extract factual information about the user from their messages. Ignore anything the user says about other users or any AI assistants. Focus on: the user's possessions, preferences, problems, goals, and personal context.",
    )

    return memories

def extract_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content

async def get_user_memories(user_id) -> list[dict[str, Any]]:
    memory = await AsyncMemory.from_config(create_config(
        bot_config.default_model_name, bot_config.default_model_name))
    
    result = await memory.get_all(user_id=user_id)
    return result.get("results", [])

async def clear_user_memory(user_id: str, persona=None):
    memory = await AsyncMemory.from_config(create_config(
        bot_config.default_model_name, bot_config.default_model_name))

    if persona:
        await memory.delete_all(user_id=user_id, agent_id=persona)
    else:
        await memory.delete_all(user_id=user_id)