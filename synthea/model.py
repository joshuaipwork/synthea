from abc import abstractmethod

from langchain_core.messages import BaseMessage
from openai.types import Model

from synthea.commands import ParsedArgs
from synthea.context_manager import DiscordMetadata
from synthea.dtos import GenerationResponse


class Model:
    @abstractmethod
    async def queue_for_generation(
        self,
        chat_history: list[BaseMessage],
        args: ParsedArgs,
        discord_metadata: DiscordMetadata,
        persona_system_prompt: str = None,
    ) -> GenerationResponse:
        pass

    @abstractmethod
    async def get_models(self) -> list[Model]:
        pass
