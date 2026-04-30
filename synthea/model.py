
from abc import abstractmethod
from typing import List

from synthea.CommandParser import ParsedArgs
from synthea.dtos.GenerationResponse import GenerationResponse
from langchain_core.messages import BaseMessage
from openai.types import Model

class Model:
    @abstractmethod
    async def queue_for_generation(self, 
                                   chat_history: list[BaseMessage],
                                   args: ParsedArgs,
                                   persona_system_prompt: str = None) -> GenerationResponse:
        pass

    @abstractmethod
    async def get_models(self) -> List[Model]:
        pass
