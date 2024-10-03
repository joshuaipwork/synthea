
from abc import abstractmethod


class Model:
    @abstractmethod
    async def queue_for_generation(self, chat_history: list[dict[str, dict[str, str]]]) -> str:
        pass