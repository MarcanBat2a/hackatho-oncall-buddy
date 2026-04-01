from abc import ABC, abstractmethod

from src.domain.models import Message


class LLMPort(ABC):
    @abstractmethod
    async def complete(self, messages: list[Message], model: str) -> str: ...
