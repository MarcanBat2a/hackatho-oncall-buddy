from abc import ABC, abstractmethod

from src.domain.models import ContextData, TicketData


class ContextPort(ABC):
    @abstractmethod
    async def gather(self, ticket: TicketData) -> ContextData: ...
