from abc import ABC, abstractmethod

from src.domain.models import TicketData


class TicketPort(ABC):
    @abstractmethod
    async def fetch(self, ticket_id: str) -> TicketData: ...
