from abc import ABC, abstractmethod

from src.domain.models import AnalysisReport


class OutputPort(ABC):
    @abstractmethod
    async def publish(self, ticket_id: str, report: AnalysisReport) -> None: ...
