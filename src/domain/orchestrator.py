from __future__ import annotations

import asyncio
import logging

from src.domain.models import AnalysisReport, ContextData, TicketData
from src.domain.debate import DebateEngine
from src.domain.synthesizer import Synthesizer
from src.ports.context import ContextPort
from src.ports.output import OutputPort
from src.ports.ticket import TicketPort

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        ticket_port: TicketPort,
        context_ports: list[ContextPort],
        output_port: OutputPort,
        debate_engine: DebateEngine,
        synthesizer: Synthesizer,
    ):
        self.ticket_port = ticket_port
        self.context_ports = context_ports
        self.output_port = output_port
        self.debate_engine = debate_engine
        self.synthesizer = synthesizer

    async def analyze(self, ticket_id: str) -> AnalysisReport:
        # 1. Fetch ticket
        logger.info("Fetching ticket %s", ticket_id)
        ticket = await self.ticket_port.fetch(ticket_id)
        logger.info("Ticket fetched: %s", ticket.title)

        # 2. Gather context from all sources in parallel (graceful degradation)
        logger.info("Gathering context from %d sources", len(self.context_ports))
        results = await asyncio.gather(
            *[port.gather(ticket) for port in self.context_ports],
            return_exceptions=True,
        )
        contexts: list[ContextData] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Context source %d failed: %s", i, result)
            else:
                contexts.append(result)
        total_files = sum(len(ctx.relevant_files) for ctx in contexts)
        logger.info("Context gathered: %d relevant files found", total_files)

        # 3. Run debate
        logger.info("Starting multi-agent debate")
        rounds = await self.debate_engine.run(ticket, contexts)
        logger.info("Debate completed: %d rounds", len(rounds))

        # 4. Synthesize
        logger.info("Synthesizing final report")
        report = await self.synthesizer.synthesize(ticket, rounds)
        logger.info("Report generated — confidence: %s", report.confidence)

        # 5. Publish (failure here should not lose the report)
        logger.info("Publishing report to %s", ticket_id)
        try:
            await self.output_port.publish(ticket_id, report)
            logger.info("Report published successfully")
        except Exception as e:
            logger.error("Failed to publish report to %s: %s", ticket_id, e)
            logger.info("Report content is still available in the return value")

        return report
