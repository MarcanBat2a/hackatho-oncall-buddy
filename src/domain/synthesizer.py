from __future__ import annotations

import json
import logging

from src.domain.models import AnalysisReport, DebateRound, Message, TicketData
from src.ports.llm import LLMPort

logger = logging.getLogger(__name__)


class Synthesizer:
    def __init__(self, model: str, prompt_template: str, llm: LLMPort):
        self.model = model
        self.prompt_template = prompt_template
        self.llm = llm

    def _build_transcript(self, rounds: list[DebateRound]) -> str:
        parts = []
        for debate_round in rounds:
            parts.append(f"## Round {debate_round.round_number}")
            parts.append(f"**Moderator question:** {debate_round.moderator_prompt}")
            for r in debate_round.responses:
                parts.append(f"\n**{r.agent_name}** ({r.model}):\n{r.content}")
            parts.append(f"\n**Moderator synthesis:** {debate_round.moderator_synthesis}")
            parts.append("---")
        return "\n\n".join(parts)

    async def synthesize(
        self,
        ticket: TicketData,
        rounds: list[DebateRound],
    ) -> AnalysisReport:
        transcript = self._build_transcript(rounds)

        all_agents = set()
        all_models = set()
        for debate_round in rounds:
            for r in debate_round.responses:
                all_agents.add(r.agent_name)
                all_models.add(r.model)

        messages = [
            Message(role="system", content=self.prompt_template),
            Message(
                role="user",
                content=(
                    f"Ticket: {ticket.ticket_id} — {ticket.title}\n\n"
                    f"Full debate transcript ({len(rounds)} rounds, "
                    f"{len(all_agents)} agents):\n\n{transcript}\n\n"
                    "Produce the final structured report. Also return a JSON block "
                    "with the following keys for programmatic extraction:\n"
                    '```json\n{\n  "domain": "...",\n  "root_cause": "...",\n'
                    '  "affected_files": [...],\n  "confidence": "High|Medium|Low",\n'
                    '  "recommendations": [...],\n  "consensus_points": [...],\n'
                    '  "divergence_points": [...]\n}\n```'
                ),
            ),
        ]

        response = await self.llm.complete(messages, self.model)
        logger.info("Synthesizer produced response (%d chars)", len(response))

        report = self._parse_response(response, ticket, rounds, all_agents, all_models)
        report.formatted_comment = self._format_jira_comment(report)
        return report

    def _parse_response(
        self,
        response: str,
        ticket: TicketData,
        rounds: list[DebateRound],
        agents: set[str],
        models: set[str],
    ) -> AnalysisReport:
        # Try to extract JSON block
        json_start = response.find("```json")
        json_end = response.find("```", json_start + 7) if json_start != -1 else -1

        if json_start != -1 and json_end != -1:
            json_str = response[json_start + 7 : json_end].strip()
            try:
                data = json.loads(json_str)
                return AnalysisReport(
                    ticket_id=ticket.ticket_id,
                    domain=data.get("domain", "Unknown"),
                    root_cause=data.get("root_cause", "Unable to determine"),
                    affected_files=data.get("affected_files", []),
                    confidence=data.get("confidence", "Low"),
                    recommendations=data.get("recommendations", []),
                    consensus_points=data.get("consensus_points", []),
                    divergence_points=data.get("divergence_points", []),
                    total_rounds=len(rounds),
                    agents_used=sorted(agents),
                    models_used=sorted(models),
                )
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from synthesizer response")

        # Fallback: create report from raw response
        logger.warning("Using fallback parsing for synthesizer response")
        return AnalysisReport(
            ticket_id=ticket.ticket_id,
            domain="Unknown",
            root_cause=response[:500],
            affected_files=[],
            confidence="Low",
            recommendations=["Review the full analysis below"],
            consensus_points=[],
            divergence_points=[],
            total_rounds=len(rounds),
            agents_used=sorted(agents),
            models_used=sorted(models),
        )

    def _format_jira_comment(self, report: AnalysisReport) -> str:
        affected = "\n".join(f"  - `{f}`" for f in report.affected_files) or "  - None identified"
        recommendations = "\n".join(
            f"{i}. {r}" for i, r in enumerate(report.recommendations, 1)
        ) or "No specific recommendations"
        consensus = "\n".join(f"  - {p}" for p in report.consensus_points) or "  - None"
        divergence = "\n".join(f"  - {p}" for p in report.divergence_points) or "  - None"

        return (
            f"## Bug Analysis — Automated Crew Report\n\n"
            f"**Domain:** {report.domain}\n\n"
            f"### Diagnosis\n"
            f"- **Probable root cause:** {report.root_cause}\n"
            f"- **Affected files:**\n{affected}\n"
            f"- **Confidence level:** {report.confidence}\n\n"
            f"### Recommendations\n{recommendations}\n\n"
            f"### Debate Summary\n"
            f"- **Consensus points:**\n{consensus}\n"
            f"- **Divergence points:**\n{divergence}\n\n"
            f"---\n"
            f"Analyzed by {len(report.agents_used)} agents over "
            f"{report.total_rounds} debate rounds\n"
            f"Models used: {', '.join(report.models_used)}"
        )
