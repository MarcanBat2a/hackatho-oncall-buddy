from __future__ import annotations

import logging

from src.domain.models import (
    AgentConfig,
    AgentResponse,
    ContextData,
    DebateRound,
    Message,
    ModeratorConfig,
    TicketData,
)
from src.ports.llm import LLMPort

logger = logging.getLogger(__name__)


class Moderator:
    def __init__(self, config: ModeratorConfig, llm: LLMPort, prompt_template: str):
        self.config = config
        self.llm = llm
        self.prompt_template = prompt_template

    def build_briefing_context(
        self, ticket: TicketData, contexts: list[ContextData]
    ) -> str:
        parts = [
            f"# Bug Ticket: {ticket.ticket_id}",
            f"**Title:** {ticket.title}",
            f"**Priority:** {ticket.priority}",
            f"**Labels:** {', '.join(ticket.labels) or 'None'}",
            f"**Components:** {', '.join(ticket.components) or 'None'}",
            f"\n**Description:**\n{ticket.description}",
        ]
        if ticket.comments:
            parts.append(f"\n**Existing comments ({len(ticket.comments)}):**")
            for i, comment in enumerate(ticket.comments[:5], 1):
                parts.append(f"  {i}. {comment[:500]}")

        for ctx in contexts:
            if ctx.relevant_files:
                parts.append(f"\n## Relevant code from {ctx.source}")
                for rf in ctx.relevant_files:
                    parts.append(
                        f"\n### {rf.path} (match: {rf.match_reason})\n```\n{rf.snippet}\n```"
                    )

        return "\n".join(parts)

    async def generate_round_prompt(
        self,
        round_number: int,
        briefing_context: str,
        previous_rounds: list[DebateRound],
    ) -> str:
        if round_number == 1:
            user_content = (
                f"Here is the bug ticket context:\n\n{briefing_context}\n\n"
                "This is Round 1. Generate a focused opening question for the agents "
                "to begin their analysis. Ask them to independently identify the domain, "
                "suspicious files, and initial root cause hypotheses."
            )
        else:
            last_round = previous_rounds[-1]
            responses_text = "\n\n".join(
                f"**{r.agent_name}** ({r.model}):\n{r.content}"
                for r in last_round.responses
            )
            user_content = (
                f"Previous round synthesis:\n{last_round.moderator_synthesis}\n\n"
                f"Agent responses from round {last_round.round_number}:\n{responses_text}\n\n"
                f"This is Round {round_number}. Based on the agents' responses, "
                "generate a targeted follow-up question. Push on areas of divergence, "
                "challenge weak arguments, and ask for specific evidence."
            )

        messages = [
            Message(role="system", content=self.prompt_template),
            Message(role="user", content=user_content),
        ]
        return await self.llm.complete(messages, self.config.model)

    async def synthesize_round(
        self,
        round_number: int,
        moderator_prompt: str,
        responses: list[AgentResponse],
    ) -> str:
        responses_text = "\n\n".join(
            f"**{r.agent_name}** ({r.model}):\n{r.content}" for r in responses
        )

        messages = [
            Message(role="system", content=self.prompt_template),
            Message(
                role="user",
                content=(
                    f"Round {round_number} question was: {moderator_prompt}\n\n"
                    f"Agent responses:\n{responses_text}\n\n"
                    "Synthesize these responses. Identify:\n"
                    "- Points of convergence\n"
                    "- Points of divergence\n"
                    "- Whether consensus has been reached\n"
                    "- Key questions for the next round (if needed)\n\n"
                    "Rules: reject bare agreement without argumentation. "
                    "Highlight position changes."
                ),
            ),
        ]
        return await self.llm.complete(messages, self.config.model)

    async def should_continue(
        self, round_number: int, synthesis: str
    ) -> bool:
        if round_number >= self.config.max_rounds:
            logger.info("Max rounds (%d) reached, ending debate", self.config.max_rounds)
            return False

        messages = [
            Message(role="system", content=self.prompt_template),
            Message(
                role="user",
                content=(
                    f"Round {round_number} synthesis:\n{synthesis}\n\n"
                    "Has sufficient consensus been reached to end the debate? "
                    "Answer ONLY 'CONTINUE' or 'CONCLUDE' (one word)."
                ),
            ),
        ]
        response = await self.llm.complete(messages, self.config.model)
        should_stop = "CONCLUDE" in response.upper()
        if should_stop:
            logger.info("Moderator concluded debate at round %d", round_number)
        return not should_stop
