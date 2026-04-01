from __future__ import annotations

import asyncio
import logging

from src.domain.models import (
    AgentConfig,
    AgentResponse,
    ContextData,
    DebateRound,
    Message,
    TicketData,
)
from src.domain.moderator import Moderator
from src.ports.llm import LLMPort

logger = logging.getLogger(__name__)

AGENT_TIMEOUT_SECONDS = 120


class DebateEngine:
    def __init__(
        self,
        moderator: Moderator,
        agents: list[AgentConfig],
        agent_prompts: dict[str, str],
        llm: LLMPort,
    ):
        self.moderator = moderator
        self.agents = agents
        self.agent_prompts = agent_prompts
        self.llm = llm

    async def _call_agent(
        self,
        agent: AgentConfig,
        round_prompt: str,
        briefing_context: str,
        previous_rounds: list[DebateRound],
    ) -> AgentResponse:
        system_prompt = self.agent_prompts.get(agent.name, "")
        history_parts = []
        for prev in previous_rounds:
            history_parts.append(f"Round {prev.round_number} question: {prev.moderator_prompt}")
            for r in prev.responses:
                history_parts.append(f"  {r.agent_name}: {r.content[:500]}")
            history_parts.append(f"  Moderator synthesis: {prev.moderator_synthesis}")

        history_text = "\n".join(history_parts) if history_parts else "No previous rounds."

        messages = [
            Message(role="system", content=system_prompt),
            Message(
                role="user",
                content=(
                    f"## Bug Context\n{briefing_context}\n\n"
                    f"## Debate History\n{history_text}\n\n"
                    f"## Current Round Question\n{round_prompt}\n\n"
                    "Provide your analysis. Remember: argue your position with evidence, "
                    "never simply agree with others."
                ),
            ),
        ]

        try:
            content = await asyncio.wait_for(
                self.llm.complete(messages, agent.model),
                timeout=AGENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("Agent %s timed out after %ds", agent.name, AGENT_TIMEOUT_SECONDS)
            content = f"[Agent {agent.name} did not respond — timed out]"
        except Exception as e:
            logger.error("Agent %s failed: %s", agent.name, e)
            content = f"[Agent {agent.name} failed: {e}]"

        return AgentResponse(
            agent_name=agent.name,
            content=content,
            model=agent.model,
        )

    async def run(
        self,
        ticket: TicketData,
        contexts: list[ContextData],
    ) -> list[DebateRound]:
        briefing_context = self.moderator.build_briefing_context(ticket, contexts)
        rounds: list[DebateRound] = []

        round_number = 1
        while True:
            logger.info("Starting debate round %d", round_number)

            round_prompt = await self.moderator.generate_round_prompt(
                round_number, briefing_context, rounds
            )
            logger.info("Round %d prompt: %s", round_number, round_prompt[:200])

            # Collect agent responses in parallel — anti-groupthink
            responses = await asyncio.gather(
                *[
                    self._call_agent(agent, round_prompt, briefing_context, rounds)
                    for agent in self.agents
                ]
            )

            active_responses = sum(
                1 for r in responses if not r.content.startswith("[Agent ")
            )
            logger.info(
                "Round %d: received %d/%d active responses",
                round_number,
                active_responses,
                len(self.agents),
            )

            synthesis = await self.moderator.synthesize_round(
                round_number, round_prompt, list(responses)
            )

            debate_round = DebateRound(
                round_number=round_number,
                moderator_prompt=round_prompt,
                responses=list(responses),
                moderator_synthesis=synthesis,
            )
            rounds.append(debate_round)

            should_continue = await self.moderator.should_continue(round_number, synthesis)
            if not should_continue:
                break

            round_number += 1

        logger.info("Debate completed after %d rounds", len(rounds))
        return rounds
