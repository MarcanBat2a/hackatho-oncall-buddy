from __future__ import annotations

import logging
import os

import aiohttp

from src.domain.models import Message
from src.ports.llm import LLMPort

logger = logging.getLogger(__name__)

COPILOT_CHAT_URL = "https://api.githubcopilot.com/chat/completions"


class CopilotAdapter(LLMPort):
    def __init__(self, github_token: str | None = None):
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN is required for CopilotAdapter")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def complete(self, messages: list[Message], model: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Content-Type": "application/json",
            "Copilot-Integration-Id": "jira-bug-analyzer",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ],
            "temperature": 0.7,
        }

        session = await self._get_session()
        async with session.post(
            COPILOT_CHAT_URL,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(
                    "Copilot API error %d for model %s: %s",
                    resp.status, model, body[:500],
                )
                raise RuntimeError(
                    f"Copilot API returned {resp.status}: {body[:200]}"
                )

            data = await resp.json()
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                logger.error("Unexpected Copilot API response structure: %s", data)
                raise RuntimeError(
                    f"Unexpected response from Copilot API: {e}"
                ) from e

            if content is None:
                logger.warning("Copilot API returned null content for model %s", model)
                content = "[No response content from LLM]"

            logger.debug(
                "Copilot response from %s: %d chars", model, len(content)
            )
            return content
