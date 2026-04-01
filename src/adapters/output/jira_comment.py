from __future__ import annotations

import asyncio
import logging

from jira import JIRA
from jira.exceptions import JIRAError

from src.domain.models import AnalysisReport
from src.ports.output import OutputPort

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
BACKOFF_BASE = 2


class JiraCommentAdapter(OutputPort):
    def __init__(self, base_url: str, email: str, api_token: str):
        self.client = JIRA(
            server=base_url,
            basic_auth=(email, api_token),
        )

    async def publish(self, ticket_id: str, report: AnalysisReport) -> None:
        logger.info("Posting analysis comment to %s", ticket_id)

        for attempt in range(MAX_RETRIES + 1):
            try:
                await asyncio.to_thread(
                    self.client.add_comment, ticket_id, report.formatted_comment
                )
                logger.info("Comment posted successfully to %s", ticket_id)
                return
            except JIRAError:
                if attempt == MAX_RETRIES:
                    raise
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "Jira API error posting to %s, retrying in %ds (attempt %d/%d)",
                    ticket_id, wait, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(wait)
