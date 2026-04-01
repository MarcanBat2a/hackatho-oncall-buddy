from __future__ import annotations

import asyncio
import logging
import time

from jira import JIRA
from jira.exceptions import JIRAError

from src.domain.models import TicketData
from src.ports.ticket import TicketPort

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
BACKOFF_BASE = 2


class JiraAdapter(TicketPort):
    def __init__(self, base_url: str, email: str, api_token: str):
        self.client = JIRA(
            server=base_url,
            basic_auth=(email, api_token),
        )

    async def fetch(self, ticket_id: str) -> TicketData:
        logger.info("Fetching Jira ticket %s", ticket_id)

        for attempt in range(MAX_RETRIES + 1):
            try:
                issue = await asyncio.to_thread(self.client.issue, ticket_id)
                comments_raw = await asyncio.to_thread(self.client.comments, issue)
                break
            except JIRAError:
                if attempt == MAX_RETRIES:
                    raise
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "Jira API error fetching %s, retrying in %ds (attempt %d/%d)",
                    ticket_id, wait, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(wait)

        comments = [comment.body for comment in comments_raw]

        attachments = []
        for attachment in getattr(issue.fields, "attachment", None) or []:
            attachments.append(attachment.filename)

        return TicketData(
            ticket_id=ticket_id,
            title=issue.fields.summary or "",
            description=issue.fields.description or "",
            labels=list(issue.fields.labels or []),
            components=[c.name for c in (issue.fields.components or [])],
            priority=getattr(issue.fields.priority, "name", "") if issue.fields.priority else "",
            comments=comments,
            attachments=attachments,
        )
