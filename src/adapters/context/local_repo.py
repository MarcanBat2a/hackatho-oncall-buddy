from __future__ import annotations

import asyncio
import logging
import re

from src.domain.models import ContextData, RelevantFile, TicketData
from src.ports.context import ContextPort

logger = logging.getLogger(__name__)


class LocalRepoAdapter(ContextPort):
    def __init__(self, repo_path: str, max_results: int = 20):
        self.repo_path = repo_path
        self.max_results = max_results

    async def gather(self, ticket: TicketData) -> ContextData:
        keywords = self._extract_keywords(ticket)
        logger.info("Searching repo with %d keywords: %s", len(keywords), keywords)

        relevant_files: list[RelevantFile] = []
        keywords_used: list[str] = []

        for keyword in keywords:
            if len(relevant_files) >= self.max_results:
                break
            results = await self._ripgrep(keyword)
            if results:
                keywords_used.append(keyword)
                for path, snippet in results:
                    if not any(rf.path == path for rf in relevant_files):
                        relevant_files.append(
                            RelevantFile(
                                path=path,
                                snippet=snippet,
                                match_reason=f"keyword: {keyword}",
                            )
                        )
                        if len(relevant_files) >= self.max_results:
                            break

        logger.info("Found %d relevant files", len(relevant_files))
        return ContextData(
            source="local_repo",
            relevant_files=relevant_files,
            keywords_used=keywords_used,
        )

    def _extract_keywords(self, ticket: TicketData) -> list[str]:
        keywords = []

        # Extract from stacktraces (file paths, class names)
        text = f"{ticket.title} {ticket.description}"
        # Match patterns like com.package.ClassName or path/to/file.ext:line
        stacktrace_patterns = re.findall(
            r'(?:[\w.]+\.(?:[A-Z]\w+))|(?:[\w/]+\.\w+:\d+)', text
        )
        keywords.extend(stacktrace_patterns[:5])

        # Extract error messages and class names
        error_patterns = re.findall(r'(?:Error|Exception|Failed|Failure)[\w]*', text)
        keywords.extend(error_patterns[:3])

        # Extract CamelCase identifiers
        camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        keywords.extend(camel_case[:5])

        # Use components and labels as keywords
        keywords.extend(ticket.components[:3])
        keywords.extend(ticket.labels[:3])

        # Fallback: significant words from the title
        if not keywords:
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                          "to", "for", "of", "with", "by", "from", "not", "bug", "issue",
                          "error", "problem", "when", "after", "before", "should", "does"}
            words = re.findall(r'\b\w{4,}\b', ticket.title.lower())
            keywords.extend(w for w in words if w not in stop_words)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for k in keywords:
            if k.lower() not in seen:
                seen.add(k.lower())
                unique.append(k)
        return unique[:10]

    async def _ripgrep(self, keyword: str) -> list[tuple[str, str]]:
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "rg",
                "--max-count", "3",
                "--max-columns", "200",
                "-n",
                "--type-not", "binary",
                "--",  # prevent keyword from being interpreted as a flag
                keyword,
                self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except FileNotFoundError:
            logger.error("ripgrep (rg) not found — install it or check PATH")
            return []
        except asyncio.TimeoutError:
            logger.warning("ripgrep timed out for keyword: %s", keyword)
            if proc:
                proc.kill()
                await proc.wait()
            return []

        if proc.returncode not in (0, 1):  # 1 = no matches
            logger.warning("ripgrep error for '%s': %s", keyword, stderr.decode()[:200])
            return []

        results: list[tuple[str, str]] = []
        current_file = None
        current_lines: list[str] = []

        for line in stdout.decode(errors="replace").splitlines():
            # rg output format: file:line:content
            parts = line.split(":", 2)
            if len(parts) >= 3:
                file_path = parts[0]
                if file_path != current_file:
                    if current_file and current_lines:
                        results.append((current_file, "\n".join(current_lines)))
                    current_file = file_path
                    current_lines = []
                current_lines.append(f"{parts[1]}: {parts[2]}")

        if current_file and current_lines:
            results.append((current_file, "\n".join(current_lines)))

        return results[:5]
