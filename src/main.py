from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

from src.adapters.context.local_repo import LocalRepoAdapter
from src.adapters.llm.copilot import CopilotAdapter
from src.adapters.output.jira_comment import JiraCommentAdapter
from src.adapters.ticket.jira_adapter import JiraAdapter
from src.domain.debate import DebateEngine
from src.domain.models import AgentConfig, CrewConfig, ModeratorConfig, SynthesizerConfig
from src.domain.moderator import Moderator
from src.domain.orchestrator import Orchestrator
from src.domain.synthesizer import Synthesizer

logger = logging.getLogger("jira_bug_analyzer")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        logger.error("Missing required environment variable: %s", name)
        raise SystemExit(f"Error: environment variable {name} is not set. "
                         f"Add it to config/.env or export it.")
    return value


def load_config(config_dir: Path) -> tuple[dict, CrewConfig]:
    settings_path = config_dir / "settings.yaml"
    crew_path = config_dir / "crew.yaml"

    with open(settings_path) as f:
        settings = yaml.safe_load(f)
    if not settings:
        raise ValueError(f"Empty or invalid settings file: {settings_path}")

    with open(crew_path) as f:
        crew_data = yaml.safe_load(f)
    if not crew_data or "crew" not in crew_data:
        raise ValueError(f"Empty or missing 'crew' key in: {crew_path}")
    crew_raw = crew_data["crew"]

    crew = CrewConfig(
        moderator=ModeratorConfig(
            model=crew_raw["moderator"]["model"],
            max_rounds=crew_raw["moderator"]["max_rounds"],
            rules=crew_raw["moderator"].get("rules", []),
        ),
        agents=[
            AgentConfig(
                name=a["name"],
                role=a["role"],
                model=a["model"],
                prompt_template=a["prompt_template"],
            )
            for a in crew_raw["agents"]
        ],
        synthesizer=SynthesizerConfig(
            model=crew_raw["synthesizer"]["model"],
            prompt_template=crew_raw["synthesizer"]["prompt_template"],
        ),
    )

    return settings, crew


def load_prompt(template_path: str, project_root: Path) -> str:
    full_path = project_root / template_path
    with open(full_path) as f:
        return f.read()


def build_pipeline(
    settings: dict,
    crew: CrewConfig,
    project_root: Path,
) -> Orchestrator:
    # LLM adapter
    llm = CopilotAdapter()

    # Load prompt templates
    moderator_prompt = load_prompt("prompts/moderator.txt", project_root)
    agent_prompts = {}
    for agent in crew.agents:
        agent_prompts[agent.name] = load_prompt(agent.prompt_template, project_root)
    synthesizer_prompt = load_prompt(
        crew.synthesizer.prompt_template, project_root
    )

    # Moderator
    moderator = Moderator(crew.moderator, llm, moderator_prompt)

    # Debate engine
    debate_engine = DebateEngine(moderator, crew.agents, agent_prompts, llm)

    # Synthesizer
    synthesizer = Synthesizer(crew.synthesizer.model, synthesizer_prompt, llm)

    # Ticket adapter
    ticket_config = settings["sources"]["ticket"]["config"]
    jira_email = _require_env("JIRA_EMAIL")
    jira_token = _require_env("JIRA_API_TOKEN")
    ticket_port = JiraAdapter(
        base_url=ticket_config["base_url"],
        email=jira_email,
        api_token=jira_token,
    )

    # Context adapters
    context_ports = []
    for ctx_cfg in settings["sources"]["context"]["adapters"]:
        if ctx_cfg.get("enabled", True) is False:
            continue
        if ctx_cfg["adapter"] == "local_repo":
            context_ports.append(
                LocalRepoAdapter(
                    repo_path=ctx_cfg["config"]["repo_path"],
                    max_results=ctx_cfg["config"].get("max_results", 20),
                )
            )

    # Output adapter
    output_port = JiraCommentAdapter(
        base_url=ticket_config["base_url"],
        email=jira_email,
        api_token=jira_token,
    )

    return Orchestrator(
        ticket_port=ticket_port,
        context_ports=context_ports,
        output_port=output_port,
        debate_engine=debate_engine,
        synthesizer=synthesizer,
    )


@click.command()
@click.option(
    "--ticket",
    required=True,
    help="Jira ticket ID to analyze (e.g., PROJ-1234)",
)
@click.option(
    "--config-dir",
    default="config",
    type=click.Path(exists=True),
    help="Path to configuration directory",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(ticket: str, config_dir: str, verbose: bool) -> None:
    """Analyze a Jira bug ticket using a multi-agent debate crew."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / "config" / ".env")

    config_path = Path(config_dir)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    try:
        settings, crew = load_config(config_path)
        pipeline = build_pipeline(settings, crew, project_root)
    except FileNotFoundError as e:
        logger.error("Configuration file not found: %s", e)
        sys.exit(1)
    except (ValueError, KeyError) as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    logger.info("Starting analysis of ticket %s", ticket)
    try:
        report = asyncio.run(pipeline.analyze(ticket))
        logger.info(
            "Analysis complete — Confidence: %s, Domain: %s",
            report.confidence,
            report.domain,
        )
        click.echo(f"\n{'='*60}")
        click.echo(report.formatted_comment)
        click.echo(f"{'='*60}\n")
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
