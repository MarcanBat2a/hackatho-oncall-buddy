from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TicketData:
    ticket_id: str
    title: str
    description: str
    labels: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    priority: str = ""
    comments: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)


@dataclass
class RelevantFile:
    path: str
    snippet: str
    match_reason: str


@dataclass
class ContextData:
    source: str
    relevant_files: list[RelevantFile] = field(default_factory=list)
    keywords_used: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    name: str
    role: str
    model: str
    prompt_template: str


@dataclass
class ModeratorConfig:
    model: str
    max_rounds: int
    rules: list[str] = field(default_factory=list)


@dataclass
class SynthesizerConfig:
    model: str
    prompt_template: str


@dataclass
class CrewConfig:
    moderator: ModeratorConfig
    agents: list[AgentConfig]
    synthesizer: SynthesizerConfig


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class AgentResponse:
    agent_name: str
    content: str
    model: str


@dataclass
class DebateRound:
    round_number: int
    moderator_prompt: str
    responses: list[AgentResponse]
    moderator_synthesis: str


@dataclass
class AnalysisReport:
    ticket_id: str
    domain: str
    root_cause: str
    affected_files: list[str]
    confidence: str  # "High", "Medium", "Low"
    recommendations: list[str]
    consensus_points: list[str]
    divergence_points: list[str]
    total_rounds: int
    agents_used: list[str]
    models_used: list[str]
    formatted_comment: str = ""
