from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Paragraph:
    pid: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}: {self.text}"


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    qtype: str = "unknown"          # bridge | comparison | 2hop | 3hop ...
    hops: int = 2
    supporting_pids: list[str] = field(default_factory=list)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.calls += other.calls
        return self


@dataclass
class Retrieval:
    pids: list[str]
    context: str
    debug: dict = field(default_factory=dict)   # e.g. seed entities, triples used


@dataclass
class PipelineResult:
    qid: str
    pipeline: str
    answer: str
    retrieved_pids: list[str]
    usage: Usage
    latency_ms: dict[str, float]
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
