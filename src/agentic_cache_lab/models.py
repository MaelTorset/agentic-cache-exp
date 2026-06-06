from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from time import time


class EventKind(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    FILE_READ = "file_read"
    COMMAND = "command"
    ERROR = "error"
    DECISION = "decision"


class SegmentKind(StrEnum):
    MESSAGE = "message"
    FILE = "file"
    COMMAND = "command"
    ERROR = "error"
    DECISION = "decision"


@dataclass(frozen=True)
class AgentEvent:
    kind: EventKind
    text: str
    source: str = "conversation"
    created_at: float = field(default_factory=time)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Segment:
    id: str
    kind: SegmentKind
    source: str
    text: str
    token_count: int
    content_hash: str
    labels: tuple[str, ...]
    importance: float
    volatility: float
    created_at: float
    last_used_at: float


@dataclass(frozen=True)
class ScoredSegment:
    segment: Segment
    score: float
    reason: str


@dataclass(frozen=True)
class PackedPrompt:
    prompt: str
    stable_prefix: str
    dynamic_suffix: str
    included: tuple[ScoredSegment, ...]
    omitted: tuple[Segment, ...]
    token_estimate: int
    stable_token_estimate: int


def estimate_tokens(text: str) -> int:
    """Cheap deterministic approximation for offline benchmarks."""
    words = text.split()
    return max(1, int(len(words) * 1.35)) if text.strip() else 0


def stable_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]
