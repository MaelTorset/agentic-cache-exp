from __future__ import annotations

from dataclasses import dataclass, field

from .models import AgentEvent, EventKind, Segment, SegmentKind, estimate_tokens, stable_hash


LABEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "auth": ("auth", "token", "login", "session", "cookie", "jwt", "password"),
    "qr": ("qr", "qrcode", "code qr", "scan"),
    "cache": ("cache", "kv", "prefix", "prompt", "prefill", "ttl"),
    "frontend": ("frontend", "component", "css", "react", "page", "button"),
    "backend": ("api", "server", "route", "database", "sql", "endpoint"),
    "billing": ("billing", "invoice", "payment", "stripe", "subscription"),
    "analytics": ("analytics", "event", "tracking", "metric", "funnel"),
    "error": ("error", "traceback", "exception", "failed", "bug"),
}


KIND_MAP = {
    EventKind.USER: SegmentKind.MESSAGE,
    EventKind.ASSISTANT: SegmentKind.MESSAGE,
    EventKind.FILE_READ: SegmentKind.FILE,
    EventKind.COMMAND: SegmentKind.COMMAND,
    EventKind.ERROR: SegmentKind.ERROR,
    EventKind.DECISION: SegmentKind.DECISION,
}


@dataclass
class SegmentStore:
    segments: dict[str, Segment] = field(default_factory=dict)

    def add_event(self, event: AgentEvent) -> Segment:
        content_hash = stable_hash(f"{event.kind.value}:{event.source}:{event.text}")
        segment_id = f"seg_{content_hash}"
        existing = self.segments.get(segment_id)
        if existing:
            return existing

        segment = Segment(
            id=segment_id,
            kind=KIND_MAP[event.kind],
            source=event.source,
            text=event.text,
            token_count=estimate_tokens(event.text),
            content_hash=content_hash,
            labels=labels_for(event.text, event.source),
            importance=importance_for(event),
            volatility=volatility_for(event),
            created_at=event.created_at,
            last_used_at=event.created_at,
        )
        self.segments[segment.id] = segment
        return segment

    def add_events(self, events: list[AgentEvent]) -> list[Segment]:
        return [self.add_event(event) for event in events]

    def all(self) -> list[Segment]:
        return sorted(self.segments.values(), key=lambda segment: segment.created_at)


def labels_for(text: str, source: str) -> tuple[str, ...]:
    haystack = f"{source} {text}".lower()
    labels = [label for label, keywords in LABEL_KEYWORDS.items() if any(keyword in haystack for keyword in keywords)]
    if not labels:
        labels.append("general")
    return tuple(labels)


def importance_for(event: AgentEvent) -> float:
    if event.kind == EventKind.DECISION:
        return 1.0
    if event.kind == EventKind.ERROR:
        return 0.85
    if event.kind == EventKind.FILE_READ:
        return 0.7
    if event.kind == EventKind.USER:
        return 0.65
    if event.kind == EventKind.ASSISTANT:
        return 0.45
    return 0.4


def volatility_for(event: AgentEvent) -> float:
    if event.kind in {EventKind.ERROR, EventKind.COMMAND}:
        return 0.85
    if event.kind == EventKind.USER:
        return 0.55
    if event.kind == EventKind.FILE_READ:
        return 0.25
    if event.kind == EventKind.DECISION:
        return 0.1
    return 0.35
