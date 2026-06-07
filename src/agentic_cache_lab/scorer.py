from __future__ import annotations

from dataclasses import dataclass
import re

from .models import ScoredSegment, Segment, SegmentKind
from .segment_store import labels_for


@dataclass(frozen=True)
class ScoreConfig:
    relevance_weight: float = 2.0
    importance_weight: float = 1.2
    recency_weight: float = 0.4
    volatility_penalty: float = 0.8
    token_penalty_per_1k: float = 0.25


class SegmentScorer:
    def __init__(self, config: ScoreConfig | None = None) -> None:
        self.config = config or ScoreConfig()

    def score(self, segments: list[Segment], query: str) -> list[ScoredSegment]:
        if not segments:
            return []

        excluded_labels = labels_to_exclude(query)
        query_labels = set(labels_for(query, "query")).difference(excluded_labels)
        newest = max(segment.created_at for segment in segments)
        oldest = min(segment.created_at for segment in segments)
        span = max(1.0, newest - oldest)

        scored = [self._score_one(segment, query, query_labels, excluded_labels, oldest, span) for segment in segments]
        return sorted(scored, key=lambda item: item.score, reverse=True)

    def _score_one(
        self,
        segment: Segment,
        query: str,
        query_labels: set[str],
        excluded_labels: set[str],
        oldest: float,
        span: float,
    ) -> ScoredSegment:
        label_overlap = len(query_labels.intersection(segment.labels))
        lexical_overlap = overlap_ratio(query, segment.text)
        relevance = min(1.0, label_overlap * 0.45 + lexical_overlap)
        recency = (segment.created_at - oldest) / span
        token_cost = segment.token_count / 1000

        score = (
            relevance * self.config.relevance_weight
            + segment.importance * self.config.importance_weight
            + recency * self.config.recency_weight
            - segment.volatility * self.config.volatility_penalty
            - token_cost * self.config.token_penalty_per_1k
        )
        excluded_overlap = excluded_labels.intersection(segment.labels)
        if excluded_overlap:
            if segment.kind in {SegmentKind.FILE, SegmentKind.COMMAND}:
                score = -999.0
            else:
                penalty = 0.7 if segment.kind == SegmentKind.DECISION else 6.0
                score -= penalty * len(excluded_overlap)

        reason = (
            f"labels={','.join(segment.labels)} relevance={relevance:.2f} "
            f"importance={segment.importance:.2f} volatility={segment.volatility:.2f}"
        )
        return ScoredSegment(segment=segment, score=score, reason=reason)


def overlap_ratio(left: str, right: str) -> float:
    left_terms = terms(left)
    if not left_terms:
        return 0.0
    right_terms = terms(right)
    return len(left_terms.intersection(right_terms)) / len(left_terms)


def terms(text: str) -> set[str]:
    return {part.strip(".,:;()[]{}<>/\\\"'").lower() for part in text.split() if len(part) > 3}


def labels_to_exclude(query: str) -> set[str]:
    exclusions: set[str] = set()
    negative_words = ("drop", "omit", "ignore", "forget", "unrelated", "without", "exclude")
    for label, aliases in {
        "qr": ("qr", "qrcode", "scanner"),
        "auth": ("auth", "authentication", "login"),
        "frontend": ("frontend", "ui"),
        "backend": ("backend", "api"),
        "billing": ("billing", "invoice", "payment", "stripe"),
        "analytics": ("analytics", "tracking", "metric"),
        "cache": ("cache", "kv"),
    }.items():
        if any(has_nearby_negative(query, alias, negative_words) for alias in aliases):
            exclusions.add(label)
    return exclusions


def has_nearby_negative(query: str, alias: str, negative_words: tuple[str, ...]) -> bool:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    alias_parts = alias.lower().split()
    if not tokens or not alias_parts:
        return False

    negative_positions = {index for index, token in enumerate(tokens) if token in negative_words}
    if not negative_positions:
        return False

    for index in range(0, len(tokens) - len(alias_parts) + 1):
        if tokens[index : index + len(alias_parts)] != alias_parts:
            continue
        if any(0 <= index - negative_index <= 4 for negative_index in negative_positions):
            return True
    return False
