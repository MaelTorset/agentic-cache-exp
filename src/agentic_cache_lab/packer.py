from __future__ import annotations

from dataclasses import dataclass

from .models import PackedPrompt, ScoredSegment, Segment, SegmentKind, estimate_tokens
from .scorer import SegmentScorer, labels_to_exclude
from .segment_store import labels_for


@dataclass(frozen=True)
class PackConfig:
    max_prompt_tokens: int = 1800
    stable_prefix_budget_ratio: float = 0.65
    critical_context_budget_ratio: float = 0.45
    min_score: float = 0.25
    critical_context: bool = False


class PromptPacker:
    def __init__(self, scorer: SegmentScorer | None = None, config: PackConfig | None = None) -> None:
        self.scorer = scorer or SegmentScorer()
        self.config = config or PackConfig()

    def pack(self, segments: list[Segment], query: str, objective: str) -> PackedPrompt:
        base = (
            "You are a local coding agent experimenting with cache-aware context routing.\n"
            f"Current objective: {objective}\n"
            "Use only the active context below. If context is missing, ask for retrieval.\n"
        )
        if self.config.critical_context:
            base += (
                "Context is ordered by priority. Preserve critical constraints exactly, "
                "especially failed tests, expected behavior, root-cause evidence, and patch targets.\n"
            )
        base_tokens = estimate_tokens(base)
        budget = max(0, self.config.max_prompt_tokens - base_tokens - estimate_tokens(query))

        scored = [item for item in self.scorer.score(segments, query) if item.score >= self.config.min_score]
        critical: list[ScoredSegment] = []
        stable: list[ScoredSegment] = []
        dynamic: list[ScoredSegment] = []
        used_ids: set[str] = set()

        critical_tokens = 0
        if self.config.critical_context:
            critical_budget = int(budget * self.config.critical_context_budget_ratio)
            critical_candidates = sorted(
                (item for item in scored if is_critical_context(item.segment, query)),
                key=lambda item: (critical_priority(item.segment, query), item.score),
                reverse=True,
            )
            for item in critical_candidates:
                segment = item.segment
                if critical_tokens + segment.token_count > critical_budget:
                    continue
                critical.append(item)
                used_ids.add(segment.id)
                critical_tokens += segment.token_count

        remaining_budget = max(0, budget - critical_tokens)
        stable_budget = int(remaining_budget * self.config.stable_prefix_budget_ratio)
        dynamic_budget = remaining_budget - stable_budget

        stable_tokens = 0
        for item in scored:
            segment = item.segment
            if segment.id in used_ids:
                continue
            if not is_stable(segment):
                continue
            if stable_tokens + segment.token_count > stable_budget:
                continue
            stable.append(item)
            used_ids.add(segment.id)
            stable_tokens += segment.token_count

        dynamic_tokens = 0
        for item in scored:
            segment = item.segment
            if segment.id in used_ids:
                continue
            if dynamic_tokens + segment.token_count > dynamic_budget:
                continue
            dynamic.append(item)
            used_ids.add(segment.id)
            dynamic_tokens += segment.token_count

        critical_section = render_section("critical context", critical) if self.config.critical_context else None
        stable_prefix = render_section("stable prefix", stable)
        dynamic_suffix = render_section("dynamic suffix", dynamic)
        sections = [base.strip()]
        if critical_section is not None:
            sections.append(critical_section)
        sections.extend([stable_prefix, dynamic_suffix, f"User query:\n{query}"])
        prompt = "\n\n".join(sections)
        omitted = tuple(segment for segment in segments if segment.id not in used_ids)
        stable_prefix_text = "\n\n".join(section for section in [base.strip(), critical_section, stable_prefix] if section)
        return PackedPrompt(
            prompt=prompt,
            stable_prefix=stable_prefix_text,
            dynamic_suffix=dynamic_suffix,
            included=tuple(critical + stable + dynamic),
            omitted=omitted,
            token_estimate=estimate_tokens(prompt),
            stable_token_estimate=estimate_tokens(stable_prefix_text),
        )


def is_stable(segment: Segment) -> bool:
    if segment.kind == SegmentKind.DECISION:
        return True
    if segment.kind == SegmentKind.FILE and segment.volatility <= 0.35:
        return True
    return segment.volatility <= 0.2


def is_critical_context(segment: Segment, query: str) -> bool:
    excluded_labels = labels_to_exclude(query)
    if segment.kind in {SegmentKind.FILE, SegmentKind.COMMAND} and excluded_labels.intersection(segment.labels):
        return False

    source = segment.source.lower()
    text = segment.text.lower()
    query_labels = set(labels_for(query, "query")).difference(excluded_labels)
    has_query_topic = bool(query_labels.intersection(segment.labels))

    if segment.kind == SegmentKind.ERROR and has_query_topic:
        return True
    if is_test_source(source) and has_query_topic:
        return True
    if has_query_topic and any(marker in text for marker in ("expected ", "root cause", "rejected", "failure")):
        return True
    if segment.kind == SegmentKind.FILE and has_query_topic and any(
        marker in source for marker in ("auth", "cookie", "session", "middleware")
    ):
        return True
    if segment.kind == SegmentKind.DECISION and has_query_topic:
        return True
    return False


def critical_priority(segment: Segment, query: str) -> int:
    source = segment.source.lower()
    text = segment.text.lower()
    if segment.kind == SegmentKind.ERROR:
        return 50
    if is_test_source(source) or "expected " in text:
        return 45
    if segment.kind == SegmentKind.FILE and any(marker in source for marker in ("auth", "cookie", "session")):
        return 40
    if "root cause" in text or "rejected" in text:
        return 35
    if segment.kind == SegmentKind.DECISION:
        return 25
    return 10


def is_test_source(source: str) -> bool:
    return any(marker in source for marker in ("/tests/", ".test.", "_test.", "test/"))


def render_section(title: str, items: list[ScoredSegment]) -> str:
    if not items:
        return f"{title}:\n(empty)"

    rendered = [f"{title}:"]
    for item in items:
        segment = item.segment
        rendered.append(
            "\n".join(
                [
                    f"- {segment.kind.value} {segment.source} labels={','.join(segment.labels)} score={item.score:.2f}",
                    indent(segment.text.strip(), "  "),
                ]
            )
        )
    return "\n".join(rendered)


def indent(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())
