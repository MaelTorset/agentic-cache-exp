from __future__ import annotations

from dataclasses import dataclass

from .models import PackedPrompt, ScoredSegment, Segment, SegmentKind, estimate_tokens
from .scorer import SegmentScorer


@dataclass(frozen=True)
class PackConfig:
    max_prompt_tokens: int = 1800
    stable_prefix_budget_ratio: float = 0.65
    min_score: float = 0.25


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
        base_tokens = estimate_tokens(base)
        budget = max(0, self.config.max_prompt_tokens - base_tokens - estimate_tokens(query))
        stable_budget = int(budget * self.config.stable_prefix_budget_ratio)
        dynamic_budget = budget - stable_budget

        scored = [item for item in self.scorer.score(segments, query) if item.score >= self.config.min_score]
        stable: list[ScoredSegment] = []
        dynamic: list[ScoredSegment] = []
        used_ids: set[str] = set()

        stable_tokens = 0
        for item in scored:
            segment = item.segment
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

        stable_prefix = render_section("stable prefix", stable)
        dynamic_suffix = render_section("dynamic suffix", dynamic)
        prompt = "\n\n".join(
            [
                base.strip(),
                stable_prefix,
                dynamic_suffix,
                f"User query:\n{query}",
            ]
        )
        omitted = tuple(segment for segment in segments if segment.id not in used_ids)
        return PackedPrompt(
            prompt=prompt,
            stable_prefix=f"{base.strip()}\n\n{stable_prefix}",
            dynamic_suffix=dynamic_suffix,
            included=tuple(stable + dynamic),
            omitted=omitted,
            token_estimate=estimate_tokens(prompt),
            stable_token_estimate=estimate_tokens(f"{base}\n{stable_prefix}"),
        )


def is_stable(segment: Segment) -> bool:
    if segment.kind == SegmentKind.DECISION:
        return True
    if segment.kind == SegmentKind.FILE and segment.volatility <= 0.35:
        return True
    return segment.volatility <= 0.2


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
