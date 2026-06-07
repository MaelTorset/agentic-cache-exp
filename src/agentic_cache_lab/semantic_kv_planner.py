from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import AgentEvent, Segment, SegmentKind
from .segment_store import SegmentStore


@dataclass(frozen=True)
class BranchPlanConfig:
    branch_labels: tuple[str, ...] = ("auth", "qr", "billing")
    measured_labels: tuple[str, ...] = ("auth", "qr")
    top_k: int = 5
    max_segments_per_branch: int = 4
    max_branch_words: int = 80


def build_semantic_kv_plan(
    events: list[AgentEvent],
    branch_labels: tuple[str, ...] = ("auth", "qr", "billing"),
    measured_labels: tuple[str, ...] = ("auth", "qr"),
    top_k: int = 5,
) -> dict[str, object]:
    config = BranchPlanConfig(branch_labels=branch_labels, measured_labels=measured_labels, top_k=top_k)
    store = SegmentStore()
    segments = store.add_events(events)

    root_text = build_root_text(segments)
    branch_texts = {
        label: build_branch_text(
            label,
            segments,
            max_segments=config.max_segments_per_branch,
            max_words=config.max_branch_words,
        )
        for label in config.branch_labels
    }

    plan_segments: list[dict[str, object]] = [{"id": "root", "text": root_text, "add_bos": True}]
    plan_segments.extend(
        {"id": f"branch_{label}", "text": text}
        for label, text in branch_texts.items()
        if text.strip()
    )
    available_labels = {
        str(segment["id"]).removeprefix("branch_")
        for segment in plan_segments
        if str(segment["id"]).startswith("branch_")
    }

    ops: list[dict[str, object]] = [{"op": "eval", "seq": 0, "segment": "root", "logits": False}]
    branch_seq_by_label = {label: index + 1 for index, label in enumerate(config.branch_labels) if label in available_labels}
    scratch_seq_start = 1 + len(branch_seq_by_label)

    for label, seq in branch_seq_by_label.items():
        ops.append({"op": "copy", "from": 0, "to": seq})

    # Evaluation order intentionally switches topics to model long-running agent behavior.
    for label in config.branch_labels:
        seq = branch_seq_by_label.get(label)
        if seq is None:
            continue
        ops.append(
            {
                "op": "eval",
                "seq": seq,
                "segment": f"branch_{label}",
                "logits": label in config.measured_labels,
            }
        )

    scratch_seq_by_label: dict[str, int] = {}
    for offset, label in enumerate(label for label in config.measured_labels if label in available_labels):
        scratch_seq = scratch_seq_start + offset
        scratch_seq_by_label[label] = scratch_seq
        ops.extend(
            [
                {"op": "eval", "seq": scratch_seq, "segment": "root", "logits": False},
                {"op": "eval", "seq": scratch_seq, "segment": f"branch_{label}", "logits": True},
                {
                    "op": "compare",
                    "left": branch_seq_by_label[label],
                    "right": scratch_seq,
                    "label": f"{label}_branch_vs_scratch",
                },
            ]
        )

    return {
        "config": {"top_k": config.top_k, "suppress_logs": True},
        "metadata": {
            "branch_labels": list(config.branch_labels),
            "measured_labels": list(config.measured_labels),
            "branch_sequences": branch_seq_by_label,
            "scratch_sequences": scratch_seq_by_label,
        },
        "segments": plan_segments,
        "ops": ops,
    }


def build_root_text(segments: list[Segment]) -> str:
    root_candidates = [
        segment
        for segment in segments
        if segment.kind in {SegmentKind.MESSAGE, SegmentKind.DECISION}
        and ("general" in segment.labels or segment.kind == SegmentKind.MESSAGE)
    ]
    selected = sorted(root_candidates, key=lambda segment: (segment.created_at, segment.source))[:2]
    if not selected and segments:
        selected = [min(segments, key=lambda segment: segment.created_at)]

    return render_segment_block("Shared session context", selected)


def build_branch_text(label: str, segments: list[Segment], max_segments: int, max_words: int) -> str:
    candidates = [
        segment
        for segment in segments
        if label in segment.labels and segment.kind in {SegmentKind.FILE, SegmentKind.ERROR, SegmentKind.DECISION, SegmentKind.MESSAGE}
    ]
    selected = sorted(candidates, key=branch_sort_key)[:max_segments]
    if not selected:
        return ""
    return truncate_words(render_segment_block(f"{label} branch context", selected), max_words)


def branch_sort_key(segment: Segment) -> tuple[float, float, str]:
    kind_priority = {
        SegmentKind.ERROR: 0.0,
        SegmentKind.FILE: 1.0,
        SegmentKind.DECISION: 2.0,
        SegmentKind.MESSAGE: 3.0,
        SegmentKind.COMMAND: 4.0,
    }
    return (kind_priority.get(segment.kind, 9.0), segment.created_at, segment.source)


def render_segment_block(title: str, segments: Iterable[Segment]) -> str:
    lines = [f"{title}:"]
    for segment in segments:
        lines.append(f"[{segment.kind.value}] {segment.source}")
        lines.append(segment.text.strip())
    return "\n".join(lines).strip() + "\n"


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n"
