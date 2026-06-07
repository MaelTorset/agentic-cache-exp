from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .event_log import dump_jsonl
from .fixture_repo import build_fixture_repo_events
from .llm_client import EchoClient, OpenAICompatibleClient
from .models import AgentEvent, estimate_tokens
from .semantic_kv_planner import build_branch_text, build_root_text, build_semantic_kv_plan
from .segment_store import SegmentStore


BUG_QUERY = (
    "You are fixing the failing auth cookie test. Answer in exactly four short bullets: "
    "1 file to patch, 2 root cause, 3 development/test behavior, 4 production behavior. "
    "Do not include unrelated QR, billing, or analytics code."
)

OBJECTIVE = "Solve the auth cookie bug in the fixture TypeScript repo."


@dataclass(frozen=True)
class BugPrompt:
    mode: str
    prompt: str
    token_estimate: int


def build_bug_resolution_prompts(events: list[AgentEvent]) -> list[BugPrompt]:
    plan = build_semantic_kv_plan(
        events,
        branch_labels=("auth", "qr", "billing", "analytics"),
        measured_labels=("auth", "qr"),
    )
    segments = {str(segment["id"]): str(segment["text"]) for segment in plan["segments"]}
    raw_history = "\n\n".join(f"{event.kind.value.upper()} {event.source}\n{event.text}" for event in events)
    full_noise = (
        "You are a coding agent. Use the complete session history, including unrelated exploration.\n"
        f"Objective: {OBJECTIVE}\n\n"
        f"{raw_history}\n\n"
        f"Task:\n{BUG_QUERY}\n"
    )
    routed = (
        "You are a coding agent. Use the shared session context and only the active auth branch.\n"
        f"Objective: {OBJECTIVE}\n\n"
        f"{segments.get('root', '')}\n"
        f"{segments.get('branch_auth', '')}\n"
        f"Task:\n{BUG_QUERY}\n"
    )

    return [
        BugPrompt("full_noise", full_noise, estimate_tokens(full_noise)),
        BugPrompt("routed_prompt", routed, estimate_tokens(routed)),
    ]


def build_native_generation_plan(events: list[AgentEvent], max_tokens: int = 96) -> dict[str, object]:
    store = SegmentStore()
    segments = store.add_events(events)
    root_text = build_root_text(segments)
    auth_text = build_branch_text("auth", segments, max_segments=4, max_words=80)
    task_text = (
        "\nTask:\n"
        f"{BUG_QUERY}\n"
        "Answer:\n"
    )

    return {
        "config": {"top_k": 5, "suppress_logs": True},
        "metadata": {
            "mode": "native_kv_branch_generation",
            "max_tokens": max_tokens,
            "branch_sequence": 1,
            "scratch_sequence": 2,
        },
        "segments": [
            {"id": "root", "text": root_text, "add_bos": True},
            {"id": "branch_auth", "text": auth_text},
            {"id": "task", "text": task_text},
        ],
        "ops": [
            {"op": "eval", "seq": 0, "segment": "root", "logits": False},
            {"op": "copy", "from": 0, "to": 1},
            {"op": "eval", "seq": 1, "segment": "branch_auth", "logits": False},
            {"op": "eval", "seq": 1, "segment": "task", "logits": True},
            {"op": "generate", "seq": 1, "label": "kv_branch_auth", "max_tokens": max_tokens},
            {"op": "eval", "seq": 2, "segment": "root", "logits": False},
            {"op": "eval", "seq": 2, "segment": "branch_auth", "logits": False},
            {"op": "eval", "seq": 2, "segment": "task", "logits": True},
            {"op": "compare", "left": 1, "right": 2, "label": "pre_generation_branch_vs_scratch"},
            {"op": "generate", "seq": 2, "label": "scratch_auth", "max_tokens": max_tokens},
        ],
    }


def score_bug_answer(text: str) -> dict[str, object]:
    lower = text.lower()
    file_hit = "backend/src/auth/cookie.ts" in lower or "auth/cookie" in lower or "cookie.ts" in lower
    cause_hit = "samesite=none" in lower and ("secure" in lower or "local" in lower or "localhost" in lower)
    dev_hit = "samesite=lax" in lower or ("lax" in lower and ("development" in lower or "test" in lower or "local" in lower))
    prod_hit = (
        "production" in lower
        and "samesite=none" in lower
        and "secure" in lower
    )
    unrelated_terms = ("qr", "billing", "analytics")
    unrelated_as_cause = any(term in lower for term in unrelated_terms) and any(
        marker in lower for marker in ("root cause", "bug is", "caused by", "fix the qr", "billing", "analytics")
    )

    positive = int(file_hit) + int(cause_hit) + int(dev_hit) + int(prod_hit)
    penalty = int(unrelated_as_cause)
    return {
        "score": max(0, positive - penalty),
        "max_score": 4,
        "file_hit": file_hit,
        "cause_hit": cause_hit,
        "dev_lax_hit": dev_hit,
        "prod_secure_hit": prod_hit,
        "unrelated_penalty": penalty,
    }


def run_bug_resolution_quality_benchmark(
    repo_root: Path,
    base_url: str,
    model: str,
    runs: int,
    warmup: int,
    max_output_tokens: int,
    temperature: float,
    seed_base: int,
    output_trace: Path,
    echo: bool = False,
    timeout_seconds: float = 600,
) -> dict[str, object]:
    events = build_fixture_repo_events(repo_root)
    output_trace.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(events, output_trace)
    prompts = build_bug_resolution_prompts(events)
    client = (
        EchoClient()
        if echo
        else OpenAICompatibleClient(
            base_url=base_url,
            model=model,
            api_key="local",
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
    )

    for warmup_index in range(warmup):
        for prompt in prompts:
            client.complete(prompt.prompt, max_tokens=max_output_tokens, seed=seed_base + warmup_index)

    samples = []
    for run_index in range(1, runs + 1):
        for prompt_index, prompt in enumerate(prompts):
            seed = seed_base + run_index * 100 + prompt_index
            result = client.complete(prompt.prompt, max_tokens=max_output_tokens, seed=seed)
            scored = score_bug_answer(result.text)
            samples.append(
                {
                    "run": run_index,
                    "mode": prompt.mode,
                    "seed": seed,
                    "prompt_tokens_estimate": prompt.token_estimate,
                    "runtime_prompt_tokens": token_value(result.usage, "prompt_tokens"),
                    "cached_prompt_tokens": cached_token_value(result.usage),
                    "completion_tokens": token_value(result.usage, "completion_tokens"),
                    "latency_ms": round(result.latency_ms, 2),
                    "score": scored,
                    "output_text": result.text,
                    "output_preview": result.text[:320],
                    "usage": result.usage,
                }
            )

    return {
        "benchmark": "fixture_repo_bug_resolution_quality",
        "repo_root": str(repo_root),
        "trace": str(output_trace),
        "model": "echo" if echo else model,
        "base_url": "echo" if echo else base_url,
        "runs": runs,
        "warmup": warmup,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "prompts": [
            {"mode": prompt.mode, "token_estimate": prompt.token_estimate}
            for prompt in prompts
        ],
        "samples": samples,
        "summary": summarize_bug_samples(samples),
    }


def summarize_bug_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    by_mode: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        by_mode.setdefault(str(sample["mode"]), []).append(sample)

    summary: dict[str, object] = {}
    for mode, mode_samples in by_mode.items():
        scores = [int(dict(sample["score"])["score"]) for sample in mode_samples]
        latencies = [float(sample["latency_ms"]) for sample in mode_samples]
        prompt_estimates = [int(sample["prompt_tokens_estimate"]) for sample in mode_samples]
        runtime_prompt_tokens = [sample.get("runtime_prompt_tokens") for sample in mode_samples]
        cached_prompt_tokens = [sample.get("cached_prompt_tokens") for sample in mode_samples]
        summary[mode] = {
            "runs": len(mode_samples),
            "score_avg": round(mean(scores), 3),
            "score_min": min(scores),
            "score_max": max(scores),
            "success_rate_score_4": round(sum(score == 4 for score in scores) / len(scores), 3),
            "latency_ms_avg": round(mean(latencies), 2),
            "latency_ms_min": round(min(latencies), 2),
            "latency_ms_max": round(max(latencies), 2),
            "prompt_tokens_estimate_avg": round(mean(prompt_estimates), 2),
            "runtime_prompt_tokens_avg": average_optional(runtime_prompt_tokens),
            "cached_prompt_tokens_avg": average_optional(cached_prompt_tokens),
            "runtime_cache_hit_ratio_avg": average_cache_ratio(runtime_prompt_tokens, cached_prompt_tokens),
        }
    return summary


def token_value(usage: dict[str, object], key: str) -> int | None:
    value = usage.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def cached_token_value(usage: dict[str, object]) -> int | None:
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    return token_value(details, "cached_tokens")


def average_optional(values: list[object]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return None
    return round(mean(numeric), 3)


def average_cache_ratio(prompt_tokens: list[object], cached_tokens: list[object]) -> float | None:
    ratios = []
    for prompt, cached in zip(prompt_tokens, cached_tokens, strict=True):
        if isinstance(prompt, int | float) and prompt:
            if isinstance(cached, int | float):
                ratios.append(float(cached) / float(prompt))
    if not ratios:
        return None
    return round(mean(ratios), 4)
