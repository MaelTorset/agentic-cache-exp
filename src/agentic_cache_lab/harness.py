from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .event_log import load_jsonl
from .llm_client import EchoClient, LLMResult, OpenAICompatibleClient
from .models import AgentEvent, estimate_tokens
from .packer import PackConfig, PromptPacker
from .segment_store import SegmentStore


@dataclass(frozen=True)
class PromptCandidate:
    mode: str
    prompt: str
    token_estimate: int
    stable_prefix_token_estimate: int


def build_prompt_candidates(
    trace_path: Path,
    query: str,
    objective: str,
    max_prompt_tokens: int,
) -> list[PromptCandidate]:
    events = load_jsonl(trace_path)
    store = SegmentStore()
    segments = store.add_events(events)
    packed_v0 = PromptPacker(config=PackConfig(max_prompt_tokens=max_prompt_tokens)).pack(
        segments,
        query=query,
        objective=objective,
    )
    packed_v1 = PromptPacker(
        config=PackConfig(max_prompt_tokens=max_prompt_tokens, critical_context=True)
    ).pack(
        segments,
        query=query,
        objective=objective,
    )

    raw_history = "\n\n".join(f"{event.kind.value.upper()} {event.source}\n{event.text}" for event in events)
    raw_prompt = (
        "You are a local coding agent experimenting with cache-aware context routing.\n"
        f"Current objective: {objective}\n"
        "Use the full active history below.\n\n"
        f"{raw_history}\n\nUser query:\n{query}"
    )
    oracle_relevant_prompt = build_oracle_relevant_prompt(events, query=query, objective=objective)
    return [
        PromptCandidate(
            mode="raw",
            prompt=raw_prompt,
            token_estimate=estimate_tokens(raw_prompt),
            stable_prefix_token_estimate=0,
        ),
        PromptCandidate(
            mode="routed",
            prompt=packed_v0.prompt,
            token_estimate=packed_v0.token_estimate,
            stable_prefix_token_estimate=packed_v0.stable_token_estimate,
        ),
        PromptCandidate(
            mode="routed_critical",
            prompt=packed_v1.prompt,
            token_estimate=packed_v1.token_estimate,
            stable_prefix_token_estimate=packed_v1.stable_token_estimate,
        ),
        PromptCandidate(
            mode="oracle_relevant",
            prompt=oracle_relevant_prompt,
            token_estimate=estimate_tokens(oracle_relevant_prompt),
            stable_prefix_token_estimate=0,
        ),
    ]


def build_oracle_relevant_prompt(events: list[AgentEvent], query: str, objective: str) -> str:
    relevant_history = "\n\n".join(
        f"{event.kind.value.upper()} {event.source}\n{event.text}" for event in events if is_oracle_relevant_event(event)
    )
    return (
        "You are a local coding agent experimenting with cache-aware context routing.\n"
        f"Current objective: {objective}\n"
        "Use only the relevant history below. This is an oracle upper-bound baseline, not a deployable policy.\n\n"
        f"{relevant_history}\n\nUser query:\n{query}"
    )


def is_oracle_relevant_event(event: AgentEvent) -> bool:
    source = event.source.lower()
    text = event.text.lower()
    if any(marker in source for marker in ("noise_file", "/qr/", "/frontend/", "/billing/", "/analytics/")):
        return False
    if any(marker in source for marker in ("backend/src/auth", "backend/tests/auth", "logs/auth")):
        return True
    if source in {"notes", "conversation"}:
        return "auth cookie" in text or "authentication cookie" in text
    return False


def run_model_harness(
    trace_path: Path,
    query: str,
    objective: str,
    max_prompt_tokens: int,
    base_url: str,
    model: str,
    api_key: str,
    runs: int,
    warmup: int,
    max_output_tokens: int,
    echo: bool = False,
    timeout_seconds: float = 300,
) -> dict[str, object]:
    client = (
        EchoClient()
        if echo
        else OpenAICompatibleClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    )
    prompts = build_prompt_candidates(
        trace_path=trace_path,
        query=query,
        objective=objective,
        max_prompt_tokens=max_prompt_tokens,
    )

    for _ in range(warmup):
        for prompt in prompts:
            client.complete(prompt.prompt, max_tokens=max_output_tokens)

    samples = []
    for run_index in range(1, runs + 1):
        for prompt in prompts:
            result = client.complete(prompt.prompt, max_tokens=max_output_tokens)
            samples.append(sample_result(run_index, prompt, result))

    return {
        "trace": str(trace_path),
        "model": "echo" if echo else model,
        "base_url": "echo" if echo else base_url,
        "runs": runs,
        "warmup": warmup,
        "max_output_tokens": max_output_tokens,
        "timeout_seconds": timeout_seconds,
        "prompts": [
            {
                "mode": prompt.mode,
                "token_estimate": prompt.token_estimate,
                "stable_prefix_token_estimate": prompt.stable_prefix_token_estimate,
            }
            for prompt in prompts
        ],
        "samples": samples,
        "summary": summarize_samples(samples),
    }


def sample_result(run_index: int, prompt: PromptCandidate, result: LLMResult) -> dict[str, object]:
    output_tokens = completion_tokens(result) or estimate_tokens(result.text)
    latency_seconds = max(result.latency_ms / 1000, 0.001)
    runtime_prompt_tokens = prompt_tokens(result)
    cached_prompt_tokens = cached_tokens(result)
    return {
        "run": run_index,
        "mode": prompt.mode,
        "prompt_tokens_estimate": prompt.token_estimate,
        "stable_prefix_tokens_estimate": prompt.stable_prefix_token_estimate,
        "runtime_prompt_tokens": runtime_prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "runtime_prompt_cache_hit_ratio": cache_hit_ratio(runtime_prompt_tokens, cached_prompt_tokens),
        "latency_ms": round(result.latency_ms, 2),
        "output_tokens_estimate": output_tokens,
        "output_tokens_per_second_estimate": round(output_tokens / latency_seconds, 2),
        "usage": result.usage,
        "output_text": result.text,
        "output_preview": result.text[:240],
    }


def completion_tokens(result: LLMResult) -> int | None:
    value = result.usage.get("completion_tokens")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def prompt_tokens(result: LLMResult) -> int | None:
    value = result.usage.get("prompt_tokens")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def cached_tokens(result: LLMResult) -> int | None:
    details = result.usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("cached_tokens")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def cache_hit_ratio(prompt_token_count: int | None, cached_token_count: int | None) -> float | None:
    if not prompt_token_count or cached_token_count is None:
        return None
    return round(cached_token_count / prompt_token_count, 4)


def summarize_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    by_mode: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        by_mode.setdefault(str(sample["mode"]), []).append(sample)

    summary = {}
    for mode, mode_samples in by_mode.items():
        latencies = [float(sample["latency_ms"]) for sample in mode_samples]
        tps = [float(sample["output_tokens_per_second_estimate"]) for sample in mode_samples]
        summary[mode] = {
            "latency_ms_avg": round(mean(latencies), 2),
            "latency_ms_min": round(min(latencies), 2),
            "latency_ms_max": round(max(latencies), 2),
            "output_tps_estimate_avg": round(mean(tps), 2),
            "runtime_prompt_cache_hit_ratio_avg": average_optional(
                sample.get("runtime_prompt_cache_hit_ratio") for sample in mode_samples
            ),
            "runs": len(mode_samples),
        }
    return summary


def average_optional(values: object) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return None
    return round(mean(numeric), 4)
