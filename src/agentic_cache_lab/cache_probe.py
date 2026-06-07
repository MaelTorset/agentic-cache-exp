from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .harness import PromptCandidate, build_prompt_candidates
from .llm_client import EchoClient, LLMResult, OpenAICompatibleClient


@dataclass(frozen=True)
class CacheProbeStep:
    name: str
    prompt_mode: str
    prompt: str
    prompt_tokens_estimate: int


def build_cache_probe_steps(
    trace_path: Path,
    query: str,
    objective: str,
    max_prompt_tokens: int,
) -> list[CacheProbeStep]:
    prompts = {candidate.mode: candidate for candidate in build_prompt_candidates(trace_path, query, objective, max_prompt_tokens)}
    full = prompts["raw"]
    forgotten = prompts["routed_critical"]
    return [
        to_step("full_history_first", full),
        to_step("forgotten_active_first", forgotten),
        to_step("full_history_return", full),
        to_step("forgotten_active_return", forgotten),
    ]


def to_step(name: str, candidate: PromptCandidate) -> CacheProbeStep:
    return CacheProbeStep(
        name=name,
        prompt_mode=candidate.mode,
        prompt=candidate.prompt,
        prompt_tokens_estimate=candidate.token_estimate,
    )


def run_cache_residency_probe(
    trace_path: Path,
    query: str,
    objective: str,
    max_prompt_tokens: int,
    base_url: str,
    model: str,
    api_key: str,
    max_output_tokens: int,
    timeout_seconds: float,
    echo: bool = False,
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
    steps = build_cache_probe_steps(
        trace_path=trace_path,
        query=query,
        objective=objective,
        max_prompt_tokens=max_prompt_tokens,
    )

    samples = []
    for step in steps:
        result = client.complete(step.prompt, max_tokens=max_output_tokens)
        samples.append(sample_probe_step(step, result))

    return {
        "benchmark": "cache_residency_probe",
        "model": "echo" if echo else model,
        "base_url": "echo" if echo else base_url,
        "trace": str(trace_path),
        "max_output_tokens": max_output_tokens,
        "samples": samples,
        "summary": summarize_probe(samples),
        "interpretation": {
            "full_history_return": "If cached_prompt_tokens is high here, the runtime kept the full-history prompt reusable after a forgotten prompt ran.",
            "forgotten_active_return": "If cached_prompt_tokens is high here, the runtime kept the shortened forgotten prompt reusable too.",
            "limitation": "This probes runtime cache residency for whole prompt shapes. It does not prove arbitrary KV-cache surgery or recombination of middle prompt spans.",
        },
    }


def sample_probe_step(step: CacheProbeStep, result: LLMResult) -> dict[str, object]:
    runtime_prompt_tokens = prompt_tokens(result)
    cached_prompt_tokens = cached_tokens(result)
    return {
        "step": step.name,
        "prompt_mode": step.prompt_mode,
        "prompt_tokens_estimate": step.prompt_tokens_estimate,
        "runtime_prompt_tokens": runtime_prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "runtime_prompt_cache_hit_ratio": cache_hit_ratio(runtime_prompt_tokens, cached_prompt_tokens),
        "latency_ms": round(result.latency_ms, 2),
        "usage": result.usage,
        "output_preview": result.text[:160],
    }


def summarize_probe(samples: list[dict[str, object]]) -> dict[str, object]:
    by_step = {str(sample["step"]): sample for sample in samples}
    full_return = by_step.get("full_history_return", {})
    forgotten_return = by_step.get("forgotten_active_return", {})
    full_ratio = optional_float(full_return.get("runtime_prompt_cache_hit_ratio"))
    forgotten_ratio = optional_float(forgotten_return.get("runtime_prompt_cache_hit_ratio"))
    return {
        "full_history_return_cache_hit_ratio": full_ratio,
        "forgotten_active_return_cache_hit_ratio": forgotten_ratio,
        "full_history_likely_resident_after_forget": full_ratio is not None and full_ratio >= 0.8,
        "forgotten_prompt_likely_resident_on_reuse": forgotten_ratio is not None and forgotten_ratio >= 0.8,
    }


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


def optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
