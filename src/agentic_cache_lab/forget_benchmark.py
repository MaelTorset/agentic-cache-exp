from __future__ import annotations

from pathlib import Path

from .cli import run_benchmark
from .harness import run_model_harness


MODE_NAMES = {
    "raw": "industry_raw_cached",
    "routed": "forget_routed_cached_v0",
    "routed_critical": "forget_critical_context_v1",
    "oracle_relevant": "oracle_relevant_only",
}

COMPARISON_MODES = (
    "forget_routed_cached_v0",
    "forget_critical_context_v1",
    "oracle_relevant_only",
)


def run_forget_vs_industry(
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
    timeout_seconds: float,
    echo: bool = False,
) -> dict[str, object]:
    harness = run_model_harness(
        trace_path=trace_path,
        query=query,
        objective=objective,
        max_prompt_tokens=max_prompt_tokens,
        base_url=base_url,
        model=model,
        api_key=api_key,
        runs=runs,
        warmup=warmup,
        max_output_tokens=max_output_tokens,
        echo=echo,
        timeout_seconds=timeout_seconds,
    )
    routing_v0 = run_benchmark(
        trace_path=trace_path,
        query=query,
        objective=objective,
        max_prompt_tokens=max_prompt_tokens,
    )
    routing_v1 = run_benchmark(
        trace_path=trace_path,
        query=query,
        objective=objective,
        max_prompt_tokens=max_prompt_tokens,
        critical_context=True,
    )

    relabel_modes(harness)
    included_sources_by_mode = {
        "forget_routed_cached_v0": [str(item["source"]) for item in routing_v0["included_segments"]],
        "forget_critical_context_v1": [str(item["source"]) for item in routing_v1["included_segments"]],
    }
    omitted_sources_by_mode = {
        "forget_routed_cached_v0": [str(item["source"]) for item in routing_v0["omitted_segments"]],
        "forget_critical_context_v1": [str(item["source"]) for item in routing_v1["omitted_segments"]],
    }

    return {
        "benchmark": "forget_vs_industry",
        "interpretation": {
            "industry_raw_cached": "Full rolling history with provider/runtime prompt cache. It keeps every useless file in the active prompt.",
            "forget_routed_cached_v0": "Same event log preserved externally, but irrelevant segments are omitted from the active prompt while runtime cache remains enabled.",
            "forget_critical_context_v1": "Adds a protected critical-context lane for failed tests, expected behavior, root-cause evidence, and likely patch targets.",
            "forget_decider": "segment_scorer_v0 + critical_context_v1",
        },
        "trace": str(trace_path),
        "query": query,
        "objective": objective,
        "included_sources": included_sources_by_mode["forget_routed_cached_v0"],
        "forgotten_sources": omitted_sources_by_mode["forget_routed_cached_v0"],
        "included_sources_by_mode": included_sources_by_mode,
        "forgotten_sources_by_mode": omitted_sources_by_mode,
        "routing": {
            "forget_routed_cached_v0": routing_summary(routing_v0),
            "forget_critical_context_v1": routing_summary(routing_v1),
        },
        "model_harness": harness,
        "comparison": compare_samples(harness["samples"]),
    }


def relabel_modes(result: dict[str, object]) -> None:
    for prompt in result["prompts"]:
        prompt["mode"] = MODE_NAMES.get(str(prompt["mode"]), str(prompt["mode"]))
    for sample in result["samples"]:
        sample["mode"] = MODE_NAMES.get(str(sample["mode"]), str(sample["mode"]))
    result["summary"] = {MODE_NAMES.get(mode, mode): value for mode, value in result["summary"].items()}


def compare_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    industry = [sample for sample in samples if sample["mode"] == "industry_raw_cached"]
    if not industry:
        return {}

    return {
        mode: compare_mode_to_industry(industry, mode_samples)
        for mode in COMPARISON_MODES
        if (mode_samples := [sample for sample in samples if sample["mode"] == mode])
    }


def compare_mode_to_industry(
    industry: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> dict[str, object]:
    industry_prompt = average_int(sample.get("runtime_prompt_tokens") for sample in industry)
    candidate_prompt = average_int(sample.get("runtime_prompt_tokens") for sample in candidate)
    industry_latency = average_float(sample["latency_ms"] for sample in industry)
    candidate_latency = average_float(sample["latency_ms"] for sample in candidate)

    return {
        "runtime_prompt_token_reduction_ratio": reduction_ratio(industry_prompt, candidate_prompt),
        "latency_reduction_ratio": reduction_ratio(industry_latency, candidate_latency),
        "industry_runtime_prompt_tokens_avg": industry_prompt,
        "candidate_runtime_prompt_tokens_avg": candidate_prompt,
        "industry_latency_ms_avg": round(industry_latency, 2) if industry_latency is not None else None,
        "candidate_latency_ms_avg": round(candidate_latency, 2) if candidate_latency is not None else None,
    }


def routing_summary(result: dict[str, object]) -> dict[str, object]:
    return {
        "strategy": result["packing_strategy"],
        "raw_tokens_estimate": result["raw_tokens_estimate"],
        "forget_tokens_estimate": result["routed_tokens_estimate"],
        "stable_prefix_tokens_estimate": result["stable_prefix_tokens_estimate"],
        "estimated_token_reduction_ratio": reduction_ratio(
            result["raw_tokens_estimate"],
            result["routed_tokens_estimate"],
        ),
    }


def average_int(values: object) -> int | None:
    numeric = [int(value) for value in values if isinstance(value, int)]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric))


def average_float(values: object) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def reduction_ratio(before: int | float | None, after: int | float | None) -> float | None:
    if not before or after is None:
        return None
    return round((before - after) / before, 4)
