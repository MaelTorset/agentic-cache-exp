from __future__ import annotations

from pathlib import Path

from .cli import run_benchmark
from .harness import run_model_harness


MODE_NAMES = {
    "raw": "industry_raw_cached",
    "routed": "forget_routed_cached",
}


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
    routing = run_benchmark(
        trace_path=trace_path,
        query=query,
        objective=objective,
        max_prompt_tokens=max_prompt_tokens,
    )

    relabel_modes(harness)
    included_sources = [str(item["source"]) for item in routing["included_segments"]]
    omitted_sources = [str(item["source"]) for item in routing["omitted_segments"]]

    return {
        "benchmark": "forget_vs_industry",
        "interpretation": {
            "industry_raw_cached": "Full rolling history with provider/runtime prompt cache. It keeps every useless file in the active prompt.",
            "forget_routed_cached": "Same event log preserved externally, but irrelevant segments are omitted from the active prompt while runtime cache remains enabled.",
            "forget_decider": "segment_scorer_v0",
        },
        "trace": str(trace_path),
        "query": query,
        "objective": objective,
        "included_sources": included_sources,
        "forgotten_sources": omitted_sources,
        "routing": {
            "raw_tokens_estimate": routing["raw_tokens_estimate"],
            "forget_tokens_estimate": routing["routed_tokens_estimate"],
            "stable_prefix_tokens_estimate": routing["stable_prefix_tokens_estimate"],
            "estimated_token_reduction_ratio": reduction_ratio(
                routing["raw_tokens_estimate"],
                routing["routed_tokens_estimate"],
            ),
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
    forget = [sample for sample in samples if sample["mode"] == "forget_routed_cached"]
    if not industry or not forget:
        return {}

    industry_prompt = average_int(sample.get("runtime_prompt_tokens") for sample in industry)
    forget_prompt = average_int(sample.get("runtime_prompt_tokens") for sample in forget)
    industry_latency = average_float(sample["latency_ms"] for sample in industry)
    forget_latency = average_float(sample["latency_ms"] for sample in forget)

    return {
        "runtime_prompt_token_reduction_ratio": reduction_ratio(industry_prompt, forget_prompt),
        "latency_reduction_ratio": reduction_ratio(industry_latency, forget_latency),
        "industry_runtime_prompt_tokens_avg": industry_prompt,
        "forget_runtime_prompt_tokens_avg": forget_prompt,
        "industry_latency_ms_avg": round(industry_latency, 2) if industry_latency is not None else None,
        "forget_latency_ms_avg": round(forget_latency, 2) if forget_latency is not None else None,
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
