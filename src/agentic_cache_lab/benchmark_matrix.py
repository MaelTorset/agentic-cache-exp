from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import tempfile

from .coding_benchmark import run_long_coding_task_benchmark


@dataclass(frozen=True)
class MatrixConfig:
    base_url: str
    model: str
    noise_files: tuple[int, ...]
    output_budgets: tuple[int, ...]
    max_prompt_tokens: int
    runs: int
    warmup: int
    timeout_seconds: float
    echo: bool = False


def run_long_coding_matrix(config: MatrixConfig) -> dict[str, object]:
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for noise_count in config.noise_files:
            for output_budget in config.output_budgets:
                trace_path = root / f"coding_noise_{noise_count}_out_{output_budget}.jsonl"
                result = run_long_coding_task_benchmark(
                    trace_path=trace_path,
                    base_url=config.base_url,
                    model=config.model,
                    api_key="local",
                    noise_files=noise_count,
                    max_prompt_tokens=config.max_prompt_tokens,
                    runs=config.runs,
                    warmup=config.warmup,
                    max_output_tokens=output_budget,
                    timeout_seconds=config.timeout_seconds,
                    echo=config.echo,
                )
                rows.append(summarize_result(noise_count, output_budget, result))

    return {
        "benchmark": "long_coding_task_matrix",
        "model": "echo" if config.echo else config.model,
        "base_url": "echo" if config.echo else config.base_url,
        "config": {
            "noise_files": list(config.noise_files),
            "output_budgets": list(config.output_budgets),
            "max_prompt_tokens": config.max_prompt_tokens,
            "runs": config.runs,
            "warmup": config.warmup,
            "timeout_seconds": config.timeout_seconds,
        },
        "rows": rows,
        "aggregate": aggregate_rows(rows),
    }


def summarize_result(noise_files: int, output_budget: int, result: dict[str, object]) -> dict[str, object]:
    comparison = result["comparison"]
    quality = result["quality"]
    routing = result["routing"]
    row = {
        "noise_files": noise_files,
        "output_budget": output_budget,
        "industry_quality_pass_rate": quality["industry_raw_cached"]["pass_rate"],
        "industry_runtime_prompt_tokens_avg": prompt_tokens_for_mode(result, "industry_raw_cached"),
        "industry_latency_ms_avg": latency_for_mode(result, "industry_raw_cached"),
    }
    for mode in ("forget_routed_cached_v0", "forget_critical_context_v1", "oracle_relevant_only"):
        mode_quality = quality[mode]["pass_rate"]
        mode_comparison = comparison[mode]
        row[f"{mode}_quality_pass_rate"] = mode_quality
        row[f"{mode}_quality_delta"] = round(mode_quality - quality["industry_raw_cached"]["pass_rate"], 4)
        row[f"{mode}_runtime_prompt_token_reduction_ratio"] = mode_comparison[
            "runtime_prompt_token_reduction_ratio"
        ]
        row[f"{mode}_latency_reduction_ratio"] = mode_comparison["latency_reduction_ratio"]
        row[f"{mode}_runtime_prompt_tokens_avg"] = mode_comparison["candidate_runtime_prompt_tokens_avg"]
        row[f"{mode}_latency_ms_avg"] = mode_comparison["candidate_latency_ms_avg"]
        row[f"{mode}_tokens_per_success"] = tokens_per_success(
            mode_comparison["candidate_runtime_prompt_tokens_avg"],
            mode_quality,
        )
        if mode in routing:
            row[f"{mode}_estimated_token_reduction_ratio"] = routing[mode]["estimated_token_reduction_ratio"]
        if mode in result["prompt_oracle_exposure"]:
            row[f"{mode}_oracle_fact_exposure_count"] = result["prompt_oracle_exposure"][mode][
                "oracle_fact_exposure_count"
            ]

    row["industry_tokens_per_success"] = tokens_per_success(
        row["industry_runtime_prompt_tokens_avg"],
        row["industry_quality_pass_rate"],
    )
    row["industry_oracle_fact_exposure_count"] = result["prompt_oracle_exposure"]["industry_raw_cached"][
        "oracle_fact_exposure_count"
    ]
    return row


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    aggregate = {
        "cases": len(rows),
        "industry_quality_pass_rate_avg": avg(row["industry_quality_pass_rate"] for row in rows),
        "industry_runtime_prompt_tokens_avg": avg_optional(row["industry_runtime_prompt_tokens_avg"] for row in rows),
    }
    for mode in ("forget_routed_cached_v0", "forget_critical_context_v1", "oracle_relevant_only"):
        aggregate[f"{mode}_quality_pass_rate_avg"] = avg(row[f"{mode}_quality_pass_rate"] for row in rows)
        aggregate[f"{mode}_quality_delta_avg"] = avg(row[f"{mode}_quality_delta"] for row in rows)
        aggregate[f"{mode}_runtime_prompt_token_reduction_ratio_avg"] = avg_optional(
            row[f"{mode}_runtime_prompt_token_reduction_ratio"] for row in rows
        )
        aggregate[f"{mode}_estimated_token_reduction_ratio_avg"] = avg_optional(
            row.get(f"{mode}_estimated_token_reduction_ratio") for row in rows
        )
        aggregate[f"{mode}_latency_reduction_ratio_avg"] = avg_optional(
            row[f"{mode}_latency_reduction_ratio"] for row in rows
        )
        aggregate[f"{mode}_tokens_per_success_avg"] = avg_optional(
            row[f"{mode}_tokens_per_success"] for row in rows
        )
        aggregate[f"{mode}_oracle_fact_exposure_count_avg"] = avg_optional(
            row[f"{mode}_oracle_fact_exposure_count"] for row in rows
        )
    return aggregate


def prompt_tokens_for_mode(result: dict[str, object], mode: str) -> int | None:
    samples = [sample for sample in result["model_harness"]["samples"] if sample["mode"] == mode]
    numeric = [int(sample["runtime_prompt_tokens"]) for sample in samples if isinstance(sample["runtime_prompt_tokens"], int)]
    if not numeric:
        return None
    return round(mean(numeric))


def latency_for_mode(result: dict[str, object], mode: str) -> float | None:
    samples = [sample for sample in result["model_harness"]["samples"] if sample["mode"] == mode]
    numeric = [float(sample["latency_ms"]) for sample in samples if isinstance(sample["latency_ms"], int | float)]
    if not numeric:
        return None
    return round(mean(numeric), 2)


def tokens_per_success(prompt_tokens: int | None, pass_rate: float) -> float | None:
    if not prompt_tokens or pass_rate <= 0:
        return None
    return round(prompt_tokens / pass_rate, 2)


def avg(values: object) -> float:
    numeric = [float(value) for value in values]
    return round(mean(numeric), 4)


def avg_optional(values: object) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return None
    return round(mean(numeric), 4)
