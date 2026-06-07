from __future__ import annotations

from pathlib import Path

from .event_log import dump_jsonl
from .forget_benchmark import run_forget_vs_industry
from .harness import build_prompt_candidates
from .synthetic import build_long_coding_task_events


CODING_QUERY = (
    "/no_think\n"
    "Solve only the authentication cookie bug. Forget QR, frontend, billing, and analytics noise. "
    "Answer with the patch target, root cause, and behavior change."
)

CODING_OBJECTIVE = (
    "Evaluate whether active-context forgetting preserves coding-task quality while reducing prompt tokens."
)


def run_long_coding_task_benchmark(
    trace_path: Path,
    base_url: str,
    model: str,
    api_key: str,
    noise_files: int,
    max_prompt_tokens: int,
    runs: int,
    warmup: int,
    max_output_tokens: int,
    timeout_seconds: float,
    echo: bool = False,
) -> dict[str, object]:
    dump_jsonl(build_long_coding_task_events(noise_files=noise_files), trace_path)
    result = run_forget_vs_industry(
        trace_path=trace_path,
        query=CODING_QUERY,
        objective=CODING_OBJECTIVE,
        max_prompt_tokens=max_prompt_tokens,
        base_url=base_url,
        model=model,
        api_key=api_key,
        runs=runs,
        warmup=warmup,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        echo=echo,
    )
    result["benchmark"] = "long_coding_task_forget_vs_industry"
    result["quality"] = evaluate_quality(result["model_harness"]["samples"])
    result["prompt_oracle_exposure"] = evaluate_prompt_oracle_exposure(
        trace_path=trace_path,
        query=CODING_QUERY,
        objective=CODING_OBJECTIVE,
        max_prompt_tokens=max_prompt_tokens,
    )
    result["task_expectations"] = {
        "file": "backend/src/auth/cookie.ts",
        "root_cause": "SameSite=None without Secure is rejected on local HTTP",
        "fix": "Use SameSite=Lax without Secure for local development; keep SameSite=None with Secure for production.",
    }
    return result


def evaluate_quality(samples: list[dict[str, object]]) -> dict[str, object]:
    by_mode = {}
    for sample in samples:
        mode = str(sample["mode"])
        by_mode.setdefault(mode, []).append(evaluate_answer(str(sample.get("output_text", ""))))

    return {
        mode: {
            "pass_rate": round(sum(item["passed"] for item in items) / len(items), 4),
            "checks": items,
        }
        for mode, items in by_mode.items()
    }


def evaluate_answer(answer: str) -> dict[str, object]:
    lowered = answer.lower()
    checks = {
        "mentions_patch_file": "backend/src/auth/cookie.ts" in lowered or "auth/cookie" in lowered,
        "mentions_samesite_none": "samesite=none" in lowered or "same site none" in lowered,
        "mentions_secure": "secure" in lowered,
        "mentions_local_lax": ("samesite=lax" in lowered or "same site lax" in lowered) and (
            "local" in lowered or "development" in lowered
        ),
        "does_not_focus_on_noise": not any(
            phrase in lowered
            for phrase in (
                "patch qr",
                "qr scanner is the root",
                "billing is the root",
                "analytics is the root",
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "answer_preview": answer[:360],
    }


def evaluate_prompt_oracle_exposure(
    trace_path: Path,
    query: str,
    objective: str,
    max_prompt_tokens: int,
) -> dict[str, object]:
    mode_names = {
        "raw": "industry_raw_cached",
        "routed": "forget_routed_cached_v0",
        "routed_critical": "forget_critical_context_v1",
        "oracle_relevant": "oracle_relevant_only",
    }
    return {
        mode_names.get(candidate.mode, candidate.mode): oracle_fact_exposure(candidate.prompt)
        for candidate in build_prompt_candidates(
            trace_path=trace_path,
            query=query,
            objective=objective,
            max_prompt_tokens=max_prompt_tokens,
        )
    }


def oracle_fact_exposure(prompt: str) -> dict[str, object]:
    lowered = prompt.lower()
    facts = {
        "patch_file": "backend/src/auth/cookie.ts" in lowered or "auth/cookie" in lowered,
        "samesite_none_without_secure": "samesite=none without secure" in lowered
        or "samesite none without secure" in lowered,
        "local_lax_expected": ("samesite=lax" in lowered or "same site lax" in lowered)
        and ("local" in lowered or "development" in lowered),
        "prod_none_secure_expected": ("samesite=none with secure" in lowered or "samesite=none${secure" in lowered)
        and ("production" in lowered or "prod" in lowered),
    }
    return {
        "oracle_fact_exposure_count": sum(facts.values()),
        "facts": facts,
    }
