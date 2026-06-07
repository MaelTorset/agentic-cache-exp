from __future__ import annotations

import json
import os
import sys
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.benchmark_matrix import MatrixConfig, run_long_coding_matrix


def main() -> None:
    config = MatrixConfig(
        base_url=os.environ.get("ACL_BASE_URL", "http://127.0.0.1:8081"),
        model=os.environ.get("ACL_MODEL", "Qwen3-4B"),
        noise_files=parse_int_list(os.environ.get("ACL_NOISE_FILES", "4,8,12")),
        output_budgets=parse_int_list(os.environ.get("ACL_OUTPUT_BUDGETS", "48,96")),
        max_prompt_tokens=int(os.environ.get("ACL_MAX_PROMPT_TOKENS", "2048")),
        runs=int(os.environ.get("ACL_RUNS", "1")),
        warmup=int(os.environ.get("ACL_WARMUP", "1")),
        timeout_seconds=float(os.environ.get("ACL_TIMEOUT_SECONDS", "600")),
        echo=os.environ.get("ACL_ECHO", "0") == "1",
    )
    result = run_long_coding_matrix(config)
    output_path = output_file(os.environ.get("ACL_OUTPUT"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nWrote result to {output_path}")


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def output_file(raw: str | None) -> Path:
    if raw:
        return Path(raw)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "benchmark-results" / f"long-coding-matrix-{stamp}.json"


if __name__ == "__main__":
    main()
