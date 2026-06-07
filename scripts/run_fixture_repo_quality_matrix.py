from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.bug_resolution_benchmark import run_bug_resolution_quality_benchmark


def main() -> None:
    repo_root = Path(os.environ.get("ACL_FIXTURE_REPO", str(ROOT / "examples" / "fixtures" / "shopbug-repo")))
    base_url = os.environ.get("ACL_BASE_URL", "http://127.0.0.1:8082")
    model = os.environ.get("ACL_MODEL", "Qwen3-4B")
    runs = int(os.environ.get("ACL_RUNS", "5"))
    warmup = int(os.environ.get("ACL_WARMUP", "1"))
    max_output_tokens = int(os.environ.get("ACL_MAX_OUTPUT_TOKENS", "128"))
    temperature = float(os.environ.get("ACL_TEMPERATURE", "0.35"))
    seed_base = int(os.environ.get("ACL_SEED_BASE", "4242"))
    timeout_seconds = float(os.environ.get("ACL_TIMEOUT_SECONDS", "900"))
    output = Path(os.environ.get("ACL_OUTPUT", str(ROOT / "benchmark-results" / "fixture-repo-quality-matrix.json")))
    trace_output = Path(os.environ.get("ACL_TRACE_OUTPUT", str(ROOT / "benchmark-results" / "fixture-repo-quality-session.jsonl")))
    include_kv = os.environ.get("ACL_INCLUDE_KV", "1") != "0"
    echo = os.environ.get("ACL_ECHO", "0") == "1"

    result = run_bug_resolution_quality_benchmark(
        repo_root=repo_root,
        base_url=base_url,
        model=model,
        runs=runs,
        warmup=warmup,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        seed_base=seed_base,
        output_trace=trace_output,
        echo=echo,
        timeout_seconds=timeout_seconds,
    )
    if include_kv and not echo:
        result["kv_branch_metrics"] = run_kv_branch_probe()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def run_kv_branch_probe() -> dict[str, object]:
    kv_output = ROOT / "benchmark-results" / "fixture-repo-quality-kv-branch.json"
    env = os.environ.copy()
    env["ACL_OUTPUT"] = str(kv_output)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_fixture_repo_branch_benchmark.py")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": completed.stderr or completed.stdout,
        }
    raw = json.loads(completed.stdout)
    return {
        "ok": True,
        "planner_summary": raw.get("planner_summary", {}),
        "comparisons": raw.get("comparisons", {}),
        "segments": raw.get("segments", {}),
        "copy_latency_ms_total": round(sum(float(op["latency_ms"]) for op in raw.get("ops", []) if op.get("op") == "copy"), 6),
        "root_eval_latency_ms": next(
            (
                float(op["latency_ms"])
                for op in raw.get("ops", [])
                if op.get("op") == "eval" and op.get("segment") == "root" and op.get("seq") == 0
            ),
            None,
        ),
    }


if __name__ == "__main__":
    main()
