from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.branch_switch_benchmark import (
    build_branch_switch_plan,
    summarize_kv_footprint,
    summarize_repeats,
    summarize_switch_latencies,
)
from agentic_cache_lab.fixture_repo import build_fixture_repo_events


def main() -> None:
    repo_root = Path(os.environ.get("ACL_FIXTURE_REPO", str(ROOT / "examples" / "fixtures" / "shopbug-repo")))
    model = os.environ.get("ACL_MODEL_PATH", "/data/llama/models/Qwen3-4B-Q4_K_M.gguf")
    runner = Path(os.environ.get("ACL_NATIVE_RUNNER", str(ROOT / "build" / "native-probes" / "semantic-kv-runner")))
    threads = os.environ.get("ACL_THREADS", "10")
    ctx = os.environ.get("ACL_CTX", "16384")
    batch = os.environ.get("ACL_BATCH", "2048")
    seqs = os.environ.get("ACL_SEQS", "8")
    switches = int(os.environ.get("ACL_SWITCHES", "4"))
    root_pad_words = int(os.environ.get("ACL_ROOT_PAD_WORDS", "1024"))
    generate_tokens = int(os.environ.get("ACL_GENERATE_TOKENS", "24"))
    repeats = int(os.environ.get("ACL_REPEATS", "5"))
    # default: Qwen3-4B f16 KV — 36 layers * 8 kv heads * 128 head dim * 2 (K+V) * 2 bytes
    kv_bytes_per_token = int(os.environ.get("ACL_KV_BYTES_PER_TOKEN", str(36 * 8 * 128 * 2 * 2)))
    output = Path(os.environ.get("ACL_OUTPUT", str(ROOT / "benchmark-results" / "branch-switch-benchmark.json")))

    events = build_fixture_repo_events(repo_root)
    plan = build_branch_switch_plan(
        events,
        switches=switches,
        root_pad_words=root_pad_words,
        generate_tokens=generate_tokens,
    )

    runs: list[dict[str, object]] = []
    segment_tokens: dict[str, object] | None = None
    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / "branch_switch_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for repeat in range(1, repeats + 1):
            raw = run_native_runner(
                runner=runner,
                model=model,
                plan_path=plan_path,
                threads=threads,
                ctx=ctx,
                batch=batch,
                seqs=seqs,
            )
            if segment_tokens is None:
                segment_tokens = dict(raw.get("segments", {}))
            kv_footprint = summarize_kv_footprint(plan, raw, kv_bytes_per_token)
            generations = dict(raw.get("generations", {}))
            gen_branch = str(dict(generations.get("gen_branch_cache", {})).get("text", ""))
            gen_scratch = str(dict(generations.get("gen_scratch", {})).get("text", ""))
            runs.append(
                {
                    "repeat": repeat,
                    "switch_summary": summarize_switch_latencies(plan, raw),
                    "comparisons": raw.get("comparisons", {}),
                    "generation_text_match": gen_branch == gen_scratch,
                    "kv_footprint": kv_footprint,
                    "sequences": raw.get("sequences", {}),
                    "native_ops": raw.get("ops", []),
                }
            )
            print(f"repeat {repeat}/{repeats} done", file=sys.stderr)

    result = {
        "benchmark": "branch_switch_latency",
        "repo_root": str(repo_root),
        "model": model,
        "config": {
            "threads": threads,
            "ctx": ctx,
            "batch": batch,
            "seqs": seqs,
            "switches": switches,
            "root_pad_words": root_pad_words,
            "generate_tokens": generate_tokens,
            "repeats": repeats,
        },
        "plan_metadata": {k: v for k, v in plan["metadata"].items() if k != "op_tags"},
        "segment_tokens": segment_tokens,
        "runs": runs,
        "summary": summarize_repeats(runs),
        "exactness": {
            "generation_text_match_all_runs": all(bool(run["generation_text_match"]) for run in runs),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "exactness": result["exactness"]}, indent=2, sort_keys=True))


def run_native_runner(
    runner: Path,
    model: str,
    plan_path: Path,
    threads: str,
    ctx: str,
    batch: str,
    seqs: str,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(runner),
            "-m",
            model,
            "--plan",
            str(plan_path),
            "--threads",
            threads,
            "--ctx",
            ctx,
            "--batch",
            batch,
            "--seqs",
            seqs,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"semantic-kv-runner failed: {completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout)


if __name__ == "__main__":
    main()
