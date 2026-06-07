from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.event_log import dump_jsonl
from agentic_cache_lab.fixture_repo import build_fixture_repo_events
from agentic_cache_lab.semantic_kv_planner import build_semantic_kv_plan


def main() -> None:
    repo_root = Path(os.environ.get("ACL_FIXTURE_REPO", str(ROOT / "examples" / "fixtures" / "shopbug-repo")))
    model = os.environ.get("ACL_MODEL_PATH", "/data/llama/models/Qwen3-4B-Q4_K_M.gguf")
    runner = Path(os.environ.get("ACL_NATIVE_RUNNER", str(ROOT / "build" / "native-probes" / "semantic-kv-runner")))
    branch_labels = parse_labels(os.environ.get("ACL_BRANCH_LABELS", "auth,qr,billing,analytics"))
    measured_labels = parse_labels(os.environ.get("ACL_MEASURED_LABELS", "auth,qr"))
    threads = os.environ.get("ACL_THREADS", "10")
    ctx = os.environ.get("ACL_CTX", "2048")
    batch = os.environ.get("ACL_BATCH", "1024")
    seqs = os.environ.get("ACL_SEQS", "8")
    top_k = int(os.environ.get("ACL_TOP_K", "5"))
    trace_output = Path(os.environ.get("ACL_TRACE_OUTPUT", str(ROOT / "benchmark-results" / "fixture-repo-session.jsonl")))
    output = os.environ.get("ACL_OUTPUT")

    events = build_fixture_repo_events(repo_root)
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(events, trace_output)

    plan = build_semantic_kv_plan(
        events,
        branch_labels=branch_labels,
        measured_labels=measured_labels,
        top_k=top_k,
    )

    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / "fixture_repo_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = run_native_runner(runner, model, plan_path, threads=threads, ctx=ctx, batch=batch, seqs=seqs)

    result["fixture_repo"] = {
        "repo_root": str(repo_root),
        "trace_output": str(trace_output),
        "event_count": len(events),
    }
    result["planner_summary"] = summarize_plan(plan, result)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))


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


def summarize_plan(plan: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    metadata = dict(plan.get("metadata", {}))
    branch_sequences = dict(metadata.get("branch_sequences", {}))
    scratch_sequences = dict(metadata.get("scratch_sequences", {}))
    segments = dict(result.get("segments", {}))
    root_tokens = int(dict(segments.get("root", {})).get("tokens", 0))

    return {
        "branch_labels": list(branch_sequences.keys()),
        "measured_labels": list(scratch_sequences.keys()),
        "root_tokens": root_tokens,
        "estimated_prefill_tokens_avoided": root_tokens * len(branch_sequences),
        "branch_sequence_count": len(branch_sequences),
        "scratch_sequence_count": len(scratch_sequences),
    }


def parse_labels(raw: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not labels:
        raise ValueError("expected at least one label")
    return labels


if __name__ == "__main__":
    main()

