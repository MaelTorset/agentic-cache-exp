from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.bug_resolution_benchmark import build_native_generation_plan, score_bug_answer
from agentic_cache_lab.event_log import dump_jsonl
from agentic_cache_lab.fixture_repo import build_fixture_repo_events


def main() -> None:
    repo_root = Path(os.environ.get("ACL_FIXTURE_REPO", str(ROOT / "examples" / "fixtures" / "shopbug-repo")))
    model = os.environ.get("ACL_MODEL_PATH", "/data/llama/models/Qwen3-4B-Q4_K_M.gguf")
    runner = Path(os.environ.get("ACL_NATIVE_RUNNER", str(ROOT / "build" / "native-probes" / "semantic-kv-runner")))
    threads = os.environ.get("ACL_THREADS", "10")
    ctx = os.environ.get("ACL_CTX", "2048")
    batch = os.environ.get("ACL_BATCH", "1024")
    seqs = os.environ.get("ACL_SEQS", "4")
    max_tokens = int(os.environ.get("ACL_MAX_OUTPUT_TOKENS", "96"))
    output = Path(os.environ.get("ACL_OUTPUT", str(ROOT / "benchmark-results" / "fixture-repo-native-generation.json")))
    trace_output = Path(os.environ.get("ACL_TRACE_OUTPUT", str(ROOT / "benchmark-results" / "fixture-repo-native-generation-session.jsonl")))

    events = build_fixture_repo_events(repo_root)
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(events, trace_output)
    plan = build_native_generation_plan(events, max_tokens=max_tokens)

    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / "native_generation_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raw = run_native_runner(
            runner=runner,
            model=model,
            plan_path=plan_path,
            threads=threads,
            ctx=ctx,
            batch=batch,
            seqs=seqs,
        )

    generations = dict(raw.get("generations", {}))
    kv_text = str(dict(generations.get("kv_branch_auth", {})).get("text", ""))
    scratch_text = str(dict(generations.get("scratch_auth", {})).get("text", ""))
    result = {
        "benchmark": "fixture_repo_native_kv_generation",
        "repo_root": str(repo_root),
        "trace": str(trace_output),
        "model": model,
        "plan_metadata": plan["metadata"],
        "native": raw,
        "quality": {
            "kv_branch_auth": score_bug_answer(kv_text),
            "scratch_auth": score_bug_answer(scratch_text),
        },
        "generation_text_match": kv_text == scratch_text,
        "generation_texts": {
            "kv_branch_auth": kv_text,
            "scratch_auth": scratch_text,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


if __name__ == "__main__":
    main()

