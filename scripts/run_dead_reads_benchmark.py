"""Dead-reads forgetting benchmark (harness scenario).

Scenario: an agent reads three large files; only one matters for the task.
How expensive is it to "forget" the two dead reads, and what does it cost in
quality to keep them?

Conditions (one native plan):
- ``noisy``   (seq 1): append-only status quo — root + dead1 + useful + dead2
  + task. Nothing forgotten, full context carried.
- ``clean``   (seq 2): branch layout — reads lived on separate KV branches, the
  dead ones were dropped for free; final state is root + useful + task.
- ``spliced`` (seq 3): a-posteriori forgetting without branch structure — start
  from the noisy state, remove the dead reads' KV, RoPE-shift the survivors
  left, then eval the task. (What "forget" costs when context is append-only.)

Metrics: pre-generation logit comparison of noisy/spliced vs clean, greedy
answers scored with the fixture bug oracle, per-op latencies, and a
cache-write model: tokens whose KV a hosted append-only harness would have to
re-write to forget the dead reads (suffix after the first removal) vs zero for
the branch layout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.bug_resolution_benchmark import score_bug_answer
from agentic_cache_lab.fixture_repo import build_fixture_repo_events
from agentic_cache_lab.segment_store import SegmentStore
from agentic_cache_lab.semantic_kv_planner import build_root_text

TASK_TEXT = (
    "\nTask:\nYou are fixing the failing auth cookie test. Answer in exactly "
    "four short bullets: 1 file to patch, 2 root cause, 3 development/test "
    "behavior, 4 production behavior.\nAnswer:\n"
)

FILLER_LINE = "// reviewed: helper follows repository conventions, no behavior change intended\n"


def read_padded(repo_root: Path, relative: str, target_words: int) -> str:
    text = (repo_root / relative).read_text(encoding="utf-8")
    padded = f"FILE {relative}\n{text}\n"
    while len(padded.split()) < target_words:
        padded += FILLER_LINE + text + "\n"
    return padded


def build_plan(repo_root: Path, read_words: int, generate_tokens: int) -> dict[str, object]:
    events = build_fixture_repo_events(repo_root)
    store = SegmentStore()
    segments = store.add_events(events)
    root_text = build_root_text(segments)

    reads = {
        "read_dead1": read_padded(repo_root, "backend/src/billing/invoice.ts", read_words),
        "read_useful": read_padded(repo_root, "backend/src/auth/cookie.ts", read_words),
        "read_dead2": read_padded(repo_root, "backend/src/analytics/events.ts", read_words),
    }

    return {
        "config": {"top_k": 20, "suppress_logs": True},
        "metadata": {"mode": "dead_reads_forgetting", "read_words": read_words},
        "segments": [
            {"id": "root", "text": root_text, "add_bos": True},
            *({"id": k, "text": v} for k, v in reads.items()),
            {"id": "task", "text": TASK_TEXT},
        ],
        "ops": [
            # trunk
            {"op": "eval", "seq": 0, "segment": "root", "pos": 0},
            {"op": "copy", "from": 0, "to": 1},
            {"op": "copy", "from": 0, "to": 2},
            # noisy: dead1, useful, dead2 in read order
            {"op": "eval", "seq": 1, "segment": "read_dead1", "start_after_segment": "root"},
            {"op": "eval", "seq": 1, "segment": "read_useful", "start_after_segment": "read_dead1"},
            {"op": "eval", "seq": 1, "segment": "read_dead2", "start_after_segment": "read_useful"},
            # spliced starts from the noisy reads state
            {"op": "copy", "from": 1, "to": 3},
            {"op": "eval", "seq": 1, "segment": "task", "start_after_segment": "read_dead2", "logits": True},
            # clean branch layout
            {"op": "eval", "seq": 2, "segment": "read_useful", "start_after_segment": "root"},
            {"op": "eval", "seq": 2, "segment": "task", "start_after_segment": "read_useful", "logits": True},
            # spliced: drop dead2 then dead1, shifting survivors left each time
            {"op": "remove", "seq": 3, "segment": "read_dead2"},
            {
                "op": "shift",
                "seq": 3,
                "start_after_segment": "read_dead2",
                "p1": -1,
                "delta_segment": "read_dead2",
                "direction": "negative",
            },
            {"op": "remove", "seq": 3, "segment": "read_dead1"},
            {
                "op": "shift",
                "seq": 3,
                "start_after_segment": "read_dead1",
                "p1": -1,
                "delta_segment": "read_dead1",
                "direction": "negative",
            },
            {"op": "eval", "seq": 3, "segment": "task", "start_after_segment": "read_useful", "logits": True},
            {"op": "compare", "left": 1, "right": 2, "label": "noisy_vs_clean"},
            {"op": "compare", "left": 3, "right": 2, "label": "spliced_vs_clean"},
            {"op": "compare", "left": 3, "right": 1, "label": "spliced_vs_noisy"},
            {"op": "generate", "seq": 1, "label": "gen_noisy", "max_tokens": generate_tokens},
            {"op": "generate", "seq": 2, "label": "gen_clean", "max_tokens": generate_tokens},
            {"op": "generate", "seq": 3, "label": "gen_spliced", "max_tokens": generate_tokens},
        ],
    }


def main() -> None:
    repo_root = Path(os.environ.get("ACL_FIXTURE_REPO", str(ROOT / "examples" / "fixtures" / "shopbug-repo")))
    model = os.environ.get("ACL_MODEL_PATH", "/data/llama/models/Qwen3-4B-Q4_K_M.gguf")
    runner = Path(os.environ.get("ACL_NATIVE_RUNNER", str(ROOT / "build" / "native-probes" / "semantic-kv-runner")))
    read_words = int(os.environ.get("ACL_READ_WORDS", "600"))
    generate_tokens = int(os.environ.get("ACL_GENERATE_TOKENS", "96"))
    output = Path(os.environ.get("ACL_OUTPUT", str(ROOT / "benchmark-results" / "dead-reads-benchmark.json")))

    plan = build_plan(repo_root, read_words, generate_tokens)
    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / "dead_reads_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [
                str(runner),
                "-m", model,
                "--plan", str(plan_path),
                "--threads", os.environ.get("ACL_THREADS", "10"),
                "--ctx", os.environ.get("ACL_CTX", "16384"),
                "--batch", os.environ.get("ACL_BATCH", "4096"),
                "--seqs", os.environ.get("ACL_SEQS", "4"),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"semantic-kv-runner failed: {completed.stdout}\n{completed.stderr}")
    raw = json.loads(completed.stdout)

    seg_tokens = {k: int(v["tokens"]) for k, v in dict(raw.get("segments", {})).items()}
    texts = {label: str(dict(g).get("text", "")) for label, g in dict(raw.get("generations", {})).items()}
    # Append-only hosted harness: forgetting dead1 invalidates everything from
    # dead1 onward; those tokens must be re-prefilled and re-written to cache.
    invalidated = sum(seg_tokens.get(k, 0) for k in ("read_dead1", "read_useful", "read_dead2"))
    result = {
        "benchmark": "dead_reads_forgetting",
        "model": model,
        "read_words": read_words,
        "segment_tokens": seg_tokens,
        "comparisons": raw.get("comparisons", {}),
        "quality": {label: score_bug_answer(text) for label, text in texts.items()},
        "generation_texts": texts,
        "greedy_match": {
            "clean_vs_spliced": texts.get("gen_clean") == texts.get("gen_spliced"),
            "clean_vs_noisy": texts.get("gen_clean") == texts.get("gen_noisy"),
        },
        "cache_write_model": {
            "append_only_tokens_invalidated_by_forget": invalidated,
            "branch_layout_tokens_invalidated": 0,
            "context_tokens_carried_noisy": seg_tokens.get("root", 0) + invalidated + seg_tokens.get("task", 0),
            "context_tokens_carried_clean": seg_tokens.get("root", 0)
            + seg_tokens.get("read_useful", 0)
            + seg_tokens.get("task", 0),
        },
        "native": raw,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("comparisons", "quality", "greedy_match", "cache_write_model")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
