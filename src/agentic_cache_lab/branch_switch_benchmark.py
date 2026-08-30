"""Branch-switching benchmark: branch-cache vs prefix-slot vs scratch prefill.

Models an agent alternating between two subtasks (auth, qr) on top of a shared
stable prefix. Three conditions are executed in one native plan:

- ``branch_cache``: root is evaluated once, each branch keeps its own resident
  KV sequence. A switch only evaluates the new turn tokens.
- ``prefix_slot``: a single sequence emulates a one-slot prefix cache (like a
  single llama.cpp server slot). Root stays resident, but switching branches
  removes the other branch suffix and re-evaluates branch plus prior turns.
- ``scratch``: every switch re-evaluates root, branch, and prior turns on a
  fresh sequence (no cache reuse at all).

The switch schedule alternates branches: auth, qr, auth, qr, ...

Exactness bonus: the final auth state under branch_cache and under scratch
contain identical tokens at identical positions, so the plan compares their
logits and generates greedily from both.
"""

from __future__ import annotations

from statistics import mean, median

from .models import AgentEvent
from .segment_store import SegmentStore
from .semantic_kv_planner import build_branch_text, build_root_text

BRANCH_CACHE_ROOT_SEQ = 0
BRANCH_CACHE_SEQS = {"auth": 1, "qr": 2}
PREFIX_SLOT_SEQ = 3
SCRATCH_SEQ_BASE = 4

ROOT_PAD_SENTENCE = (
    "Project convention: keep changes minimal, respect existing style, run the "
    "test suite before concluding, and never touch unrelated modules. "
)


def build_turn_text(branch: str, turn_index: int) -> str:
    return (
        f"\nTurn {turn_index} on {branch} branch: re-read the failing detail, "
        f"note one new observation, and plan fix step {turn_index}.\n"
    )


def build_branch_switch_plan(
    events: list[AgentEvent],
    switches: int = 4,
    root_pad_words: int = 1024,
    generate_tokens: int = 24,
) -> dict[str, object]:
    store = SegmentStore()
    segments = store.add_events(events)
    root_text = build_root_text(segments)
    if root_pad_words > 0:
        pad_repeats = max(1, root_pad_words // len(ROOT_PAD_SENTENCE.split()))
        root_text = ROOT_PAD_SENTENCE * pad_repeats + "\n" + root_text
    branch_texts = {
        name: build_branch_text(name, segments, max_segments=4, max_words=80)
        for name in ("auth", "qr")
    }

    schedule = [("auth", "qr")[k % 2] for k in range(switches)]
    turn_counters = {"auth": 0, "qr": 0}
    turn_ids: list[str] = []
    plan_segments: list[dict[str, object]] = [
        {"id": "root", "text": root_text, "add_bos": True},
        {"id": "branch_auth", "text": branch_texts["auth"]},
        {"id": "branch_qr", "text": branch_texts["qr"]},
    ]
    for branch in schedule:
        turn_counters[branch] += 1
        turn_id = f"turn_{branch}_{turn_counters[branch]}"
        turn_ids.append(turn_id)
        plan_segments.append({"id": turn_id, "text": build_turn_text(branch, turn_counters[branch])})

    ops: list[dict[str, object]] = []
    op_tags: list[dict[str, object]] = []

    def emit(op: dict[str, object], condition: str, stage: str, switch: int | None = None) -> None:
        op_tags.append({"index": len(ops), "condition": condition, "stage": stage, "switch": switch})
        ops.append(op)

    # --- branch_cache -------------------------------------------------------
    emit({"op": "eval", "seq": BRANCH_CACHE_ROOT_SEQ, "segment": "root", "pos": 0}, "branch_cache", "setup")
    for branch in ("auth", "qr"):
        seq = BRANCH_CACHE_SEQS[branch]
        emit({"op": "copy", "from": BRANCH_CACHE_ROOT_SEQ, "to": seq}, "branch_cache", "setup")
        emit(
            {"op": "eval", "seq": seq, "segment": f"branch_{branch}", "start_after_segment": "root"},
            "branch_cache",
            "setup",
        )

    bc_last_segment = {"auth": "branch_auth", "qr": "branch_qr"}
    for k, (branch, turn_id) in enumerate(zip(schedule, turn_ids, strict=True), start=1):
        seq = BRANCH_CACHE_SEQS[branch]
        emit(
            {
                "op": "eval",
                "seq": seq,
                "segment": turn_id,
                "start_after_segment": bc_last_segment[branch],
                "logits": True,
            },
            "branch_cache",
            "switch",
            k,
        )
        bc_last_segment[branch] = turn_id

    # --- prefix_slot --------------------------------------------------------
    emit({"op": "eval", "seq": PREFIX_SLOT_SEQ, "segment": "root", "pos": 0}, "prefix_slot", "setup")
    slot_resident: list[str] = []
    branch_history: dict[str, list[str]] = {"auth": [], "qr": []}
    for k, (branch, turn_id) in enumerate(zip(schedule, turn_ids, strict=True), start=1):
        for stale in slot_resident:
            emit({"op": "remove", "seq": PREFIX_SLOT_SEQ, "segment": stale}, "prefix_slot", "switch", k)
        branch_history[branch].append(turn_id)
        chain = [f"branch_{branch}", *branch_history[branch]]
        previous = "root"
        for i, segment_id in enumerate(chain):
            emit(
                {
                    "op": "eval",
                    "seq": PREFIX_SLOT_SEQ,
                    "segment": segment_id,
                    "start_after_segment": previous,
                    "logits": i == len(chain) - 1,
                },
                "prefix_slot",
                "switch",
                k,
            )
            previous = segment_id
        slot_resident = chain

    # --- scratch ------------------------------------------------------------
    scratch_history: dict[str, list[str]] = {"auth": [], "qr": []}
    final_scratch_seq: dict[str, int] = {}
    for k, (branch, turn_id) in enumerate(zip(schedule, turn_ids, strict=True), start=1):
        seq = SCRATCH_SEQ_BASE + k - 1
        scratch_history[branch].append(turn_id)
        final_scratch_seq[branch] = seq
        chain = ["root", f"branch_{branch}", *scratch_history[branch]]
        previous: str | None = None
        for i, segment_id in enumerate(chain):
            op: dict[str, object] = {
                "op": "eval",
                "seq": seq,
                "segment": segment_id,
                "logits": i == len(chain) - 1,
            }
            if previous is None:
                op["pos"] = 0
            else:
                op["start_after_segment"] = previous
            emit(op, "scratch", "switch", k)
            previous = segment_id

    # --- exactness bonus ----------------------------------------------------
    last_branch = schedule[-1]
    exact_seq_branch = BRANCH_CACHE_SEQS[last_branch]
    exact_seq_scratch = final_scratch_seq[last_branch]
    emit(
        {
            "op": "compare",
            "left": exact_seq_branch,
            "right": exact_seq_scratch,
            "label": "final_switch_branch_vs_scratch",
        },
        "exactness",
        "compare",
    )
    if generate_tokens > 0:
        emit(
            {
                "op": "generate",
                "seq": exact_seq_branch,
                "label": "gen_branch_cache",
                "max_tokens": generate_tokens,
            },
            "exactness",
            "generate",
        )
        emit(
            {
                "op": "generate",
                "seq": exact_seq_scratch,
                "label": "gen_scratch",
                "max_tokens": generate_tokens,
            },
            "exactness",
            "generate",
        )

    return {
        "config": {"top_k": 5, "suppress_logs": True},
        "metadata": {
            "mode": "branch_switch_benchmark",
            "switches": switches,
            "schedule": schedule,
            "root_pad_words": root_pad_words,
            "generate_tokens": generate_tokens,
            "op_tags": op_tags,
        },
        "segments": plan_segments,
        "ops": ops,
    }


def summarize_switch_latencies(plan: dict[str, object], native: dict[str, object]) -> dict[str, object]:
    """Aggregate per-op native latencies into per-condition per-switch costs."""
    op_tags = list(plan["metadata"]["op_tags"])  # type: ignore[index]
    native_ops = {int(op["index"]): op for op in native.get("ops", [])}

    per_switch: dict[str, dict[int, float]] = {}
    setup_cost: dict[str, float] = {}
    for tag in op_tags:
        native_op = native_ops.get(int(tag["index"]))
        if native_op is None:
            continue
        latency = float(native_op.get("latency_ms", 0.0))
        condition = str(tag["condition"])
        if tag["stage"] == "setup":
            setup_cost[condition] = setup_cost.get(condition, 0.0) + latency
        elif tag["stage"] == "switch":
            per_switch.setdefault(condition, {})[int(tag["switch"])] = (
                per_switch.setdefault(condition, {}).get(int(tag["switch"]), 0.0) + latency
            )

    summary: dict[str, object] = {}
    for condition, switches in per_switch.items():
        values = [switches[k] for k in sorted(switches)]
        summary[condition] = {
            "setup_ms": round(setup_cost.get(condition, 0.0), 3),
            "per_switch_ms": {str(k): round(v, 3) for k, v in sorted(switches.items())},
            "switch_ms_mean": round(mean(values), 3),
            "switch_ms_median": round(median(values), 3),
            "switch_ms_total": round(sum(values), 3),
        }
    return summary


def summarize_kv_footprint(
    plan: dict[str, object],
    native: dict[str, object],
    kv_bytes_per_token: int,
) -> dict[str, object]:
    """Tokens resident per condition after the run, plus a KV byte estimate.

    Uses the native ``sequences`` output (``next_pos`` = tokens resident in a
    split KV cache). ``scratch`` sequences are counted together because a real
    no-cache baseline would free them; the number reported is what this run
    actually held.
    """
    sequences = {int(seq): dict(state) for seq, state in dict(native.get("sequences", {})).items()}
    switches = int(plan["metadata"]["switches"])  # type: ignore[index]

    groups = {
        "branch_cache": [BRANCH_CACHE_ROOT_SEQ, *BRANCH_CACHE_SEQS.values()],
        "prefix_slot": [PREFIX_SLOT_SEQ],
        "scratch": [SCRATCH_SEQ_BASE + k for k in range(switches)],
    }
    footprint: dict[str, object] = {"kv_bytes_per_token": kv_bytes_per_token}
    for condition, seqs in groups.items():
        tokens = sum(int(sequences.get(seq, {}).get("next_pos", 0)) for seq in seqs)
        footprint[condition] = {
            "sequences": len(seqs),
            "tokens_resident": tokens,
            "kv_mb_estimate": round(tokens * kv_bytes_per_token / (1024 * 1024), 1),
        }
    return footprint


def summarize_repeats(runs: list[dict[str, object]]) -> dict[str, object]:
    """Median-of-runs aggregation across repeated native executions."""
    conditions: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        for condition, stats in dict(run["switch_summary"]).items():  # type: ignore[index]
            bucket = conditions.setdefault(str(condition), {"setup_ms": [], "switch_ms_mean": [], "switch_ms_total": []})
            stats = dict(stats)
            bucket["setup_ms"].append(float(stats["setup_ms"]))
            bucket["switch_ms_mean"].append(float(stats["switch_ms_mean"]))
            bucket["switch_ms_total"].append(float(stats["switch_ms_total"]))

    return {
        condition: {
            "runs": len(values["switch_ms_mean"]),
            "setup_ms_median": round(median(values["setup_ms"]), 3),
            "switch_ms_mean_median": round(median(values["switch_ms_mean"]), 3),
            "switch_ms_total_median": round(median(values["switch_ms_total"]), 3),
        }
        for condition, values in conditions.items()
    }
