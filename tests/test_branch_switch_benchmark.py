from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.branch_switch_benchmark import (
    build_branch_switch_plan,
    summarize_switch_latencies,
)
from agentic_cache_lab.fixture_repo import build_fixture_repo_events

FIXTURE_REPO = Path(__file__).resolve().parents[1] / "examples" / "fixtures" / "shopbug-repo"


def build_plan(**kwargs):
    events = build_fixture_repo_events(FIXTURE_REPO)
    return build_branch_switch_plan(events, **kwargs)


def test_plan_has_three_conditions_and_alternating_schedule() -> None:
    plan = build_plan(switches=4, root_pad_words=64)
    metadata = plan["metadata"]
    assert metadata["schedule"] == ["auth", "qr", "auth", "qr"]
    conditions = {tag["condition"] for tag in metadata["op_tags"]}
    assert conditions == {"branch_cache", "prefix_slot", "scratch", "exactness"}


def test_op_tags_align_with_ops() -> None:
    plan = build_plan(switches=4, root_pad_words=64)
    op_tags = plan["metadata"]["op_tags"]
    assert len(op_tags) == len(plan["ops"])
    assert [tag["index"] for tag in op_tags] == list(range(len(plan["ops"])))


def test_branch_cache_switch_is_single_turn_eval() -> None:
    plan = build_plan(switches=4, root_pad_words=64)
    ops = plan["ops"]
    switch_ops = [
        ops[tag["index"]]
        for tag in plan["metadata"]["op_tags"]
        if tag["condition"] == "branch_cache" and tag["stage"] == "switch"
    ]
    assert len(switch_ops) == 4
    for op in switch_ops:
        assert op["op"] == "eval"
        assert op["segment"].startswith("turn_")
        assert op["logits"] is True


def test_scratch_reprefills_root_each_switch() -> None:
    plan = build_plan(switches=4, root_pad_words=64)
    ops = plan["ops"]
    root_evals = [
        ops[tag["index"]]
        for tag in plan["metadata"]["op_tags"]
        if tag["condition"] == "scratch" and ops[tag["index"]].get("segment") == "root"
    ]
    assert len(root_evals) == 4
    assert len({op["seq"] for op in root_evals}) == 4


def test_final_states_share_token_content() -> None:
    plan = build_plan(switches=4, root_pad_words=64)
    ops = plan["ops"]
    tags = plan["metadata"]["op_tags"]
    compare = next(ops[tag["index"]] for tag in tags if tag["stage"] == "compare")
    left_chain = [
        op["segment"]
        for op in ops
        if op["op"] == "eval" and op["seq"] == compare["left"]
    ]
    right_chain = [
        op["segment"]
        for op in ops
        if op["op"] == "eval" and op["seq"] == compare["right"]
    ]
    # branch_cache seq inherits root via copy; scratch evals it explicitly.
    assert ["root", *left_chain] == right_chain


def test_summarize_switch_latencies_groups_by_condition() -> None:
    plan = build_plan(switches=2, root_pad_words=64)
    native = {
        "ops": [
            {"index": tag["index"], "latency_ms": 10.0}
            for tag in plan["metadata"]["op_tags"]
        ]
    }
    summary = summarize_switch_latencies(plan, native)
    assert set(summary) == {"branch_cache", "prefix_slot", "scratch"}
    assert summary["branch_cache"]["per_switch_ms"] == {"1": 10.0, "2": 10.0}
    assert summary["scratch"]["switch_ms_total"] > summary["branch_cache"]["switch_ms_total"]
