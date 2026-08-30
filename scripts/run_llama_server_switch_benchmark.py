"""Replay the branch-switch schedule against a running llama-server.

The native branch-switch benchmark compares resident KV branches against a
*single* prefix-cache slot. A real llama.cpp deployment is not limited to one
slot: ``llama-server -np N`` keeps N independent slots, each with its own KV
and its own prefix matching, which is the natural baseline for "one resident
conversation per subtask".

This script replays the exact same segment texts and the same
``auth -> qr -> auth -> qr`` schedule through llama-server under two
conditions:

- ``server_slots``: each branch is pinned to its own slot (``id_slot``), so a
  switch only prefills the new turn tokens.
- ``server_single_slot``: both branches share one slot, which is what the
  native ``prefix_slot`` condition models.

Per-switch cost is ``timings.prompt_ms``; ``timings.cache_n`` reports how many
prompt tokens were served from the slot's KV, and ``timings.prompt_n`` how many
were actually evaluated.

Start the server first, matching the native run's parameters, e.g.:

    llama-server -m Qwen3-4B-Q4_K_M.gguf -c 16384 -t 10 -b 2048 -np 4 \
        --host 127.0.0.1 --port 8099 --slots
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.branch_switch_benchmark import build_branch_switch_plan
from agentic_cache_lab.fixture_repo import build_fixture_repo_events

BRANCH_SLOTS = {"auth": 0, "qr": 1}
SINGLE_SLOT = 2


def post_json(url: str, payload: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def erase_slot(base_url: str, slot: int) -> None:
    request = urllib.request.Request(
        f"{base_url}/slots/{slot}?action=erase", data=b"{}", headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(request, timeout=60).read()
    except urllib.error.HTTPError as error:  # slot never used yet
        if error.code not in (400, 404):
            raise


def complete(base_url: str, prompt: str, slot: int) -> dict:
    """Prefill ``prompt`` on ``slot`` and return the server timings."""
    result = post_json(
        f"{base_url}/completion",
        {
            "prompt": prompt,
            "n_predict": 1,
            "cache_prompt": True,
            "id_slot": slot,
            "temperature": 0.0,
        },
    )
    timings = dict(result["timings"])
    return {
        "prompt_ms": float(timings["prompt_ms"]),
        "prompt_n": int(timings["prompt_n"]),
        "cache_n": int(timings.get("cache_n", 0)),
        "id_slot": int(result.get("id_slot", slot)),
    }


def build_prompts(plan: dict) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Return per-branch setup prompts and the ordered (branch, prompt) switches."""
    segments = {str(segment["id"]): str(segment["text"]) for segment in plan["segments"]}
    schedule = list(plan["metadata"]["schedule"])

    setup = {branch: segments["root"] + segments[f"branch_{branch}"] for branch in ("auth", "qr")}

    history: dict[str, list[str]] = {"auth": [], "qr": []}
    counters = {"auth": 0, "qr": 0}
    switches: list[tuple[str, str]] = []
    for branch in schedule:
        counters[branch] += 1
        history[branch].append(segments[f"turn_{branch}_{counters[branch]}"])
        switches.append((branch, setup[branch] + "".join(history[branch])))
    return setup, switches


def run_condition(base_url: str, setup: dict[str, str], switches: list[tuple[str, str]], *, shared: bool) -> dict:
    """Run one condition; ``shared=True`` forces both branches onto one slot."""
    slots = {branch: SINGLE_SLOT for branch in BRANCH_SLOTS} if shared else dict(BRANCH_SLOTS)
    for slot in sorted(set(slots.values())):
        erase_slot(base_url, slot)

    setup_ms = 0.0
    for branch, prompt in setup.items():
        setup_ms += complete(base_url, prompt, slots[branch])["prompt_ms"]

    per_switch = []
    for index, (branch, prompt) in enumerate(switches, start=1):
        measurement = complete(base_url, prompt, slots[branch])
        measurement.update({"switch": index, "branch": branch})
        per_switch.append(measurement)

    values = [entry["prompt_ms"] for entry in per_switch]
    return {
        "setup_ms": round(setup_ms, 3),
        "per_switch": per_switch,
        "switch_ms_mean": round(sum(values) / len(values), 3),
        "switch_ms_median": round(median(values), 3),
        "switch_ms_total": round(sum(values), 3),
    }


def main() -> None:
    base_url = os.environ.get("ACL_SERVER_URL", "http://127.0.0.1:8099").rstrip("/")
    repo_root = Path(os.environ.get("ACL_FIXTURE_REPO", str(ROOT / "examples" / "fixtures" / "shopbug-repo")))
    switches_count = int(os.environ.get("ACL_SWITCHES", "4"))
    root_pad_words = int(os.environ.get("ACL_ROOT_PAD_WORDS", "1024"))
    repeats = int(os.environ.get("ACL_REPEATS", "5"))
    output = Path(
        os.environ.get("ACL_OUTPUT", str(ROOT / "benchmark-results" / "llama-server-switch-benchmark.json"))
    )

    events = build_fixture_repo_events(repo_root)
    plan = build_branch_switch_plan(
        events,
        switches=switches_count,
        root_pad_words=root_pad_words,
        generate_tokens=0,
    )
    setup, switch_prompts = build_prompts(plan)

    runs: list[dict] = []
    for repeat in range(1, repeats + 1):
        started = time.time()
        run = {
            "repeat": repeat,
            "server_slots": run_condition(base_url, setup, switch_prompts, shared=False),
            "server_single_slot": run_condition(base_url, setup, switch_prompts, shared=True),
        }
        run["wall_seconds"] = round(time.time() - started, 2)
        runs.append(run)
        print(
            f"repeat {repeat}/{repeats}: "
            f"slots {run['server_slots']['switch_ms_mean']:.1f} ms/switch, "
            f"single {run['server_single_slot']['switch_ms_mean']:.1f} ms/switch",
            flush=True,
        )

    summary = {
        condition: {
            "runs": repeats,
            "setup_ms_median": round(median([run[condition]["setup_ms"] for run in runs]), 3),
            "switch_ms_mean_median": round(median([run[condition]["switch_ms_mean"] for run in runs]), 3),
            "switch_ms_total_median": round(median([run[condition]["switch_ms_total"] for run in runs]), 3),
        }
        for condition in ("server_slots", "server_single_slot")
    }

    payload = {
        "metadata": {
            "mode": "llama_server_switch_benchmark",
            "server_url": base_url,
            "switches": switches_count,
            "schedule": plan["metadata"]["schedule"],
            "root_pad_words": root_pad_words,
            "repeats": repeats,
        },
        "summary": summary,
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
