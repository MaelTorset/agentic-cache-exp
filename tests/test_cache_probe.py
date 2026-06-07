from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.cache_probe import build_cache_probe_steps, summarize_probe
from agentic_cache_lab.coding_benchmark import CODING_OBJECTIVE, CODING_QUERY
from agentic_cache_lab.event_log import dump_jsonl
from agentic_cache_lab.synthetic import build_long_coding_task_events


class CacheProbeTest(unittest.TestCase):
    def test_cache_probe_sequence_alternates_full_and_forgotten_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            dump_jsonl(build_long_coding_task_events(noise_files=4), trace_path)
            steps = build_cache_probe_steps(
                trace_path=trace_path,
                query=CODING_QUERY,
                objective=CODING_OBJECTIVE,
                max_prompt_tokens=2048,
            )

        self.assertEqual(
            [step.name for step in steps],
            [
                "full_history_first",
                "forgotten_active_first",
                "full_history_return",
                "forgotten_active_return",
            ],
        )
        self.assertEqual(steps[0].prompt_mode, "raw")
        self.assertEqual(steps[1].prompt_mode, "routed_critical")
        self.assertGreater(steps[0].prompt_tokens_estimate, steps[1].prompt_tokens_estimate)

    def test_summarize_probe_marks_high_cache_hit_as_resident(self) -> None:
        summary = summarize_probe(
            [
                {"step": "full_history_first", "runtime_prompt_cache_hit_ratio": 0.0},
                {"step": "forgotten_active_first", "runtime_prompt_cache_hit_ratio": 0.0},
                {"step": "full_history_return", "runtime_prompt_cache_hit_ratio": 0.95},
                {"step": "forgotten_active_return", "runtime_prompt_cache_hit_ratio": 0.9},
            ]
        )

        self.assertTrue(summary["full_history_likely_resident_after_forget"])
        self.assertTrue(summary["forgotten_prompt_likely_resident_on_reuse"])


if __name__ == "__main__":
    unittest.main()
