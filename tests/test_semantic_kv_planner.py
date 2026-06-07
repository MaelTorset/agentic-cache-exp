from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.semantic_kv_planner import build_semantic_kv_plan
from agentic_cache_lab.synthetic import build_long_coding_task_events


class SemanticKvPlannerTest(unittest.TestCase):
    def test_planner_emits_branches_scratch_and_compares(self) -> None:
        plan = build_semantic_kv_plan(
            build_long_coding_task_events(noise_files=8),
            branch_labels=("auth", "qr", "billing"),
            measured_labels=("auth", "qr"),
            top_k=5,
        )

        segment_ids = {segment["id"] for segment in plan["segments"]}
        self.assertIn("root", segment_ids)
        self.assertIn("branch_auth", segment_ids)
        self.assertIn("branch_qr", segment_ids)
        self.assertIn("branch_billing", segment_ids)

        eval_segments = [op["segment"] for op in plan["ops"] if op["op"] == "eval"]
        for segment in eval_segments:
            self.assertIn(segment, segment_ids)

        comparisons = [op for op in plan["ops"] if op["op"] == "compare"]
        self.assertEqual([op["label"] for op in comparisons], ["auth_branch_vs_scratch", "qr_branch_vs_scratch"])

        metadata = plan["metadata"]
        self.assertEqual(metadata["branch_sequences"]["auth"], 1)
        self.assertEqual(metadata["branch_sequences"]["qr"], 2)
        self.assertGreater(metadata["scratch_sequences"]["auth"], metadata["branch_sequences"]["billing"])

    def test_planner_omits_missing_branch_labels(self) -> None:
        plan = build_semantic_kv_plan(
            build_long_coding_task_events(noise_files=0),
            branch_labels=("auth", "nonexistent"),
            measured_labels=("auth", "nonexistent"),
            top_k=3,
        )

        segment_ids = {segment["id"] for segment in plan["segments"]}
        self.assertIn("branch_auth", segment_ids)
        self.assertNotIn("branch_nonexistent", segment_ids)
        self.assertEqual(plan["config"]["top_k"], 3)
        self.assertEqual([op["label"] for op in plan["ops"] if op["op"] == "compare"], ["auth_branch_vs_scratch"])


if __name__ == "__main__":
    unittest.main()
