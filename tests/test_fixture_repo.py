from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.fixture_repo import build_fixture_repo_events
from agentic_cache_lab.models import EventKind
from agentic_cache_lab.semantic_kv_planner import build_semantic_kv_plan


ROOT = Path(__file__).resolve().parents[1]


class FixtureRepoTest(unittest.TestCase):
    def test_fixture_repo_trace_contains_real_file_reads(self) -> None:
        events = build_fixture_repo_events(ROOT / "examples" / "fixtures" / "shopbug-repo")

        file_reads = [event for event in events if event.kind == EventKind.FILE_READ]
        self.assertGreaterEqual(len(file_reads), 6)
        self.assertTrue(any(event.source == "backend/src/auth/cookie.ts" for event in file_reads))
        self.assertTrue(any("SameSite=None" in event.text for event in file_reads))

    def test_fixture_repo_trace_emits_semantic_branches(self) -> None:
        events = build_fixture_repo_events(ROOT / "examples" / "fixtures" / "shopbug-repo")
        plan = build_semantic_kv_plan(
            events,
            branch_labels=("auth", "qr", "billing", "analytics"),
            measured_labels=("auth", "qr"),
        )

        segment_ids = {segment["id"] for segment in plan["segments"]}
        self.assertIn("branch_auth", segment_ids)
        self.assertIn("branch_qr", segment_ids)
        self.assertIn("branch_billing", segment_ids)
        self.assertIn("branch_analytics", segment_ids)
        self.assertEqual([op["label"] for op in plan["ops"] if op["op"] == "compare"], ["auth_branch_vs_scratch", "qr_branch_vs_scratch"])


if __name__ == "__main__":
    unittest.main()

