from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.models import AgentEvent, EventKind
from agentic_cache_lab.scorer import SegmentScorer, labels_to_exclude
from agentic_cache_lab.segment_store import SegmentStore


class SegmentScorerTest(unittest.TestCase):
    def test_exclusion_is_directional(self) -> None:
        self.assertEqual(labels_to_exclude("Fix auth. Omit QR scanner context."), {"qr"})
        self.assertEqual(labels_to_exclude("Omit auth login context."), {"auth"})

    def test_excluded_qr_file_scores_below_auth_file(self) -> None:
        events = [
            AgentEvent(EventKind.FILE_READ, "QR scanner mentions auth but is unrelated.", "qr.tsx", 1),
            AgentEvent(EventKind.FILE_READ, "Auth middleware verifies session cookies.", "auth.ts", 2),
        ]
        segments = SegmentStore().add_events(events)
        scores = SegmentScorer().score(segments, "Fix auth cookies. Omit QR scanner context.")

        by_source = {item.segment.source: item.score for item in scores}
        self.assertGreater(by_source["auth.ts"], by_source["qr.tsx"])


if __name__ == "__main__":
    unittest.main()
