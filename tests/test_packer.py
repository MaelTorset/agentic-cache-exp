from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.models import AgentEvent, EventKind
from agentic_cache_lab.packer import PackConfig, PromptPacker
from agentic_cache_lab.segment_store import SegmentStore


class PromptPackerTest(unittest.TestCase):
    def test_packer_keeps_auth_and_omits_unrelated_qr_when_budget_is_small(self) -> None:
        events = [
            AgentEvent(EventKind.FILE_READ, "QR scanner camera component with qrcode display details.", "qr.tsx", 1),
            AgentEvent(EventKind.FILE_READ, "Auth middleware verifies JWT session cookie and user login.", "auth.ts", 2),
            AgentEvent(EventKind.ERROR, "Login returns 200 but cookie is missing on /api/me.", "dev.log", 3),
            AgentEvent(EventKind.DECISION, "Decision: focus on auth cookie delivery, QR can wait.", "notes", 4),
        ]
        store = SegmentStore()
        segments = store.add_events(events)

        packed = PromptPacker(config=PackConfig(max_prompt_tokens=120)).pack(
            segments,
            query="Fix the auth cookie bug. Omit QR scanner context.",
            objective="Resolve authentication only.",
        )

        included_sources = {item.segment.source for item in packed.included}
        omitted_sources = {segment.source for segment in packed.omitted}

        self.assertIn("auth.ts", included_sources)
        self.assertIn("notes", included_sources)
        self.assertIn("qr.tsx", omitted_sources)

    def test_stable_prefix_contains_durable_decisions(self) -> None:
        events = [
            AgentEvent(EventKind.DECISION, "Decision: auth and QR are separate concerns.", "notes", 1),
            AgentEvent(EventKind.ERROR, "Transient stack trace for one failing request.", "logs", 2),
        ]
        store = SegmentStore()
        packed = PromptPacker(config=PackConfig(max_prompt_tokens=300)).pack(
            store.add_events(events),
            query="Continue auth work.",
            objective="Keep durable context stable.",
        )

        self.assertIn("Decision: auth and QR are separate concerns.", packed.stable_prefix)
        self.assertIn("Transient stack trace", packed.dynamic_suffix)


if __name__ == "__main__":
    unittest.main()
