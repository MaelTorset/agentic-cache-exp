from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.event_log import dump_jsonl
from agentic_cache_lab.harness import build_prompt_candidates
from agentic_cache_lab.synthetic import build_long_context_events


class SyntheticTraceTest(unittest.TestCase):
    def test_long_context_trace_makes_routing_savings_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            dump_jsonl(build_long_context_events(noise_blocks=24), trace_path)
            prompts = build_prompt_candidates(
                trace_path=trace_path,
                query="Focus on auth cookies. Omit QR scanner and frontend noise.",
                objective="Measure routing savings.",
                max_prompt_tokens=2048,
            )
            by_mode = {prompt.mode: prompt for prompt in prompts}

        raw = by_mode["raw"]
        routed = by_mode["routed"]
        self.assertEqual(raw.mode, "raw")
        self.assertEqual(routed.mode, "routed")
        self.assertGreater(raw.token_estimate, routed.token_estimate * 2)
        self.assertGreater(routed.stable_prefix_token_estimate, 0)


if __name__ == "__main__":
    unittest.main()
