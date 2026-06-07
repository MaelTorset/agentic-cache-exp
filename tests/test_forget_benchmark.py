from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.event_log import dump_jsonl
from agentic_cache_lab.forget_benchmark import run_forget_vs_industry
from agentic_cache_lab.synthetic import build_long_context_events


class ForgetBenchmarkTest(unittest.TestCase):
    def test_forget_vs_industry_echo_reports_forgotten_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            dump_jsonl(build_long_context_events(noise_blocks=4), trace_path)
            result = run_forget_vs_industry(
                trace_path=trace_path,
                query="Focus only on auth cookies. Forget QR scanner and frontend noise.",
                objective="Compare active-context forgetting.",
                max_prompt_tokens=1024,
                base_url="http://127.0.0.1:8081",
                model="Qwen3-4B",
                api_key="local",
                runs=1,
                warmup=0,
                max_output_tokens=8,
                timeout_seconds=1,
                echo=True,
            )

        self.assertEqual(result["benchmark"], "forget_vs_industry")
        self.assertIn("industry_raw_cached", result["model_harness"]["summary"])
        self.assertIn("forget_routed_cached_v0", result["model_harness"]["summary"])
        self.assertIn("forget_critical_context_v1", result["model_harness"]["summary"])
        self.assertIn("oracle_relevant_only", result["model_harness"]["summary"])
        self.assertGreater(len(result["forgotten_sources"]), 0)
        self.assertGreater(result["routing"]["forget_routed_cached_v0"]["estimated_token_reduction_ratio"], 0.35)
        self.assertGreater(result["routing"]["forget_critical_context_v1"]["estimated_token_reduction_ratio"], 0.2)


if __name__ == "__main__":
    unittest.main()
