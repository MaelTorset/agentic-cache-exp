from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.coding_benchmark import evaluate_answer, run_long_coding_task_benchmark


class CodingBenchmarkTest(unittest.TestCase):
    def test_quality_evaluator_passes_expected_answer(self) -> None:
        result = evaluate_answer(
            "Patch backend/src/auth/cookie.ts. Root cause is SameSite=None without Secure "
            "on local HTTP. Use SameSite=Lax in local development and keep Secure production cookies."
        )

        self.assertTrue(result["passed"])

    def test_long_coding_task_echo_reports_token_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_long_coding_task_benchmark(
                trace_path=Path(directory) / "trace.jsonl",
                base_url="http://127.0.0.1:8081",
                model="Qwen3-4B",
                api_key="local",
                noise_files=8,
                max_prompt_tokens=1536,
                runs=1,
                warmup=0,
                max_output_tokens=16,
                timeout_seconds=1,
                echo=True,
            )

        self.assertEqual(result["benchmark"], "long_coding_task_forget_vs_industry")
        self.assertGreater(result["routing"]["forget_routed_cached_v0"]["estimated_token_reduction_ratio"], 0.4)
        self.assertIn("forget_critical_context_v1", result["quality"])
        self.assertIn("oracle_relevant_only", result["quality"])
        self.assertIn("prompt_oracle_exposure", result)
        self.assertTrue(any("noise_file" in source for source in result["forgotten_sources"]))


if __name__ == "__main__":
    unittest.main()
