from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.benchmark_matrix import MatrixConfig, run_long_coding_matrix


class BenchmarkMatrixTest(unittest.TestCase):
    def test_matrix_echo_aggregates_multiple_cases(self) -> None:
        result = run_long_coding_matrix(
            MatrixConfig(
                base_url="http://127.0.0.1:8081",
                model="Qwen3-4B",
                noise_files=(2, 4),
                output_budgets=(16,),
                max_prompt_tokens=1024,
                runs=1,
                warmup=0,
                timeout_seconds=1,
                echo=True,
            )
        )

        self.assertEqual(result["benchmark"], "long_coding_task_matrix")
        self.assertEqual(result["aggregate"]["cases"], 2)
        self.assertGreater(
            result["aggregate"]["forget_routed_cached_v0_estimated_token_reduction_ratio_avg"],
            0,
        )
        self.assertIn("forget_critical_context_v1_quality_pass_rate_avg", result["aggregate"])
        self.assertIn("oracle_relevant_only_tokens_per_success_avg", result["aggregate"])


if __name__ == "__main__":
    unittest.main()
