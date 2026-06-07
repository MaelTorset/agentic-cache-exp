from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NativePlanTest(unittest.TestCase):
    def test_example_plans_have_required_shape(self) -> None:
        for path in sorted((ROOT / "examples" / "native").glob("*.json")):
            with self.subTest(path=path.name):
                plan = json.loads(path.read_text(encoding="utf-8"))
                self.assertIsInstance(plan.get("segments"), list)
                self.assertIsInstance(plan.get("ops"), list)
                self.assertGreater(len(plan["segments"]), 0)
                self.assertGreater(len(plan["ops"]), 0)

                segment_ids = {segment["id"] for segment in plan["segments"]}
                for segment in plan["segments"]:
                    self.assertIsInstance(segment["text"], str)

                for op in plan["ops"]:
                    self.assertIn(op["op"], {"eval", "copy", "remove", "shift", "keep", "compare", "generate"})
                    if op["op"] == "eval":
                        self.assertIn(op["segment"], segment_ids)

    def test_native_runner_help_when_built(self) -> None:
        runner = ROOT / "build" / "native-probes" / "semantic-kv-runner"
        if not runner.exists():
            self.skipTest("semantic-kv-runner has not been built")

        result = subprocess.run(
            [str(runner), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--plan", result.stderr)


if __name__ == "__main__":
    unittest.main()
