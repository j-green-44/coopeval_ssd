from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), "/home/jack/phd/tiny_cooperative_vlm/src"]

from run_grid_context import DETAILED_PROMPT_PATH, INSTRUCTIONS  # noqa: E402


class DetailedPromptTests(unittest.TestCase):
    def test_reuses_prior_cleanup_policy_prompt_with_grid_override(self) -> None:
        source = DETAILED_PROMPT_PATH.read_text(encoding="utf-8")
        self.assertIn("maximise YOUR OWN cumulative reward", source)
        self.assertTrue(INSTRUCTIONS.startswith(source))
        self.assertIn("GRID INTERFACE OVERRIDE", INSTRUCTIONS)
        self.assertIn("do NOT receive an RGB image", INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
