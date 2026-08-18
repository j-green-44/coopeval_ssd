from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grid_prompt import build_request_text  # noqa: E402


class GridPromptTests(unittest.TestCase):
    def test_request_embeds_grid_json_and_requires_allowed_action(self) -> None:
        grid = {"schema_version": "cleanup_grid_context_v1", "cells": [], "valid_actions": ["NOOP", "FORWARD"]}
        text = build_request_text("Choose a safe action.", grid)

        self.assertIn(json.dumps(grid, sort_keys=True), text)
        self.assertIn('"action"', text)
        self.assertIn("FORWARD", text)
        self.assertNotIn("RGB", text)
        self.assertNotIn("world grid", text.lower())


if __name__ == "__main__":
    unittest.main()
