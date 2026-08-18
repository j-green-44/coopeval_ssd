from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from action_outcome import build_action_outcome


class ActionOutcomeTests(unittest.TestCase):
    def test_reports_blocked_movement_without_exposing_coordinates(self) -> None:
        outcome = build_action_outcome("STEP_RIGHT", moved=False, local_view_changed=False, dirt_before=2, dirt_after=2)
        self.assertEqual(outcome["movement_outcome"], "blocked")
        self.assertFalse(outcome["local_view_changed"])
        self.assertNotIn("position", outcome)

    def test_reports_cleaning_target_removed(self) -> None:
        outcome = build_action_outcome("FIRE_CLEAN", moved=False, local_view_changed=True, dirt_before=3, dirt_after=2)
        self.assertEqual(outcome["movement_outcome"], "not_applicable")
        self.assertEqual(outcome["cleaning_effect"], "one_or_more_visible_targets_removed")
        self.assertEqual(outcome["visible_live_dirt_before"], 3)
        self.assertEqual(outcome["visible_live_dirt_after"], 2)


if __name__ == "__main__":
    unittest.main()
