from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from affordances import build_local_affordances


class AffordanceTests(unittest.TestCase):
    def test_marks_blocked_moves_and_counts_only_forward_beam_dirt(self) -> None:
        names = ("wall", "sand", "river", "dirt_live")
        grid = np.zeros((5, 5, 4), dtype=np.uint8)
        grid[1, 2, 3] = 1  # forward, shootable dirt
        grid[0, 2, 3] = 1  # also in beam
        grid[1, 2, 2] = 1  # river is traversable terrain
        grid[2, 3, 0] = 1  # step right blocked by wall
        # A clean beam stops at the first live-dirt target; the second dirt is beyond it.
        ids = np.zeros((5, 5), dtype=np.int32); ids[2, 2] = 1
        value = build_local_affordances(grid, names, ids, self_id=1, ready_to_shoot=True)
        self.assertEqual(value["movement_affordances"]["step_right"], "blocked")
        self.assertEqual(value["movement_affordances"]["forward"], "free")
        self.assertTrue(value["cleaning_affordance"]["visible_live_dirt_in_cleaning_beam"])
        self.assertEqual(value["cleaning_affordance"]["visible_live_dirt_targets_in_beam"], 1)


if __name__ == "__main__":
    unittest.main()
