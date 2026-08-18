from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class CommonsHarvestTests(unittest.TestCase):
    def test_prompt_explains_regrowth_restraint_and_controller_schema(self) -> None:
        from run_commons_harvest import build_commons_harvest_prompt

        prompt = build_commons_harvest_prompt(
            visual_summary={"apple_like_cells": 3, "grass_like_cells": 8, "wall_like_cells": 1},
            history=[],
            ready_to_shoot=True,
            valid_actions=["NOOP", "FORWARD", "FIRE_ZAP"],
        )

        self.assertIn("regrowth", prompt.lower())
        self.assertIn("restraint", prompt.lower())
        self.assertIn("local", prompt.lower())
        self.assertIn('{"action":"ONE_VALID_ACTION"}', prompt)
        self.assertNotIn("WORLD.RGB", prompt)
        self.assertNotIn("global coordinates", prompt.lower())

    def test_local_rgb_summary_counts_only_local_visual_evidence(self) -> None:
        import numpy as np
        from run_commons_harvest import summarise_local_rgb

        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[0:8, 0:8] = (214, 88, 88)  # Apple palette.
        rgb[0:8, 8:16] = (44, 160, 44)  # Grass-like patch.
        rgb[8:16, 0:8] = (80, 80, 80)  # Wall-like patch.

        summary = summarise_local_rgb(rgb, tile_size=8)

        self.assertEqual(summary["view_tiles"], {"height": 2, "width": 2})
        self.assertEqual(summary["apple_like_cells"], 1)
        self.assertEqual(summary["grass_like_cells"], 1)
        self.assertEqual(summary["wall_like_cells"], 1)
        self.assertNotIn("world", str(summary).lower())

    def test_supervisor_strategy_mode_uses_approved_commons_harvest_prompt(self) -> None:
        from run_commons_harvest import build_commons_harvest_prompt

        prompt = build_commons_harvest_prompt(
            visual_summary={"apple_like_cells": 4, "grass_like_cells": 8, "wall_like_cells": 2},
            history=[{"step": 0, "action": "FORWARD", "reward": 0.0}],
            ready_to_shoot=True,
            valid_actions=["FORWARD", "TURN_LEFT", "FIRE_ZAP"],
            policy_mode="supervisor_strategy",
        )

        self.assertIn("Maximise your own cumulative individual reward", prompt)
        self.assertIn("Apples regrow only when enough nearby apples remain", prompt)
        self.assertIn("When local apple evidence is sparse or falling", prompt)
        self.assertIn("LOCAL_RGB_SUMMARY_JSON", prompt)
        self.assertNotIn("WORLD.RGB", prompt)
        self.assertIn('{"action":"ONE_VALID_ACTION"}', prompt)

    def test_action_mapping_resolves_all_eight_commons_harvest_actions(self) -> None:
        from commons_harvest import action_mapping_from_action_set

        action_set = [
            {"move": 0, "turn": 0, "fireZap": 0},
            {"move": 1, "turn": 0, "fireZap": 0},
            {"move": 3, "turn": 0, "fireZap": 0},
            {"move": 4, "turn": 0, "fireZap": 0},
            {"move": 2, "turn": 0, "fireZap": 0},
            {"move": 0, "turn": -1, "fireZap": 0},
            {"move": 0, "turn": 1, "fireZap": 0},
            {"move": 0, "turn": 0, "fireZap": 1},
        ]

        mapping = action_mapping_from_action_set(action_set)

        self.assertEqual(set(mapping.name_to_index), {"NOOP", "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", "TURN_LEFT", "TURN_RIGHT", "FIRE_ZAP"})
        self.assertEqual(mapping.index("FIRE_ZAP"), 7)


if __name__ == "__main__":
    unittest.main()
