from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grid_context import build_grid_message  # noqa: E402


class GridContextTests(unittest.TestCase):
    def test_message_is_egocentric_and_omits_world_and_future_state(self) -> None:
        names = ("wall", "sand", "grass", "river", "apple_live", "dirt_live", "avatar")
        semantic = np.zeros((3, 3, len(names)), dtype=np.uint8)
        semantic[1, 1, names.index("grass")] = 1
        semantic[0, 1, names.index("dirt_live")] = 1
        semantic[1, 2, names.index("apple_live")] = 1
        semantic[1, 1, names.index("avatar")] = 1
        semantic[2, 0, names.index("avatar")] = 1
        agent_ids = np.array([[0, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=np.int32)
        orientations = np.zeros((3, 3), dtype=np.int32)

        message = build_grid_message(
            semantic=semantic,
            channel_names=names,
            agent_ids=agent_ids,
            orientations=orientations,
            self_id=1,
            frame=7,
            previous_action="FORWARD",
            previous_reward=0.0,
            cumulative_reward=1.5,
            ready_to_shoot=True,
            valid_actions=("NOOP", "FORWARD", "FIRE_CLEAN"),
        )

        self.assertEqual(message["schema_version"], "cleanup_grid_context_v1")
        self.assertEqual(message["frame"], 7)
        self.assertEqual(message["self"], {"row": 1, "column": 1, "orientation": "NORTH"})
        self.assertEqual(message["view"], {"height": 3, "width": 3})
        self.assertEqual(message["cells"][0], {"row": 0, "column": 1, "terrain": "none", "objects": ["dirt_live"]})
        self.assertIn(
            {"row": 2, "column": 0, "terrain": "none", "objects": ["other_agent"]},
            message["cells"],
        )
        self.assertEqual(message["valid_actions"], ["NOOP", "FORWARD", "FIRE_CLEAN"])
        forbidden = {"rgb", "world_position", "world_grid", "future_reward", "teacher_action", "optimal_action"}
        self.assertFalse(forbidden.intersection(message))

    def test_hides_inactive_resources_and_spawn_points(self) -> None:
        names = ("sand", "spawn_point", "apple_inactive", "apple_live", "dirt_inactive", "dirt_live")
        semantic = np.zeros((2, 3, len(names)), dtype=np.uint8)
        semantic[0, 0, names.index("spawn_point")] = 1
        semantic[0, 1, names.index("apple_inactive")] = 1
        semantic[0, 2, names.index("dirt_inactive")] = 1
        semantic[1, 0, names.index("apple_live")] = 1
        semantic[1, 1, names.index("dirt_live")] = 1
        agent_ids = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)

        message = build_grid_message(
            semantic=semantic, channel_names=names, agent_ids=agent_ids,
            orientations=np.zeros((2, 3), dtype=np.int32), self_id=1, frame=0,
            previous_action="NOOP", previous_reward=0.0, cumulative_reward=0.0,
            ready_to_shoot=False, valid_actions=("NOOP",),
        )
        visible = {(cell["row"], cell["column"]): cell for cell in message["cells"]}
        self.assertNotIn((0, 0), visible)
        self.assertNotIn((0, 1), visible)
        self.assertNotIn((0, 2), visible)
        self.assertEqual(visible[(1, 0)]["objects"], ["apple_live", "self"])
        self.assertEqual(visible[(1, 1)]["objects"], ["dirt_live"])

    def test_message_requires_exactly_one_self_avatar(self) -> None:
        semantic = np.zeros((2, 2, 1), dtype=np.uint8)
        agent_ids = np.zeros((2, 2), dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_grid_message(
                semantic=semantic,
                channel_names=("sand",),
                agent_ids=agent_ids,
                orientations=np.zeros((2, 2), dtype=np.int32),
                self_id=1,
                frame=0,
                previous_action="NOOP",
                previous_reward=0.0,
                cumulative_reward=0.0,
                ready_to_shoot=False,
                valid_actions=("NOOP",),
            )


if __name__ == "__main__":
    unittest.main()
