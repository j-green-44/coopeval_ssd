"""Compact policy-visible movement and cleaning affordances from a local grid."""
from __future__ import annotations

from typing import Sequence
import numpy as np


def build_local_affordances(semantic: np.ndarray, channel_names: Sequence[str], agent_ids: np.ndarray, *, self_id: int, ready_to_shoot: bool) -> dict[str, object]:
    row, col = (int(v) for v in np.argwhere(agent_ids == self_id)[0])
    blocked = {"wall"}
    name_to_index = {name: i for i, name in enumerate(channel_names)}

    def cell_blocked(r: int, c: int) -> bool:
        if not (0 <= r < semantic.shape[0] and 0 <= c < semantic.shape[1]): return True
        return any(semantic[r, c, name_to_index[name]] for name in blocked if name in name_to_index) or bool(agent_ids[r, c] and agent_ids[r, c] != self_id)

    offsets = {"forward": (-1, 0), "backward": (1, 0), "step_left": (0, -1), "step_right": (0, 1)}
    moves = {name: "blocked" if cell_blocked(row + dy, col + dx) else "free" for name, (dy, dx) in offsets.items()}
    dirt_index = name_to_index.get("dirt_live")
    targets = 0
    if dirt_index is not None:
        for r in range(row - 1, -1, -1):
            if cell_blocked(r, col): break
            if semantic[r, col, dirt_index]:
                targets += 1
                break
    return {"movement_affordances": moves, "cleaning_affordance": {"ready_to_shoot": bool(ready_to_shoot), "visible_live_dirt_in_cleaning_beam": targets > 0, "visible_live_dirt_targets_in_beam": targets}}
