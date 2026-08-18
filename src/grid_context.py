"""Policy-visible Cleanup semantic-grid serialisation.

This module deliberately turns an already egocentric local simulator grid into a
small JSON-safe message. It does not accept world coordinates, RGB, future
state, or teacher actions.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


TERRAIN_NAMES = frozenset({"wall", "sand", "grass", "river"})
# These semantic channels describe latent map/resource state rather than an
# actionable visible target. Keep them out of the policy interface.
HIDDEN_SEMANTIC_NAMES = frozenset({"spawn_point", "apple_inactive", "dirt_inactive"})


def build_grid_message(
    *,
    semantic: np.ndarray,
    channel_names: Sequence[str],
    agent_ids: np.ndarray,
    orientations: np.ndarray,
    self_id: int,
    frame: int,
    previous_action: str,
    previous_reward: float,
    cumulative_reward: float,
    valid_actions: Sequence[str],
    previous_outcome: dict[str, object] | None = None,
    local_affordances: dict[str, object] | None = None,
    ready_to_shoot: bool | None = None,
    orientation_codes: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe, local-only context message for one agent.

    ``semantic``, ``agent_ids``, and ``orientations`` must already be expressed
    in the controlled agent's egocentric view. The only identity exposed is
    whether an avatar is self or another visible agent.
    """
    if semantic.ndim != 3:
        raise ValueError("semantic must have shape [rows, columns, channels]")
    rows, columns, channels = semantic.shape
    if len(channel_names) != channels:
        raise ValueError("channel_names must match semantic channel count")
    if agent_ids.shape != (rows, columns) or orientations.shape != (rows, columns):
        raise ValueError("agent grids must match semantic height and width")

    self_cells = np.argwhere(agent_ids == self_id)
    if len(self_cells) != 1:
        raise ValueError("local agent ID grid must contain exactly one self avatar")
    self_row, self_column = (int(value) for value in self_cells[0])
    orientation_code = int(orientations[self_row, self_column])
    orientation = (orientation_codes or {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}).get(
        orientation_code, "UNKNOWN"
    )

    cells: list[dict[str, Any]] = []
    for row in range(rows):
        for column in range(columns):
            active = [channel_names[index] for index in range(channels) if bool(semantic[row, column, index]) and channel_names[index] not in HIDDEN_SEMANTIC_NAMES]
            terrain = next((name for name in active if name in TERRAIN_NAMES), "none")
            objects = [name for name in active if name not in TERRAIN_NAMES and name != "avatar"]
            if bool(agent_ids[row, column]):
                objects.append("self" if int(agent_ids[row, column]) == self_id else "other_agent")
            if terrain != "none" or objects:
                cells.append({"row": row, "column": column, "terrain": terrain, "objects": objects})

    return {
        "schema_version": "cleanup_grid_context_v1",
        "frame": int(frame),
        "self": {"row": self_row, "column": self_column, "orientation": orientation},
        "view": {"height": rows, "width": columns},
        "cells": cells,
        "local_affordances": local_affordances or {},
        "agent_state": {
            "previous_action": previous_action,
            "previous_reward": float(previous_reward),
            "cumulative_reward": float(cumulative_reward),
            "previous_outcome": previous_outcome or {},
            "ready_to_shoot": ready_to_shoot,
        },
        "valid_actions": list(valid_actions),
    }
