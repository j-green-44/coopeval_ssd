"""Text-only policy prompt for the explicit symbolic upper-bound experiment."""
from __future__ import annotations

import json
from typing import Any


def build_request_text(instructions: str, grid_message: dict[str, Any]) -> str:
    """Build the entire text policy input without image or world-state references."""
    grid_json = json.dumps(grid_message, sort_keys=True)
    return (
        f"INSTRUCTIONS:\n{instructions.strip()}\n\n"
        "You receive only the current egocentric symbolic grid below. "
        "Do not assume unseen cells, hidden state, or future rewards.\n"
        f"GRID_CONTEXT_JSON:\n{grid_json}\n\n"
        "Choose exactly one action from valid_actions. Return JSON only: "
        '{"action":"...","public_message":"","intent":""}. '
        "Keep public_message and intent short."
    )
