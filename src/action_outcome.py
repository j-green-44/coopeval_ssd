"""Policy-visible outcome summaries derived from a completed Cleanup action."""
from __future__ import annotations


MOVEMENT_ACTIONS = frozenset({"FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT"})


def build_action_outcome(action: str, *, moved: bool, local_view_changed: bool, dirt_before: int, dirt_after: int) -> dict[str, object]:
    """Summarise past local action effects without exposing world coordinates."""
    action = action.upper()
    movement_outcome = "moved" if moved else "blocked" if action in MOVEMENT_ACTIONS else "not_applicable"
    cleaning_effect = "not_applicable"
    if action == "FIRE_CLEAN":
        if dirt_after < dirt_before:
            cleaning_effect = "one_or_more_visible_targets_removed"
        elif dirt_after > dirt_before:
            cleaning_effect = "more_visible_dirt_than_before"
        else:
            cleaning_effect = "no_visible_target_change"
    return {
        "movement_outcome": movement_outcome,
        "local_view_changed": bool(local_view_changed),
        "visible_live_dirt_before": int(dirt_before),
        "visible_live_dirt_after": int(dirt_after),
        "cleaning_effect": cleaning_effect,
    }
