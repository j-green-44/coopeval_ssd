"""Pure data handling for opt-in, role-level Cleanup mediation."""
from __future__ import annotations

import json
from typing import Any

CHOICES = {"JOIN", "CONTINUE", "LEAVE"}
ROLES = {"CLEAN", "HARVEST", "FLEX"}


def parse_mediation_choice(raw: str, was_enrolled: bool, is_initial_epoch: bool) -> dict[str, Any]:
    """Parse one voluntary mediation choice without changing enrolment on error."""
    try:
        value = json.loads(raw)
        choice = str(value["mediation_choice"]).upper()
    except Exception as error:
        return {"valid": False, "choice": None, "enrolled": was_enrolled, "error": f"invalid mediation choice: {error}"}

    allowed = {"JOIN", "LEAVE"} if (is_initial_epoch or not was_enrolled) else {"CONTINUE", "LEAVE"}
    if choice not in CHOICES or choice not in allowed:
        return {
            "valid": False,
            "choice": choice,
            "enrolled": was_enrolled,
            "error": f"choice {choice!r} is invalid; allowed choices are {sorted(allowed)}",
        }
    return {"valid": True, "choice": choice, "enrolled": choice != "LEAVE", "error": None}


def validate_mediator_plan(raw: str, participants: set[int], interval_steps: int) -> dict[str, Any]:
    """Validate a bounded assignment plan; invalid plans must be safely abstained."""
    if len(participants) < 2:
        return {"valid": False, "assignments": [], "error": "at least two participants are required for mediation"}
    try:
        value = json.loads(raw)
        valid_for_steps = int(value["valid_for_steps"])
        assignments = value["assignments"]
    except Exception as error:
        return {"valid": False, "assignments": [], "error": f"invalid mediator JSON: {error}"}

    if valid_for_steps != interval_steps:
        return {"valid": False, "assignments": [], "error": f"valid_for_steps must be {interval_steps}"}
    if not isinstance(assignments, list):
        return {"valid": False, "assignments": [], "error": "assignments must be a list"}

    normalised: list[dict[str, Any]] = []
    assigned_agents: list[int] = []
    for item in assignments:
        if not isinstance(item, dict):
            return {"valid": False, "assignments": [], "error": "each assignment must be an object"}
        try:
            agent_index = int(item["agent_index"])
            role = str(item["role"]).upper()
            objective = str(item["objective"])
        except Exception as error:
            return {"valid": False, "assignments": [], "error": f"invalid assignment: {error}"}
        if agent_index not in participants:
            return {"valid": False, "assignments": [], "error": "assignments must target current participants only"}
        if role not in ROLES:
            return {"valid": False, "assignments": [], "error": f"unsupported role: {role}"}
        normalised.append(
            {
                "agent_index": agent_index,
                "role": role,
                "objective": objective[:300],
                "fairness_basis": str(item.get("fairness_basis", ""))[:300],
            }
        )
        assigned_agents.append(agent_index)

    if set(assigned_agents) != participants or len(assigned_agents) != len(set(assigned_agents)):
        return {"valid": False, "assignments": [], "error": "assignments must contain each current participant exactly once"}

    roles = {item["role"] for item in normalised}
    if len(participants) == 2 and roles != {"CLEAN", "HARVEST"}:
        return {"valid": False, "assignments": [], "error": "two participants require exactly CLEAN and HARVEST assignments"}
    if len(participants) >= 3:
        if not {"CLEAN", "HARVEST"}.issubset(roles) or any(item["role"] not in {"CLEAN", "HARVEST", "FLEX"} for item in normalised):
            return {"valid": False, "assignments": [], "error": "three or more participants require CLEAN, HARVEST and FLEX-only remaining assignments"}
        if sum(item["role"] == "CLEAN" for item in normalised) != 1 or sum(item["role"] == "HARVEST" for item in normalised) != 1:
            return {"valid": False, "assignments": [], "error": "three or more participants require exactly one CLEAN and one HARVEST assignment"}
        if len(participants) > 2 and sum(item["role"] == "FLEX" for item in normalised) != len(participants) - 2:
            return {"valid": False, "assignments": [], "error": "all remaining participants must receive FLEX assignments"}
    return {"valid": True, "assignments": sorted(normalised, key=lambda item: item["agent_index"]), "error": None}


def build_two_cleaner_rotation_plan(
    participants: list[int], ledger: list[dict[str, Any]], interval_steps: int
) -> dict[str, Any]:
    """Allocate one harvest opportunity and two clean roles by causal round-robin."""
    ordered_participants = sorted(set(participants))
    if len(ordered_participants) != 3 or len(ordered_participants) != len(participants):
        return {
            "valid": False,
            "valid_for_steps": interval_steps,
            "assignments": [],
            "error": "two-cleaner rotation requires exactly three current participants",
        }

    harvest_counts = {index: 0 for index in ordered_participants}
    for epoch in ledger:
        for assignment in epoch.get("assignments", []):
            index = int(assignment.get("agent_index", -1))
            if index in harvest_counts and assignment.get("role") == "HARVEST":
                harvest_counts[index] += 1
    harvester = min(ordered_participants, key=lambda index: (harvest_counts[index], index))
    assignments = [
        {
            "agent_index": index,
            "role": "HARVEST" if index == harvester else "CLEAN",
            "objective": (
                "Prioritize locally observable apples while they are available."
                if index == harvester
                else "Prioritize locally visible pollution removal to restore the shared river."
            ),
            "fairness_basis": (
                f"Deterministic rotation: agent {harvester} has the fewest prior HARVEST intervals "
                f"({harvest_counts[harvester]}), with agent index as the tie-breaker."
            ),
        }
        for index in ordered_participants
    ]
    return {"valid": True, "valid_for_steps": interval_steps, "assignments": assignments, "error": None}


def close_epoch(epoch: dict[str, Any], end_step: int, outcomes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Close a completed mediated interval and preserve role/outcome attribution."""
    assignments = epoch.get("assignments", [])
    realised_outcomes: list[dict[str, Any]] = []
    for assignment in assignments:
        agent_index = int(assignment["agent_index"])
        outcome = outcomes.get(agent_index, {})
        realised_outcomes.append(
            {
                "agent_index": agent_index,
                "assigned_role": assignment["role"],
                "reward": float(outcome.get("reward", 0.0)),
                "confirmed_clean_removals": int(outcome.get("confirmed_clean_removals", 0)),
            }
        )
    return {
        "epoch": int(epoch["epoch"]),
        "start_step": int(epoch["start_step"]),
        "end_step": int(end_step),
        "participants": list(epoch.get("participants", [])),
        "assignments": [{"agent_index": item["agent_index"], "role": item["role"]} for item in assignments],
        "realised_outcomes": realised_outcomes,
    }


def participant_fairness_summary(ledger: list[dict[str, Any]], participants: list[int]) -> list[dict[str, Any]]:
    """Summarise all completed mediated assignments for current participants."""
    summaries: dict[int, dict[str, Any]] = {
        index: {
            "agent_index": index,
            "mediated_intervals": 0,
            "clean_intervals": 0,
            "harvest_intervals": 0,
            "flex_intervals": 0,
            "reward_while_mediated": 0.0,
            "confirmed_clean_removals_while_mediated": 0,
            "last_assigned_role": None,
        }
        for index in participants
    }
    for epoch in ledger:
        outcomes = {int(item["agent_index"]): item for item in epoch.get("realised_outcomes", [])}
        for assignment in epoch.get("assignments", []):
            index = int(assignment["agent_index"])
            if index not in summaries:
                continue
            role = assignment["role"]
            summary = summaries[index]
            summary["mediated_intervals"] += 1
            summary[f"{role.lower()}_intervals"] += 1
            summary["last_assigned_role"] = role
            outcome = outcomes.get(index, {})
            summary["reward_while_mediated"] += float(outcome.get("reward", 0.0))
            summary["confirmed_clean_removals_while_mediated"] += int(outcome.get("confirmed_clean_removals", 0))
    return [summaries[index] for index in sorted(summaries)]
