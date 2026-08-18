"""Bounded, directed, one-step-delayed cheap-talk message handling."""
from __future__ import annotations

from typing import Any

MAX_MESSAGE_CHARS = 160


def parse_directed_messages(value: Any, sender: int, agent_count: int) -> dict[str, list[dict[str, Any]]]:
    """Validate at most one bounded text message for each other agent."""
    if value is None:
        return {"messages": [], "dropped": []}
    if not isinstance(value, list):
        return {"messages": [], "dropped": [{"reason": "messages must be a list"}]}

    messages: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    recipients: set[int] = set()
    for item in value:
        if not isinstance(item, dict):
            dropped.append({"reason": "message must be an object"})
            continue
        try:
            recipient = int(item["to"])
            text = str(item["text"])
        except Exception:
            dropped.append({"reason": "message requires to and text"})
            continue
        if recipient == sender:
            dropped.append({"to": recipient, "reason": "self messages are not allowed"})
        elif recipient < 0 or recipient >= agent_count:
            dropped.append({"to": recipient, "reason": "recipient is out of range"})
        elif recipient in recipients:
            dropped.append({"to": recipient, "reason": "one message per recipient per step"})
        elif not text or len(text) > MAX_MESSAGE_CHARS:
            dropped.append({"to": recipient, "reason": f"text must contain 1..{MAX_MESSAGE_CHARS} characters"})
        else:
            recipients.add(recipient)
            messages.append({"to": recipient, "text": text})
    return {"messages": messages, "dropped": dropped}


def deliver_messages(sender: int, step: int, messages: list[dict[str, Any]], next_inbox: list[list[dict[str, Any]]]) -> None:
    """Queue validated messages for recipient visibility on the next simulator turn."""
    for message in messages:
        next_inbox[int(message["to"])].append({"from": sender, "sent_at_step": step, "text": message["text"]})
