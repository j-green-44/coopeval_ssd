from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_pi_two_agents import json_safe, normalise_event_item


class EventNormalisationTests(unittest.TestCase):
    def test_decodes_lab2d_compact_dict_event(self) -> None:
        raw = [b"dict", b"player_index", 3.0]
        self.assertEqual(normalise_event_item(raw), {"player_index": 3.0})

    def test_json_safe_encodes_unknown_bytes(self) -> None:
        self.assertEqual(json_safe({"raw": b"abc"}), {"raw": "616263"})


if __name__ == "__main__":
    unittest.main()
