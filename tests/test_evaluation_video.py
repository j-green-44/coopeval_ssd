from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), "/home/jack/phd/tiny_cooperative_vlm/src"]

from run_grid_context import save_evaluation_video  # noqa: E402


class EvaluationVideoTests(unittest.TestCase):
    def test_save_evaluation_video_writes_lossless_mkv(self) -> None:
        frames = [
            np.zeros((8, 12, 3), dtype=np.uint8),
            np.full((8, 12, 3), 255, dtype=np.uint8),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episode_000.lossless.mkv"
            save_evaluation_video(frames, output, fps=10)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            metadata = json.loads(
                __import__("subprocess").check_output(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,nb_frames", "-of", "json", str(output)],
                    text=True,
                )
            )
            stream = metadata["streams"][0]
            self.assertEqual(stream["codec_name"], "ffv1")
            self.assertEqual((stream["width"], stream["height"]), (12, 8))


if __name__ == "__main__":
    unittest.main()
