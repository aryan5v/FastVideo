from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

BRIDGE_PATH = Path(__file__).parents[1] / "bridge" / "fastvideo_mlx_bridge.py"
SPEC = importlib.util.spec_from_file_location("fastvideo_mlx_bridge", BRIDGE_PATH)
assert SPEC is not None and SPEC.loader is not None
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


class BridgeTest(unittest.TestCase):

    def test_resolve_checkpoint_prefers_variant_specific_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "mlx_dit_raw"
            checkpoint.mkdir()
            self.assertEqual(BRIDGE.resolve_checkpoint(root, "raw"), checkpoint)
            self.assertIsNone(BRIDGE.resolve_checkpoint(root, "ema"))

    def test_generation_command_enables_atomic_live_previews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entrypoint = root / "examples" / "inference" / "basic" / "mlx_wan_prompt_to_video.py"
            entrypoint.parent.mkdir(parents=True)
            entrypoint.write_text("# test")
            output = root / "output" / "video.mp4"
            request = {
                "repo_root": str(root),
                "prompt": "A paper boat.",
                "model_root": str(root / "model"),
                "checkpoint_path": str(root / "model" / "mlx_dit_raw"),
                "output_path": str(output),
            }
            command = BRIDGE._generation_command(request, root / "request.json")
            self.assertIn("--preview-dir", command)
            preview_index = command.index("--preview-dir") + 1
            self.assertEqual(command[preview_index], str(output.parent / "previews"))
            self.assertEqual(command[command.index("--preview-every") + 1], "1")

    def test_preview_event_shape_is_json_line_safe(self) -> None:
        event = {
            "type": "preview",
            "preview_path": "/tmp/preview-step-1.mp4",
            "current": 1,
            "total": 3,
            "fraction": 0.37,
        }
        encoded = json.dumps(event)
        self.assertNotIn("\n", encoded)
        self.assertEqual(json.loads(encoded)["preview_path"], event["preview_path"])


if __name__ == "__main__":
    unittest.main()
