from __future__ import annotations

import argparse
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

BRIDGE_PATH = Path(__file__).parents[1] / "bridge" / "fastvideo_mlx_bridge.py"
VIEWS_PATH = Path(__file__).parents[1] / "Sources" / "FastVideoMac" / "Views.swift"
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
            self.assertNotIn("--fast", command)

    def test_generation_command_enables_fast_mode_with_first_party_rife_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entrypoint = root / "examples" / "inference" / "basic" / "mlx_wan_prompt_to_video.py"
            entrypoint.parent.mkdir(parents=True)
            entrypoint.write_text("# test")
            model_root = root / "model"
            rife_weights = model_root / "rife" / "RIFE-4.25"
            rife_weights.mkdir(parents=True)
            request = {
                "repo_root": str(root),
                "prompt": "A fast paper boat.",
                "model_root": str(model_root),
                "checkpoint_path": str(root / "checkpoint"),
                "output_path": str(root / "video.mp4"),
                "fast": True,
            }
            command = BRIDGE._generation_command(request, root / "request.json")
            self.assertIn("--fast", command)
            self.assertEqual(
                command[command.index("--fast-rife-weights-dir") + 1],
                str(rife_weights),
            )

    def test_checkpoint_validation_matches_native_mlx_checkpoint_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "mlx_dit.json").write_text("{}")
            (checkpoint / "mlx_dit.safetensors").write_bytes(b"test")
            self.assertTrue(BRIDGE.checkpoint_is_valid(checkpoint))
            (checkpoint / "mlx_dit.json").unlink()
            self.assertFalse(BRIDGE.checkpoint_is_valid(checkpoint))

    def test_ffmpeg_falls_back_to_imageio_binary(self) -> None:
        executable = BRIDGE.resolve_ffmpeg()
        if executable is not None:
            self.assertTrue(Path(executable).is_file())

    def test_video_surface_avoids_swiftui_videoplayer_on_macos_27(self) -> None:
        source = VIEWS_PATH.read_text()
        self.assertIn("private struct VideoSurface: NSViewRepresentable", source)
        self.assertIn("AVPlayerView()", source)
        self.assertNotIn("VideoPlayer(", source)

    def test_first_party_release_installs_shared_and_ema_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared_payload = root / "shared-payload"
            (shared_payload / "tokenizer").mkdir(parents=True)
            (shared_payload / "text_encoder").mkdir()
            (shared_payload / "transformer").mkdir()
            (shared_payload / "transformer" / "config.json").write_text("{}")
            ema_payload = root / "ema-payload"
            ema_payload.mkdir()
            (ema_payload / "mlx_dit.json").write_text("{}")
            (ema_payload / "mlx_dit.safetensors").write_bytes(b"weights")
            rife_payload = root / "rife-payload"
            rife_payload.mkdir()
            (rife_payload / "config.json").write_text("{}")
            (rife_payload / "model.safetensors").write_bytes(b"rife weights")

            shared_archive = root / "shared.tar.gz"
            ema_archive = root / "ema.tar.gz"
            rife_archive = root / "rife.tar.gz"
            with tarfile.open(shared_archive, "w:gz") as archive:
                archive.add(shared_payload, arcname="shared")
            with tarfile.open(ema_archive, "w:gz") as archive:
                archive.add(ema_payload, arcname="ema")
            with tarfile.open(rife_archive, "w:gz") as archive:
                archive.add(rife_payload, arcname="rife")

            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps({
                    "shared": {
                        "url": shared_archive.as_uri(),
                        "sha256": "",
                        "bytes": 0
                    },
                    "fast_mode": {
                        "url": rife_archive.as_uri(),
                        "sha256": "",
                        "bytes": 0
                    },
                    "variants": {
                        "ema": {
                            "url": ema_archive.as_uri(),
                            "sha256": "",
                            "bytes": 0
                        },
                        "raw": {
                            "url": ema_archive.as_uri(),
                            "sha256": "",
                            "bytes": 0
                        },
                    },
                }))
            model_root = root / "installed-shared"
            checkpoint_root = root / "installed-ema"
            result = BRIDGE.command_install_release(
                argparse.Namespace(
                    catalog=str(catalog),
                    variant="ema",
                    model_root=str(model_root),
                    checkpoint_root=str(checkpoint_root),
                ))
            self.assertEqual(result, 0)
            self.assertTrue(BRIDGE.model_components_present(model_root))
            self.assertTrue(BRIDGE.rife_weights_present(model_root))
            self.assertTrue(BRIDGE.checkpoint_is_valid(checkpoint_root))

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
