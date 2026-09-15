from __future__ import annotations

import io
import importlib
import json
import sys
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIVE_URL = "http://localhost:5001"


class WhisperServerTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.server = importlib.import_module("servers.whisper.server")
        except (ImportError, ModuleNotFoundError) as exc:
            raise unittest.SkipTest(f"server deps not installed: {exc}") from None
        self.client = self.server.app.test_client()

    def test_health_endpoint_reports_ok(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(str(payload["model"]).startswith("whisper:"))

    def test_transcribe_returns_text(self) -> None:
        class DummyModel:
            def transcribe(self, path, language="en"):
                return {"text": "hello world"}

        with patch("servers.whisper.server.get_model", return_value=DummyModel()):
            data = {
                "file": (io.BytesIO(b"dummy"), "audio.wav"),
            }
            resp = self.client.post("/transcribe", data=data, content_type="multipart/form-data")

        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload["transcription"], "hello world")


def _server_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{LIVE_URL}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


@unittest.skipUnless(_server_reachable(), "Whisper server not running on :5001")
class WhisperLiveServerTests(unittest.TestCase):
    """Live tests: hit the real server (loads the model, uses the GPU).

    Watch `nvidia-smi` inside WSL while these run to confirm GPU usage.
    First request is slow — it downloads/loads the whisper model.
    """

    def test_live_health(self) -> None:
        with urllib.request.urlopen(f"{LIVE_URL}/health", timeout=5) as resp:
            payload = json.load(resp)
        self.assertEqual(payload["status"], "ok")

    def test_live_transcribe_voice_wav(self) -> None:
        voice_bytes = (REPO_ROOT / "voice.wav").read_bytes()
        boundary = "----testboundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="voice.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode() + voice_bytes + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{LIVE_URL}/transcribe",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.load(resp)

        transcription = payload.get("transcription", "").lower()
        self.assertIn("struggling", transcription)
        self.assertIn("out of line", transcription)


if __name__ == "__main__":
    unittest.main()
