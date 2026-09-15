from __future__ import annotations

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

LIVE_URL = "http://localhost:5000"
# Transcript of voice.wav (see test_cosyvoice_server.py)
VOICE_TRANSCRIPT = (
    "I'm really struggling to stay calm right now "
    "because what you did was totally out of line."
)


class Qwen3ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.server = importlib.import_module("servers.qwen3.server")
        except (ImportError, ModuleNotFoundError) as exc:
            raise unittest.SkipTest(f"server deps not installed: {exc}") from None
        self.client = self.server.app.test_client()

    def test_health_endpoint_reports_ok(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(str(payload["model"]).startswith("qwen3:"))

    def test_generate_uses_model_and_returns_wav(self) -> None:
        class DummyModel:
            def synthesize(self, text, ref_audio=None, ref_text="", speed=1.0,
                           temperature=0.9, language="en"):
                return b"dummy"

        with patch("servers.qwen3.server.get_model", return_value=DummyModel()):
            resp = self.client.post(
                "/generate",
                data={
                    "text": "Hello",
                    "style": "neutral",
                    "language": "en",
                    "speed_ref": "1.0",
                    "temperature": "0.9",
                    "ref_text": "reference",
                },
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "audio/wav")


def _server_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{LIVE_URL}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


@unittest.skipUnless(_server_reachable(), "Qwen3 server not running on :5000")
class Qwen3LiveServerTests(unittest.TestCase):
    """Live tests: hit the real server (loads the model, uses the GPU).

    Watch `nvidia-smi` inside WSL while these run to confirm GPU usage.
    First request is slow — it downloads/loads Qwen3-TTS-12Hz-1.7B-Base.
    """

    def test_live_health(self) -> None:
        with urllib.request.urlopen(f"{LIVE_URL}/health", timeout=5) as resp:
            payload = json.load(resp)
        self.assertEqual(payload["status"], "ok")

    def test_live_generate_voice_clone(self) -> None:
        voice_bytes = (REPO_ROOT / "voice.wav").read_bytes()
        boundary = "----testboundary"
        parts = []
        for name, value in {
            "text": "Hello, this is a Qwen3 TTS GPU test.",
            "language": "en",
            "ref_text": VOICE_TRANSCRIPT,
            "temperature": "0.9",
            "speed_ref": "1.0",
        }.items():
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            )
        body = "".join(parts).encode()
        body += (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="reference_audio"; filename="voice.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode() + voice_bytes + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{LIVE_URL}/generate",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=900) as resp:
            wav_bytes = resp.read()

        self.assertEqual(wav_bytes[:4], b"RIFF")
        self.assertGreater(len(wav_bytes), 10_000)
        (REPO_ROOT / "qwen3_test.wav").write_bytes(wav_bytes)


if __name__ == "__main__":
    unittest.main()
