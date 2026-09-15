from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIVE_URL = "http://localhost:5002"


class _FakeTensor:
    """Minimal stand-in for the tensor returned by ChatterboxTTS."""

    def squeeze(self, dim):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self


class ChatterboxServerTests(unittest.TestCase):
    def setUp(self) -> None:
        # server.py imports soundfile lazily inside synthesize(), so swap in
        # a fake module unconditionally (works whether or not the real
        # soundfile is installed in this environment).
        self._orig_soundfile = sys.modules.get("soundfile")
        fake = types.ModuleType("soundfile")

        def write(file_obj, data, samplerate, format="WAV"):
            file_obj.write(b"RIFFTEST")

        fake.write = write
        sys.modules["soundfile"] = fake

        try:
            self.server = importlib.import_module("servers.chatterbox.server")
        except (ImportError, ModuleNotFoundError) as exc:
            # tearDown won't run on SkipTest, so restore soundfile now.
            if self._orig_soundfile is not None:
                sys.modules["soundfile"] = self._orig_soundfile
            else:
                sys.modules.pop("soundfile", None)
            raise unittest.SkipTest(f"server deps not installed: {exc}") from None
        self.client = self.server.app.test_client()

    def tearDown(self) -> None:
        if self._orig_soundfile is not None:
            sys.modules["soundfile"] = self._orig_soundfile
        else:
            sys.modules.pop("soundfile", None)

    def test_health_endpoint_reports_ok(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model"], "chatterbox")

    def test_synthesize_returns_wav_response(self) -> None:
        class DummyModel:
            sr = 16000

            def generate(self, text, audio_prompt_path=None, exaggeration=0.5,
                         cfg_weight=0.5, temperature=0.8):
                return _FakeTensor()

        with patch("servers.chatterbox.server.get_model", return_value=DummyModel()):
            resp = self.client.post(
                "/synthesize",
                data={
                    "text": "Hello",
                    "exaggeration": "0.5",
                    "cfg_weight": "0.5",
                    "temperature": "0.8",
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


@unittest.skipUnless(_server_reachable(), "Chatterbox server not running on :5002")
class ChatterboxLiveServerTests(unittest.TestCase):
    """Live tests: hit the real server (loads the model, uses the GPU).

    Watch `nvidia-smi` inside WSL while these run to confirm GPU usage.
    First request is slow — it downloads/loads the Chatterbox model.
    """

    def test_live_health(self) -> None:
        with urllib.request.urlopen(f"{LIVE_URL}/health", timeout=5) as resp:
            payload = json.load(resp)
        self.assertEqual(payload["status"], "ok")

    def test_live_synthesize_voice_clone(self) -> None:
        voice_bytes = (REPO_ROOT / "voice.wav").read_bytes()
        boundary = "----testboundary"
        parts = []
        for name, value in {
            "text": "Hello, this is a Chatterbox GPU test.",
            "exaggeration": "0.5",
            "cfg_weight": "0.5",
            "temperature": "0.8",
        }.items():
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            )
        body = "".join(parts).encode()
        body += (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="audio_prompt"; filename="voice.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode() + voice_bytes + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{LIVE_URL}/synthesize",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            wav_bytes = resp.read()

        self.assertEqual(wav_bytes[:4], b"RIFF")
        self.assertGreater(len(wav_bytes), 10_000)
        (REPO_ROOT / "chatterbox_test.wav").write_bytes(wav_bytes)


if __name__ == "__main__":
    unittest.main()
