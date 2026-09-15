from __future__ import annotations

import base64
import importlib
import io
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

LIVE_URL = "http://localhost:5003"


class _FakeTensor:
    """Minimal stand-in for the tensor returned by CosyVoice."""

    def cpu(self):
        return self

    def numpy(self):
        return self

    @property
    def T(self):
        return self


class CosyVoiceServerTests(unittest.TestCase):
    """Unit tests: exercise the Flask app with the model mocked out."""

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
            self.server = importlib.import_module("servers.cosyvoice.server")
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
        self.assertTrue(str(payload["model"]).startswith("cosyvoice:"))

    def test_synthesize_returns_audio_b64(self) -> None:
        dummy_audio = b"\x00\x01\x02"
        audio_b64 = base64.b64encode(dummy_audio).decode("ascii")

        class DummyModel:
            sample_rate = 16000

            def inference_zero_shot(self, tts_text, prompt_text, ref_path, speed=1.0):
                return [{"tts_speech": _FakeTensor()}]

        with patch("servers.cosyvoice.server.get_model", return_value=DummyModel()):
            resp = self.client.post(
                "/synthesize",
                json={
                    "tts_text": "Hello",
                    "prompt_text": "Sample",
                    "prompt_audio_b64": audio_b64,
                    "speed": 1.0,
                },
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertIn("audio_b64", payload)
        self.assertTrue(payload["audio_b64"])


def _server_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{LIVE_URL}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


@unittest.skipUnless(_server_reachable(), "CosyVoice server not running on :5003")
class CosyVoiceLiveServerTests(unittest.TestCase):
    """Live tests: hit the real server (loads the model, uses the GPU).

    Watch `nvidia-smi` inside WSL while these run to confirm GPU usage.
    First request is slow — it loads CosyVoice2-0.5B onto the GPU.
    """

    def test_live_health(self) -> None:
        with urllib.request.urlopen(f"{LIVE_URL}/health", timeout=5) as resp:
            payload = json.load(resp)
        self.assertEqual(payload["status"], "ok")

    def test_live_synthesize_returns_valid_wav(self) -> None:
        voice = REPO_ROOT / "voice.wav"
        audio_b64 = base64.b64encode(voice.read_bytes()).decode("ascii")
        body = json.dumps(
            {
                "tts_text": "Hello, this is a CosyVoice GPU test.",
                # Transcript of voice.wav — REQUIRED for zero-shot cloning;
                # without it the voice matches but the text is gibberish.
                "prompt_text": (
                    "I'm really struggling to stay calm right now "
                    "because what you did was totally out of line."
                ),
                "prompt_audio_b64": audio_b64,
                "speed": 1.0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{LIVE_URL}/synthesize",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.load(resp)

        self.assertIn("audio_b64", payload)
        wav_bytes = base64.b64decode(payload["audio_b64"])
        # Valid WAV starts with the RIFF magic bytes.
        self.assertEqual(wav_bytes[:4], b"RIFF")
        self.assertGreater(len(wav_bytes), 10_000)
        # Save so the result can be listened to after the test run.
        (REPO_ROOT / "synth_test.wav").write_bytes(wav_bytes)


if __name__ == "__main__":
    unittest.main()
