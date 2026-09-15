from __future__ import annotations

import base64
import os

import requests


class CosyVoiceBackend:
    """Client for the CosyVoice zero-shot TTS server (servers/cosyvoice/server.py, port 5003)."""

    name = "cosyvoice"

    def __init__(self, endpoint: str = "http://localhost:5003") -> None:
        self._endpoint = endpoint.rstrip("/")

    def synthesize(
        self,
        text: str,
        reference_audio_path: str,
        reference_text: str | None = None,
        *,
        language: str = "en",
        style: str = "neutral",
        speed: float = 1.0,
        extra: dict | None = None,
    ) -> bytes:
        with open(reference_audio_path, "rb") as fh:
            audio_b64 = base64.b64encode(fh.read()).decode("utf-8")
        payload = {
            "tts_text": text,
            "prompt_audio_b64": audio_b64,
            "prompt_text": reference_text or "",
            "speed": speed,
        }
        response = requests.post(
            f"{self._endpoint}/synthesize",
            json=payload,
            timeout=(10, float(os.getenv("COSYVOICE_TTS_TIMEOUT", "900"))),
        )
        response.raise_for_status()
        data = response.json()
        return base64.b64decode(data["audio_b64"])

    def health(self) -> bool:
        try:
            response = requests.get(f"{self._endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
