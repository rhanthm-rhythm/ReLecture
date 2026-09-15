from __future__ import annotations

import os

import requests


class ChatterboxBackend:
    """Client for the Chatterbox TTS server (servers/chatterbox/server.py, port 5002)."""

    name = "chatterbox"

    def __init__(self, endpoint: str = "http://localhost:5002") -> None:
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
        extra = extra or {}
        payload = {
            "text": text,
            "exaggeration": extra.get("exaggeration", 0.5),
            "cfg_weight": extra.get("cfg_weight", 0.5),
            "temperature": extra.get("temperature", 0.8),
        }
        with open(reference_audio_path, "rb") as fh:
            response = requests.post(
                f"{self._endpoint}/synthesize",
                data=payload,
                files={"audio_prompt": fh},
                timeout=(10, float(os.getenv("CHATTERBOX_TTS_TIMEOUT", "900"))),
            )
        response.raise_for_status()
        return response.content

    def health(self) -> bool:
        try:
            response = requests.get(f"{self._endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
