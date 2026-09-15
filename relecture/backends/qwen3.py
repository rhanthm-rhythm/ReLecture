from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

# Qwen3-TTS-1.7B can take a long time per chunk (model load + long text),
# so the read timeout is generous and configurable.
_TIMEOUT = float(os.getenv("QWEN3_TTS_TIMEOUT", "1800"))


class Qwen3Backend:
    """Client for the Qwen3-based TTS server (servers/qwen3/server.py, port 5000)."""

    name = "qwen3"

    def __init__(self, endpoint: str = "http://localhost:5000") -> None:
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
            "style": style,
            "language": language,
            "model": "qwen3",
            "speed_ref": speed,
            "temperature": extra.get("temperature", 0.9),
            "top_p": extra.get("top_p", 1.0),
            "repetition_penalty": extra.get("repetition_penalty", 1.05),
        }
        if reference_text:
            payload["ref_text"] = reference_text
        logger.info("POST %s/generate — %d chars of text, sending request…", self._endpoint, len(text))
        started = time.perf_counter()
        with open(reference_audio_path, "rb") as fh:
            response = requests.post(
                f"{self._endpoint}/generate",
                data=payload,
                files={"reference_audio": fh},
                timeout=(10, _TIMEOUT),
            )
        elapsed = time.perf_counter() - started
        logger.info("Qwen3 /generate responded: HTTP %d in %.1fs", response.status_code, elapsed)
        response.raise_for_status()
        return response.content

    def health(self) -> bool:
        try:
            response = requests.get(f"{self._endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
