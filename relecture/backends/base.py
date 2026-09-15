from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSBackend(Protocol):
    """HTTP contract every TTS server must satisfy.

    Each backend implementation is a thin client that speaks to a companion
    server running in its own venv under servers/<name>/server.py.
    The server must expose POST /synthesize and GET /health.
    """
    name: str

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
        """Return raw WAV bytes for the synthesized audio."""
        ...

    def health(self) -> bool:
        """Return True if the backend server is reachable."""
        ...
