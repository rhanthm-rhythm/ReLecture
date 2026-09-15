from __future__ import annotations

import os
from dataclasses import dataclass, field


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _maybe_use_local_ca_bundle() -> None:
    """Trust a locally-generated CA bundle if present.

    On networks with TLS-inspecting proxies (e.g. corporate firewalls),
    Windows may trust the intercepting root CA while Python's bundled
    `certifi` store does not, causing SSL errors when calling hosted APIs
    (Nebius, etc.). If `.certs/mas-tts-ca-bundle.pem` exists (certifi's
    public roots + the local corporate root, see servers/whisper docs),
    point `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` at it unless already set.
    """
    bundle = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".certs", "mas-tts-ca-bundle.pem")
    if os.path.isfile(bundle):
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)


_maybe_load_dotenv()
_maybe_use_local_ca_bundle()


@dataclass(frozen=True)
class LLMBackendConfig:
    api_key: str | None
    base_url: str | None
    model_name: str | None


@dataclass(frozen=True)
class VisionBackendConfig:
    api_key: str | None
    base_url: str | None
    model_name: str | None


@dataclass(frozen=True)
class TTSBackendConfig:
    """Configuration for a TTS backend server.

    Each backend runs in its own venv (see servers/<name>/). The endpoint
    URL points to its HTTP server. Extra holds backend-specific params like
    exaggeration (Chatterbox) or cfg_weight.
    """
    backend: str  # "qwen3" | "chatterbox" | "cosyvoice"
    endpoint_url: str
    extra: dict = field(default_factory=dict)

    def __hash__(self):
        return hash((self.backend, self.endpoint_url))

    def __eq__(self, other):
        if not isinstance(other, TTSBackendConfig):
            return NotImplemented
        return self.backend == other.backend and self.endpoint_url == other.endpoint_url


@dataclass(frozen=True)
class AppConfig:
    llm: LLMBackendConfig
    vision: VisionBackendConfig
    whisper_endpoint: str
    tts: TTSBackendConfig
    language: str


def load_config() -> AppConfig:
    tts_backend = os.getenv("TTS_BACKEND", "qwen3")
    tts_endpoints = {
        "qwen3": os.getenv("QWEN3_TTS_ENDPOINT", "http://localhost:5000"),
        "chatterbox": os.getenv("CHATTERBOX_ENDPOINT", "http://localhost:5002"),
        "cosyvoice": os.getenv("COSYVOICE_ENDPOINT", "http://localhost:5003"),
    }
    return AppConfig(
        llm=LLMBackendConfig(
            api_key=os.getenv("NEBIUS_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.studio.nebius.com/v1/"),
            model_name=os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b"),
        ),
        vision=VisionBackendConfig(
            api_key=os.getenv("VISION_API_KEY", "EMPTY"),
            base_url=os.getenv("VISION_BASE_URL", "http://localhost:8005/v1"),
            model_name=os.getenv("VISION_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"),
        ),
        whisper_endpoint=os.getenv("WHISPER_ENDPOINT", "http://localhost:5001/transcribe"),
        tts=TTSBackendConfig(
            backend=tts_backend,
            endpoint_url=tts_endpoints.get(tts_backend, "http://localhost:5000"),
        ),
        language=os.getenv("LANGUAGE", "en"),
    )
