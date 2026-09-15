"""Qwen3 TTS server — runs in its own venv.

Install: pip install -r servers/qwen3/requirements.txt
Start:   python servers/qwen3/server.py

Exposes:
  POST /generate   multipart: text, reference_audio, ref_text, style, language, speed_ref, temperature
  GET  /health
"""
from __future__ import annotations

import io
import os
import tempfile
import threading

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
_model = None
_lock = threading.Lock()
QWEN3_MODEL = os.getenv("QWEN3_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# Flask form language codes -> Qwen3-TTS language names
LANG_MAP = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}


class _VoiceCloneModel:
    """Adapter exposing synthesize(...) -> wav bytes over qwen_tts."""

    def __init__(self, model):
        self._model = model

    def synthesize(
        self,
        text,
        ref_audio=None,
        ref_text="",
        speed=1.0,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        max_new_tokens=None,
        language="en",
    ):
        import soundfile as sf

        ref_path = None
        if ref_audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(ref_audio)
                ref_path = tmp.name
        try:
            # Dynamic max token bounding to protect against runaway loops:
            # At 12Hz speech, normal speaking speed is ~2.5 words/sec (~5 tokens/word).
            # We allow up to 18 tokens/word + 120 buffer clamped between 240 and 2048.
            word_count = max(len(text.split()), 1)
            token_limit = max_new_tokens or min(2048, max(240, int(word_count * 18) + 120))

            kwargs = {
                "text": text,
                "language": LANG_MAP.get(language, language),
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": token_limit,
            }
            if ref_path:
                kwargs["ref_audio"] = ref_path
                clean_ref_text = (ref_text or "").strip()
                if clean_ref_text:
                    kwargs["ref_text"] = clean_ref_text
                else:
                    # No transcript or empty: speaker-embedding-only mode
                    kwargs["x_vector_only_mode"] = True
            wavs, sr = self._model.generate_voice_clone(**kwargs)
        finally:
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        return buf.getvalue()


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from qwen_tts import Qwen3TTSModel

                model_path = _resolve_local_model_path(QWEN3_MODEL)
                model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                _model = _VoiceCloneModel(model)
    return _model


def _resolve_local_model_path(repo_id: str) -> str:
    """Resolve `repo_id` to its local HF cache snapshot directory when it is
    already downloaded, so `from_pretrained` sees a local path and never
    attempts any network call (some tokenizer code paths in transformers,
    e.g. the mistral-regex patch, hit the Hub API even when the files are
    fully cached, whenever they're given a bare repo id instead of a path).
    Falls back to `repo_id` unchanged (normal online-capable behavior) if the
    snapshot isn't cached locally yet.
    """
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return repo_id


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": f"qwen3:{QWEN3_MODEL}"})


@app.post("/generate")
def generate():
    import time
    t0 = time.time()

    text = request.form.get("text", "")
    style = request.form.get("style", "neutral")
    language = request.form.get("language", "en")
    speed = float(request.form.get("speed_ref", 1.0))
    temperature = float(request.form.get("temperature", 0.7))
    top_p = float(request.form.get("top_p", 0.95))
    repetition_penalty = float(request.form.get("repetition_penalty", 1.1))
    max_tokens_arg = request.form.get("max_new_tokens")
    max_new_tokens = int(max_tokens_arg) if max_tokens_arg else None

    ref_text = request.form.get("ref_text", "")
    ref_audio = request.files.get("reference_audio")
    ref_bytes = ref_audio.read() if ref_audio else None

    print(
        f"[Qwen3-TTS] Synthesis request: words={len(text.split())}, "
        f"ref_text_len={len(ref_text)}, has_audio={bool(ref_bytes)}, "
        f"temp={temperature}, rep_pen={repetition_penalty}",
        flush=True,
    )

    model = get_model()
    with _lock:
        audio_bytes = model.synthesize(
            text,
            ref_audio=ref_bytes,
            ref_text=ref_text,
            speed=speed,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            language=language,
        )

    elapsed = time.time() - t0
    print(f"[Qwen3-TTS] Synthesis completed in {elapsed:.2f}s ({len(audio_bytes)} bytes)", flush=True)
    return send_file(io.BytesIO(audio_bytes), mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
