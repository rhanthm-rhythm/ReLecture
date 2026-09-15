from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import wave
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, whisper_endpoint: str = "http://localhost:5001/transcribe"):
        self.whisper_endpoint = whisper_endpoint

    def health(self) -> bool:
        endpoint = self.whisper_endpoint.rstrip("/")
        if endpoint.endswith("/transcribe"):
            endpoint = endpoint[: -len("/transcribe")]
        try:
            response = requests.get(f"{endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        data = {"language": language or "en"}
        try:
            with open(audio_path, "rb") as audio_file:
                response = requests.post(
                    self.whisper_endpoint,
                    files={"file": audio_file},
                    data=data or None,
                    timeout=300,
                )
            response.raise_for_status()
            payload = response.json()
            return (payload.get("transcription") or "").strip()
        except requests.RequestException as exc:
            raise RuntimeError(f"Whisper transcription failed for '{audio_path}': {exc}") from exc


class TransformationService:
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    @classmethod
    def from_config(cls, config=None) -> "TransformationService":
        from openai import OpenAI
        from .config import load_config

        if config is None:
            config = load_config()
        client = OpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
        )
        return cls(client, config.llm.model_name or "openai/gpt-oss-120b")

    def build_transformation_prompt(
        self,
        target_audience: str = "general audience",
        describe_visuals: bool = False,
        custom_instructions: Optional[str] = None,
        background_profile: Optional[str] = None,
        accessibility_profile: Optional[str] = None,
    ) -> str:
        if accessibility_profile == "visual_impairment":
            describe_visuals = True
        if background_profile == "cs_background" and target_audience == "general audience":
            target_audience = "learners without a CS background"

        rules = [
            f"You are adapting a lecture transcript for {target_audience}.",
            "Explain technical terms and concepts clearly and naturally for this audience.",
            "Return plain spoken prose only. Do not emit markdown formatting, code blocks, bullet points, or raw symbolic math.",
            "Preserve technical correctness and the lecturer's original pedagogical intent.",
        ]
        if accessibility_profile != "visual_impairment":
            rules.append(
                "IMPORTANT LENGTH CONSTRAINT: Your output must be at most 1.5 to 2 times the word count of the original transcript. "
                "Be concise — clarify terms inline without lengthy digressions. Do not pad with filler or unnecessary elaboration."
            )
        if describe_visuals:
            rules.extend([
                "The lecture includes slide visual context.",
                "Explain the meaning, flow, and functional relationships of visual elements (diagrams, tables, equations) naturally as a human lecturer would.",
                "Never use robotic visual meta-clichés such as 'as you can see on the slide', 'in this diagram', or 'looking at the screen'.",
                "If the slide contains a distinct diagram or figure that requires audio description, structure your response as:",
                "[Narrator]: <Concise, natural third-person audio description of the visual figure>",
                "[Lecturer]: <The adapted spoken lecture speech for the target audience>",
                "If no separate visual description is needed, output only the spoken lecture prose without tags.",
            ])
        if custom_instructions:
            rules.append(f"Follow these additional adaptation instructions: {custom_instructions}")

        return " ".join(rules)

    def transform_content(
        self,
        transcript: str,
        visual_context: Optional[str] = None,
        target_audience: str = "general audience",
        describe_visuals: bool = False,
        custom_instructions: Optional[str] = None,
        background_profile: Optional[str] = None,
        accessibility_profile: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        if accessibility_profile == "visual_impairment":
            describe_visuals = True

        system_prompt = self.build_transformation_prompt(
            target_audience=target_audience,
            describe_visuals=describe_visuals,
            custom_instructions=custom_instructions,
            background_profile=background_profile,
            accessibility_profile=accessibility_profile,
        )
        if visual_context and describe_visuals:
            user_message = f"Transcript:\n{transcript}\n\nVisual_Context:\n{visual_context}"
        else:
            user_message = f"Transcript:\n{transcript}"
        if accessibility_profile != "visual_impairment":
            word_count = len(transcript.split())
            max_words = int(word_count * 2)
            user_message += f"\n\n[Original word count: {word_count}. Keep output under {max_words} words.]"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
        except Exception:
            logger.exception("Transformation LLM call failed (model=%s)", self.model_name)
            return None, None

        content = response.choices[0].message.content if response and response.choices else None
        if not content:
            logger.warning("Transformation returned empty content (model=%s)", self.model_name)
            return None, None

        content = content.strip()
        narrator_text = None
        lecturer_text = content

        if describe_visuals and "[Narrator]:" in content and "[Lecturer]:" in content:
            parts = content.split("[Lecturer]:")
            narrator_part = parts[0].replace("[Narrator]:", "").strip()
            lecturer_part = parts[1].strip() if len(parts) > 1 else ""
            if narrator_part:
                narrator_text = narrator_part
            if lecturer_part:
                lecturer_text = lecturer_part

        return lecturer_text, narrator_text

    def sanitize_for_speech(self, transcript: str) -> str:
        if not transcript or not transcript.strip():
            return transcript

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict text preprocessor for speech synthesis. "
                            "Your ONLY job is to convert LaTeX, mathematical symbols, equations, and technical formulas "
                            "into natural, phonetically spoken English words (for example: 'x^2' -> 'x squared', "
                            "'\\alpha' -> 'alpha', 'A = B' -> 'A equals B').\n"
                            "CRITICAL RULES:\n"
                            "- Do NOT reply conversationally.\n"
                            "- Do NOT answer questions, acknowledge the speaker, or add greetings.\n"
                            "- Preserve all non-mathematical words, sentences, punctuation, and wording EXACTLY as given.\n"
                            "- Output ONLY the normalized transcript text and nothing else."
                        ),
                    },
                    {"role": "user", "content": f"Text to normalize:\n{transcript}"},
                ],
                temperature=0.0,
                max_tokens=1200,
            )
        except Exception as exc:
            logger.warning("Math sanitization LLM call failed, keeping original transcript: %s", exc)
            return transcript
        content = response.choices[0].message.content if response and response.choices else None
        return content.strip() if content else transcript

    def identify_segments_needing_sanitization(self, items: list[tuple[int, str]]) -> set[int]:
        """Layer 1: Inspect transcript segments in a single batched prompt
        and return the set of segment IDs that contain mathematical formulas,
        calculus expressions, equations, or technical notation requiring normalization."""
        non_empty = [(seg_id, text.strip()) for seg_id, text in items if text and text.strip()]
        if not non_empty:
            return set()

        flagged_ids: set[int] = set()
        chunk_size = 50
        for i in range(0, len(non_empty), chunk_size):
            chunk = non_empty[i : i + chunk_size]
            formatted = "\n".join(f"[{seg_id}]: {text}" for seg_id, text in chunk)
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert speech synthesis preprocessor reviewing lecture transcript segments.\n"
                                "Your task is to identify which segments contain mathematical notation, calculus expressions, "
                                "equations, formulas, or mathematical variables that must be rewritten into natural spoken English "
                                "for text-to-speech synthesis (e.g. 'd/dx' -> 'd over d x', 'partial I partial x' -> 'partial I with respect to x', 'x^2' -> 'x squared').\n\n"
                                "REQUIRE sanitization:\n"
                                "- Calculus notation: 'd/dx', 'dy/dx', 'partial of z with respect to y', derivatives, integrals, limits\n"
                                "- Formulas & equations: 'x^2 + y = z', 'A_i', 'alpha = beta', fractions, summations, matrices\n"
                                "- Spoken or written mathematical expressions and technical variables\n\n"
                                "Do NOT require sanitization:\n"
                                "- Plain English conversational speech, speaker introductions, slide titles, general academic prose without math\n\n"
                                "Output ONLY a valid JSON object in this format:\n"
                                '{"sanitize_segment_ids": [id1, id2, ...]}\n'
                                'If no segments require sanitization, return: {"sanitize_segment_ids": []}'
                            ),
                        },
                        {"role": "user", "content": f"Segments:\n{formatted}"},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content if response and response.choices else None
                if content:
                    parsed = json.loads(content)
                    ids = parsed.get("sanitize_segment_ids", [])
                    flagged_ids.update(int(x) for x in ids if str(x).isdigit())
            except Exception as exc:
                logger.warning("Layer 1 math assessment failed, falling back to sanitizing all non-empty segments: %s", exc)
                return {seg_id for seg_id, _ in non_empty}

        return flagged_ids


class VisionService:
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    @classmethod
    def from_config(cls, config=None) -> "VisionService | None":
        from openai import OpenAI
        from .config import load_config

        if config is None:
            config = load_config()
        vision_cfg = config.vision
        if vision_cfg.base_url and vision_cfg.model_name:
            client = OpenAI(
                api_key=vision_cfg.api_key or "EMPTY",
                base_url=vision_cfg.base_url,
            )
            return cls(client, vision_cfg.model_name)
        return None

    def extract_visual_context(self, slide_image_path: str) -> Optional[str]:
        if not slide_image_path or not os.path.exists(slide_image_path):
            return None
        with open(slide_image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type = "image/png" if slide_image_path.lower().endswith(".png") else "image/jpeg"
        prompt = (
            "Analyze this lecture slide and return only semantic content, spoken math, diagrams, and key text "
            "in structured prose suitable for accessibility adaptation."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception:
            return None
        content = response.choices[0].message.content if response and response.choices else None
        if not content:
            return None
        # Some VLMs (e.g. reasoning models like MiniCPM-V) emit a <think>...</think>
        # block ahead of the actual answer; strip it so raw reasoning never leaks
        # into the visual context fed to transformation/TTS downstream.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if not content:
            return None
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            parsed = json.loads(content)
            return json.dumps(parsed, indent=2)
        except Exception:
            return content.strip()


class AudioProcessingService:
    @staticmethod
    def ensure_directory_exists(filepath: str) -> str:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        return directory

    @staticmethod
    def get_audio_info(filepath: str) -> dict:
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise RuntimeError("pydub is required for audio inspection.") from exc
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        audio = AudioSegment.from_file(filepath)
        return {
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "duration_seconds": len(audio) / 1000.0,
            "sample_width": audio.sample_width,
            "file_size_bytes": os.path.getsize(filepath),
        }

    @staticmethod
    def merge_wav_files(file_paths: list[str], output_path: str, silence_duration_ms: int = 400) -> str:
        if not file_paths:
            raise ValueError("No files to merge")
        data: list[bytes] = []
        params = None
        for index, path in enumerate(file_paths):
            with wave.open(path, "rb") as wav_file:
                if params is None:
                    params = wav_file.getparams()
                if index > 0 and silence_duration_ms > 0:
                    frame_rate = params.framerate
                    sample_width = params.sampwidth
                    channels = params.nchannels
                    silent_frames = int(frame_rate * (silence_duration_ms / 1000.0))
                    data.append(b"\x00" * silent_frames * sample_width * channels)
                data.append(wav_file.readframes(wav_file.getnframes()))
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setparams(params)
            for chunk in data:
                wav_file.writeframes(chunk)
        return output_path


class TextChunker:
    def split_text(self, text: str, max_chars: int = 350) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        chunks: list[str] = []
        current = ""
        for sentence in text.replace("? ", "?|").replace(". ", ".|").replace("! ", "!|").split("|"):
            if len(current) + len(sentence) + 1 <= max_chars:
                current += sentence + " "
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence + " "
        if current:
            chunks.append(current.strip())
        return chunks


def generate_speech(
    text: str,
    output_path: str,
    style: str = "neutral",
    language: str = "en",
    reference_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    model: str = "qwen3",
    speed: float = 1.0,
    temperature: float = 0.9,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
) -> str:
    url = "http://localhost:5000/generate"
    payload = {
        "text": text,
        "style": style,
        "language": language,
        "model": model,
        "speed_ref": speed,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    if ref_text:
        payload["ref_text"] = ref_text
    files = {}
    if reference_audio_path:
        if not os.path.exists(reference_audio_path):
            return f"Error: Reference audio file not found at {reference_audio_path}"
        files["reference_audio"] = open(reference_audio_path, "rb")
    try:
        response = requests.post(url, data=payload, files=files or None, timeout=300)
    except Exception as exc:
        if files:
            files["reference_audio"].close()
        return f"Error generating audio: {exc}"
    if files:
        files["reference_audio"].close()
    if response.status_code != 200:
        return f"Error: TTS API returned status {response.status_code}: {response.text}"
    with open(output_path, "wb") as handle:
        handle.write(response.content)
    return f"Success: Audio saved to {output_path}"
