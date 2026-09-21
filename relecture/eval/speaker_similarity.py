from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from resemblyzer import VoiceEncoder


def _resolve(base_dir: Path, path_str: str) -> Path:
    """Resolve a manifest path that may use Windows backslashes (WSL-safe)."""
    normalized = path_str.replace("\\", "/")
    p = Path(normalized)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _cosine(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) if len(values) > 1 else 0.0
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "n": len(values),
    }


def compute_speaker_similarity(
    project_file: str,
    encoder: "VoiceEncoder | None" = None,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Compute lecturer voice similarity for all synthesized segments.

    Uses lecture-kind chunks only — narration chunks (different voice) are
    excluded so they don't pollute the similarity score. Each lecture chunk is
    embedded individually and similarities are averaged per segment.

    Falls back to the full merged segment WAV if chunk-level audio is missing
    (e.g. for segments that were already synthesized before chunking was logged).

    Args:
        project_file: Path to project.json (WSL path /mnt/c/... is fine).
        encoder: Pre-loaded VoiceEncoder to reuse across projects.
        progress_cb: Optional callable(message) for progress updates.

    Returns:
        Result dict. Caller is responsible for writing to disk.
    """
    try:
        from resemblyzer import VoiceEncoder as _VoiceEncoder, preprocess_wav
    except ImportError as exc:
        raise RuntimeError(
            "resemblyzer is required: pip install webrtcvad-wheels resemblyzer"
        ) from exc

    from ..storage import ensure_project_manifest, load_stage_manifest, project_paths

    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)
    project_dir = Path(paths.project_dir)

    manifest = load_stage_manifest(project, project_file, "synthesis")
    backend = manifest.backend

    enc = encoder or _VoiceEncoder()

    # --- Lecturer voice reference ---
    ref_embed = None
    ref_path_str = manifest.resolved_voice_reference.path if manifest.resolved_voice_reference else None
    if ref_path_str:
        ref_path = _resolve(project_dir, ref_path_str)
        if ref_path.exists():
            try:
                ref_embed = enc.embed_utterance(preprocess_wav(str(ref_path)))
                _log(f"  lecturer ref: {ref_path.name}")
            except Exception as exc:
                _log(f"  WARNING: could not embed lecturer reference {ref_path}: {exc}")
        else:
            _log(f"  WARNING: lecturer reference not found: {ref_path}")

    if ref_embed is None:
        _log("  WARNING: no lecturer reference — speaker_similarity will not be computed")

    # --- Narrator voice reference (bundled asset) ---
    narrator_embed = None
    narrator_ref_path = Path(__file__).resolve().parents[2] / "assets" / "narrator_voice.wav"
    if narrator_ref_path.exists():
        try:
            narrator_embed = enc.embed_utterance(preprocess_wav(str(narrator_ref_path)))
            _log(f"  narrator ref: {narrator_ref_path.name}")
        except Exception as exc:
            _log(f"  WARNING: could not embed narrator reference: {exc}")
    else:
        _log(f"  WARNING: narrator reference not found at {narrator_ref_path} — narration_similarity will not be computed")

    # --- Counts ---
    n_results = len(manifest.results)
    total_lecture = sum(sum(1 for c in r.chunks if c.kind == "lecture") for r in manifest.results)
    total_narration = sum(sum(1 for c in r.chunks if c.kind == "narration") for r in manifest.results)
    _log(f"  {n_results} segments — {total_lecture} lecture chunks, {total_narration} narration chunks")

    per_segment = []
    seg_lec_sims: list[float] = []
    seg_nar_sims: list[float] = []

    for idx, result in enumerate(manifest.results, 1):
        seg_id = result.segment_id
        lecture_chunks = [c for c in result.chunks if c.kind == "lecture"]
        narration_chunks = [c for c in result.chunks if c.kind == "narration"]

        entry: dict = {
            "segment_id": seg_id,
            "duration_seconds": result.duration_seconds,
            "n_lecture_chunks": len(lecture_chunks),
            "n_narration_chunks": len(narration_chunks),
        }

        # --- Lecturer similarity ---
        if ref_embed is not None:
            if lecture_chunks:
                targets = [(f"chunk_{c.index}", _resolve(project_dir, c.audio.path)) for c in lecture_chunks]
            else:
                # Fall back to full segment WAV if no chunk-level audio recorded.
                targets = [("segment_wav", _resolve(project_dir, result.audio.path))]
                entry["lecturer_similarity_source"] = "full_segment_wav_fallback"

            lec_sims: list[float] = []
            for label, audio_path in targets:
                if not audio_path.exists():
                    _log(f"  WARNING: seg {seg_id} {label} not found: {audio_path}")
                    continue
                try:
                    sim = _cosine(ref_embed, enc.embed_utterance(preprocess_wav(str(audio_path))))
                    lec_sims.append(sim)
                except Exception as exc:
                    _log(f"  WARNING: seg {seg_id} {label} embed failed: {exc}")

            if lec_sims:
                seg_lec_sim = sum(lec_sims) / len(lec_sims)
                entry["speaker_similarity"] = round(seg_lec_sim, 6)
                entry["lecture_chunk_similarities"] = [round(s, 6) for s in lec_sims]
                seg_lec_sims.append(seg_lec_sim)

        # --- Narrator similarity ---
        if narrator_embed is not None and narration_chunks:
            nar_sims: list[float] = []
            for chunk in narration_chunks:
                audio_path = _resolve(project_dir, chunk.audio.path)
                if not audio_path.exists():
                    _log(f"  WARNING: seg {seg_id} narration chunk_{chunk.index} not found")
                    continue
                try:
                    sim = _cosine(narrator_embed, enc.embed_utterance(preprocess_wav(str(audio_path))))
                    nar_sims.append(sim)
                except Exception as exc:
                    _log(f"  WARNING: seg {seg_id} narration chunk_{chunk.index} embed failed: {exc}")

            if nar_sims:
                seg_nar_sim = sum(nar_sims) / len(nar_sims)
                entry["narration_similarity"] = round(seg_nar_sim, 6)
                entry["narration_chunk_similarities"] = [round(s, 6) for s in nar_sims]
                seg_nar_sims.append(seg_nar_sim)

        # --- Log line ---
        lec_str = f"lec={entry['speaker_similarity']:.4f}" if "speaker_similarity" in entry else "lec=n/a"
        nar_str = f"  nar={entry['narration_similarity']:.4f}" if "narration_similarity" in entry else ""
        _log(f"  seg {idx}/{n_results}  id={seg_id}  {lec_str}{nar_str}")

        per_segment.append(entry)

    aggregates: dict = {}
    if seg_lec_sims:
        aggregates["speaker_similarity"] = _stats(seg_lec_sims)
    if seg_nar_sims:
        aggregates["narration_similarity"] = _stats(seg_nar_sims)

    return {
        "project": project_file,
        "backend": backend,
        "language": manifest.language,
        "aggregates": aggregates,
        "per_segment": per_segment,
    }
