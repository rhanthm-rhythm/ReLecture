from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Callable


def _normalize(text: str) -> str:
    """
    Lowercase and reduce to bare ASCII words only.

    Strategy:
    - Dashes that join words (em, en, hyphen variants) → space so
      "word—word" becomes "word word" not "wordword"
    - All Unicode whitespace variants → space
    - NFKD decomposition (accented letters → base + combining mark)
    - Keep only ASCII letters and digits; drop everything else
      (all punctuation, curly quotes, combining marks, symbols)
    """
    # Dash-like chars that join words: replace with space before any other step
    for ch in "—–‒‑‐-":  # em, en, figure, nb-, ‐, hyphen
        text = text.replace(ch, " ")
    # Unicode whitespace variants → space
    text = text.replace(" ", " ")  # narrow no-break space
    text = text.replace(" ", " ")  # no-break space
    # NFKD: decomposes accented letters into base char + combining mark
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    # Keep only ASCII alnum and spaces; everything else (curly quotes, combining
    # marks, punctuation, symbols) is dropped
    out = []
    for ch in text:
        if ch == " ":
            out.append(" ")
        elif ch.isascii() and ch.isalnum():
            out.append(ch)
    return " ".join("".join(out).split())


def _edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    """Return (substitutions, deletions, insertions, ref_words)."""
    try:
        import jiwer
    except ImportError as exc:
        raise RuntimeError("jiwer is required: pip install jiwer") from exc
    ref_words = len(reference.split()) if reference.strip() else 0
    if ref_words == 0:
        return 0, 0, 0, 0
    m = jiwer.process_words(reference, hypothesis)
    return m.substitutions, m.deletions, m.insertions, ref_words


def _acc_key() -> dict:
    return {"S": 0, "D": 0, "I": 0, "N": 0}


def _wer_dict(acc: dict, *, include_counts: bool = True) -> dict:
    total_edits = acc["S"] + acc["D"] + acc["I"]
    wer_val = total_edits / acc["N"] if acc["N"] > 0 else 0.0
    d: dict = {"wer": round(wer_val, 4), "edits": total_edits, "ref_words": acc["N"]}
    if include_counts:
        d["substitutions"] = acc["S"]
        d["deletions"] = acc["D"]
        d["insertions"] = acc["I"]
    return d


def _add(acc: dict, S: int, D: int, I: int, N: int) -> None:
    acc["S"] += S
    acc["D"] += D
    acc["I"] += I
    acc["N"] += N


def compute_wer(
    project_file: str,
    whisper_endpoint: str = "http://localhost:5001/transcribe",
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Compute WER for all chunks in the synthesis manifest of a project.

    Transcribes each chunk WAV via the running Whisper HTTP service and measures
    WER against the chunk reference text from the synthesis manifest.

    Args:
        project_file: Path to project.json
        whisper_endpoint: URL of the Whisper transcription service.
        progress_cb: Optional callable(message) for progress updates.

    Returns:
        Result dict with per-segment WER, per-chunk transcriptions, and aggregates.
        Caller is responsible for writing to disk.
    """
    from ..services import TranscriptionService
    from ..storage import ensure_project_manifest, load_stage_manifest, project_paths
    from ..utils import resolve_project_path

    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)
    project_dir = Path(paths.project_dir)

    manifest = load_stage_manifest(project, project_file, "synthesis")
    backend = manifest.backend

    svc = TranscriptionService(whisper_endpoint)

    seg_durations = {seg.id: seg.duration for seg in manifest.segments}
    total_chunks = sum(len(r.chunks) for r in manifest.results)
    _log(f"  {len(manifest.results)} segments, {total_chunks} chunks")

    corpus: dict[str, dict] = {"lecture": _acc_key(), "narration": _acc_key()}
    per_segment = []
    chunk_counter = 0

    for result in manifest.results:
        seg_id = result.segment_id
        seg_acc: dict[str, dict] = {"lecture": _acc_key(), "narration": _acc_key()}
        chunk_details = []

        for chunk in result.chunks:
            audio_path = resolve_project_path(project_dir, chunk.audio.path)
            if not audio_path or not Path(audio_path).exists():
                continue

            # chunk.kind is "lecture" or "narration" (visual description track)
            raw_kind = chunk.kind
            kind = "narration" if raw_kind == "narration" else "lecture"
            ref = _normalize(chunk.text)

            try:
                hyp = _normalize(svc.transcribe(audio_path, language=manifest.language))
            except Exception as exc:
                _log(f"  WARNING: chunk {chunk.index} (seg {seg_id}) transcription failed: {exc}")
                chunk_details.append({
                    "chunk_index": chunk.index,
                    "kind": kind,
                    "reference": ref,
                    "hypothesis": None,
                    "error": str(exc),
                    "wer": None,
                    "edits": None,
                    "ref_words": len(ref.split()) if ref.strip() else 0,
                })
                chunk_counter += 1
                continue

            S, D, I, N = _edit_counts(ref, hyp)
            _add(seg_acc[kind], S, D, I, N)
            _add(corpus[kind], S, D, I, N)

            chunk_counter += 1
            chunk_wer = (S + D + I) / N if N > 0 else 0.0
            chunk_details.append({
                "chunk_index": chunk.index,
                "kind": kind,
                "reference": ref,
                "hypothesis": hyp,
                "wer": round(chunk_wer, 4),
                "edits": S + D + I,
                "ref_words": N,
            })

            if chunk_counter % 10 == 0:
                _log(f"  chunk {chunk_counter}/{total_chunks}  seg {seg_id}")

        combined_seg = {k: seg_acc["lecture"][k] + seg_acc["narration"][k] for k in ("S", "D", "I", "N")}
        seg_wer = _wer_dict(combined_seg, include_counts=False)["wer"]
        _log(f"  seg {seg_id} done — WER={seg_wer:.4f}")

        per_segment.append({
            "segment_id": seg_id,
            "duration_seconds": result.duration_seconds,
            "original_duration": seg_durations.get(seg_id, 0.0),
            "wer": {
                "lecture": _wer_dict(seg_acc["lecture"], include_counts=False),
                "narration": _wer_dict(seg_acc["narration"], include_counts=False),
                "combined": _wer_dict(combined_seg, include_counts=False),
            },
            "chunks": chunk_details,
        })

    combined_corpus = {k: corpus["lecture"][k] + corpus["narration"][k] for k in ("S", "D", "I", "N")}
    return {
        "project": project_file,
        "backend": backend,
        "language": manifest.language,
        "whisper_model": "base.en",
        "aggregates": {
            "wer": {
                "lecture": _wer_dict(corpus["lecture"]),
                "narration": _wer_dict(corpus["narration"]),
                "combined": _wer_dict(combined_corpus),
            }
        },
        "per_segment": per_segment,
    }
