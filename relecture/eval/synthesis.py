from __future__ import annotations

import json
from pathlib import Path

from ..storage import project_paths, ensure_project_manifest, load_stage_manifest
from ..utils import resolve_project_path


def run_synthesis_eval(project_file: str) -> dict:
    """Evaluate synthesis quality: speaker similarity via resemblyzer.

    UTMOS requires a separate package (speechmos); will be added when available.
    Requires: pip install resemblyzer
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("resemblyzer is required: pip install resemblyzer") from exc

    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)
    manifest = load_stage_manifest(project, project_file, "synthesis")

    encoder = VoiceEncoder()

    ref_path = resolve_project_path(
        paths.project_dir,
        manifest.resolved_voice_reference.path if manifest.resolved_voice_reference else "",
    )

    ref_embed = None
    if ref_path and Path(ref_path).exists():
        ref_wav = preprocess_wav(ref_path)
        ref_embed = encoder.embed_utterance(ref_wav)

    results = []
    for result in manifest.results:
        audio_path = resolve_project_path(paths.project_dir, result.audio.path)
        if not Path(audio_path).exists():
            continue
        entry = {"segment_id": result.segment_id, "duration_seconds": result.duration_seconds}
        if ref_embed is not None:
            synth_wav = preprocess_wav(audio_path)
            synth_embed = encoder.embed_utterance(synth_wav)
            similarity = float(
                np.dot(ref_embed, synth_embed)
                / (np.linalg.norm(ref_embed) * np.linalg.norm(synth_embed))
            )
            entry["speaker_similarity"] = similarity
        results.append(entry)

    sim_values = [r["speaker_similarity"] for r in results if "speaker_similarity" in r]
    aggregates = {}
    if sim_values:
        aggregates["speaker_similarity"] = {"mean": sum(sim_values) / len(sim_values), "n": len(sim_values)}

    output = {
        "project": project_file,
        "backend": manifest.backend,
        "language": manifest.language,
        "aggregates": aggregates,
        "per_segment": results,
    }

    out_dir = Path(paths.project_dir) / "eval" / "synthesis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"quality.{manifest.backend}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Synthesis eval saved to {out_path}")
    return output
