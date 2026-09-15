from __future__ import annotations

import json
from pathlib import Path

from ..storage import load_json, project_paths, ensure_project_manifest, load_stage_manifest


def run_transcription_eval(project_file: str, ground_truth_file: str) -> dict:
    """Compute WER between Whisper transcripts and VTT-derived ground truth.

    Requires: pip install jiwer
    """
    try:
        from jiwer import wer
    except ImportError as exc:
        raise RuntimeError("jiwer is required: pip install jiwer") from exc

    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)
    manifest = load_stage_manifest(project, project_file, "transcription")
    ground_truth = load_json(ground_truth_file)  # {segment_id_str: text}

    results = []
    all_hyp = []
    all_ref = []
    for segment in manifest.segments:
        hyp = (segment.clean_transcript or segment.raw_transcript or "").strip()
        ref = ground_truth.get(str(segment.id), "").strip()
        if not ref:
            continue
        segment_wer = float(wer(ref, hyp))
        results.append({
            "segment_id": segment.id,
            "wer": segment_wer,
            "ref_len": len(ref.split()),
            "hyp_len": len(hyp.split()),
        })
        all_hyp.append(hyp)
        all_ref.append(ref)

    aggregate_wer = float(wer(" ".join(all_ref), " ".join(all_hyp))) if all_ref else None
    output = {
        "project": project_file,
        "aggregate_wer": aggregate_wer,
        "per_segment": results,
    }

    out_dir = Path(paths.project_dir) / "eval" / "transcription"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wer.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if aggregate_wer is not None:
        print(f"WER: {aggregate_wer:.4f} — saved to {out_path}")
    else:
        print(f"No ground truth matches found — saved to {out_path}")
    return output
