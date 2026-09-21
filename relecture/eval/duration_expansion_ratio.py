"""
Duration Expansion Ratio for synthesized speech segments.

Computes the ratio of synthesized segment duration to original segment duration,
using only data already present in the synthesis manifest — no external libraries
or audio processing required.

Does NOT touch pipeline code. Only reads project manifests.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    mean = sum(values) / len(values)
    std = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if len(values) > 1
        else 0.0
    )
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "n": len(values),
    }


def compute_duration_expansion_ratio(
    project_file: str,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Compute duration expansion ratio for all synthesized segments in a project.

    Ratio = synthesized_duration_seconds / original_segment_duration_seconds.

    A ratio of 3.5 means the synthesized audio takes 3.5× longer than the
    original lecture segment — expected for adapted VI transcripts which are
    3–5× longer.

    Args:
        project_file: Path to project.json.
        progress_cb: Optional callable(message) for progress updates.

    Returns:
        Result dict with per-segment ratios and corpus aggregates.
        Caller is responsible for writing to disk.
    """
    from ..storage import ensure_project_manifest, load_stage_manifest, project_paths

    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)  # noqa: F841 — kept for consistency with other modules
    manifest = load_stage_manifest(project, project_file, "synthesis")
    backend = manifest.backend

    # Build segment-id → original duration lookup from the manifest
    seg_durations: dict[int, float] = {
        seg.id: seg.duration for seg in manifest.segments if seg.duration
    }

    n_results = len(manifest.results)
    _log(f"  {n_results} segments")

    per_segment: list[dict] = []
    all_ratios: list[float] = []

    for idx, result in enumerate(manifest.results, 1):
        seg_id = result.segment_id
        synth_dur = result.duration_seconds
        orig_dur = seg_durations.get(seg_id)

        entry: dict = {
            "segment_id": seg_id,
            "synth_duration_seconds": synth_dur,
            "original_duration_seconds": orig_dur,
        }

        if orig_dur and orig_dur > 0 and synth_dur is not None:
            ratio = synth_dur / orig_dur
            entry["ratio"] = round(ratio, 6)
            all_ratios.append(ratio)
            _log(f"  seg {idx}/{n_results}  id={seg_id}  ratio={ratio:.4f}  "
                 f"synth={synth_dur:.1f}s  orig={orig_dur:.1f}s")
        else:
            entry["ratio"] = None
            reason = "orig_duration=0" if (orig_dur is not None and orig_dur == 0) else "orig_duration=missing"
            entry["warning"] = reason
            _log(f"  seg {idx}/{n_results}  id={seg_id}  ratio=N/A  ({reason})")

        per_segment.append(entry)

    return {
        "project": project_file,
        "backend": backend,
        "language": manifest.language,
        "aggregates": {
            "duration_expansion_ratio": _stats(all_ratios),
        },
        "per_segment": per_segment,
    }
