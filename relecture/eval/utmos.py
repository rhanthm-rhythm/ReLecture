"""
UTMOS naturalness scoring for synthesized speech segments.

Uses the `speechmos` package (tarepan/SpeechMOS) to predict Mean Opinion Scores
for each segment-level WAV in the synthesis manifest.

Does NOT touch pipeline code. Only reads project manifests and segment WAV files.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable


def _resolve(base_dir: Path, path_str: str) -> Path:
    """Resolve a manifest path that may use Windows backslashes (WSL-safe)."""
    normalized = path_str.replace("\\", "/")
    p = Path(normalized)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


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


_UTMOS_CKPT_URL = (
    "https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt"
)


def _load_utmos_model(device: str):
    """Download (once) and load the UTMOS22 Strong model checkpoint."""
    import urllib.request
    from pathlib import Path
    import torch
    from speechmos.utmos22.strong.model import UTMOS22Strong

    cache_dir = Path.home() / ".cache" / "utmos22"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / "utmos22_strong_step7459_v1.pt"

    if not ckpt_path.exists():
        print(f"[UTMOS] Downloading checkpoint to {ckpt_path} ...", flush=True)
        urllib.request.urlretrieve(_UTMOS_CKPT_URL, ckpt_path)
        print("[UTMOS] Download complete.", flush=True)

    model = UTMOS22Strong()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # checkpoint may be a raw state_dict or wrapped in a dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def _load_predictor():
    """Return a callable(wav_path: str) -> float UTMOS predictor.

    Uses UTMOS22 Strong from speechmos + checkpoint from tarepan/SpeechMOS releases.
    Runs on GPU (CUDA) if available, otherwise CPU.
    """
    try:
        import torch
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "torch and soundfile are required for UTMOS scoring.\n"
            "  Run: pip install torch soundfile\n"
            f"  Root cause: {exc}"
        ) from exc

    try:
        from speechmos.utmos22.strong.model import UTMOS22Strong  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "speechmos (tarepan version) is required for UTMOS scoring.\n"
            "  Run: pip install git+https://github.com/tarepan/SpeechMOS@v1.0.0\n"
            f"  Root cause: {exc}"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[UTMOS] Using device: {device}", flush=True)
    if device == "cuda":
        print(f"[UTMOS] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model = _load_utmos_model(device)

    def _predict(wav_path: str) -> float:
        wave, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)  # stereo → mono
        wave_t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            score = model(wave_t, sr)
        return float(score.item())

    return _predict


def compute_utmos(
    project_file: str,
    predictor=None,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Compute UTMOS naturalness scores for all synthesized segments in a project.

    Scores each segment-level WAV with the UTMOS predictor and aggregates
    mean/std/min/max across all segments within the presentation.

    Args:
        project_file: Path to project.json.
        predictor: Pre-loaded callable(wav_path) -> float to reuse across projects.
                   If None, loads the speechmos predictor on first call.
        progress_cb: Optional callable(message) for progress updates.

    Returns:
        Result dict with per-segment scores and corpus aggregates.
        Caller is responsible for writing to disk.
    """
    from ..storage import ensure_project_manifest, load_stage_manifest, project_paths

    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)
    project_dir = Path(paths.project_dir)

    manifest = load_stage_manifest(project, project_file, "synthesis")
    backend = manifest.backend

    score_fn = predictor or _load_predictor()

    n_results = len(manifest.results)
    total_dur = sum(r.duration_seconds for r in manifest.results if r.duration_seconds)
    _log(f"  {n_results} segments, {total_dur:.0f}s total audio")

    per_segment: list[dict] = []
    all_scores: list[float] = []
    import time as _time
    t_video_start = _time.time()

    for idx, result in enumerate(manifest.results, 1):
        seg_id = result.segment_id
        t_seg_start = _time.time()

        audio_path = _resolve(project_dir, result.audio.path)
        if not audio_path.exists():
            _log(f"  WARNING: seg {seg_id} WAV not found: {audio_path}")
            per_segment.append({
                "segment_id": seg_id,
                "duration_seconds": result.duration_seconds,
                "utmos": None,
                "error": f"WAV not found: {audio_path}",
            })
            continue

        _log(f"  seg {idx}/{n_results}  id={seg_id}  scoring {result.duration_seconds:.1f}s audio ...")
        try:
            score = score_fn(str(audio_path))
            all_scores.append(score)
            seg_elapsed = _time.time() - t_seg_start
            video_elapsed = _time.time() - t_video_start
            avg_per_seg = video_elapsed / idx
            eta_secs = avg_per_seg * (n_results - idx)
            running_mean = sum(all_scores) / len(all_scores)
            _log(
                f"  seg {idx}/{n_results}  id={seg_id}  "
                f"UTMOS={score:.4f}  [{seg_elapsed:.1f}s]  "
                f"running_mean={running_mean:.4f}  "
                f"ETA {eta_secs:.0f}s"
            )
            per_segment.append({
                "segment_id": seg_id,
                "duration_seconds": result.duration_seconds,
                "utmos": round(score, 6),
            })
        except Exception as exc:
            _log(f"  WARNING: seg {seg_id} scoring failed: {exc}")
            per_segment.append({
                "segment_id": seg_id,
                "duration_seconds": result.duration_seconds,
                "utmos": None,
                "error": str(exc),
            })

    return {
        "project": project_file,
        "backend": backend,
        "language": manifest.language,
        "aggregates": {
            "utmos": _stats(all_scores),
        },
        "per_segment": per_segment,
    }
