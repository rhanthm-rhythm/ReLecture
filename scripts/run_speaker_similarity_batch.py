"""
Batch speaker-similarity evaluation for the first N presentations (alphabetically).
Uses resemblyzer d-vector embeddings. Loads the VoiceEncoder once and reuses it.

Run this from WSL/Linux where resemblyzer installs cleanly:
    pip install webrtcvad-wheels resemblyzer
    python scripts/run_speaker_similarity_batch.py \\
        --dataset eval_result/chatterbox/independent/visual_impairment \\
        [--output-root eval_result/synthesis/speaker_similarity] \\
        [--limit 50] \\
        [--skip-existing]

Results are written to:
  <output-root>/<videoID>/speaker_similarity.chatterbox.json
  <output-root>/_aggregate.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "eval_result" / "synthesis" / "speaker_similarity"


def _load_valid(path: Path) -> dict | None:
    """Load and validate a result JSON. Returns None if missing or invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _ = data["aggregates"]["speaker_similarity"]["mean"]
        return data
    except Exception:
        return None


def _nar_mean(result: dict) -> float | None:
    agg = result["aggregates"].get("narration_similarity")
    return agg["mean"] if agg else None


def _agg_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(values) / len(values)
    import math
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) if len(values) > 1 else 0.0
    return {"mean": round(mean, 6), "std": round(std, 6), "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch speaker-similarity evaluation (resemblyzer d-vectors)."
    )
    parser.add_argument("--dataset", required=True, help="Root dir containing project subdirs.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max projects (alphabetical).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos with a valid result file.")
    parser.add_argument("--tag", default="chatterbox", help="Model tag used in the output filename (e.g. qwen3).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    projects = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith("_")],
        key=lambda d: d.name,
    )[: args.limit]

    print(f"Projects : {len(projects)} (first {args.limit} alphabetically)")
    print(f"Output   : {output_root}")
    print()

    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        print(
            "ERROR: resemblyzer not installed.\n"
            "  Run: pip install webrtcvad-wheels resemblyzer",
            file=sys.stderr,
        )
        sys.exit(1)

    from relecture.eval.speaker_similarity import compute_speaker_similarity

    print("Loading VoiceEncoder ...", flush=True)
    encoder = VoiceEncoder()
    print("VoiceEncoder ready.\n", flush=True)

    all_results: list[dict] = []
    failed: list[dict] = []
    elapsed_times: list[float] = []
    batch_start = time.time()

    for idx, project_dir in enumerate(projects, 1):
        video_id = project_dir.name
        project_file = project_dir / "project.json"
        out_file = output_root / video_id / f"speaker_similarity.{args.tag}.json"

        if not project_file.exists():
            print(f"[{idx:>2}/{len(projects)}] SKIP {video_id} — no project.json")
            continue

        if args.skip_existing:
            existing = _load_valid(out_file)
            if existing is not None:
                mean_sim = existing["aggregates"]["speaker_similarity"]["mean"]
                print(f"[{idx:>2}/{len(projects)}] SKIP {video_id} — already done  mean={mean_sim:.4f}")
                all_results.append(existing)
                continue
            elif out_file.exists():
                print(f"[{idx:>2}/{len(projects)}] REPROCESS {video_id} — existing file invalid or incomplete")

        print(f"[{idx:>2}/{len(projects)}] {video_id}", flush=True)
        t0 = time.time()

        def _progress(msg: str) -> None:
            print(f"  {msg}", flush=True)

        try:
            result = compute_speaker_similarity(
                str(project_file),
                encoder=encoder,
                progress_cb=_progress,
            )
            elapsed = time.time() - t0
            elapsed_times.append(elapsed)

            agg = result["aggregates"].get("speaker_similarity")
            nar_agg = result["aggregates"].get("narration_similarity")
            if agg:
                remaining = len(projects) - idx
                eta_s = (sum(elapsed_times) / len(elapsed_times)) * remaining
                eta_str = f"{eta_s/60:.1f} min" if eta_s >= 60 else f"{eta_s:.0f}s"
                nar_str = f"  narr={nar_agg['mean']:.4f}" if nar_agg else ""
                print(
                    f"  DONE  lec={agg['mean']:.4f}  std={agg['std']:.4f}"
                    f"  min={agg['min']:.4f}  max={agg['max']:.4f}  n={agg['n']}"
                    f"{nar_str}  [{elapsed:.1f}s elapsed, ETA {eta_str}]"
                )
            else:
                print(f"  DONE  (no similarity computed — voice reference missing?)  [{elapsed:.1f}s]")

            out_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
            os.replace(tmp, out_file)
            all_results.append(result)

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.1f}s): {exc}")
            traceback.print_exc()
            failed.append({"project": video_id, "error": str(exc)})

        print()

    # --- Corpus aggregate ---
    if not all_results:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    per_project_rows = []
    all_lec_means: list[float] = []
    all_nar_means: list[float] = []
    for r in all_results:
        video_id = Path(r["project"]).parent.name
        agg = r["aggregates"].get("speaker_similarity")
        nar_agg = r["aggregates"].get("narration_similarity")
        row: dict = {"video_id": video_id}
        if agg:
            row["lecturer_similarity_mean"] = agg["mean"]
            row["lecturer_similarity_std"] = agg["std"]
            row["n_segments"] = agg["n"]
            all_lec_means.append(agg["mean"])
        else:
            row["lecturer_similarity_mean"] = None
        if nar_agg:
            row["narration_similarity_mean"] = nar_agg["mean"]
            row["narration_similarity_std"] = nar_agg["std"]
            all_nar_means.append(nar_agg["mean"])
        else:
            row["narration_similarity_mean"] = None
        per_project_rows.append(row)

    aggregate = {
        "dataset": str(dataset_dir),
        "n_projects": len(all_results),
        "corpus_speaker_similarity": _agg_stats(all_lec_means),
        "corpus_narration_similarity": _agg_stats(all_nar_means),
        "per_project": per_project_rows,
        "failed": failed,
    }

    agg_path = output_root / "_aggregate.json"
    agg_tmp = agg_path.with_suffix(".json.tmp")
    agg_tmp.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    os.replace(agg_tmp, agg_path)

    total_elapsed = time.time() - batch_start
    print("=" * 60)
    print(f"Aggregate saved : {agg_path}")
    print(f"Projects done   : {aggregate['n_projects']}")
    print(f"Total wall time : {total_elapsed/60:.1f} min")
    cs = aggregate["corpus_speaker_similarity"]
    cn = aggregate["corpus_narration_similarity"]
    if cs["mean"] is not None:
        print(f"Lecturer sim    : {cs['mean']:.4f}  (std={cs['std']:.4f}  n={cs['n']})")
    else:
        print("Lecturer sim    : N/A")
    if cn["mean"] is not None:
        print(f"Narration sim   : {cn['mean']:.4f}  (std={cn['std']:.4f}  n={cn['n']})")
    else:
        print("Narration sim   : N/A (no narration chunks found)")

    if failed:
        print(f"\n{len(failed)} project(s) failed:")
        for f in failed:
            print(f"  {f['project']}: {f['error']}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
