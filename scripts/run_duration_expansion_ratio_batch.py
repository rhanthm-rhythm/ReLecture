"""
Batch duration-expansion-ratio evaluation for the first N presentations (alphabetically).
No external libraries required — reads directly from synthesis manifests.

Does NOT touch pipeline code. Only reads project manifests.

Run from the project root (Windows or WSL):
    uv run python scripts/run_duration_expansion_ratio_batch.py \\
        --dataset eval_result/chatterbox/independent/visual_impairment \\
        [--output-root eval_result/synthesis/duration_expansion_ratio] \\
        [--limit 50] \\
        [--skip-existing]

Results are written to:
  <output-root>/<videoID>/duration_expansion_ratio.chatterbox.json
  <output-root>/_aggregate.json

Crash-safe: each result is written atomically (.tmp then rename).
Interrupted runs can be resumed with --skip-existing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "eval_result" / "synthesis" / "duration_expansion_ratio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agg_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(values) / len(values)
    std = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if len(values) > 1
        else 0.0
    )
    return {"mean": round(mean, 6), "std": round(std, 6), "n": len(values)}


def _load_valid(path: Path) -> dict | None:
    """Load and validate a result JSON. Returns None if missing or incomplete."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        mean = data["aggregates"]["duration_expansion_ratio"]["mean"]
        if mean is None:
            return None
        return data
    except Exception:
        return None


def _write_atomic(path: Path, obj: dict) -> None:
    """Write JSON atomically: write .tmp then os.replace (NTFS-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch duration-expansion-ratio evaluation from synthesis manifests."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Root dir containing project subdirs (e.g. eval_result/chatterbox/independent/visual_impairment).",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Max number of projects to process (alphabetical order, default: 50).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip videos that already have a valid result file.",
    )
    parser.add_argument("--tag", default="chatterbox", help="Model tag used in the output filename (e.g. qwen3).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # First N alphabetically; skip dirs with _ prefix (aggregates etc.)
    projects = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith("_")],
        key=lambda d: d.name,
    )[: args.limit]

    print(f"Projects : {len(projects)} (first {args.limit} alphabetically)")
    print(f"Dataset  : {dataset_dir}")
    print(f"Output   : {output_root}")
    print()

    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    try:
        from relecture.eval.duration_expansion_ratio import compute_duration_expansion_ratio
    except ImportError as exc:
        print(f"ERROR: could not import duration_expansion_ratio module: {exc}", file=sys.stderr)
        sys.exit(1)

    all_results: list[dict] = []
    failed: list[dict] = []
    elapsed_times: list[float] = []
    batch_start = time.time()

    for idx, project_dir in enumerate(projects, 1):
        video_id = project_dir.name
        project_file = project_dir / "project.json"
        out_file = output_root / video_id / f"duration_expansion_ratio.{args.tag}.json"

        if not project_file.exists():
            print(f"[{idx:>2}/{len(projects)}] SKIP {video_id} — no project.json")
            continue

        # --- Skip / reprocess logic ---
        if args.skip_existing:
            existing = _load_valid(out_file)
            if existing is not None:
                agg = existing["aggregates"]["duration_expansion_ratio"]
                print(
                    f"[{idx:>2}/{len(projects)}] SKIP {video_id} — already done  "
                    f"mean={agg['mean']:.4f}  std={agg['std']:.4f}  n={agg['n']}"
                )
                all_results.append(existing)
                continue
            elif out_file.exists():
                print(
                    f"[{idx:>2}/{len(projects)}] REPROCESS {video_id} — existing file invalid or incomplete"
                )

        print(f"[{idx:>2}/{len(projects)}] {video_id}", flush=True)
        t0 = time.time()

        def _progress(msg: str) -> None:
            print(f"  {msg}", flush=True)

        try:
            result = compute_duration_expansion_ratio(
                str(project_file),
                progress_cb=_progress,
            )
            elapsed = time.time() - t0
            elapsed_times.append(elapsed)

            agg = result["aggregates"]["duration_expansion_ratio"]
            remaining = len(projects) - idx
            eta_s = (sum(elapsed_times) / len(elapsed_times)) * remaining if elapsed_times else 0.0
            eta_str = f"{eta_s / 60:.1f} min" if eta_s >= 60 else f"{eta_s:.0f}s"

            print(
                f"  DONE  mean={agg['mean']:.4f}  std={agg['std']:.4f}"
                f"  min={agg['min']:.4f}  max={agg['max']:.4f}  n={agg['n']}"
                f"  [{elapsed:.2f}s elapsed, ETA {eta_str}]"
            )

            _write_atomic(out_file, result)
            all_results.append(result)

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  FAILED ({elapsed:.2f}s): {exc}")
            traceback.print_exc()
            failed.append({"project": video_id, "error": str(exc)})

        print()

    # --- Corpus aggregate ---
    if not all_results:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    per_project_rows: list[dict] = []
    all_means: list[float] = []

    for r in all_results:
        video_id = Path(r["project"]).parent.name
        agg = r["aggregates"].get("duration_expansion_ratio", {})
        row: dict = {
            "video_id": video_id,
            "ratio_mean": agg.get("mean"),
            "ratio_std": agg.get("std"),
            "n_segments": agg.get("n"),
        }
        if agg.get("mean") is not None:
            all_means.append(agg["mean"])
        per_project_rows.append(row)

    aggregate = {
        "dataset": str(dataset_dir),
        "n_projects": len(all_results),
        "corpus_duration_expansion_ratio": _agg_stats(all_means),
        "per_project": per_project_rows,
        "failed": failed,
    }

    agg_path = output_root / "_aggregate.json"
    _write_atomic(agg_path, aggregate)

    total_elapsed = time.time() - batch_start
    print("=" * 60)
    print(f"Aggregate saved : {agg_path}")
    print(f"Projects done   : {aggregate['n_projects']}")
    print(f"Total wall time : {total_elapsed:.1f}s")
    ca = aggregate["corpus_duration_expansion_ratio"]
    if ca["mean"] is not None:
        print(f"Corpus ratio    : {ca['mean']:.4f}×  (std={ca['std']:.4f}  n={ca['n']})")
    else:
        print("Corpus ratio    : N/A")

    if failed:
        print(f"\n{len(failed)} project(s) failed:")
        for f in failed:
            print(f"  {f['project']}: {f['error']}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
