"""
Batch WER evaluation for the first N presentations (alphabetically) in a dataset directory.
Uses the running Whisper HTTP service (same one the pipeline uses).

Does NOT touch pipeline code. Only reads project manifests and chunk WAV files.

Results are written to:
  <output-root>/<videoID>/wer.chatterbox.json   — per-video WER + transcriptions
  <output-root>/_aggregate.json                 — corpus-level WER across all videos

Usage:
    uv run python scripts/run_wer_batch.py \\
        --dataset eval_result/chatterbox/independent/visual_impairment \\
        [--output-root eval_result/groundtruth/synthesis/transcription/whisper] \\
        [--limit 50] \\
        [--skip-existing] \\
        [--whisper-endpoint http://localhost:5001/transcribe]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "eval_result" / "groundtruth" / "synthesis" / "transcription" / "whisper"


def _wer_dict(acc: dict) -> dict:
    total = acc["S"] + acc["D"] + acc["I"]
    wer = total / acc["N"] if acc["N"] else 0.0
    return {
        "wer": round(wer, 4),
        "edits": total,
        "ref_words": acc["N"],
        "substitutions": acc["S"],
        "deletions": acc["D"],
        "insertions": acc["I"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch WER evaluation across a dataset of projects.")
    parser.add_argument("--dataset", required=True, help="Root dir containing project subdirs.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Directory for per-video result folders (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max projects to process (alphabetical order).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos that already have a result file.")
    parser.add_argument("--whisper-endpoint", default="http://localhost:5001/transcribe", help="Whisper service URL.")
    parser.add_argument("--tag", default="chatterbox", help="Model tag used in the output filename (e.g. qwen3).")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # First 50 alphabetically, skip _ prefixed dirs (aggregates etc.)
    projects = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith("_")],
        key=lambda d: d.name,
    )[: args.limit]

    print(f"Projects : {len(projects)} (first {args.limit} alphabetically)")
    print(f"Output   : {output_root}")
    print(f"Whisper  : {args.whisper_endpoint}")
    print()

    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    # Health check
    try:
        import requests
        health_url = args.whisper_endpoint.replace("/transcribe", "/health")
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
            print("Whisper service: OK\n")
        else:
            print(f"WARNING: Whisper health check returned {resp.status_code}\n", file=sys.stderr)
    except Exception as exc:
        print(f"WARNING: Whisper health check failed: {exc}\n", file=sys.stderr)

    from relecture.eval.wer import compute_wer

    all_results: list[dict] = []
    failed: list[dict] = []

    for idx, project_dir in enumerate(projects, 1):
        video_id = project_dir.name
        project_file = project_dir / "project.json"
        out_file = output_root / video_id / f"wer.{args.tag}.json"

        if not project_file.exists():
            print(f"[{idx:>2}/{len(projects)}] SKIP {video_id} — no project.json")
            continue

        if args.skip_existing and out_file.exists():
            try:
                result = json.loads(out_file.read_text(encoding="utf-8"))
                # Validate it has the required fields before trusting it
                _ = result["aggregates"]["wer"]["combined"]["wer"]
                print(f"[{idx:>2}/{len(projects)}] SKIP {video_id} — already done  WER={_:.4f}")
                all_results.append(result)
                continue
            except Exception as exc:
                print(f"[{idx:>2}/{len(projects)}] REPROCESS {video_id} — existing file invalid: {exc}")

        print(f"[{idx:>2}/{len(projects)}] {video_id}", flush=True)

        def _progress(msg: str, _vid=video_id) -> None:
            print(f"  {msg}", flush=True)

        try:
            result = compute_wer(
                str(project_file),
                whisper_endpoint=args.whisper_endpoint,
                progress_cb=_progress,
            )
            agg = result["aggregates"]["wer"]
            print(
                f"  DONE  combined={agg['combined']['wer']:.4f}  "
                f"lecture={agg['lecture']['wer']:.4f}  "
                f"narration={agg['narration']['wer']:.4f}"
            )

            out_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = out_file.with_suffix(".json.tmp")
            tmp_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            os.replace(tmp_file, out_file)  # atomic on NTFS
            all_results.append(result)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            failed.append({"project": video_id, "error": str(exc)})

        print()

    # Aggregate
    if not all_results:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    agg_corpus: dict[str, dict] = {k: {"S": 0, "D": 0, "I": 0, "N": 0} for k in ("lecture", "narration")}
    for result in all_results:
        for kind in ("lecture", "narration"):
            w = result["aggregates"]["wer"].get(kind, {})
            agg_corpus[kind]["S"] += w.get("substitutions", 0)
            agg_corpus[kind]["D"] += w.get("deletions", 0)
            agg_corpus[kind]["I"] += w.get("insertions", 0)
            agg_corpus[kind]["N"] += w.get("ref_words", 0)

    combined = {k: agg_corpus["lecture"][k] + agg_corpus["narration"][k] for k in ("S", "D", "I", "N")}
    aggregate = {
        "dataset": str(dataset_dir),
        "n_projects": len(all_results),
        "whisper_model": "base.en",
        "corpus_wer": {
            "lecture": _wer_dict(agg_corpus["lecture"]),
            "narration": _wer_dict(agg_corpus["narration"]),
            "combined": _wer_dict(combined),
        },
        "per_project": [
            {
                "video_id": Path(r["project"]).parent.name,
                "wer": r["aggregates"]["wer"]["combined"]["wer"],
                "lecture_wer": r["aggregates"]["wer"]["lecture"]["wer"],
                "narration_wer": r["aggregates"]["wer"]["narration"]["wer"],
            }
            for r in all_results
        ],
        "failed": failed,
    }

    agg_path = output_root / "_aggregate.json"
    agg_tmp = agg_path.with_suffix(".json.tmp")
    agg_tmp.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    os.replace(agg_tmp, agg_path)

    print("=" * 60)
    print(f"Aggregate saved : {agg_path}")
    print(f"Projects done   : {aggregate['n_projects']}")
    print(f"Combined WER    : {aggregate['corpus_wer']['combined']['wer']:.4f}")
    print(f"  lecture       : {aggregate['corpus_wer']['lecture']['wer']:.4f}")
    print(f"  narration     : {aggregate['corpus_wer']['narration']['wer']:.4f}")

    if failed:
        print(f"\n{len(failed)} project(s) failed:")
        for f in failed:
            print(f"  {f['project']}: {f['error']}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
