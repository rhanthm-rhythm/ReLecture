from __future__ import annotations

"""Batch pipeline runner for the Underline dataset.

Usage:
    python scripts/batch_run.py \
        --dataset data/dataset/dataset_index.json \
        --plan plans/full_eval.json \
        --output-root data/processed/
"""

import argparse
import json
import sys
import traceback
from pathlib import Path


def load_plan(plan_path: str) -> dict:
    return json.loads(Path(plan_path).read_text(encoding="utf-8"))


def run_item(item: dict, plan: dict, output_root: Path) -> dict:
    from relecture.stages.segment import run_segmentation
    from relecture.stages.transcribe import run_transcription
    from relecture.stages.transform import run_transformation
    from relecture.stages.synthesize import run_synthesis

    video_path = item.get("files", {}).get("video")
    if not video_path or not Path(video_path).exists():
        return {"status": "skip", "reason": "video not found"}

    lecture_id = item.get("lecture_id") or item.get("asset_hash", "unknown")
    project_dir = output_root / str(lecture_id)
    project_file = str(project_dir / "project.json")

    stages = plan.get("stages", ["segment", "transcribe", "transform", "synthesize"])

    if "segment" in stages:
        run_segmentation(project_file, source_video=video_path)

    if "transcribe" in stages:
        run_transcription(project_file)

    transform_cfg = plan.get("transformation", {})
    if "transform" in stages:
        strategies = transform_cfg.get("strategies", ["independent"])
        profiles = transform_cfg.get("profiles", [{"background": "none", "accessibility": "none"}])
        for strategy in strategies:
            for profile in profiles:
                run_transformation(
                    project_file,
                    background_profile=profile.get("background", "none"),
                    accessibility_profile=profile.get("accessibility", "none"),
                    target_audience=profile.get("target_audience", "general audience"),
                    strategy=strategy,
                )

    synth_cfg = plan.get("synthesis", {})
    if "synthesize" in stages:
        backends = synth_cfg.get("backends", ["qwen3"])
        mode = synth_cfg.get("mode", "full")
        for backend in backends:
            run_synthesis(project_file, mode=mode, backend=backend)

    return {"status": "ok"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch pipeline runner.")
    parser.add_argument("--dataset", required=True, help="Path to dataset_index.json")
    parser.add_argument("--plan", required=True, help="Path to pipeline plan JSON")
    parser.add_argument("--output-root", default="data/processed", help="Root dir for processed projects")
    parser.add_argument("--limit", type=int, help="Process only first N items")
    args = parser.parse_args(argv)

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    plan = load_plan(args.plan)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    items = dataset[:args.limit] if args.limit else dataset
    results = []
    ok = skip = error = 0

    for index, item in enumerate(items, start=1):
        label = item.get("title") or item.get("asset_hash", "?")
        print(f"[{index}/{len(items)}] {label}")
        try:
            result = run_item(item, plan, output_root)
        except Exception as exc:
            result = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
            error += 1
        else:
            if result["status"] == "ok":
                ok += 1
            else:
                skip += 1
        result["item"] = item.get("lecture_id") or item.get("asset_hash")
        results.append(result)

    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDone: ok={ok} skip={skip} error={error}  summary={summary_path}")
    return 0 if error == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
