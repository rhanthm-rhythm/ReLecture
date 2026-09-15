from __future__ import annotations

import json
from pathlib import Path


def _collect_eval_files(dataset_dir: Path, eval_type: str, pattern: str) -> list[dict]:
    records = []
    for project_dir in sorted(dataset_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        eval_path = project_dir / "eval" / eval_type
        if not eval_path.exists():
            continue
        for json_file in sorted(eval_path.glob(pattern)):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                data["_source_file"] = str(json_file)
                data["_project_dir"] = str(project_dir)
                records.append(data)
            except Exception:
                pass
    return records


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": round(mean, 4), "std": round(variance ** 0.5, 4), "n": len(values)}


def _latex_row(label: str, values: dict) -> str:
    mean = values.get("mean")
    std = values.get("std")
    n = values.get("n", 0)
    if mean is None:
        return f"{label} & -- & -- & {n} \\\\"
    return f"{label} & {mean:.4f} & {std:.4f} & {n} \\\\"


def run_aggregate(dataset_dir: str, output_dir: str) -> None:
    dataset_path = Path(dataset_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Transcription WER
    wer_records = _collect_eval_files(dataset_path, "transcription", "wer.json")
    wer_values = [r["aggregate_wer"] for r in wer_records if isinstance(r.get("aggregate_wer"), float)]
    wer_summary = {
        "aggregate": _mean_std(wer_values),
        "per_project": [
            {"project": r["_project_dir"], "wer": r.get("aggregate_wer")} for r in wer_records
        ],
    }
    (out_path / "transcription_wer.json").write_text(json.dumps(wer_summary, indent=2), encoding="utf-8")

    # Transformation judge scores
    transform_records = _collect_eval_files(dataset_path, "transformation", "judge.*.json")
    by_strategy: dict[str, dict[str, list[float]]] = {}
    for r in transform_records:
        strategy = r.get("strategy", "unknown")
        if strategy not in by_strategy:
            by_strategy[strategy] = {}
        for key, val in (r.get("aggregates") or {}).items():
            by_strategy[strategy].setdefault(key, []).append(val.get("mean", 0))
    transform_summary = {
        strategy: {key: _mean_std(vals) for key, vals in metrics.items()}
        for strategy, metrics in by_strategy.items()
    }
    (out_path / "transformation_judge.json").write_text(json.dumps(transform_summary, indent=2), encoding="utf-8")

    # Synthesis similarity
    synth_records = _collect_eval_files(dataset_path, "synthesis", "quality.*.json")
    by_backend: dict[str, list[float]] = {}
    for r in synth_records:
        backend = r.get("backend", "unknown")
        sim = r.get("aggregates", {}).get("speaker_similarity", {}).get("mean")
        if sim is not None:
            by_backend.setdefault(backend, []).append(sim)
    synth_summary = {backend: _mean_std(vals) for backend, vals in by_backend.items()}
    (out_path / "synthesis_quality.json").write_text(json.dumps(synth_summary, indent=2), encoding="utf-8")

    # LaTeX tables
    latex_lines = [
        "% Transcription WER",
        "\\begin{tabular}{lrrr}",
        "\\hline",
        "Metric & Mean & Std & N \\\\",
        "\\hline",
        _latex_row("WER", wer_summary["aggregate"]),
        "\\hline",
        "\\end{tabular}",
        "",
        "% Transformation quality by strategy",
        "\\begin{tabular}{llrrr}",
        "\\hline",
        "Strategy & Metric & Mean & Std & N \\\\",
        "\\hline",
    ]
    for strategy, metrics in transform_summary.items():
        for metric, vals in metrics.items():
            mean = vals.get("mean")
            std = vals.get("std")
            n = vals.get("n", 0)
            mean_s = f"{mean:.4f}" if mean is not None else "--"
            std_s = f"{std:.4f}" if std is not None else "--"
            latex_lines.append(f"{strategy} & {metric} & {mean_s} & {std_s} & {n} \\\\")
    latex_lines += [
        "\\hline",
        "\\end{tabular}",
        "",
        "% Synthesis speaker similarity",
        "\\begin{tabular}{lrrr}",
        "\\hline",
        "Backend & Mean & Std & N \\\\",
        "\\hline",
    ]
    for backend, vals in synth_summary.items():
        latex_lines.append(_latex_row(backend, vals))
    latex_lines += ["\\hline", "\\end{tabular}"]

    (out_path / "tables.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    print(f"Aggregation complete. Results in {out_path}")
