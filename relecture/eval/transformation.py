from __future__ import annotations

import json
from pathlib import Path

from ..storage import project_paths, ensure_project_manifest, load_stage_manifest


JUDGE_RUBRIC = """You are evaluating an adapted lecture transcript. Score the adaptation on four dimensions.
Return JSON only, with exactly these keys and integer scores:

{{
  "semantic_faithfulness": <0-5>,
  "accessibility_improvement": <0-5>,
  "hallucination_free": <0 or 1>,
  "speech_readiness": <0-5>
}}

Original transcript:
{original}

Adapted transcript:
{adapted}

Target audience: {audience}
Accessibility profile: {accessibility}
"""


def _judge_segment(client, model: str, original: str, adapted: str, audience: str, accessibility: str) -> dict:
    prompt = JUDGE_RUBRIC.format(
        original=original, adapted=adapted, audience=audience, accessibility=accessibility
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as exc:
        return {"error": str(exc)}
    content = response.choices[0].message.content if response and response.choices else ""
    try:
        if "```json" in content:
            content = content[content.find("```json") + 7: content.rfind("```")].strip()
        return json.loads(content)
    except Exception:
        return {"raw": content}


def run_transformation_eval(project_file: str, judge_model: str | None = None) -> dict:
    """LLM-as-judge evaluation of transformation quality."""
    from openai import OpenAI
    from ..config import load_config

    config = load_config()
    client = OpenAI(api_key=config.llm.api_key, base_url=config.llm.base_url)
    model = judge_model or config.llm.model_name or "openai/gpt-oss-120b"

    project = ensure_project_manifest(project_file)
    paths = project_paths(project_file)
    manifest = load_stage_manifest(project, project_file, "transformation")

    results = []
    for segment in manifest.segments:
        original = (segment.clean_transcript or segment.raw_transcript or "").strip()
        adapted = (segment.transformed_transcript or "").strip()
        if not original or not adapted or original == adapted:
            continue
        scores = _judge_segment(client, model, original, adapted, manifest.target_audience, manifest.accessibility_profile)
        scores["segment_id"] = segment.id
        results.append(scores)

    numeric_keys = ["semantic_faithfulness", "accessibility_improvement", "hallucination_free", "speech_readiness"]
    aggregates = {}
    for key in numeric_keys:
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if values:
            aggregates[key] = {"mean": sum(values) / len(values), "n": len(values)}

    output = {
        "project": project_file,
        "strategy": manifest.strategy,
        "target_audience": manifest.target_audience,
        "describe_visuals": getattr(manifest, "describe_visuals", manifest.accessibility_profile == "visual_impairment"),
        "background_profile": manifest.background_profile,
        "accessibility_profile": manifest.accessibility_profile,
        "aggregates": aggregates,
        "per_segment": results,
    }

    out_dir = Path(paths.project_dir) / "eval" / "transformation"
    out_dir.mkdir(parents=True, exist_ok=True)
    strategy_slug = manifest.strategy or "independent"
    out_path = out_dir / f"judge.{strategy_slug}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Transformation eval saved to {out_path}")
    return output
