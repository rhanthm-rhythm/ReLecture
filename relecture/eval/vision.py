from __future__ import annotations

import json
from pathlib import Path

from ..storage import project_paths


VISION_JUDGE_RUBRIC = """You are evaluating a visual description extracted from a lecture slide for accessibility.
The description should help a visually impaired student understand the slide content.

Score on three dimensions. Return JSON only:
{{
  "completeness": <0-5>,
  "accuracy": <0-5>,
  "spoken_clarity": <0-5>
}}

Visual description:
{description}
"""


def _judge_vision(client, model: str, description: str) -> dict:
    prompt = VISION_JUDGE_RUBRIC.format(description=description)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
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


def run_vision_eval(project_file: str, vision_manifest_path_arg: str, judge_model: str | None = None) -> dict:
    """LLM-as-judge evaluation of VLM visual descriptions."""
    from openai import OpenAI
    from ..config import load_config
    from ..storage import load_vision_manifest

    config = load_config()
    client = OpenAI(api_key=config.llm.api_key, base_url=config.llm.base_url)
    model = judge_model or config.llm.model_name or "openai/gpt-oss-120b"

    paths = project_paths(project_file)
    vision_manifest = load_vision_manifest(vision_manifest_path_arg)

    results = []
    for result in vision_manifest.results:
        if not result.visual_context:
            continue
        scores = _judge_vision(client, model, result.visual_context)
        scores["segment_id"] = result.segment_id
        results.append(scores)

    numeric_keys = ["completeness", "accuracy", "spoken_clarity"]
    aggregates = {}
    for key in numeric_keys:
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if values:
            aggregates[key] = {"mean": sum(values) / len(values), "n": len(values)}

    output = {
        "project": project_file,
        "vision_model": vision_manifest.model_name,
        "aggregates": aggregates,
        "per_segment": results,
    }

    out_dir = Path(paths.project_dir) / "eval" / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = vision_manifest.model_name.replace("/", "-").replace(".", "-")
    out_path = out_dir / f"judge.{model_slug}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Vision eval saved to {out_path}")
    return output
