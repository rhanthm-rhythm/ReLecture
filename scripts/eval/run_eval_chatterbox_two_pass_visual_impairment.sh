#!/bin/bash
cd "$(dirname "$0")/../.."
exec uv run python scripts/eval_runner.py \
    --backend chatterbox --strategy two_pass \
    --audience-slug visual_impairment \
    --target-audience "visually impaired students" \
    --accessibility visual_impairment \
    --describe-visuals \
    "$@"
