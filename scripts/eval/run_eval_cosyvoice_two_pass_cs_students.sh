#!/bin/bash
cd "$(dirname "$0")/../.."
exec uv run python scripts/eval_runner.py \
    --backend cosyvoice --strategy two_pass \
    --audience-slug cs_students \
    --target-audience "students with Computer Science background" \
    --accessibility none \
    "$@"
