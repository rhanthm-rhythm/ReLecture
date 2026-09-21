#!/bin/bash
cd "$(dirname "$0")/../.."
exec uv run python scripts/eval_runner.py \
    --backend chatterbox --strategy independent \
    --audience-slug cs_students \
    --target-audience "students with Computer Science background" \
    --accessibility none \
    "$@"
