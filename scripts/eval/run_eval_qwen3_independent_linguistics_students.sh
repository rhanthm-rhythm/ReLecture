#!/bin/bash
cd "$(dirname "$0")/../.."
exec uv run python scripts/eval_runner.py \
    --backend qwen3 --strategy independent \
    --audience-slug linguistics_students \
    --target-audience "students with linguistics background" \
    --accessibility none \
    "$@"
