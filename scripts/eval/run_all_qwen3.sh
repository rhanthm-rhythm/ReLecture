#!/bin/bash
# Run all 6 configs for qwen3 (localhost:5000) sequentially.
# Launch this once — it chains independent then two_pass for each audience.
set -o pipefail
cd "$(dirname "$0")/../.."

echo "=== QWEN3 EVAL: starting all 6 configs ==="
echo "Started: $(date)"

for script in \
    run_eval_qwen3_independent_cs_students.sh \
    run_eval_qwen3_independent_linguistics_students.sh \
    run_eval_qwen3_independent_visual_impairment.sh \
    run_eval_qwen3_two_pass_cs_students.sh \
    run_eval_qwen3_two_pass_linguistics_students.sh \
    run_eval_qwen3_two_pass_visual_impairment.sh \
; do
    echo ""
    echo ">>> Starting: $script ($(date))"
    bash "scripts/eval/$script"
    rc=$?
    echo ">>> Finished: $script (exit $rc, $(date))"
done

echo ""
echo "=== QWEN3 EVAL: all 6 configs done ==="
echo "Finished: $(date)"
