#!/bin/bash
# Run all 6 configs for chatterbox (localhost:5002) sequentially.
set -o pipefail
cd "$(dirname "$0")/../.."

echo "=== CHATTERBOX EVAL: starting all 6 configs ==="
echo "Started: $(date)"

for script in \
    run_eval_chatterbox_independent_cs_students.sh \
    run_eval_chatterbox_independent_linguistics_students.sh \
    run_eval_chatterbox_independent_visual_impairment.sh \
    run_eval_chatterbox_two_pass_cs_students.sh \
    run_eval_chatterbox_two_pass_linguistics_students.sh \
    run_eval_chatterbox_two_pass_visual_impairment.sh \
; do
    echo ""
    echo ">>> Starting: $script ($(date))"
    bash "scripts/eval/$script"
    rc=$?
    echo ">>> Finished: $script (exit $rc, $(date))"
done

echo ""
echo "=== CHATTERBOX EVAL: all 6 configs done ==="
echo "Finished: $(date)"
