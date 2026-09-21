#!/bin/bash
# Run all 6 configs for cosyvoice (localhost:5003) sequentially.
set -o pipefail
cd "$(dirname "$0")/../.."

echo "=== COSYVOICE EVAL: starting all 6 configs ==="
echo "Started: $(date)"

for script in \
    run_eval_cosyvoice_independent_cs_students.sh \
    run_eval_cosyvoice_independent_linguistics_students.sh \
    run_eval_cosyvoice_independent_visual_impairment.sh \
    run_eval_cosyvoice_two_pass_cs_students.sh \
    run_eval_cosyvoice_two_pass_linguistics_students.sh \
    run_eval_cosyvoice_two_pass_visual_impairment.sh \
; do
    echo ""
    echo ">>> Starting: $script ($(date))"
    bash "scripts/eval/$script"
    rc=$?
    echo ">>> Finished: $script (exit $rc, $(date))"
done

echo ""
echo "=== COSYVOICE EVAL: all 6 configs done ==="
echo "Finished: $(date)"
