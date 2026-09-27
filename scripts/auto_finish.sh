#!/usr/bin/env bash
# Unattended: wait for round 4, run round 5 (bigger stage 2), keep the best.
set -uo pipefail
cd ~/Business-entity-challenge
CFG=configs/aws_cpu.yaml
ts() { date '+%H:%M:%S'; }

# memory watchdog: kill stage2 before the instance freezes
( while sleep 10; do
    avail=$(awk '/MemAvailable/{a=$2} /MemTotal/{t=$2} END{print int(100*a/t)}' /proc/meminfo)
    if [ "$avail" -lt 10 ]; then
      echo "[$(ts)] WATCHDOG: ${avail}% RAM free -> killing stage2"
      pkill -9 -f stage2.py || true
    fi
  done ) &
WD=$!
trap 'kill $WD 2>/dev/null || true' EXIT

echo "[$(ts)] waiting for round 4 to finish..."
while pgrep -f run_round4.sh >/dev/null; do sleep 60; done
echo "[$(ts)] round 4 finished:"
grep -E "stage-1 OOF|outside top-k|ALL DONE|Error|Traceback" round4.log | tail -8
python scripts/pick_best.py

if [ -f artifacts/models/verifier_v4/stage1_fold2.txt ]; then
  echo "[$(ts)] ROUND 5: stage 2 big (topk 50, 1500 trees, 192 leaves)"
  if python scripts/stage2.py train --config $CFG --model-dir models/verifier_v5 \
        --reuse-folds models/verifier_v4 --topk 50 --n-estimators 1500 \
        --leaves 192 --n-jobs 8 \
     && python scripts/stage2.py predict --config $CFG --model-dir models/verifier_v5 \
        --topk 50 --n-jobs 8 \
     && python scripts/validate_submission.py --config $CFG; then
    python scripts/diag_by_country.py
    mkdir -p submissions/step5_stage2_big
    cp output/*.tsv submissions/step5_stage2_big/
    echo "[$(ts)] round 5 done"
  else
    echo "[$(ts)] round 5 FAILED (kept earlier submissions)"
  fi
else
  echo "[$(ts)] no v4 fold models -> skipping round 5"
fi

python scripts/pick_best.py
echo "[$(ts)] FINISHED -> upload submissions/best/matching_results.tsv (see SOURCE.txt)"
