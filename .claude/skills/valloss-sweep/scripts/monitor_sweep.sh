#!/bin/bash
# monitor_sweep.sh [GRID_TAG] — one-shot progress snapshot for a running sweep.
# Prints the SLURM queue plus, per replay log, how many ensemble eval points are
# done and whether the replay finished. Run repeatedly (or under /loop) to watch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO"
TAG="${1:-}"

echo "=== queue ($(squeue -u "$USER" -h | wc -l) jobs) ==="
squeue -u "$USER" -o "%.12i %.22j %.9T %.8M %R" 2>/dev/null | grep -E "JOBID|fd_train|fd_replay" || echo "  (no train/replay jobs)"

echo
echo "=== replay progress (ens=2 line count per task) ==="
shopt -s nullglob
logs=(experiments/logs/fd_replay*_*.out)
[ -n "$TAG" ] && logs=(experiments/logs/fd_replay*"$TAG"*_*.out experiments/logs/fd_replay20m_*_*.out)
done=0; total=0
for f in $(ls -t experiments/logs/fd_replay*_*.out 2>/dev/null | head -40); do
  total=$((total+1))
  c=$(grep -c 'Fused replay complete' "$f" 2>/dev/null || true)
  done=$((done+c))
  printf "  %-46s pts=%-4s done=%s\n" "$(basename "$f")" "$(grep -c '\[step [0-9]* ens=2\]\|\[epoch [0-9]* ens=2\]' "$f" 2>/dev/null || echo 0)" "$c"
done
echo "  --> $done complete (of $total replay logs shown)"

echo
echo "=== training: resume + epoch evidence (latest logs) ==="
grep -hE "Resumed weights from|SKIP: final|Epoch [0-9]+/[0-9]+|Traceback|Error" \
  $(ls -t experiments/logs/fd_train_*_*.out 2>/dev/null | head -10) 2>/dev/null \
  | grep -E "Resumed weights|SKIP|Traceback|Error" | sort -u | head -12 || echo "  (no training logs yet)"
