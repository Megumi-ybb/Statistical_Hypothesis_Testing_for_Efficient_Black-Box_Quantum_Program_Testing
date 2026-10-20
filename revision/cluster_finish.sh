#!/bin/bash
set -euo pipefail
export ARROW_DEFAULT_MEMORY_POOL=system
export LOKY_MAX_CPU_COUNT="${NSLOTS:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
cd "$(dirname "$0")"
PY=harness/.venv/bin/python

campaigns=(I4E1 I4E1B I5 I8 I4E2 I4E2b I4E3 I9C3 I7size I7seq)
expected() {
  case "$1" in
    I4E1) echo 351 ;;  I4E1B) echo 58 ;;  I5) echo 108 ;;  I8) echo 9 ;;
    I4E2) echo 39 ;;   I4E2b) echo 39 ;;  I4E3) echo 289 ;;
    I9C3) echo 25 ;;   I7size) echo 116 ;; I7seq) echo 22 ;;
    *) echo 0 ;;
  esac
}
have() { ls "results/$1"/*.parquet 2>/dev/null | wc -l; }

deadline=$(( $(date +%s) + 96*3600 ))
while :; do
  missing=0; status=""
  for c in "${campaigns[@]}"; do
    w=$(expected "$c"); h=$(have "$c")
    status+="$c:$h/$w  "
    [ "$h" -lt "$w" ] && missing=1
  done
  echo "$(date '+%F %T')  $status" >> logs/finish_progress.txt
  [ "$missing" -eq 0 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    { echo "TIMEOUT after 96h — analysing partial results. Shortfall:"; echo "$status"; } > logs/INCOMPLETE_WARNING.txt
    break
  fi
  sleep 300
done

echo "=== completeness ==="
$PY -c "from harness import io_store; df = io_store.read_grid('results/I4E1'); print(df.shape); print(df.groupby('method')[['TP','FP','TN','FN']].sum())"
echo "=== re-analysis ==="
$PY -m harness.ci_reanalysis --campaign results/I4E1 --out-dir analysis_out
$PY -m harness.crossover_analysis --campaign results/I7size --out-dir analysis_out
$PY -m harness.i2_verification --out-dir analysis_out
$PY -m harness.r19_lumping_breakdown --out analysis_out/r19_lumping_breakdown.csv
echo "=== E[N] summary (I7seq: the D1 sequential-efficiency question) ==="
$PY -c "
from harness import io_store
df = io_store.read_grid('results/I7seq')
df = df[df['method'] != '_error']
t = df.groupby(['method', 'circuit', 'd'])[['n_used_mean']].mean().reset_index()
t.to_csv('analysis_out/en_summary.csv', index=False)
p = t.pivot_table(index=['circuit', 'd'], columns='method', values='n_used_mean')
print(p.round(1).to_string())
print()
for m in p.columns:
    print(f'{m}: overall mean E[N] = {p[m].mean():.1f}')
"
echo "=== figures (every campaign; families without data are skipped) ==="
for camp in results/*/; do
  $PY -m harness.figures --campaign "$camp" --out-dir figures || true
done
echo "FINISHED — see analysis_out/ and figures/"
echo
echo ">>> LAST STEP (manual): push the outputs so they leave the cluster:"
echo ">>>   git add -f analysis_out figures harness/manifest.txt harness/g3_report*.csv harness/pilot_report.csv"
echo ">>>   git commit -m 'campaign analysis results' && git push"
