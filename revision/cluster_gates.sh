#!/bin/bash
set -euo pipefail
export ARROW_DEFAULT_MEMORY_POOL=system
export LOKY_MAX_CPU_COUNT="${NSLOTS:-8}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
cd "$(dirname "$0")"
PY=harness/.venv/bin/python

echo "=== test suites (fast) ==="
./harness/run_tests.sh
echo "=== test suites (slow gate G2, ~20 min) ==="
./harness/run_tests.sh -m slow
echo "=== gate G3: oracle equality + golden cells ==="
$PY harness/g3_golden.py --part a --out harness/g3_report_oracle.csv
$PY harness/g3_golden.py --full --part b --jobs -1 --out harness/g3_report.csv
echo "=== applicability probe + manifest ==="
$PY -m harness.campaigns --probe --out harness/applicability.json
$PY -m harness.campaigns --manifest --out harness/manifest.txt
cat harness/manifest.txt
echo "=== gate G4 pilot (measures real throughput) ==="
$PY harness/pilot_g4.py --cluster --n-jobs -1 --cores 1000 --out harness/pilot_report.csv
echo "=== C3 classification + E3 shot-free screen (zero-compute companions) ==="
$PY harness/i9_c3.py --classify --manifest
$PY -m harness.campaigns --screen --out harness/e3_screen.csv
echo "ALL GATES DONE"
