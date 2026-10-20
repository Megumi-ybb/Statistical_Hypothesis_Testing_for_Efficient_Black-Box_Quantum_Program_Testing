#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export ARROW_DEFAULT_MEMORY_POOL=system
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
mkdir -p results/I5

P=$(( ${NSLOTS:-8} < 36 ? ${NSLOTS:-8} : 36 ))

harness/.venv/bin/python -c "
import shlex
from harness import submit_grid as s
from pathlib import Path
for j in s.i5_jobs(worker_form='module'):
    if not Path(j.out_paths[0]).exists():
        print(shlex.join(list(j.argv)))
" > logs/i5_commands.txt

n=$(wc -l < logs/i5_commands.txt)
echo "I5: running $n cell(s) at concurrency $P"
xargs -P "$P" -I CMD bash -c 'CMD' < logs/i5_commands.txt
echo "I5 DONE: $(ls results/I5/*.parquet 2>/dev/null | wc -l)/108 parquet files"
