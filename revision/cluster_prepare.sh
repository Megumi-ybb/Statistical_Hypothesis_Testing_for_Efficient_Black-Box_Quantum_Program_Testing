#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export ARROW_DEFAULT_MEMORY_POOL=system

if [ ! -x harness/.venv/bin/python ]; then
  ln -sfn "$HOME/.conda/envs/quantum-testing" harness/.venv
fi
harness/.venv/bin/python -c "import numpy, scipy, pandas, pyarrow, joblib, matplotlib, qiskit, qiskit_aer; print('env OK, qiskit', qiskit.__version__)"

mkdir -p jobs results logs analysis_out figures
PY=harness/.venv/bin/python
$PY -c "from harness import submit_grid; submit_grid.write_worker_script()"
$PY -m harness.campaigns --probe --out harness/applicability.json
echo "job files ready:"
$PY -m harness.make_coarse_jobs --ncpus 100 --mem 200G --out-dir jobs
echo
echo "PREPARED. Now run:  ./cluster_submit_all.sh"
