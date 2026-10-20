#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export ARROW_DEFAULT_MEMORY_POOL=system
for f in jobs/campaign_core.sh jobs/campaign_small.sh jobs/campaign_frontiers.sh jobs/campaign_rest.sh; do
  [ -f "$f" ] || { echo "missing $f — run ./cluster_prepare.sh first"; exit 1; }
  bash "$f"
done
echo "resubmitted all campaign jobs (completed slices no-op via --skip-existing)."
echo "If the auto-finish job is no longer queued (check qstat), resubmit it:"
echo "  grid_run --grid_submit=batch --grid_ncpus=8 --grid_mem=64G ./cluster_finish.sh"
