#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for f in jobs/campaign_core.sh jobs/campaign_small.sh jobs/campaign_frontiers.sh jobs/campaign_rest.sh; do
  [ -f "$f" ] || { echo "missing $f — run ./cluster_prepare.sh first"; exit 1; }
done

echo "submitting gates job..."
grid_run --grid_submit=batch --grid_ncpus=100 --grid_mem=200G ./cluster_gates.sh

echo "submitting all campaign jobs..."
bash jobs/campaign_core.sh
bash jobs/campaign_small.sh
bash jobs/campaign_frontiers.sh
bash jobs/campaign_rest.sh

echo "submitting auto-finish job (waits for outputs, then analysis + figures)..."
grid_run --grid_submit=batch --grid_ncpus=8 --grid_mem=64G ./cluster_finish.sh

echo
echo "ALL SUBMITTED. Walk away."
echo "  peek anytime:   qstat"
echo "  when the queue is empty, results are in analysis_out/ and figures/"
echo "  fetch from your Mac:  sftp <user>@<file-server>"
