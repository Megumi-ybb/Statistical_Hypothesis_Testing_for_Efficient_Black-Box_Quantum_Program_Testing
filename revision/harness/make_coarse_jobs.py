#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from harness import campaigns, runner

campaigns.register_with_runner()

GROUPS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("campaign_core.sh", ("I4E1", "I4E1B"), True),
    ("campaign_frontiers.sh", ("I4E2", "I4E2b", "I4E3"), True),
    ("campaign_rest.sh", ("I9C3", "I7size", "I7seq", "I8"), False),
)
I5_FILE = "campaign_small.sh"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ncpus", type=int, default=100)
    p.add_argument("--mem", default="200G")
    p.add_argument("--out-dir", default="jobs")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (f"grid_run --grid_submit=batch --grid_ncpus={args.ncpus} "
              f"--grid_mem={args.mem} ./grid_worker.sh runner")

    total = 0
    for fname, camps, per_circuit in GROUPS:
        lines: list[str] = []
        for camp in camps:
            cells = list(campaigns.cells(camp))
            if per_circuit:
                for circ in sorted({c.circuit for c in cells}):
                    n = sum(1 for c in cells if c.circuit == circ)
                    lines.append(
                        f"{prefix} --campaign {camp} --circuit {circ} "
                        f"--out results/{camp} --n-reps 1000 --skip-existing"
                        f"   # {n} cells")
            else:
                lines.append(
                    f"{prefix} --campaign {camp} --out results/{camp} "
                    f"--n-reps 1000 --skip-existing   # {len(cells)} cells")
        (out_dir / fname).write_text("\n".join(lines) + "\n")
        print(f"  {fname:<26} {len(lines)} job(s)")
        total += len(lines)

    n_i5 = len(__import__("harness.submit_grid", fromlist=["i5_jobs"]).i5_jobs())
    (out_dir / I5_FILE).write_text(
        f"grid_run --grid_submit=batch --grid_ncpus={args.ncpus} "
        f"--grid_mem={args.mem} ./i5_all.sh   # {n_i5} cells run inside\n")
    print(f"  {I5_FILE:<26} 1 job ({n_i5} cells inside)")
    print(f"TOTAL: {total + 1} coarse jobs, every one {args.ncpus} cores / {args.mem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
