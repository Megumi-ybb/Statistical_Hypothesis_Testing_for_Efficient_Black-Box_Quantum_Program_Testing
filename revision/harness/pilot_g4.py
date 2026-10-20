
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness
from harness import runner, submit_grid
from harness.runner import Cell

HARNESS_DIR = Path(__file__).resolve().parent

DEFAULT_CIRCUIT = "QPE"
DEFAULT_CAMPAIGN = "I4E1"

DEFAULT_D = max(runner.JOINT_D)

LOCAL_BUDGET_RUNS = 5_000

SMOKE_N_REPS = 50
SMOKE_M = (50, 200)
SMOKE_D = 3




def applicable_operator(circuit: str, d: int, operators: Sequence[str]) -> str:
    from harness import base_circuits, mutants

    spec = base_circuits.build(circuit, d, seed=0)
    for operator in operators:
        try:
            mutants.mutate(spec, operator, rng=np.random.default_rng(0))
        except mutants.OperatorNotApplicable:
            continue
        return operator
    raise RuntimeError(f"no operator applies to {circuit} at d={d}")


def largest_supported_d(circuit: str, d_max: int) -> int:
    from harness import base_circuits

    for d in range(d_max, 1, -1):
        try:
            base_circuits.build(circuit, d, seed=0)
        except base_circuits.UnsupportedWidth:
            continue
        return d
    raise RuntimeError(f"{circuit} builds at no width <= {d_max}")


def worst_case_d(campaign: str, circuit: str) -> int:
    cells = submit_grid.campaign_cells(campaign, circuit=circuit)
    ds = [c.d for c in cells if c.paradigm == "joint"] or [c.d for c in cells]
    if not ds:
        raise SystemExit(f"{campaign} contains no {circuit} cells")
    d = max(ds)
    return d if submit_grid.cell_source() == "campaigns" else largest_supported_d(circuit, d)


def worst_case_operator(campaign: str, circuit: str, d: int) -> str:
    cells = submit_grid.campaign_cells(campaign, circuit=circuit, d=d)
    operators = sorted({c.operator for c in cells if c.operator != runner.BASELINE_OPERATOR})
    if submit_grid.cell_source() == "campaigns" and operators:
        return operators[0]
    return applicable_operator(circuit, d, operators or list(runner.OPERATORS))


def companion_chunk(campaign: str, primary: Sequence[Cell], *, n_reps: int) -> list[Cell]:
    have = max((len(c.methods) for c in primary if c.paradigm == "marginal"), default=0)
    pool = [c for c in submit_grid.campaign_cells(campaign, n_reps=n_reps)
            if c.paradigm == "marginal"]
    if not pool:
        return []
    richest = max(len(c.methods) for c in pool)
    if have >= richest:
        return []

    best = [c for c in pool if len(c.methods) == richest]
    circuit = sorted({c.circuit for c in best})[0]
    d = max(c.d for c in best if c.circuit == circuit)
    candidates = [c for c in best if c.circuit == circuit and c.d == d]
    operator = sorted({c.operator for c in candidates})[0]
    return [c for c in candidates if c.operator == operator]


def pilot_cells(
    *,
    campaign: str,
    circuit: str,
    d: int,
    operator: str,
    n_reps: int,
    m_points: Sequence[int] | None = None,
) -> list[Cell]:
    cells = submit_grid.campaign_cells(
        campaign, circuit=circuit, d=d, operator=operator, n_reps=n_reps
    )
    if m_points is not None:
        keep = set(m_points)
        cells = [c for c in cells if c.m in keep]
    if not cells:
        raise SystemExit(
            f"no cells for {campaign} {circuit} d={d} {operator}"
            + (f" m in {sorted(m_points)}" if m_points else "")
        )
    keys = {c.job_key for c in cells}
    if len(keys) != 1:
        raise SystemExit(f"pilot must be ONE chunk, got {len(keys)}: {sorted(map(str, keys))}")
    return cells




def _rows(results, *, job_wall_s: float, n_jobs: int, meta: Mapping[str, object],
          chunk: str = "primary") -> list[dict]:
    rows: list[dict] = []
    for res in results:
        cell = res.cell
        per_run_cell = res.wall_s / cell.n_reps if cell.n_reps else float("nan")
        for name, mres in res.methods.items():
            rows.append(
                {
                    **meta,
                    "chunk": chunk,
                    "campaign": cell.campaign,
                    "circuit": cell.circuit,
                    "d": cell.d,
                    "m": cell.m,
                    "paradigm": cell.paradigm,
                    "operator": cell.operator,
                    "case": cell.case,
                    "n_reps": cell.n_reps,
                    "method": name,
                    "TP": mres.TP,
                    "FP": mres.FP,
                    "TN": mres.TN,
                    "FN": mres.FN,
                    "mean_p": mres.mean_p,
                    "method_wall_s": mres.wall_s,
                    "cell_wall_s": res.wall_s,
                    "s_per_run_cell": per_run_cell,
                    "s_per_call_method": (
                        mres.wall_s / mres.n_evaluated if mres.n_evaluated else float("nan")
                    ),
                    "job_wall_s": job_wall_s,
                    "n_jobs": n_jobs,
                }
            )
    return rows


def cost_model(rows: Sequence[Mapping]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    seen: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (row["paradigm"], int(row["m"]))
        seen[key] = max(seen.get(key, 0.0), float(row["s_per_run_cell"]))
    model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for paradigm in sorted({p for p, _ in seen}):
        ms = np.array(sorted(m for p, m in seen if p == paradigm), dtype=float)
        cost = np.array([seen[(paradigm, int(m))] for m in ms], dtype=float)
        model[paradigm] = (ms, cost)
    return model


def is_sequential(cell: Cell) -> bool:
    mod = submit_grid.campaigns_module()
    if mod is None:
        return False
    if hasattr(mod, "is_sequential_cell"):
        return bool(mod.is_sequential_cell(cell))
    return cell.campaign in getattr(mod, "SEQUENTIAL_CAMPAIGNS", frozenset())


def price(cells: Sequence[Cell], model: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> float:
    total = 0.0
    for cell in cells:
        entry = model.get(cell.paradigm)
        if entry is None or is_sequential(cell):
            return float("nan")
        ms, cost = entry
        total += cell.n_reps * float(np.interp(cell.m, ms, cost))
    return total


def plan_price(cells: Sequence[Cell], derate: float) -> float:
    total = 0.0
    for cell in cells:
        per_run = submit_grid.PLAN_S_PER_RUN.get(cell.paradigm)
        if per_run is None or is_sequential(cell):
            return float("nan")
        total += cell.n_reps * per_run * derate
    return total




def schedule(
    model: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    derate: float,
    n_reps: int,
    campaigns: Sequence[str] | None = None,
    measured_methods: Iterable[str] = (),
) -> list[dict]:
    names = list(campaigns or submit_grid.CAMPAIGN_ORDER)
    measured = set(measured_methods)
    out: list[dict] = []
    for campaign in names:
        row = {
            "day": submit_grid.day_of(campaign),
            "campaign": campaign,
            "source": submit_grid.cell_source(),
            "cells": 0,
            "runs": 0,
            "core_s_measured": float("nan"),
            "core_s_plan": float("nan"),
            "status": "",
        }
        if campaign in submit_grid.SCRIPT_CAMPAIGNS:
            jobs = submit_grid.i5_jobs(n_reps=n_reps)
            row.update(
                cells=sum(j.n_cells for j in jobs),
                runs=sum(j.n_runs for j in jobs),
                source="noisy_grover",
                status="aer-bound: not priced by the count-kernel pilot",
            )
            out.append(row)
            continue
        try:
            cells = submit_grid.campaign_cells(campaign, n_reps=n_reps)
        except KeyError:
            row["status"] = "pending campaigns.py"
            out.append(row)
            continue
        row.update(
            cells=len(cells),
            runs=runner.total_runs(cells),
            core_s_measured=price(cells, model),
            core_s_plan=plan_price(cells, derate),
        )
        notes = []
        if any(is_sequential(c) for c in cells):
            notes.append("sequential: E[N]-driven, not priced off the m-ladder")
        elif any(c.m > max(model[c.paradigm][0]) for c in cells if c.paradigm in model):
            notes.append("m beyond the pilot ladder: cost clamped (under-estimate)")
        if measured:
            unmeasured = sorted({m for c in cells for m in c.methods} - measured)
            if unmeasured:
                notes.append(f"methods the pilot did not time: {', '.join(unmeasured)}")
        row["status"] = "; ".join(notes)
        out.append(row)
    return out


def format_schedule(rows: Sequence[Mapping], *, cores: int) -> list[str]:
    lines = [
        f"{'day':<12} {'campaign':<9} {'cells':>7} {'runs':>12} {'measured':>10} "
        f"{'planned':>10} {'wall':>8}  status",
        "-" * 92,
    ]
    day_totals: dict[str, float] = {}
    for row in rows:
        measured = float(row["core_s_measured"])
        planned = float(row["core_s_plan"])
        wall = measured / 3600 / cores if measured == measured else float("nan")
        if measured == measured:
            day_totals[row["day"]] = day_totals.get(row["day"], 0.0) + measured
        lines.append(
            f"{row['day']:<12} {row['campaign']:<9} {row['cells']:>7,d} {row['runs']:>12,d} "
            f"{_h(measured):>10} {_h(planned):>10} {_w(wall):>8}  {row['status']}"
        )
    lines.append("-" * 92)
    for day, core_s in day_totals.items():
        lines.append(
            f"{day}: {core_s / 3600:,.0f} grid-core-h = {core_s / 3600 / cores:.2f} h wall "
            f"on {cores:,d} cores at full utilisation"
        )
    return lines


def _h(seconds: float) -> str:
    return "n/a" if seconds != seconds else f"{seconds / 3600:,.1f} h"


def _w(hours: float) -> str:
    return "n/a" if hours != hours else f"{hours:.2f} h"




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python harness/pilot_g4.py",
        description="Gate G4: measure one worst-case chunk on the grid and re-derive "
                    "the Day-2/3 schedule.",
    )
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--circuit", default=DEFAULT_CIRCUIT)
    parser.add_argument("--d", type=int,
                        help=f"width (default: the largest the campaign contains for "
                             f"--circuit, at most {DEFAULT_D})")
    parser.add_argument("--operator", default="auto",
                        help="operator, or 'auto' for the first the campaign runs")
    parser.add_argument("--m", type=int, nargs="*", help="override the m-ladder")
    parser.add_argument("--n-reps", type=int, default=runner.DEFAULT_N_REPS)
    parser.add_argument("--alpha", type=float, default=runner.DEFAULT_ALPHA)
    parser.add_argument("--n-jobs", type=int, default=-1, help="joblib n_jobs (default -1)")
    parser.add_argument("--cores", type=int, default=1000,
                        help="grid size for the wall-clock projection (default 1000)")
    parser.add_argument("--derate", type=float, default=submit_grid.DEFAULT_DERATE,
                        help="the ASSUMED derating this gate replaces "
                             f"(default {submit_grid.DEFAULT_DERATE})")
    parser.add_argument("--out", type=Path, default=HARNESS_DIR / "pilot_report.csv")
    parser.add_argument("--schedule-out", type=Path,
                        help="default: pilot_schedule.csv beside --out")
    parser.add_argument("--no-companion", action="store_true",
                        help="skip the marginal companion chunk (see companion_chunk)")
    parser.add_argument("--cluster", action="store_true",
                        help="CLUSTER ONLY: lift the local run-count guard")
    parser.add_argument("--smoke", action="store_true",
                        help=f"local validation only: d={SMOKE_D}, m={SMOKE_M}, "
                             f"{SMOKE_N_REPS} reps")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    dist_provider: Callable[[Cell], tuple] | None = None,
    method_impls: Mapping[str, Callable[..., dict]] | None = None,
    draw_fn: Callable[..., tuple] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    injected = dist_provider is not None

    n_reps = SMOKE_N_REPS if args.smoke else args.n_reps
    m_points = tuple(args.m) if args.m else (SMOKE_M if args.smoke else None)
    if args.d is not None:
        d = args.d
    elif args.smoke or injected:
        d = SMOKE_D
    else:
        d = worst_case_d(args.campaign, args.circuit)

    operator = args.operator
    if operator == "auto":
        operator = (
            runner.OPERATORS[0] if injected
            else worst_case_operator(args.campaign, args.circuit, d)
        )

    cells = pilot_cells(
        campaign=args.campaign,
        circuit=args.circuit,
        d=d,
        operator=operator,
        n_reps=n_reps,
        m_points=m_points,
    )
    companion = [] if args.no_companion else companion_chunk(
        args.campaign, cells, n_reps=n_reps
    )
    if companion and m_points is not None:
        companion = [c for c in companion if c.m in set(m_points)]

    planned_runs = runner.total_runs(cells) + runner.total_runs(companion)
    print(f"G4 pilot chunk: {args.campaign} {args.circuit} d={d} {operator} -- "
          f"{len(cells)} cells, {runner.total_runs(cells):,d} runs, n_jobs={args.n_jobs}")
    for cell in cells:
        print(f"  {cell.label()}")

    if planned_runs > LOCAL_BUDGET_RUNS and not args.cluster:
        print(
            f"\nREFUSED: {planned_runs:,d} runs exceeds the local budget of "
            f"{LOCAL_BUDGET_RUNS:,d}.\n"
            "  --smoke     tiny chunk, runs here in seconds\n"
            "  --cluster   the real gate -- CLUSTER ONLY, do not run on the laptop",
            file=sys.stderr,
        )
        return 2

    meta = {
        "run_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 0,
        "python_version": ".".join(str(v) for v in sys.version_info[:3]),
        "harness_version": harness.__version__,
        "smoke": bool(args.smoke),
    }

    hooks = {"dist_provider": dist_provider, "cell_runner": None}
    if dist_provider is None:
        mod = submit_grid.campaigns_module()
        if mod is not None:
            hooks = {"dist_provider": mod.distributions, "cell_runner": mod.run_any_cell}

    def execute(chunk_cells, label):
        t0 = time.perf_counter()
        out, _ = runner.run_job(
            chunk_cells,
            None,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            method_impls=method_impls,
            draw_fn=draw_fn,
            **hooks,
        )
        wall = time.perf_counter() - t0
        print(f"chunk {label}: {wall:,.1f} s wall, "
              f"{sum(r.wall_s for r in out):,.1f} core-s over {len(chunk_cells)} cells")
        return out, wall

    results, job_wall_s = execute(cells, "primary")
    rows = _rows(results, job_wall_s=job_wall_s, n_jobs=args.n_jobs, meta=meta)

    if companion:
        key = companion[0]
        print(f"\ncompanion chunk (the primary's marginal cells drop methods this "
              f"campaign runs elsewhere): {key.circuit} d={key.d} {key.operator}, "
              f"{len(companion)} cells, {runner.total_runs(companion):,d} runs, "
              f"methods={key.methods}")
        comp_results, comp_wall = execute(companion, "companion")
        rows += _rows(comp_results, job_wall_s=comp_wall, n_jobs=args.n_jobs,
                      meta=meta, chunk="companion")
        results = list(results) + list(comp_results)
        job_wall_s += comp_wall

    core_s = sum(r.wall_s for r in results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")

    print(f"total {job_wall_s:,.1f} s wall; {core_s:,.1f} core-s over "
          f"{len(results)} cells; parallel speed-up {core_s / job_wall_s:,.1f}x on "
          f"{meta['cpu_count']} cpus")

    model = cost_model(rows)
    print("\nmeasured cost per run (single grid core), by paradigm and m:")
    for paradigm, (ms, cost) in model.items():
        print(f"  {paradigm:<8} " + "  ".join(f"m={int(m)}:{c * 1000:,.1f}ms"
                                              for m, c in zip(ms, cost)))
        plan = submit_grid.PLAN_S_PER_RUN.get(paradigm)
        if plan:
            mean = float(np.mean(cost))
            print(f"  {'':<8} mean {mean * 1000:,.1f} ms vs plan "
                  f"{plan * 1000:,.0f} ms x {args.derate:g} = "
                  f"{plan * args.derate * 1000:,.0f} ms -> MEASURED derating "
                  f"{mean / plan:,.2f}x")

    rows_sched = schedule(
        model,
        derate=args.derate,
        n_reps=args.n_reps,
        campaigns=None,
        measured_methods={r["method"] for r in rows},
    )
    print()
    for line in format_schedule(rows_sched, cores=args.cores):
        print(line)

    sched_path = args.schedule_out or args.out.with_name("pilot_schedule.csv")
    with sched_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_sched[0]))
        writer.writeheader()
        writer.writerows(rows_sched)
    print(f"\nwrote {sched_path} ({len(rows_sched)} campaigns)")

    derates = [
        float(np.mean(cost)) / submit_grid.PLAN_S_PER_RUN[p]
        for p, (_, cost) in model.items()
        if p in submit_grid.PLAN_S_PER_RUN
    ]
    if derates:
        measured_derate = max(derates)
        print(f"\nRE-DERIVED DERATING: {measured_derate:.2f}x "
              f"(assumed {args.derate:g}x).  Regenerate the Day-2/3 jobs with:\n"
              f"  harness/.venv/bin/python harness/submit_grid.py --derate "
              f"{measured_derate:.2f} --campaign {','.join(submit_grid.DAY_PLAN['Day 2'])} "
              f"--emit jobs/day2.sh")
    if args.smoke:
        print("\n--smoke: timings are laptop-scale and NOT a gate result; "
              "the schedule above is for wiring validation only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
