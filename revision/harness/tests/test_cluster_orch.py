
from __future__ import annotations

import re
import shlex
import time
from pathlib import Path

import numpy as np
import pytest

from harness import pilot_g4, runner, submit_grid
from harness.runner import Cell

REVISION_DIR = Path(__file__).resolve().parents[2]
HARNESS_DIR = REVISION_DIR / "harness"
RUNBOOK = HARNESS_DIR / "RUNBOOK.md"




def test_generation_is_fast_and_covers_every_available_campaign():
    t0 = time.perf_counter()
    jobs, notes = submit_grid.build_jobs(submit_grid.available_campaigns())
    elapsed = time.perf_counter() - t0

    assert elapsed < 30.0, f"job generation took {elapsed:.1f} s"
    assert jobs, notes
    generated = {j.campaign for j in jobs}
    assert generated == set(submit_grid.available_campaigns()), notes


@pytest.mark.parametrize(
    "campaign",
    [c for c in submit_grid.available_campaigns() if c not in submit_grid.SCRIPT_CAMPAIGNS],
)
def test_job_counts_are_consistent_with_the_cell_source(campaign):
    cells = submit_grid.campaign_cells(campaign)
    jobs = submit_grid.cell_jobs(campaign, cells)

    keys = {c.job_key for c in cells}
    out_paths = [p for j in jobs for p in j.out_paths]
    assert sum(j.n_cells for j in jobs) == len(cells)
    assert sum(j.n_runs for j in jobs) == runner.total_runs(cells)
    assert len({j.label for j in jobs}) == len(jobs)
    assert len(set(out_paths)) == len(out_paths) == len(keys)


def test_i4e1_matches_the_plan_upper_bound():
    jobs = submit_grid.cell_jobs("I4E1", submit_grid.campaign_cells("I4E1"))
    n_cells, n_runs = sum(j.n_cells for j in jobs), sum(j.n_runs for j in jobs)
    if submit_grid.cell_source() == "campaigns":
        assert n_cells <= 8_400 and n_runs <= 8_400_000
    else:
        assert (n_cells, n_runs) == (8_400, 8_400_000)
        assert len(jobs) == 6 * 11 * 7


def test_counts_agree_with_the_campaigns_manifest_when_it_exists():
    rows = submit_grid.manifest_rows()
    if rows is None:
        pytest.skip("campaigns.py not written yet (RUNBOOK.md section 8)")
    cell_keys = ("cells", "n_cells", "realized_cells")
    run_keys = ("runs", "n_runs", "realized_runs")
    checked = 0
    for row in rows:
        campaign = row.get("campaign")
        if campaign is None or campaign in submit_grid.SCRIPT_CAMPAIGNS:
            continue
        try:
            jobs = submit_grid.cell_jobs(campaign, submit_grid.campaign_cells(campaign))
        except KeyError:
            continue
        for key in cell_keys:
            if isinstance(row.get(key), (int, np.integer)):
                assert sum(j.n_cells for j in jobs) == int(row[key]), campaign
                checked += 1
                break
        for key in run_keys:
            if isinstance(row.get(key), (int, np.integer)):
                assert sum(j.n_runs for j in jobs) == int(row[key]), campaign
                checked += 1
                break
    assert checked, f"manifest rows carried no recognisable counts: {rows[:1]}"


def test_commands_follow_the_repo_grid_idiom():
    jobs, _ = submit_grid.build_jobs(["I4E1", "I5"], limit=40)
    for job in jobs:
        tokens = shlex.split(job.command())
        assert tokens[0] == "grid_run"
        assert tokens[1] == "--grid_submit=batch"
        assert tokens[2].startswith("--grid_ncpus=")
        assert tokens[3].startswith("--grid_mem=")
        assert tokens[4] == "./" + submit_grid.WORKER_SCRIPT_NAME
        assert tokens[5] in ("runner", "noisy")
        assert "--out" in tokens

    fallback, _ = submit_grid.build_jobs(["I4E1"], limit=3, worker_form="module")
    for job in fallback:
        tokens = shlex.split(job.command())
        assert tokens[4].endswith("python")


def test_per_job_cores_never_exceed_the_cells_they_can_parallelise():
    jobs, _ = submit_grid.build_jobs(["I4E1", "I4E3"])
    for job in jobs:
        assert 1 <= job.ncpus <= min(job.n_cells, submit_grid.DEFAULT_NCPUS_CAP)
    forced, _ = submit_grid.build_jobs(["I4E3"], grid_ncpus=100, limit=3)
    assert all(j.ncpus == 100 for j in forced)


def _a_real_slice(campaign: str = "I4E1"):
    cell = submit_grid.campaign_cells(campaign)[0]
    return cell.circuit, cell.d, cell.operator


def test_filters_and_limit():
    circuit, d, operator = _a_real_slice()
    jobs, _ = submit_grid.build_jobs(["I4E1"], circuit=circuit, d=d, operator=operator)
    assert len(jobs) == 1
    assert jobs[0].label == f"I4E1__{circuit}__d{d:02d}__{operator}"
    assert "--circuit" in jobs[0].argv and circuit in jobs[0].argv

    limited, notes = submit_grid.build_jobs(["I4E1"], limit=5)
    assert len(limited) == 5
    assert any("withheld" in n for n in notes)

    missing, notes = submit_grid.build_jobs(["I4E1"], circuit="BV", operator="O1")
    assert missing == []
    assert any("no cells match" in n for n in notes)


def test_skip_existing_drops_finished_jobs(tmp_path):
    circuit, d, _ = _a_real_slice()
    jobs, _ = submit_grid.build_jobs(["I4E1"], circuit=circuit, d=d)
    for path in jobs[0].out_paths:
        done = tmp_path / path
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_bytes(b"")

    remaining, notes = submit_grid.build_jobs(
        ["I4E1"], circuit=circuit, d=d, skip_existing_root=tmp_path
    )
    assert len(remaining) == len(jobs) - 1
    assert jobs[0].label not in {j.label for j in remaining}
    assert any("already have their parquet" in n for n in notes)


def test_worker_forms():
    script, _ = submit_grid.build_jobs(["I8"])
    assert script
    for job in script:
        assert job.argv[0] == "./" + submit_grid.WORKER_SCRIPT_NAME
        assert job.argv[1] == "runner"

    module, _ = submit_grid.build_jobs(["I8"], worker_form="module")
    assert module
    for job in module:
        assert job.argv[1:3] == ("-m", "harness.runner")

    inline, _ = submit_grid.build_jobs(["I8"], worker_form="inline")
    for job in inline:
        bootstrap = job.argv[job.argv.index("-c") + 1]
        assert "campaigns.register_with_runner()" in bootstrap
        assert "runner.main()" in bootstrap

    worker = submit_grid.write_worker_script()
    text = worker.read_text()
    assert text.startswith("#!/bin/bash")
    assert "campaigns.register_with_runner()" in text
    assert "noisy_grover.py" in text
    assert worker.stat().st_mode & 0o111


def test_every_generated_campaign_is_runnable_by_the_worker():
    jobs, _ = submit_grid.build_jobs(submit_grid.available_campaigns())
    unverified = {j.campaign for j in jobs if not j.entry_verified}
    assert unverified <= set(submit_grid.SCRIPT_CAMPAIGNS), unverified


def test_synthetic_circuit_campaigns_chunk_per_shape_arm():
    jobs, _ = submit_grid.build_jobs(["I8"])
    cells = submit_grid.campaign_cells("I8")
    arms = {c.circuit for c in cells}
    assert arms - set(runner.CIRCUITS)
    assert len(jobs) == len({c.job_key for c in cells})
    assert {j.argv[j.argv.index("--circuit") + 1] for j in jobs} == arms
    assert all(len(j.out_paths) == 1 for j in jobs)
    assert sum(j.n_cells for j in jobs) == len(cells)


@pytest.mark.parametrize(
    "campaign,expect_cells",
    [("I8", 4), ("I4E1", 20)],
)
def test_emitted_command_enumerates_the_filtered_grid(campaign, expect_cells):
    import subprocess

    jobs, _ = submit_grid.build_jobs([campaign], limit=1)
    argv = list(jobs[0].argv)
    if argv[0].endswith(submit_grid.WORKER_SCRIPT_NAME):
        submit_grid.write_worker_script()
    argv[0] = str(REVISION_DIR / argv[0])
    out = subprocess.run(
        argv + ["--dry-run", "--list-limit", "0"],
        cwd=REVISION_DIR, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert f"{expect_cells} cells" in out.stdout, out.stdout[-2000:]
    assert out.stdout.count("master=") == expect_cells


def test_i5_jobs_cover_the_committed_noisy_grid():
    jobs = submit_grid.i5_jobs()
    assert len(jobs) == 108
    assert all(j.ncpus == 1 and j.n_cells == 1 for j in jobs)
    assert len({p for j in jobs for p in j.out_paths}) == 108
    for job in jobs:
        tokens = shlex.split(job.command())
        assert "./" + submit_grid.WORKER_SCRIPT_NAME in tokens
        assert "noisy" in tokens
        for flag in ("--arm", "--strength-level", "--m", "--n-reps", "--out"):
            assert flag in tokens, flag
    assert any("fake" in shlex.split(j.command()) for j in jobs)
    module_jobs = submit_grid.i5_jobs(worker_form="module")
    assert all("harness/noisy_grover.py" in shlex.split(j.command())
               for j in module_jobs)


def test_emitted_script_is_runnable_and_self_describing(tmp_path):
    circuit, d, _ = _a_real_slice()
    expected, _ = submit_grid.build_jobs(["I4E1"], circuit=circuit, d=d)
    out = tmp_path / "jobs" / "day2.sh"
    rc = submit_grid.main([
        "--campaign", "I4E1", "--circuit", circuit, "--d", str(d),
        "--emit", str(out), "--print-limit", "2",
    ])
    assert rc == 0
    text = out.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert out.stat().st_mode & 0o111
    assert 'cd "${REVISION_DIR:-' in text
    assert "mkdir -p results/I4E1" in text
    assert text.count("grid_run ") == len(expected)
    assert "idempotent" in text


def test_module_has_no_submission_path():
    source = (HARNESS_DIR / "submit_grid.py").read_text()
    for forbidden in (
        r"^\s*import\s+subprocess",
        r"^\s*from\s+subprocess",
        r"subprocess\.",
        r"os\.system\(",
        r"\.popen\(",
        r"os\.exec",
    ):
        assert re.search(forbidden, source, re.M) is None, forbidden




def _p0(d: int) -> dict[str, float]:
    keys = [format(i, f"0{d}b") for i in range(2 ** d)]
    w = np.arange(1, len(keys) + 1, dtype=float)
    return dict(zip(keys, w / w.sum()))


def stub_dist(cell: Cell):
    return _p0(cell.d), _p0(cell.d)


def stub_draw(p_draw, m, rng):
    keys = list(p_draw)
    probs = np.fromiter((p_draw[k] for k in keys), dtype=float, count=len(keys))
    probs = probs / probs.sum()
    full = rng.multinomial(m, probs)
    nz = np.flatnonzero(full)
    states = np.array([[int(c) for c in keys[i]] for i in nz], dtype=np.uint8)
    counts = full[nz].astype(np.int64)
    X = np.repeat(states, counts, axis=0)
    rng.shuffle(X, axis=0)
    return states, counts, X


def stub_stat(draw, *, alpha, rng):
    p = float(rng.random())
    return {"p_value": p, "reject": p <= alpha}


STUB_IMPLS = {name: stub_stat for name in runner.JOINT_METHODS + runner.MARGINAL_METHODS}


def test_pilot_smoke_writes_a_report_and_a_schedule(tmp_path):
    out = tmp_path / "pilot_report.csv"
    rc = pilot_g4.main(
        ["--smoke", "--n-jobs", "1", "--out", str(out)],
        dist_provider=stub_dist,
        method_impls=STUB_IMPLS,
        draw_fn=stub_draw,
    )
    assert rc == 0

    import csv

    primary = pilot_g4.pilot_cells(
        campaign="I4E1", circuit="QPE", d=pilot_g4.SMOKE_D, operator=runner.OPERATORS[0],
        n_reps=pilot_g4.SMOKE_N_REPS, m_points=pilot_g4.SMOKE_M,
    )
    companion = [
        c for c in pilot_g4.companion_chunk("I4E1", primary, n_reps=pilot_g4.SMOKE_N_REPS)
        if c.m in set(pilot_g4.SMOKE_M)
    ]
    assert companion, "QPE's degenerate null should trigger the marginal companion"

    rows = list(csv.DictReader(out.open()))
    assert len(rows) == sum(len(c.methods) for c in primary + companion)
    for column in ("s_per_run_cell", "cell_wall_s", "TP", "FN", "job_wall_s", "chunk"):
        assert column in rows[0]
    assert {r["chunk"] for r in rows} == {"primary", "companion"}
    for row in rows:
        assert int(row["n_reps"]) == pilot_g4.SMOKE_N_REPS
        assert int(row["TP"]) + int(row["FN"]) == pilot_g4.SMOKE_N_REPS

    schedule = tmp_path / "pilot_schedule.csv"
    sched_rows = list(csv.DictReader(schedule.open()))
    assert {r["campaign"] for r in sched_rows} == set(submit_grid.CAMPAIGN_ORDER)
    assert {r["day"] for r in sched_rows} <= set(submit_grid.DAY_PLAN) | {"unscheduled"}


def test_companion_covers_the_marginal_methods_the_primary_drops():
    primary = pilot_g4.pilot_cells(
        campaign="I4E1", circuit="QPE", d=10,
        operator=pilot_g4.worst_case_operator("I4E1", "QPE", 10), n_reps=1000,
    )
    companion = pilot_g4.companion_chunk("I4E1", primary, n_reps=1000)
    primary_methods = set().union(*(c.methods for c in primary if c.paradigm == "marginal"))
    companion_methods = set().union(*(c.methods for c in companion))
    assert "bootstrap" not in primary_methods
    assert "bootstrap" in companion_methods
    assert len({c.job_key for c in companion}) == 1


def test_pilot_refuses_a_full_chunk_without_the_cluster_flag(tmp_path):
    operator = pilot_g4.worst_case_operator("I4E1", "QPE", 10)
    rc = pilot_g4.main([
        "--circuit", "QPE", "--d", "10", "--operator", operator,
        "--out", str(tmp_path / "nope.csv"),
    ])
    assert rc == 2
    assert not (tmp_path / "nope.csv").exists()


def test_pilot_chunk_is_exactly_one_job_slice():
    d = pilot_g4.worst_case_d("I4E1", "QPE")
    operator = pilot_g4.worst_case_operator("I4E1", "QPE", d)
    cells = pilot_g4.pilot_cells(
        campaign="I4E1", circuit="QPE", d=d, operator=operator, n_reps=1000
    )
    assert len({c.job_key for c in cells}) == 1
    assert len(cells) == len(runner.M_POINTS) * 2
    assert runner.total_runs(cells) == 20_000


def test_cost_model_prices_cells_by_paradigm_and_m():
    rows = [
        {"paradigm": "joint", "m": 50, "s_per_run_cell": 0.10},
        {"paradigm": "joint", "m": 2000, "s_per_run_cell": 0.50},
        {"paradigm": "marginal", "m": 50, "s_per_run_cell": 0.20},
        {"paradigm": "marginal", "m": 2000, "s_per_run_cell": 0.20},
    ]
    model = pilot_g4.cost_model(rows)
    joint = [Cell(campaign="X", circuit="BV", d=3, m=50, paradigm="joint",
                  operator="O1", n_reps=10)]
    assert pilot_g4.price(joint, model) == pytest.approx(1.0)

    beyond = [Cell(campaign="X", circuit="BV", d=3, m=5000, paradigm="joint",
                   operator="O1", n_reps=10)]
    assert pilot_g4.price(beyond, model) == pytest.approx(5.0)


def test_schedule_flags_pending_campaigns_and_keeps_the_day_split():
    model = {
        "joint": (np.array([50.0, 2000.0]), np.array([0.1, 0.5])),
        "marginal": (np.array([50.0, 2000.0]), np.array([0.2, 0.2])),
    }
    rows = pilot_g4.schedule(model, derate=2.5, n_reps=1000)
    by_name = {r["campaign"]: r for r in rows}
    assert set(by_name) == set(submit_grid.CAMPAIGN_ORDER)
    assert by_name["I4E1"]["day"] == "Day 2"
    assert by_name["I4E2"]["day"] == "Day 3"
    assert by_name["I5"]["runs"] == 108_000
    if submit_grid.campaigns_module() is None:
        for name in ("I8", "I9", "I7size", "I7seq"):
            assert by_name[name]["status"] == "pending campaigns.py"
    assert by_name["I4E1"]["core_s_measured"] > 0



_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.S)
_TEXT_BLOCK = re.compile(r"```text\n(.*?)```", re.S)
_ADD_ARGUMENT = re.compile(r"""add_argument\(\s*["'](--[A-Za-z0-9][A-Za-z0-9-]*)["']""")


def runbook_commands(text: str) -> list[str]:
    out: list[str] = []
    for block in _BASH_BLOCK.findall(text):
        pending = ""
        for raw in block.splitlines():
            line = (pending + " " + raw.strip()).strip() if pending else raw.strip()
            pending = ""
            if line.endswith("\\"):
                pending = line[:-1].strip()
                continue
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def pending_modules(text: str) -> set[str]:
    return {
        line.strip()
        for block in _TEXT_BLOCK.findall(text)
        for line in block.splitlines()
        if line.strip().startswith("harness/") and line.strip().endswith(".py")
    }


def _targets(tokens: list[str]) -> list[tuple[str, int]]:
    found = []
    for i, token in enumerate(tokens):
        if token.endswith((".py", ".sh")) and "harness/" in token:
            found.append((token[token.index("harness/"):], i))
        elif token == "-m" and i + 1 < len(tokens) and tokens[i + 1].startswith("harness."):
            found.append((tokens[i + 1].replace(".", "/") + ".py", i + 1))
    return found


def test_every_runbook_command_exists_with_the_flags_it_uses():
    text = RUNBOOK.read_text()
    pending = pending_modules(text)
    commands = runbook_commands(text)
    assert len(commands) > 15, "RUNBOOK lost its command blocks"

    checked, skipped = 0, []
    for command in commands:
        tokens = shlex.split(command, comments=True)
        for target, index in _targets(tokens):
            path = REVISION_DIR / target
            if not path.exists():
                assert target in pending, (
                    f"{target} is referenced by RUNBOOK.md, does not exist, and is not "
                    f"listed in the pending-modules block: {command}"
                )
                skipped.append(target)
                continue
            checked += 1
            if target.endswith(".sh"):
                continue
            declared = set(_ADD_ARGUMENT.findall(path.read_text()))
            used = {
                t.split("=", 1)[0]
                for t in tokens[index + 1:]
                if t.startswith("--") and t != "--"
            }
            missing = used - declared
            assert not missing, f"{target} has no {sorted(missing)} (RUNBOOK: {command})"
    assert checked >= 10, f"only {checked} runbook commands were verifiable"


def test_runbook_states_the_binding_rules():
    text = RUNBOOK.read_text()
    for marker in (
        "ARROW_DEFAULT_MEMORY_POOL=system",
        "harness/run_tests.sh",
        "--full",
        "pilot_g4.py --cluster",
        "idempotent",
        "--skip-existing",
        "seed_master",
    ):
        assert marker in text, marker
    for day in ("Day 1", "Day 2", "Day 3", "Day 4", "Day 5"):
        assert day in text, day


def test_runbook_job_counts_match_the_generator():
    text = RUNBOOK.read_text()
    for campaign in submit_grid.available_campaigns():
        jobs, _ = submit_grid.build_jobs([campaign])
        row = f"| {campaign} | {len(jobs):,d} | {sum(j.n_cells for j in jobs):,d} | " \
              f"{sum(j.n_runs for j in jobs):,d} |"
        assert row in text, f"stale RUNBOOK row for {campaign}: expected {row!r}"
    everything, _ = submit_grid.build_jobs(submit_grid.available_campaigns())
    total = f"| **total** | **{len(everything):,d}** | " \
            f"**{sum(j.n_cells for j in everything):,d}** | " \
            f"**{sum(j.n_runs for j in everything):,d}** |"
    assert total in text, f"stale RUNBOOK total: expected {total!r}"
