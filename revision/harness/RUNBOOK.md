# RUNBOOK — 5-day campaign on the 1000-core research grid

IEEE Access revision **Access-2026-23764**. Everything below runs from the
`revision/` directory of the checkout **on the cluster**, with the pinned
interpreter `harness/.venv/bin/python` (Python 3.11).

Plan of record: `revision/revision_plan.html` (Compute budget) and
`revision/impl_data.json` (`_summary.compute_budget.schedule_5day_1000core`).
Code contract: `harness/SPEC.md`. Realized cell counts come from
`harness/campaigns.py`, never from the plan tables.

---

## 0. Binding rule: what runs where

**The author's laptop runs correctness checks only.** Every command in this
runbook whose wall time is not seconds-scale is a CLUSTER command. Locally you
may run: unit tests (via `harness/run_tests.sh`, never the combined suite in one
process), `--smoke` / `--quick` / `--dry-run` modes, and job *generation*.
`harness/submit_grid.py` has no submission path at all — it prints or writes
command lines, and the emitted script is executed on the cluster login node.

Two macOS-ARM mitigations are baked in and are **harmless on Linux**, so leave
them in place on the cluster:

* `ARROW_DEFAULT_MEMORY_POOL=system` — pyarrow's bundled mimalloc corrupts
  parquet writes (and then segfaults) on macOS ARM. `io_store.py`,
  `noisy_grover.py` and `tests/conftest.py` set it at import time, which only
  works if nothing has already initialised libarrow; exporting it in the job
  environment is the belt-and-braces version.
* one interpreter process per test file (`harness/run_tests.sh`) — GC freeing
  qiskit-Rust `CircuitData` during a later lazy import segfaults in qiskit's
  bundled mimalloc on macOS ARM.

Linux grid nodes are *believed* unaffected by both; neither mitigation costs
anything there, so nothing is conditional on the platform.

---

## 1. Environment (once, on the cluster)

> **GRID@CBS users: the operative copy-paste file is `START_ON_CLUSTER.txt`.**
> That grid requires every login-node command to go through `grid_run`
> (interactive for checks, batch for campaigns) and submits executables in the
> `./script` form — `submit_grid.py` therefore emits job lines as
> `grid_run ... ./grid_worker.sh runner|noisy <args>` (the wrapper is
> auto-generated and chmod +x). Connection: `ssh <user>@<login-host>`
> (site VPN required); conda base is `/apps/anaconda3`; user limit
> 500 cores / 1000 GB RAM, so budget ~3-4 h of campaign wall-clock. The generic
> commands below describe the same sequence for a conventional cluster.

Conda is the preferred route (`harness/environment.yml` creates the
`quantum-testing` env; the symlink makes every hardcoded
`harness/.venv/bin/python` path in this runbook resolve to it, and grid jobs
pick up the right interpreter by path with no activation):

```bash
cd "$HOME/Quantum-Testing/revision"
conda env create -f harness/environment.yml
ln -sfn "$(conda info --base)/envs/quantum-testing" harness/.venv
harness/.venv/bin/python -c "import numpy, scipy, pandas, pyarrow, joblib, qiskit, qiskit_aer; print(qiskit.__version__)"
export ARROW_DEFAULT_MEMORY_POOL=system
```

Do NOT reuse the legacy `quantum-env` (qiskit 0.45, pre-1.0 API): the harness
requires qiskit >= 1.0 and the gates were validated on 2.5.1. `quantum-env`
stays untouched as the environment that reproduces the original production
scripts. If the cluster has no conda, the venv fallback is equivalent:

```bash
python3.11 -m venv harness/.venv
harness/.venv/bin/python -m pip install --upgrade pip
harness/.venv/bin/python -m pip install -r harness/requirements.txt
harness/.venv/bin/python -c "import numpy, scipy, pandas, pyarrow, joblib, qiskit, qiskit_aer; print(qiskit.__version__)"
export ARROW_DEFAULT_MEMORY_POOL=system
```

`harness/requirements.txt` pins the versions the gates were validated against
(numpy 2.4.6, scipy 1.17.1, pandas 3.0.5, pyarrow 25.0.0, joblib 1.5.3,
qiskit 2.5.1, qiskit-aer 0.17.2). Put the `export` in the profile the grid jobs
inherit. Every result row records `git_describe` plus all package versions
(`io_store.provenance()`), so environment drift is detectable after the fact —
but do not upgrade anything between Day 1 and Day 4.

---

## 2. Day 1 — freeze and gates (G1–G4)

### 2.1 Fast test suite, then the slow gate

```bash
harness/run_tests.sh
harness/run_tests.sh -m slow
```

The second form overrides the script's built-in `-m "not slow"` (pytest keeps
the last `-m`), so it runs **only** the slow gates — G2 costs ~20 minutes
because it calls the production MMDAgg. Both invocations must end with every
file green. G1 and G2 already passed on the laptop; re-running them here proves
the cluster environment, which is what the campaign actually executes.

### 2.2 Gate G3 — golden-cell regression, full configuration

```bash
harness/.venv/bin/python harness/g3_golden.py --part a --out harness/g3_report_oracle.csv
harness/.venv/bin/python harness/g3_golden.py --full --part b --jobs -1 --out harness/g3_report.csv
```

Part (a) is deterministic and sub-second (15/15 oracle equalities against the
repo's benchmark QASM). Part (b) is the ~340 s CPU Monte-Carlo regression that
the local budget guard refuses without `--full`. Expect the campaign-source `p0`
comparisons to sit inside the binomial band and the builder-source ones to
differ — that difference is the documented `produce_state` provenance finding,
not a defect.

### 2.3 Applicability probe (single source of truth for every cell count)

```bash
harness/.venv/bin/python -m harness.campaigns --probe --out harness/applicability.json
harness/.venv/bin/python -m harness.campaigns --manifest --out harness/manifest.txt
```

The probe builds every `(circuit, d)` and tries every operator, caching the
result; the manifest prints realized vs plan cells and runs per campaign with
every exclusion attributed. `submit_grid.py` and `pilot_g4.py` both read the
same enumeration, so if the manifest and the job table disagree, stop.

### 2.4 Gate G4 — worst-case grid pilot (the schedule-defining measurement)

```bash
harness/.venv/bin/python harness/pilot_g4.py --cluster --n-jobs -1 --cores 1000 --out harness/pilot_report.csv
```

Primary chunk: QPE at the largest supported width (d = 10), the first applicable
operator, the full ten-point m-ladder, both paradigms, 1000 replications —
20 cells, 20,000 runs. It then runs a **companion** chunk (QFT at d = 12,
marginal only, 10 cells) because QPE's null is degenerate, so campaigns.py drops
the copula bootstrap and the sequential test from its marginal cells — and the
bootstrap is what the 0.20 s/run marginal planning figure is made of. Pricing the
campaign's marginal half off the QPE chunk alone would under-estimate it
several-fold. `--no-companion` skips it.

Outputs `harness/pilot_report.csv` (per cell and method: confusion counts, wall
time, seconds per run and per call, `chunk` = primary/companion) and
`harness/pilot_schedule.csv`, and prints the re-derived schedule: each campaign
priced at the **measured** cost for its own (paradigm, m) instead of the plan's
flat 0.16 s / 0.20 s per run times the assumed 2.5× core derating.

Its last line is the `submit_grid.py` command for Day 2, carrying the measured
derating. **Read the re-derived Day-2/3 totals before submitting anything.** The
plan tolerates a 10× error (10 × 1,800 core-hours is still ~18 h wall on 1000
cores); if the pilot is worse than ~10× the planning figures, stop and re-scope
with the leader rather than submitting.

### 2.5 I5 sizing (the one campaign the pilot cannot price)

`harness/noisy_grover.py` is Aer-simulator bound, so the count-kernel pilot says
nothing about it. Time one cell at 20 replications and scale by 50:

```bash
harness/.venv/bin/python harness/noisy_grover.py --arm B --strength-level 2 --m 200 --n-reps 20 --out results/sizing/i5_probe.parquet
```

It prints `N cell(s) in X s`; a full I5 job is 1000 replications of one cell, so
its wall is ≈ 50 × X. If that exceeds ~2 h, split I5 further (its jobs are
already one m-point each — shard by `--n-reps` and merge, or drop the two
largest m).

### 2.6 Job generation dry run

```bash
harness/.venv/bin/python harness/submit_grid.py --campaign I4E1 --limit 3
harness/.venv/bin/python harness/submit_grid.py --dry-run --cores 1000
```

The second command prints every job plus the per-campaign table (jobs, cells,
runs, planned core-hours, per-job wall quantiles). Check three things: the first
line says `cell source: campaigns.py`; no campaign is reported as skipped; the
per-job wall estimates sit in the 5–20 minute target band (tune with
`--ncpus-cap`, default 32, or force `--grid-ncpus`).

**Day-1 artifacts:** `harness/g3_report_oracle.csv`, `harness/g3_report.csv`,
`harness/applicability.json`, `harness/manifest.txt`, `harness/pilot_report.csv`,
`harness/pilot_schedule.csv`, `results/sizing/i5_probe.parquet`, and green test
output for every test file.

---

## 3. Day 2 — the core pass (I4 E1 + I7 E-base, then I5 and I8)

Dependency order matters: **I4E1 and the I4E1B baseline pass go first**, in one
harness pass that co-computes the I7 E-base baseline statistics (energy
distance, G-test, MC-multinomial, TV bound) on the *same* draws. Nothing else
may start before those parquets exist, because I7's "rides on I4 samples" claim
and the I11 CI re-analysis both read them.

Substitute the derating `pilot_g4.py` measured for the `2.5` below.

```bash
harness/.venv/bin/python harness/submit_grid.py --campaign I4E1,I4E1B --derate 2.5 --grid-mem 64G --emit jobs/day2_core.sh
bash jobs/day2_core.sh
```

409 jobs (351 + 58), 7,460 cells, 7.46M runs. One job = one
`(campaign, circuit, d, operator)` slice run by the runner CLI, which
parallelises over the slice's cells with joblib, so `--grid_ncpus` is **per
job**: `min(cells in the slice, 32)` — typically 20 for a d ≤ 10 slice (10
m-points × 2 paradigms) and 10 for the marginal-only widths d ∈ {11, 12}.
Grid-wide parallelism comes from the ~1,000 jobs, not from any one job.

Then the two small independent campaigns:

```bash
harness/.venv/bin/python harness/submit_grid.py --campaign I5,I8 --emit jobs/day2_small.sh
bash jobs/day2_small.sh
```

I5 is 108 single-core jobs, one `(arm, noise level, m)` cell each —
`harness/noisy_grover.py` has no in-process joblib parallelism, so its
parallelism is the job count. I8 is 9 jobs, one per `(shape arm, width)`; its
cells carry synthetic circuit labels (`shapeA_dense`, `shapeB_sparse`,
`shapeC_ghz`) in the circuit field, which the runner CLI selects like any other
circuit. Its two permutation budgets are *not* a chunk axis — they ride in the
method name (`mmdagg_B500`, `mmdagg_B2000`), so both land in one file as separate
method rows sharing a seed (common random numbers), and a B=2000 cell costs about
4× a B=500 one at equal m.

Progress check (no site-specific queue command assumed):

```bash
ls results/I4E1/*.parquet | wc -l
ls results/I4E1B/*.parquet | wc -l
ls results/I5/*.parquet | wc -l
```

Targets: 351, 58, 108.

**Day-2 artifacts:** `results/I4E1/I4E1__<circuit>__d<NN>__<operator>.parquet`
(351), `results/I4E1B/…` (58),
`results/I5/I5__arm<A|B|P>__<aer0..aer4|fake>__m<mmmm>.parquet` (108),
`results/I8/I8__shape<A|B|C>_…__d<NN>__baseline.parquet` (9).

---

## 4. Day 3 — frontiers, detector, C3, and the I7 competitors

```bash
harness/.venv/bin/python harness/submit_grid.py --campaign I4E2,I4E2b,I4E3 --derate 2.5 --emit jobs/day3_frontiers.sh
bash jobs/day3_frontiers.sh
harness/.venv/bin/python harness/submit_grid.py --campaign I9C3,I7size,I7seq --derate 2.5 --emit jobs/day3_rest.sh
bash jobs/day3_rest.sh
```

367 jobs for the frontiers plus detector (39 + 39 + 289; 2,239 cells, 2.24M
runs) and 163 for the rest (25 + 116 + 22; 736 cells, 736K runs). I4E3's cells
are one per `(circuit, d, operator)` at m = 2000 with the chi-square detector
only, so those 289 jobs are single-core.

Two cautions on the second command:

* the pilot's cost model is clamped at m = 2000, so I7size's m ∈
  {2500, …, 5000} points are **under-priced** in the re-derived schedule —
  `pilot_g4.py` says so in its `status` column;
* I7seq's cells draw shots until their stopping rule fires (`m` = 5,000 is only a
  horizon), so their cost is `E[N]`-driven and cannot be read off an m-ladder.
  The pilot reports them as **unpriced** rather than guessing. Order of
  magnitude from the design: at d = 2 and |Δp| = 0.02 the SPRT needs ~6.9k shots
  to accept under H0, against a 50,000-shot cap — so give those 22 jobs generous
  queue time and watch the first few before assuming the rest fit.

The C3 applicability theorem and the shot-free screens produce no grid jobs:

```bash
harness/.venv/bin/python harness/i9_c3.py --classify --manifest
harness/.venv/bin/python -m harness.campaigns --screen --out harness/e3_screen.csv
```

The first prints which benchmark nulls admit a dependence-only alternative (a
concentrated null does not — its joint is determined by its marginals) and the
realized I9 counts; the second is the E3 fidelity/TVD screen over the 289 mutant
pairs, computed from statevectors at zero Monte-Carlo cost.

**Day-3 artifacts:** `results/I4E2/…` (39 jobs), `results/I4E2b/…` (39),
`results/I4E3/…` (289), `results/I9C3/…` (25), `results/I7size/…` (116),
`results/I7seq/…` (22), `harness/e3_screen.csv`.

---

## 5. Day 4 — aggregation, CIs, figures

### 5.1 Verify completeness, then combine

```bash
ls results/I4E1/*.parquet | wc -l
harness/.venv/bin/python -c "from harness import io_store; df = io_store.read_grid('results/I4E1'); print(df.shape); print(df.groupby('method')[['TP','FP','TN','FN']].sum())"
```

`io_store.read_grid` accepts a directory, a glob or a list. Any campaign short
of its job count above is a re-run (§6), not a partial aggregation.

### 5.2 Re-analysis

```bash
harness/.venv/bin/python -m harness.ci_reanalysis --campaign results/I4E1 --out-dir analysis_out
harness/.venv/bin/python -m harness.crossover_analysis --campaign results/I7size --out-dir analysis_out
harness/.venv/bin/python -m harness.i2_verification --out-dir analysis_out
harness/.venv/bin/python -m harness.r19_lumping_breakdown --out analysis_out/r19_lumping_breakdown.csv
```

These read the legacy on-disk CSVs *and* the new parquet grid, and refuse rather
than silently round when reconstructed counts disagree with stored rates — pass
`--lax` only with the leader's sign-off. Outputs: `analysis_out/per_cell_ci.csv`,
`analysis_out/representative_cells_ci.csv`,
`analysis_out/a_priori_vs_empirical_crossover.csv`,
`analysis_out/a_priori_sizes.csv`, plus the I2 and R19 tables.

### 5.3 Figures

```bash
harness/.venv/bin/python -m harness.figures --campaign results/I4E1 --out-dir figures --families all
harness/.venv/bin/python -m harness.figures --campaign results/I9C3 --out-dir figures
harness/.venv/bin/python -m harness.figures --campaign results/I5 --out-dir figures
```

### 5.4 Re-runs

Everything flagged in §5.1 goes back through §6 **today**, so Day 5 stays free.

**Day-4 artifacts:** `analysis_out/*.csv`, `figures/*.png`.

---

## 6. Retry and resume

**Jobs are idempotent.** A cell's entire random stream is derived from its
identity: `seed_master = SHA-256(campaign|circuit|d|m|operator|strength|case)`
truncated to 8 bytes, then `SeedSequence(master).spawn(n_reps)` gives one
generator per replication, and each statistic's own randomness comes from
`SeedSequence([master, rep, hash(method name)])` (`runner.py`, "Seed policy
(BINDING)"). Nothing consumes wall-clock time, a PID, worker count, ordering or
global numpy state, so re-running a job reproduces its statistics exactly and
overwrites its parquet with an equivalent file. The only columns that differ
between two runs of one job are timing and stamps (`method_wall_s`,
`cell_wall_s`, `created_utc`).

Consequences:

* a killed, evicted or timed-out job needs **no** cleanup — resubmit the same
  line;
* a partially written parquet is safe to overwrite (one file per job, written
  once at the end of the job);
* re-running after a code fix is legitimate and reproducible; re-running after an
  *environment* change is not — check `git_describe` and the version columns
  first.

Regenerate only the missing jobs:

```bash
harness/.venv/bin/python harness/submit_grid.py --campaign I4E1 --skip-existing --emit jobs/retry_I4E1.sh
bash jobs/retry_I4E1.sh
```

`--skip-existing` drops every job whose output parquet already exists under
`--workdir`. For one known-bad slice:

```bash
harness/.venv/bin/python harness/submit_grid.py --campaign I4E1 --circuit QPE --d 10 --operator O1 --emit jobs/retry_one.sh
bash jobs/retry_one.sh
```

If the scheduler is congested, stage with `--limit N` and `--sleep 0.2` (the
emitted script then sleeps between submissions).

---

## 7. Day 5 — buffer

Reserved. Uses in priority order: (1) finish anything that slipped from Day 3;
(2) re-runs; (3) fold measured numbers, CIs and figures into the manuscript and
response letter; (4) nothing else — do not start new experiments on Day 5.

---

## 8. Worker invocation, and one resolved defect

Jobs run SPEC.md's form, `python -m harness.runner --campaign … --circuit … --d …
--operator … --out …`. A fallback exists — `submit_grid.py --worker inline`
wraps the same CLI in `python -c "from harness import campaigns, runner;
campaigns.register_with_runner(); raise SystemExit(runner.main())"` — and both
forms are checked against the enumeration by the test suite.

The fallback exists because the module form was briefly wrong: under `-m`,
runner.py is the `__main__` module object while the
`campaigns.register_with_runner()` call inside its `main()` mutated the
separately imported `harness.runner`, so the registration never reached the
parser. In that state `--campaign I8` (and I7size, I7seq, I9C3) was rejected
outright, and the I4 campaigns silently fell back to runner.py's **unfiltered**
plan grid — jobs would have received cells whose operator has no structural site
(dying in `mutants.mutate`) and kept the bootstrap and sequential tests on
degenerate marginal nulls that campaigns.py drops. Fixed in runner.py on
2026-07-26: the `__main__` guard delegates to the canonical module and raises
loudly if `main()` ever runs from the wrong one, and a campaigns.py import error
is no longer swallowed into the unfiltered grid.

Verification, should the symptom ever reappear (a job enumerating more cells than
the manifest says):

```bash
harness/.venv/bin/python -m harness.runner --campaign I8 --dry-run --list-limit 0
harness/.venv/bin/python -m harness.runner --campaign I4E1 --circuit BV --d 3 --operator O1 --dry-run
```

The first must report 36 cells; the second must report **0** (BV has no rotation
gate, so O1 has no site). If either differs, stop and re-run
`harness/tests/test_cluster_orch.py`, whose subprocess tests cover exactly this.

Pending modules: none — every module this runbook references exists in the
checkout, so the list below is empty. Anything added to it is a file the runbook
uses that has not landed yet; the test suite enforces the distinction.

```text
(none)
```

---

## 9. Totals

Realized counts (from `campaigns.py`, i.e. after width and operator
applicability filtering; the plan tables are upper bounds):

| campaign | jobs | cells | runs |
|---|---:|---:|---:|
| I4E1 | 351 | 6,400 | 6,400,000 |
| I4E1B | 58 | 1,060 | 1,060,000 |
| I5 | 108 | 108 | 108,000 |
| I8 | 9 | 36 | 36,000 |
| I4E2 | 39 | 975 | 975,000 |
| I4E2b | 39 | 975 | 975,000 |
| I4E3 | 289 | 289 | 289,000 |
| I9C3 | 25 | 250 | 250,000 |
| I7size | 116 | 464 | 464,000 |
| I7seq | 22 | 22 | 22,000 |
| **total** | **1,056** | **10,579** | **10,579,000** |

Two reconciliations for the leader and the response letter:

1. The plan's headline is 12,822,000 runs; the realized campaign is 10,579,000.
   The reduction is **not** a scope cut — it is the applicability matrix
   (Adder exists only at d ∈ {4, 7, 10}; O1 has no rotation gate on BV, Grover
   or Adder; O4 no asymmetric two-qubit gate on QFT or QPE; O7 no routing SWAP
   on BV) plus the I9 applicability theorem, all attributed per campaign in
   `harness/manifest.txt`.
2. The realized total **includes 1,060,000 runs for the I4E1B baseline
   (Type I) pass, which the plan never priced** — I7's E-base explicitly rides
   on "the I4 E1 pass (power) *and* the baseline-case pass (Type I)". At the
   planning figures that pass costs ~134 grid-core-hours (~8 minutes of
   1000-core wall), so it does not threaten the window, but any run-count total
   quoted in the manuscript must include it.

At the plan's derating the whole campaign is ~1,318 grid-core-hours ≈ 1.3 h of
1000-core wall-clock, excluding I5 (Aer-bound, sized on Day 1) and queueing.
Gate G4 replaces that estimate with measurement before anything is submitted.
