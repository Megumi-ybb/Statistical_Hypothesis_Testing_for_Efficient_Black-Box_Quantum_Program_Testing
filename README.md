# Statistical Hypothesis Testing for Efficient Black-Box Quantum Program Testing

Code, benchmark circuits, and experimental data for the article

> T. Jin, B. Yang, H. Yu, and J. Zhao, "Statistical Hypothesis Testing for Efficient Black-Box Quantum Program Testing," *IEEE Access*, 2026. DOI: *to be added on publication*.

Everything needed to reproduce the article's tables and figures is here: the experiment code, the stored per-cell results, and the scripts that turn those results into the published tables and figures.

## What the article does

Black-box testing runs a quantum program many times and decides from the measurement outcomes alone whether the program is correct. The article casts that decision as a statistical hypothesis test with a stated Type I error. It covers two paradigms:

- **Marginal testing** checks every qubit's outcome distribution. The methods are an exact binomial test with a Bonferroni correction, an adaptive exact/chi-square selector, a parametric bootstrap test, and a sequential design that stops early with a Bonferroni split of α.
- **Joint testing** checks the full distribution over all 2^d outcomes. The methods are a chi-square multinomial test with state lumping and MMDAgg, an aggregated kernel two-sample test.

Both paradigms are compared with tolerance-based checking, which is common in practice, and with classical baselines: the G-test, a Monte Carlo multinomial test, the energy distance test, a total variation bound, and the sequential probability ratio test.

The evidence comes from two simulation campaigns, each with 1,000 replications per cell at α = 0.05.

- **Original campaign.** This campaign samples a random-state ensemble at d = 2 to 12 qubits for marginal testing and d = 2 to 10 for joint testing. Faults are injected as bit-flip, asymmetric, and depolarizing corruption of the outcomes. Sample sizes are m = 50 to 2,000.
- **Revised campaign.** This campaign uses six named benchmarks: Bernstein–Vazirani, Grover's search, QFT, QPE, a quantum walk, and an adder. It covers:
  - structural mutations O1 to O7, which are gate-angle perturbation, gate deletion, gate insertion, control–target swap, phase error, wrong oracle encoding, and a routing defect
  - a fault-strength frontier
  - dependence-only faults that separate marginal from joint testing
  - the sequential design against fixed-sample sizes
  - the permutation budget of MMDAgg
  - a noisy-simulator study

  The revised campaign totals about 10.6 million Monte Carlo runs.

## Repository layout

| Path | Contents |
|---|---|
| `Joint/Experiment/`, `Marginal/Experiment/` | Original campaign. There is one script per experimental group, with `*_falsecase.py` for the faulty arm. The OPENQASM benchmark instances are in `benchmark/`, and the raw result CSVs are in `*_results/` folders. |
| `Joint/combined_results*.csv`, `Marginal/combined_results*.csv` | Summaries of the original campaign behind Tables 5 to 9 and Figures 6 to 17. |
| `revision/harness/` | The revised campaign's harness: circuit builders, mutation operators, oracles, sampling, test statistics, campaign runner, sample-size formulas, tests, and environment files. `RUNBOOK.md` documents the cluster run. |
| `revision/results/<campaign>/` | One Parquet file per job of the revised campaign, described under Data format below. |
| `revision/analysis_out/` | Summaries derived from the results. These include the a priori sample sizes, the confidence intervals, and the LaTeX tables in `resub2/`. |
| `revision/figures/` | The revised-campaign figures printed in the article, and the graphical abstract. |
| `revision/*.py` | Scripts that build the article's tables and figures. |
| `revision/*.sh` | The SGE scripts used to run the revised campaign, kept as examples. |

## Installation

Python 3.11. From `revision/`:

```bash
python3.11 -m venv harness/.venv
harness/.venv/bin/python -m pip install -r harness/requirements.txt
export ARROW_DEFAULT_MEMORY_POOL=system
```

The pinned versions include Qiskit 2.5.1, Qiskit Aer 0.17.2, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.5, PyArrow 25.0.0, and Matplotlib 3.11.1.

`harness/environment.yml` creates the same environment with conda. After creating it, link the environment with `ln -sfn "$(conda info --base)/envs/quantum-testing" harness/.venv`.

The original campaign ran with Qiskit 0.45.0 and Qiskit Aer 0.12.2. The root `environment.yml` records that Linux environment for reference.

## Reproduce the tables and figures

No simulation is needed for this step. Set `PY=revision/harness/.venv/bin/python` and run the commands from the repository root. Commands that start with `cd revision` run inside that folder.

| Article | Content | Command | Output |
|---|---|---|---|
| Table 1 | MMDAgg Type I error against the permutation budget | `$PY revision/resub2_tables.py` | `revision/analysis_out/resub2/tab_bsweep.tex` |
| Table 2 | Type I guarantee of each procedure | derived in the article | |
| Table 3 | Provenance of every table and figure | stated in the article | |
| Table 4 | Fault injection parameters | set in `Joint/Experiment/*_falsecase.py` and `Marginal/Experiment/*_falsecase.py` | |
| Tables 5 to 9 | Original campaign, median Type I error and power | `$PY revision/original_tables.py` | `revision/analysis_out/original_tables/` |
| Tables 10 and 11 | Power against structural faults by operator | `$PY revision/resub2_tables.py` | `tab_struct_joint.tex`, `tab_struct_marginal.tex` |
| Table 12, Figure 4 | Expected number of measurements of the sequential design | same run | `tab_en_width.tex`, `revision/figures/en_by_width.png` |
| Table 13 | Type I error on the named benchmarks | same run | `tab_baseline_typei.tex` |
| Table 14 | Noisy simulator validation | same run | `tab_noisy.tex` |
| Figure 1 | Method overview | static diagram | |
| Figure 2 | Joint tests against the gate-angle operator on QFT, by fault strength | `cd revision && harness/.venv/bin/python -m harness.figures --campaign results/I4E2 --out-dir figures --families frontier` | `revision/figures/joint_frontier_QFT_O1.png` |
| Figure 3 | Dependence-only faults in QFT: marginal against joint tests | `cd revision && harness/.venv/bin/python -m harness.figures --campaign results/I9C3 --out-dir figures --families c3` | `revision/figures/c3_separation_QFT.png` |
| Figure 5 | Two-regime noisy validation | `cd revision && harness/.venv/bin/python -m harness.figures --campaign results/I5 --out-dir figures --families noisy` | `revision/figures/noisy_two_regime_aer2.png` |
| Figures 6 to 17 | Per-group results of the original campaign | `$PY revision/resub2_appendix_figs.py` | `revision/analysis_out/resub2/appendix/` |
| Graphical abstract | Figure 4 at 660 × 295 pixels | `$PY revision/resub2_tables.py` | `revision/figures/graphical_abstract_fig4.jpg` |

**About these scripts:**
- One run of `resub2_tables.py` writes all six LaTeX tables, Figure 4, and the graphical abstract.
- That run also draws `frontier_QWALK_O1.png`, a view of the same frontier on the quantum walk that the article does not print.
- The `--install` and `--audit` options of `resub2_tables.py` act on the manuscript source, which is not part of this repository.
- `original_tables.py` ends by checking its values against those printed in the article.
- `harness.figures` draws every circuit of a family, so it writes more figures than the article prints.

The a priori sample sizes in `revision/analysis_out/a_priori_sizes.csv` come from `cd revision && harness/.venv/bin/python -m harness.crossover_analysis --campaign results/I7size --out-dir analysis_out`. The article reports the normal-approximation column. `revision/harness/sample_size.py` also computes the exact sizes.

## Rerun the experiments

### Revised campaign

A cell is identified by the key `campaign|circuit|d|m|operator|strength|case`. Its master seed is the first 64 bits of the SHA-256 hash of that key. `numpy.random.SeedSequence(master).spawn(n_reps)` gives one independent stream per replication. Each statistic draws its permutations from `SeedSequence([master, rep, method key])`. The mutation site is seeded without `m`, so all sample sizes of a cell test the same mutant. The noisy study adds the arm and the noise setting to the key and transpiles with seed 7. Rerunning a cell therefore reproduces its counts exactly.

From `revision/`:

```bash
# list one cell and its master seed
harness/.venv/bin/python -m harness.runner --campaign I4E1 --circuit QFT --d 4 --m 100 --operator O2 --paradigm joint --dry-run
# rerun that cell with 1,000 replications
harness/.venv/bin/python -m harness.runner --campaign I4E1 --circuit QFT --d 4 --m 100 --operator O2 --paradigm joint --out OUT
# rerun one campaign, skipping finished jobs
harness/.venv/bin/python -m harness.runner --campaign I8 --out results/I8 --skip-existing
# the sequential study has one cell per job, so parallelize over processes
seq 2 12 | xargs -P 11 -I{} harness/.venv/bin/python -m harness.runner --campaign I7seq --d {} --out results/I7seq
# the noisy study, 108 cells
./i5_all.sh
# realized cell counts of every campaign and the reason for each exclusion
harness/.venv/bin/python -m harness.campaigns --manifest
```

| Campaign | Study |
|---|---|
| `I4E1`, `I4E1B` | Structural mutations and their null pass |
| `I4E2`, `I4E2b` | Fault-strength frontiers, joint and marginal |
| `I4E3` | Detector run at m = 2,000 |
| `I5` | Noisy-simulator validation |
| `I7seq`, `I7size` | Sequential design and fixed-sample sizes |
| `I8` | MMDAgg permutation budget |
| `I9C3` | Dependence-only faults |

The full campaign took a few hours on a 1,000-core grid. `harness/applicability.json` fixes the realized cell set. Rebuild it with `-m harness.campaigns --probe --out harness/applicability.json` only if the circuit builders change. The `cluster_*.sh` scripts and `harness/RUNBOOK.md` show the SGE workflow used. For other schedulers, `harness/submit_grid.py` emits job files.

### Original campaign

Run one experimental group from its folder, for example `cd Joint/Experiment && python joint_QFTtestfalse.py 4`, with `d` as the argument. Results go to timestamped CSVs next to the script. Combine them with:

- `python Joint/Experiment/df_combine.py --output-wide ../combined_results.csv --output-long ../combined_results_long.csv` for the joint groups
- `python Marginal/df_combine.py` for the marginal groups

The measurement draws of this campaign are seeded from the base seed 801 and the replication index. The random states come from an unseeded generator, so the campaign reproduces in distribution and not run by run. The committed summaries are therefore the reference for Tables 5 to 9. `revision/harness/g3_report.csv` is the golden-cell regression of the revised harness against these summaries. It was produced by `revision/harness/g3_golden.py` and agrees within binomial bands.

## Data format

Each row of a Parquet file in `revision/results/` is one method in one cell.

| Columns | Meaning |
|---|---|
| `campaign`, `circuit`, `d`, `m`, `paradigm`, `operator`, `strength`, `case` | The cell. `case` is `baseline` for the correct program and `falsecase` for the faulty one. |
| `method`, `n_reps` | The test and the number of replications |
| `TP`, `FP`, `TN`, `FN`, `n_evaluated` | Rejection counts over the replications |
| `mean_p`, `n_used_mean`, `n_used_median` | Mean p-value, and the sample size used by sequential methods |
| `seed_master` | Master seed of the cell |
| `harness_version`, `git_describe`, `python_version`, `numpy_version`, `scipy_version`, `pandas_version`, `pyarrow_version`, `joblib_version`, `qiskit_version`, `qiskit_aer_version`, `created_utc` | Provenance of the run |

The Type I error of a `baseline` cell is FP / (FP + TN). The power of a `falsecase` cell is TP / (TP + FN).

The original campaign's `combined_results.csv` files have one row per group, `n_qubits`, `m_samples`, and `test_method`. They carry `type_i_error`, `power_bitflip`, `power_asymmetric`, and `power_depolarize`.

## Tests

`cd revision && harness/run_tests.sh` runs each test file in its own process. `harness/run_tests.sh -m slow` runs only the 20-minute gate that checks the permutation law.

## License

The code is released under the MIT License, in `LICENSE`. The data and figures are released under CC BY 4.0, in `LICENSE-DATA`, which also defines which files each license covers. The OPENQASM benchmark instances keep the terms of their original source.

## Citation

```bibtex
@article{jin2026statistical,
  author  = {Jin, Tiancheng and Yang, Bo and Yu, Haibo and Zhao, Jianjun},
  title   = {Statistical Hypothesis Testing for Efficient Black-Box Quantum Program Testing},
  journal = {IEEE Access},
  year    = {2026},
  note    = {DOI to be added on publication}
}
```
