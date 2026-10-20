
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from harness import ci_reanalysis, crossover_analysis, figures, i2_verification, io_store

REPO_ROOT = ci_reanalysis.REPO_ROOT

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")




def _row(**kw) -> dict:
    row = {
        "campaign": "SMOKE",
        "circuit": "BV",
        "d": 2,
        "m": 50,
        "paradigm": "joint",
        "operator": "baseline",
        "strength": float("nan"),
        "case": "baseline",
        "methods": "chi_square,mmdagg",
        "n_reps": 100,
        "method": "chi_square",
        "TP": 0,
        "FP": 5,
        "TN": 95,
        "FN": 0,
        "n_evaluated": 100,
        "mean_p": 0.5,
        "n_used_mean": float("nan"),
        "n_used_median": float("nan"),
        "method_wall_s": 0.01,
        "cell_wall_s": 0.02,
        "seed_master": 1234,
    }
    row.update(io_store.provenance())
    row["created_utc"] = "2026-07-26T00:00:00+00:00"
    row.update(kw)
    return row


def _tiny_rows() -> list[dict]:
    rows: list[dict] = []
    spec = {"joint": ("chi_square", "mmdagg"), "marginal": ("adaptive", "sequential")}
    for paradigm, methods in spec.items():
        for circuit in ("BV", "QFT"):
            for d in (2, 3):
                for m in (50, 100):
                    for method in methods:
                        rows.append(_row(
                            paradigm=paradigm, circuit=circuit, d=d, m=m,
                            case="baseline", operator="baseline",
                            strength=float("nan"), method=method,
                            methods=",".join(methods),
                            TP=0, FP=4 + d, TN=96 - d, FN=0,
                            n_used_mean=float(m) * 0.6 if method == "sequential"
                            else float("nan"),
                        ))
                        for strength in (0.1, 0.4):
                            tp = int(round(100 * (0.2 + 2.0 * strength * (1 if paradigm
                                                                          == "joint" else 0.2))))
                            tp = min(tp, 100)
                            rows.append(_row(
                                paradigm=paradigm, circuit=circuit, d=d, m=m,
                                case="falsecase", operator="O1", strength=strength,
                                method=method, methods=",".join(methods),
                                TP=tp, FP=0, TN=0, FN=100 - tp,
                                n_used_mean=float(m) * 0.6 if method == "sequential"
                                else float("nan"),
                            ))
    return rows


def _noisy_rows() -> list[dict]:
    rows = []
    for arm, base in (("A", 0.2), ("B", 0.05), ("P", 0.9)):
        for m in (50, 200):
            rate = min(0.99, base * (2.0 if arm == "A" and m == 200 else 1.0))
            k = int(round(rate * 100))
            rows.append(_row(
                campaign="I5", circuit="Grover", d=5, m=m, paradigm="joint",
                case="falsecase" if arm == "P" else "baseline",
                operator="O6" if arm == "P" else "baseline",
                method="mmdagg", methods="mmdagg",
                TP=k if arm == "P" else 0, FP=0 if arm == "P" else k,
                TN=0 if arm == "P" else 100 - k, FN=100 - k if arm == "P" else 0,
                arm=arm, noise_label="low",
            ))
    return rows


@pytest.fixture(scope="module")
def tiny_grid(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tiny_grid")
    io_store.write_job(path / "smoke.parquet", _tiny_rows())
    return path


@pytest.fixture(scope="module")
def tiny_noisy(tmp_path_factory) -> Path:
    import pandas as pd

    path = tmp_path_factory.mktemp("tiny_noisy")
    rows = _noisy_rows()
    extras = pd.DataFrame([{"arm": r.pop("arm"), "noise_label": r.pop("noise_label")}
                           for r in rows])
    df = io_store.to_frame(rows)
    df = pd.concat([df.reset_index(drop=True), extras], axis=1)
    df.to_parquet(path / "i5.parquet", index=False, engine="pyarrow")
    return path


@pytest.fixture(scope="module")
def legacy_cells():
    return ci_reanalysis.load_legacy_cells(REPO_ROOT)



_CASES = [(0, 100), (1, 100), (5, 100), (50, 100), (99, 100), (100, 100),
          (48, 1000), (60, 1000), (500, 1000), (870, 1000)]


@pytest.mark.parametrize("k,n", _CASES)
def test_wilson_matches_score_equation_roots(k, n):
    from scipy.optimize import brentq
    from scipy.stats import norm

    z = float(norm.isf(ci_reanalysis.CI_ALPHA / 2.0))
    phat = k / n

    def f(p):
        return (phat - p) ** 2 * n - z * z * p * (1.0 - p)

    lo, hi = ci_reanalysis.wilson_ci(k, n)
    eps = 1e-12
    expected_lo = 0.0 if k == 0 else brentq(f, 1e-15, min(phat, 1 - eps), xtol=1e-14)
    expected_hi = 1.0 if k == n else brentq(f, max(phat, eps), 1 - 1e-15, xtol=1e-14)
    assert float(lo) == pytest.approx(expected_lo, abs=1e-10)
    assert float(hi) == pytest.approx(expected_hi, abs=1e-10)


@pytest.mark.parametrize("k,n", _CASES)
def test_clopper_pearson_matches_binomial_tail_identities(k, n):
    from scipy.stats import binom

    alpha = ci_reanalysis.CI_ALPHA
    lo, hi = ci_reanalysis.clopper_pearson_ci(k, n)
    lo, hi = float(lo), float(hi)
    if k > 0:
        assert binom.sf(k - 1, n, lo) == pytest.approx(alpha / 2.0, abs=1e-12)
    else:
        assert lo == 0.0
    if k < n:
        assert binom.cdf(k, n, hi) == pytest.approx(alpha / 2.0, abs=1e-12)
    else:
        assert hi == 1.0


@pytest.mark.parametrize("k,n", [(k, n) for k, n in _CASES if 0 < k < n])
def test_clopper_pearson_is_wider_than_wilson_in_the_interior(k, n):
    w_lo, w_hi = ci_reanalysis.wilson_ci(k, n)
    c_lo, c_hi = ci_reanalysis.clopper_pearson_ci(k, n)
    assert float(c_hi) - float(c_lo) >= float(w_hi) - float(w_lo) - 1e-12


def test_clopper_pearson_is_narrower_than_wilson_at_the_boundary():
    w_lo, w_hi = ci_reanalysis.wilson_ci(0, 100)
    c_lo, c_hi = ci_reanalysis.clopper_pearson_ci(0, 100)
    assert float(c_lo) == float(w_lo) == 0.0
    assert float(c_hi) < float(w_hi)


def test_mc_se_matches_closed_form():
    k, n = np.array([0, 50, 500, 1000]), 1000
    got = ci_reanalysis.mc_se(k, n)
    want = np.sqrt((k / n) * (1 - k / n) / n)
    assert np.allclose(got, want)


def test_wilson_half_width_expectations():
    assert ci_reanalysis.wilson_half_width(0.5, 1000) <= 0.031
    assert ci_reanalysis.wilson_half_width(0.05, 1000) == pytest.approx(0.014, abs=0.002)
    assert ci_reanalysis.wilson_half_width(0.5, 4000) < 0.55 * ci_reanalysis.wilson_half_width(0.5, 1000)


def test_counts_from_rate_roundtrip_and_guard():
    rates = np.array([0.0, 0.123, 0.5, 1.0])
    assert np.array_equal(ci_reanalysis.counts_from_rate(rates, 1000),
                          np.array([0.0, 123.0, 500.0, 1000.0]))
    with pytest.raises(ValueError, match="cannot reconstruct counts"):
        ci_reanalysis.counts_from_rate(np.array([0.1234567]), 1000)




def test_legacy_directories_cover_all_twelve_runs():
    dirs = ci_reanalysis.legacy_directories(REPO_ROOT)
    assert len(dirs) == 24
    seen = {(paradigm, circuit, case) for _, paradigm, circuit, case in dirs}
    for paradigm in ("joint", "marginal"):
        for circuit in ("BV", "Grover", "QFT", "QPE", "QWALK", "Adder"):
            for case in ("baseline", "falsecase"):
                assert (paradigm, circuit, case) in seen


def test_legacy_cells_reproduce_the_combined_csvs(legacy_cells):
    import pandas as pd

    for folder, paradigm in (("Joint", "joint"), ("Marginal", "marginal")):
        combined = pd.read_csv(REPO_ROOT / folder / "combined_results.csv")
        combined["circuit"] = combined["experiment"].map(ci_reanalysis.CIRCUIT_ALIASES)
        assert combined["circuit"].notna().all()
        mine = legacy_cells[legacy_cells["paradigm"] == paradigm]

        base = combined[combined["case"] == "baseline"]
        merged = base.merge(
            mine[mine["metric"] == "type_i_error"][["circuit", "d", "m", "method", "rate"]],
            left_on=["circuit", "n_qubits", "m_samples", "test_method"],
            right_on=["circuit", "d", "m", "method"], how="outer", indicator=True)
        assert (merged["_merge"] == "both").all()
        assert np.allclose(merged["type_i_error"], merged["rate"])

        false = combined[combined["case"] == "falsecase"]
        long = false.melt(id_vars=["circuit", "n_qubits", "m_samples", "test_method"],
                          value_vars=list(ci_reanalysis.FAULT_COLUMNS),
                          var_name="metric", value_name="csv_rate")
        merged = long.merge(
            mine[mine["metric"] != "type_i_error"][["circuit", "d", "m", "method",
                                                    "metric", "rate"]],
            left_on=["circuit", "n_qubits", "m_samples", "test_method", "metric"],
            right_on=["circuit", "d", "m", "method", "metric"], how="outer",
            indicator=True)
        assert (merged["_merge"] == "both").all()
        assert np.allclose(merged["csv_rate"], merged["rate"])


def test_legacy_baseline_counts_sum_to_the_replication_budget(legacy_cells):
    baseline = legacy_cells[legacy_cells["case"] == "baseline"]
    assert (baseline["n"] == ci_reanalysis.LEGACY_N_REPS).all()
    assert not baseline["reconstructed"].any()
    assert legacy_cells[legacy_cells["case"] == "falsecase"]["reconstructed"].all()


def test_ci_reanalysis_cli_writes_both_csvs(tmp_path):
    import pandas as pd

    rc = ci_reanalysis.main(["--out-dir", str(tmp_path)])
    assert rc == 0
    per_cell = pd.read_csv(tmp_path / "per_cell_ci.csv")
    reps = pd.read_csv(tmp_path / "representative_cells_ci.csv")
    assert len(per_cell) == 17_040
    assert len(reps) > 0
    assert set(reps["rule"]) <= {rule for rule, _ in ci_reanalysis.REPRESENTATIVE_RULES}
    assert (per_cell["wilson_lo"] <= per_cell["rate"] + 1e-12).all()
    assert (per_cell["rate"] <= per_cell["wilson_hi"] + 1e-12).all()
    interior = per_cell[(per_cell["k"] > 0) & (per_cell["k"] < per_cell["n"])]
    assert (interior["cp_half"] >= interior["wilson_half"] - 1e-12).all()
    assert per_cell["wilson_half"].max() <= 0.031


def test_ci_reanalysis_reads_harness_parquet(tiny_grid):
    cells = ci_reanalysis.load_harness_cells(tiny_grid)
    assert len(cells) == len(_tiny_rows())
    assert not cells["reconstructed"].any()
    assert (cells["n"] == 100).all()
    baseline = cells[cells["case"] == "baseline"]
    assert (baseline["metric"] == "type_i_error").all()
    assert np.allclose(baseline["k"].to_numpy(dtype=float),
                       baseline["rate"].to_numpy(dtype=float) * 100)


def test_add_intervals_and_representatives_on_parquet(tiny_grid):
    ci = ci_reanalysis.add_intervals(ci_reanalysis.load_harness_cells(tiny_grid))
    reps = ci_reanalysis.representative_cells(ci)
    assert not reps.empty
    assert "rule" in reps.columns
    summary = ci_reanalysis.precision_summary(ci)
    assert summary["n_cells"] == len(ci)
    assert all(ok for _, ok, _ in ci_reanalysis.check_expected(summary))




def test_crossover_of_a_hand_built_curve():
    m = np.array([50, 100, 200, 400], dtype=float)
    k = np.array([100, 500, 850, 990], dtype=float)
    n = np.full(4, 1000.0)
    out = crossover_analysis._crossover_one(m, k, n)
    assert out["censoring"] == "interior"
    assert out["m_cross_first"] == 200.0
    assert out["m_cross_interp"] == pytest.approx(100 + (0.80 - 0.5) / 0.35 * 100, abs=1e-9)
    assert out["monotone_in_m"] is True
    assert out["m_cross_wilson_lo"] == 200.0


def test_crossover_censoring_flags():
    n = np.full(3, 1000.0)
    m = np.array([50, 100, 200], dtype=float)
    left = crossover_analysis._crossover_one(m, np.array([900.0, 950, 990]), n)
    assert left["censoring"] == "left" and left["m_cross_first"] == 50.0
    right = crossover_analysis._crossover_one(m, np.array([10.0, 20, 30]), n)
    assert right["censoring"] == "right" and math.isnan(right["m_cross_first"])


def test_apriori_sizes_reproduce_i1(tmp_path):
    reference = crossover_analysis.apriori_reference_table()
    checks = crossover_analysis.check_expected(reference)
    failures = [(claim, detail) for claim, ok, detail in checks if not ok]
    assert not failures, failures


def test_implied_effect_inverts_the_sizing_formula():
    from harness import sample_size

    w = crossover_analysis.implied_w(500, 6)
    assert 0.0 < w < 1.0
    assert sample_size.chisq_joint_n(w, 2 ** 6 - 1) == pytest.approx(500, abs=2)
    delta = crossover_analysis.implied_delta(1000, 4)
    assert 0.0 < delta < 0.5
    assert sample_size.exact_binomial_power(1000, 0.5, delta, 0.05 / 4) >= 0.80


def test_crossover_cli_on_real_csvs(tmp_path):
    import pandas as pd

    rc = crossover_analysis.main(["--out-dir", str(tmp_path)])
    assert rc == 0
    table = pd.read_csv(tmp_path / "a_priori_vs_empirical_crossover.csv")
    assert len(table) == (4 * 6 * 11 + 3 * 6 * 9) * 3 == 1278
    assert set(table["censoring"]) <= {"left", "interior", "right"}
    assert (table.loc[table["paradigm"] == "marginal", "a_priori_m"] >= 5900).all()
    interior = table[table["censoring"] == "interior"]
    assert interior["implied_effect"].notna().all()
    assert (interior["m_cross_first"] >= interior["m_min"]).all()


def test_crossover_on_parquet_only(tiny_grid):
    table = crossover_analysis.build(None, tiny_grid)
    assert not table.empty
    assert {"a_priori_m", "implied_effect", "censoring"} <= set(table.columns)
    assert table["paradigm"].nunique() == 2




def test_i2_claims_all_reproduce():
    baseline = i2_verification.load_baseline()
    claims = i2_verification.build_claims(baseline)
    failed = claims[claims["verdict"] == "FAIL"]
    assert failed.empty, failed.to_string(index=False)
    assert len(claims) == len(i2_verification.MANUSCRIPT)


def test_i2_combined_csv_still_matches_the_raw_counts():
    baseline = i2_verification.load_baseline()
    ok, detail = i2_verification.cross_check_against_raw(baseline)
    assert ok, detail


def test_i2_adaptive_type_i_rises_with_d():
    baseline = i2_verification.load_baseline()
    per_d, trend = i2_verification.d_trend(baseline, "auto")
    assert trend["spearman_rho"] > 0
    assert trend["spearman_p"] < 0.01
    assert per_d["median_type_i"].iloc[-1] > per_d["median_type_i"].iloc[0]


def test_i2_partition_difference_is_small_and_non_directional():
    baseline = i2_verification.load_baseline()
    partition = i2_verification.partition_means(baseline)
    assert partition["abs_difference"].max() < 0.005
    assert partition["difference"].min() < 0 < partition["difference"].max()
    brittle = i2_verification.partition_brittleness(baseline)
    assert brittle["n_partitions"] == 31
    assert 0.25 < brittle["frac_entangling_larger"] < 0.75


def test_i2_cli_returns_zero(tmp_path, capsys):
    rc = i2_verification.main(["--out-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "7/7 claims reproduce" in out
    assert (tmp_path / "i2_claims.csv").exists()




def _is_png(path: Path) -> bool:
    return path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figures_render_every_family_on_the_fixture(tiny_grid, tmp_path):
    df = figures.load_campaign(tiny_grid)
    written = figures.render(df, tmp_path)
    for family in ("power_typei", "frontier", "c3", "en"):
        assert written[family], f"{family} produced no figure"
        for path in written[family]:
            assert path.exists() and _is_png(path) and path.stat().st_size > 1000
    assert written["noisy"] == []


def test_noisy_family_renders_on_the_i5_fixture(tiny_noisy, tmp_path):
    df = figures.load_campaign(tiny_noisy)
    paths = figures.noisy_two_regime(df, tmp_path)
    assert len(paths) == 1
    assert _is_png(paths[0])


def test_figures_are_deterministic(tiny_grid, tmp_path):
    df = figures.load_campaign(tiny_grid)
    first = figures.power_typei_grid(df, tmp_path / "a")
    second = figures.power_typei_grid(df, tmp_path / "b")
    assert [p.name for p in first] == [p.name for p in second]
    for a, b in zip(first, second):
        assert a.read_bytes() == b.read_bytes(), f"{a.name} is not byte-reproducible"


def test_power_typei_figure_ignores_the_other_paradigms_methods(tiny_grid, tmp_path):
    full = figures.load_campaign(tiny_grid)
    joint_only = full[full["paradigm"] == "joint"]
    a = figures.power_typei_grid(full, tmp_path / "full")
    b = figures.power_typei_grid(joint_only, tmp_path / "joint")
    joint_a = {p.name: p for p in a if p.name.startswith("joint_")}
    joint_b = {p.name: p for p in b}
    assert set(joint_a) == set(joint_b) and joint_b
    for name, path in joint_b.items():
        assert joint_a[name].read_bytes() == path.read_bytes()


def test_c3_declines_a_single_paradigm_grid(tiny_grid, tmp_path):
    df = figures.load_campaign(tiny_grid)
    assert figures.c3_separation(df, tmp_path / "both")
    assert figures.c3_separation(df[df["paradigm"] == "joint"], tmp_path / "one") == []


def test_figure_rates_carry_wilson_intervals(tiny_grid):
    df = figures.load_campaign(tiny_grid)
    assert (df["wilson_lo"] <= df["rate"] + 1e-12).all()
    assert (df["rate"] <= df["wilson_hi"] + 1e-12).all()
    assert (df["n_cell"] == 100).all()


def test_method_colors_are_stable_regardless_of_the_grid():
    assert figures.method_color("mmdagg") == figures.method_color("mmdagg", ("a", "b"))
    unknown = figures.method_color("zzz_new_test", ("zzz_new_test", "aaa_other"))
    assert unknown == figures.method_color("zzz_new_test", ("aaa_other", "zzz_new_test"))


def test_unknown_family_is_rejected(tiny_grid, tmp_path):
    df = figures.load_campaign(tiny_grid)
    with pytest.raises(ValueError, match="unknown figure family"):
        figures.render(df, tmp_path, ("nope",))


def test_figures_cli(tiny_grid, tmp_path, capsys):
    rc = figures.main(["--campaign", str(tiny_grid), "--out-dir", str(tmp_path),
                       "--families", "power_typei,en"])
    assert rc == 0
    assert "figure(s) ->" in capsys.readouterr().out
    assert list(tmp_path.glob("*.png"))
