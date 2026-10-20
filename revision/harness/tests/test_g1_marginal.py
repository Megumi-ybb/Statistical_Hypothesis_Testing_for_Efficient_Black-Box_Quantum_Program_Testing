
import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.random import default_rng
from scipy.stats import binom, binomtest, chi2

_HARNESS = Path(__file__).resolve().parents[1]
_REPO = _HARNESS.parents[1]
if str(_HARNESS.parent) not in sys.path:
    sys.path.insert(0, str(_HARNESS.parent))

from harness import sample_size as ss
from harness import stats_marginal as sm

PROD_EXP = _REPO / "Marginal" / "Experiment" / "marginal_QFTtestfalse.py"
PROD_SIM = _REPO / "Marginal" / "Simulation" / "marginal_simulation.py"
BENCH = _HARNESS / "bench_runtime.py"

_PROD_NAMES = ["chi_square_marginal_test", "exact_binomial_test",
               "chi_square_marginal_auto_test", "sequential_chi_square_test",
               "_sanitize_R", "_nearest_correlation", "chi_square_bootstrap_test",
               "sample_bernoulli_copula"]


def _load_extract_defs():
    tree = ast.parse(BENCH.read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "extract_defs")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(BENCH), "exec"), ns)
    return ns["extract_defs"]


def _load_production(path):
    extract_defs = _load_extract_defs()
    ns = {"np": np, "chi2": chi2, "binom": binom, "math": math,
          "default_rng": default_rng}
    exec(extract_defs(str(path), _PROD_NAMES), ns)
    return ns


PROD = _load_production(PROD_SIM)
PROD_E = _load_production(PROD_EXP)



def _cells(n=60, seed=20260726):
    rng = default_rng(seed)
    out = []
    for i in range(n):
        d = int(rng.integers(2, 13))
        m = int(rng.choice([20, 60, 200, 800]))
        p0 = rng.uniform(0.05, 0.95, d)
        if i % 3 == 0:
            p_gen = p0
        elif i % 3 == 1:
            p_gen = np.clip(p0 + rng.normal(0.0, 0.08, d), 0.02, 0.98)
        else:
            p_gen = p0.copy()
            p_gen[rng.integers(d)] = float(np.clip(p0[0] + 0.25, 0.02, 0.98))
        X = (rng.random((m, d)) < p_gen).astype(int)
        out.append((X, p0, m, d))
    return out


CELLS = _cells()
CELL_IDS = [f"c{i}_d{c[3]}_m{c[2]}" for i, c in enumerate(CELLS)]


def test_cells_span_the_grid():
    ds = {c[3] for c in CELLS}
    ms = {c[2] for c in CELLS}
    assert len(CELLS) >= 50
    assert min(ds) == 2 and max(ds) >= 11
    assert ms == {20, 60, 200, 800}



@pytest.mark.parametrize("cell", CELLS, ids=CELL_IDS)
def test_exact_binomial_matches_production(cell):
    X, p0, _, _ = cell
    ref = PROD["exact_binomial_test"](X, p0, alpha=0.05)
    ref_e = PROD_E["exact_binomial_test"](X, p0, alpha=0.05)
    got = sm.exact_binomial_bonferroni(X, p0, alpha=0.05)
    assert got["p_value"] == ref["p_value"] == ref_e["p_value"]
    assert np.array_equal(got["p_values_individual"],
                          np.asarray(ref["p_values_individual"]))
    assert got["reject"] == ("Fail to reject" not in ref["conclusion"])


@pytest.mark.parametrize("alternative", ["less", "greater"])
def test_exact_binomial_one_sided_matches_production(alternative):
    X, p0, _, _ = CELLS[1]
    ref = PROD["exact_binomial_test"](X, p0, alpha=0.05, alternative=alternative)
    got = sm.exact_binomial_bonferroni(X, p0, alpha=0.05, alternative=alternative)
    assert got["p_value"] == ref["p_value"]


def test_exact_binomial_midp_matches_production():
    X, p0, _, _ = CELLS[2]
    ref = PROD["exact_binomial_test"](X, p0, alpha=0.05, midp=True)
    got = sm.exact_binomial_bonferroni(X, p0, alpha=0.05, midp=True)
    assert got["p_value"] == ref["p_value"]



@pytest.mark.parametrize("cell", CELLS, ids=CELL_IDS)
def test_adaptive_matches_production(cell):
    X, p0, _, _ = cell
    ref = PROD["chi_square_marginal_auto_test"](X, p0, alpha=0.05, verbose=False)
    ref_e = PROD_E["chi_square_marginal_auto_test"](X, p0, alpha=0.05, verbose=False)
    got = sm.adaptive_test(X, p0, alpha=0.05)
    assert got["p_value"] == ref["p_value"] == ref_e["p_value"]
    assert got["test_name"] == ref["test_name"]
    assert got["reject"] == ("Fail to reject" not in ref["conclusion"])
    assert got["branch"] == ("pearson" if ref["test_name"] == "Chi-square Test"
                             else "exact_binomial")


def test_adaptive_exercises_both_branches():
    branches = {sm.adaptive_test(X, p0, alpha=0.05)["branch"] for X, p0, _, _ in CELLS}
    assert branches == {"pearson", "exact_binomial"}


@pytest.mark.parametrize("cell", CELLS[::5], ids=CELL_IDS[::5])
def test_cochran_gate_matches_production(cell):
    X, p0, m, _ = cell
    gate = sm.cochran_gate(m, p0)
    ref = PROD["chi_square_marginal_auto_test"](X, p0, alpha=0.05, verbose=False)
    used_chi2 = ref["test_name"] == "Chi-square Test"
    assert (gate["condition_i"] and gate["condition_ii"]) == used_chi2



@pytest.mark.parametrize("cell", CELLS, ids=CELL_IDS)
def test_sequential_matches_production(cell):
    X, p0, _, _ = cell
    ref = PROD["sequential_chi_square_test"](X, p0, alpha=0.05, min_n=30,
                                             check_interval=50)
    got = sm.sequential_test(X, p0, alpha=0.05, min_n=30, check_interval=50)
    assert got["p_value"] == ref["p_value"]
    assert got["n_used"] == ref["n_used"]
    assert got["decision"] == ref["decision"]
    assert got["n_looks_planned"] == ref["n_looks_planned"]
    assert got["n_looks_used"] == ref["n_looks_used"]
    assert got["reject"] == ("Fail to reject" not in ref["conclusion"])
    assert [h["p_value"] for h in got["look_history"]] == \
           [h["p_value"] for h in ref["look_history"]]


def test_sequential_default_check_interval_is_the_call_site_value():
    X, p0, _, _ = CELLS[4]
    assert sm.sequential_test(X, p0) == sm.sequential_test(X, p0, check_interval=50)


def test_sequential_n_used_bounded_by_m():
    for X, p0, m, _ in CELLS[:20]:
        res = sm.sequential_test(X, p0, alpha=0.05)
        assert 1 <= res["n_used"] <= m
        assert res["stopped_early"] == (res["n_used"] < m)



@pytest.mark.parametrize("cell", CELLS[::2], ids=CELL_IDS[::2])
def test_bootstrap_matches_production(cell):
    X, p0, _, _ = cell
    R = sm.estimate_R(X)
    ref = PROD["chi_square_bootstrap_test"](X, p0, alpha=0.05, R_assumed=R,
                                            n_bootstrap=100, seed=17)
    got = sm.bootstrap_test(X, p0, alpha=0.05, n_bootstrap=100,
                            rng=default_rng(17), R_assumed=R)
    auto = sm.bootstrap_test(X, p0, alpha=0.05, n_bootstrap=100, rng=default_rng(17))
    assert got["p_value"] == ref["p_value"] == auto["p_value"]
    assert got["reject"] == ("Fail to reject" not in ref["conclusion"])


def test_bootstrap_helpers_are_verbatim():
    rng = default_rng(0)
    R = np.clip(rng.normal(0, 0.3, (6, 6)), -0.9, 0.9)
    assert np.array_equal(sm._sanitize_R(R), PROD["_sanitize_R"](R))
    assert np.array_equal(sm._nearest_correlation(R), PROD["_nearest_correlation"](R))
    Rn = sm._nearest_correlation(R)
    p = np.full(6, 0.4)
    a = sm.sample_bernoulli_copula(50, p, Rn, default_rng(5))
    b = PROD["sample_bernoulli_copula"](50, p, Rn, default_rng(5))
    assert np.array_equal(a, b)


def test_estimate_R_matches_experiment_call_site():
    n_guarded = 0
    for X, _, _, _ in CELLS:
        if np.any(np.std(X, axis=0) < 1e-10):
            n_guarded += 1
            expect = np.eye(X.shape[1])
        else:
            expect = np.corrcoef(X, rowvar=False)
            expect = 0.5 * (expect + expect.T)
            np.fill_diagonal(expect, 1.0)
        assert np.array_equal(sm.estimate_R(X), expect)
    assert n_guarded > 0, "grid never exercised the constant-column guard"
    X_const = np.zeros((40, 3), dtype=int)
    X_const[:20, 0] = 1
    X_const[:10, 1] = 1
    assert np.array_equal(sm.estimate_R(X_const), np.eye(3))


def test_bootstrap_rejects_small_b():
    X, p0, _, _ = CELLS[0]
    with pytest.raises(ValueError):
        sm.bootstrap_test(X, p0, n_bootstrap=99, rng=default_rng(0))



def test_scipy_convention_equals_binomtest():
    rng = default_rng(11)
    for _ in range(300):
        n = int(rng.integers(5, 400))
        p = float(rng.uniform(0.01, 0.99))
        k = int(rng.integers(0, n + 1))
        got = sm._binom_two_sided_p(k, n, p, convention="scipy")
        assert got == pytest.approx(binomtest(k, n, p).pvalue, abs=1e-13)


def test_production_convention_differs_only_in_the_deep_tail():
    rng = default_rng(12)
    n_diff = 0
    n_flip = 0
    for _ in range(2000):
        n = int(rng.integers(20, 400))
        p = float(rng.uniform(0.05, 0.95))
        k = int(rng.integers(0, n + 1))
        a = sm._binom_two_sided_p(k, n, p, convention="production")
        b = sm._binom_two_sided_p(k, n, p, convention="scipy")
        if a != b:
            n_diff += 1
            assert max(a, b) < 1e-12, (n, p, k, a, b)
        for alpha in (0.05, 0.05 / 12, 0.05 / 17):
            n_flip += (a < alpha) != (b < alpha)
    assert n_diff > 0
    assert n_flip == 0


def test_convention_flag_is_plumbed_through():
    X, p0, _, _ = CELLS[7]
    for fn in (sm.exact_binomial_bonferroni, sm.adaptive_test):
        prod = fn(X, p0, alpha=0.05, two_sided_convention="production")
        sci = fn(X, p0, alpha=0.05, two_sided_convention="scipy")
        assert prod["reject"] == sci["reject"]
    with pytest.raises(ValueError):
        sm.exact_binomial_bonferroni(X, p0, two_sided_convention="nope")



def test_g_test_closed_form():
    X = np.zeros((100, 2), dtype=int)
    X[:60, 0] = 1
    X[:50, 1] = 1
    p0 = np.array([0.5, 0.5])
    res = sm.g_test_marginal(X, p0, alpha=0.05)
    g0 = 2 * (60 * math.log(60 / 50) + 40 * math.log(40 / 50))
    assert res["statistics"][0] == pytest.approx(g0, rel=1e-12)
    assert res["statistics"][1] == pytest.approx(0.0, abs=1e-12)
    p_ind = chi2.sf(np.array([g0, 0.0]), 1)
    assert res["p_value"] == pytest.approx(min(1.0, 2 * p_ind.min()), rel=1e-12)


def test_g_test_impossible_cell_rejects():
    X = np.zeros((50, 2), dtype=int)
    X[0, 0] = 1
    res = sm.g_test_marginal(X, np.array([0.0, 0.5]), alpha=0.05)
    assert not np.isfinite(res["statistics"][0])
    assert res["p_value"] == 0.0 and res["reject"]


def test_g_test_agrees_with_exact_binomial_when_asymptotics_hold():
    rng = default_rng(21)
    disagree = 0
    for _ in range(200):
        d = 4
        p0 = rng.uniform(0.2, 0.8, d)
        p_gen = p0 if rng.random() < 0.5 else np.clip(p0 + rng.normal(0, 0.05, d), 0.05, 0.95)
        X = (rng.random((2000, d)) < p_gen).astype(int)
        g = sm.g_test_marginal(X, p0, alpha=0.05)
        e = sm.exact_binomial_bonferroni(X, p0, alpha=0.05)
        disagree += g["reject"] != e["reject"]
    assert disagree <= 10


def test_g_test_type_i_is_near_nominal():
    rng = default_rng(22)
    d, m, reps = 4, 500, 400
    p0 = np.array([0.3, 0.5, 0.6, 0.45])
    rej = sum(sm.g_test_marginal((rng.random((m, d)) < p0).astype(int), p0,
                                 alpha=0.05)["reject"] for _ in range(reps))
    assert rej / reps < 0.12



def _sprt_mc(p_gen, p0, delta, *, d=1, alpha=0.05, beta=0.2,
             directions=("upper", "lower"), reps=1500, seed=0, batch=1):
    rng = default_rng(seed)
    p0v = np.full(d, p0)
    ns, rej, trunc = [], 0, 0
    for _ in range(reps):
        res = sm.sprt_test(sm.bernoulli_stream(np.full(d, p_gen), d, rng), p0v,
                           delta=delta, alpha=alpha, beta=beta,
                           directions=directions, batch=batch, max_n=200_000)
        ns.append(res["n_used"])
        rej += res["reject"]
        trunc += res["truncated"]
    return float(np.mean(ns)), rej / reps, trunc


def test_sprt_boundaries_and_alpha_split():
    res = sm.sprt_test(sm.bernoulli_stream(0.5, 4, default_rng(0)), np.full(4, 0.5),
                       delta=0.1, alpha=0.05, beta=0.2)
    assert res["alpha_dir"] == pytest.approx(0.05 / 4 / 2)
    assert res["log_A"] == pytest.approx(math.log(0.8 / res["alpha_dir"]))
    assert res["log_B"] == pytest.approx(math.log(0.2 / (1 - res["alpha_dir"])))
    no_split = sm.sprt_test(sm.bernoulli_stream(0.5, 4, default_rng(0)),
                            np.full(4, 0.5), delta=0.1, direction_split=False)
    assert no_split["alpha_dir"] == pytest.approx(0.05 / 4)
    assert math.isnan(res["p_value"]) and isinstance(res["reject"], bool)


@pytest.mark.parametrize("under,p_gen", [("H0", 0.5), ("H1", 0.6)])
def test_sprt_expected_n_matches_wald(under, p_gen):
    approx = sm.sprt_expected_n(0.5, 0.1, 0.05, 0.2, direction="upper", under=under)
    mean_n, _, trunc = _sprt_mc(p_gen, 0.5, 0.1, directions=("upper",),
                                reps=1500, seed=31)
    assert trunc == 0
    assert 1.0 <= mean_n / approx <= 1.25, (mean_n, approx)


def test_sprt_error_rates_one_sided():
    _, type_i, _ = _sprt_mc(0.5, 0.5, 0.1, directions=("upper",), reps=1500, seed=32)
    _, power, _ = _sprt_mc(0.6, 0.5, 0.1, directions=("upper",), reps=1500, seed=33)
    assert type_i <= 0.075
    assert power >= 0.75


def test_sprt_two_sided_detects_both_directions():
    _, type_i, _ = _sprt_mc(0.5, 0.5, 0.1, reps=1200, seed=34)
    _, up, _ = _sprt_mc(0.6, 0.5, 0.1, reps=1200, seed=35)
    _, down, _ = _sprt_mc(0.4, 0.5, 0.1, reps=1200, seed=36)
    assert type_i <= 0.075
    assert up >= 0.75 and down >= 0.75


def test_sprt_bonferroni_controls_family_type_i():
    _, type_i, _ = _sprt_mc(0.5, 0.5, 0.1, d=4, reps=1000, seed=37)
    assert type_i <= 0.075


def test_sprt_batching_and_truncation():
    X = np.tile(np.array([[0], [1], [1], [1], [1], [1], [1], [1], [1], [1]]), (40, 1))
    r1 = sm.sprt_test(sm.array_stream(X), np.array([0.5]), delta=0.1, batch=1)
    r5 = sm.sprt_test(sm.array_stream(X), np.array([0.5]), delta=0.1, batch=5)
    assert r1["decision"] == r5["decision"] == "reject"
    assert r5["n_used"] >= r1["n_used"]
    short = sm.sprt_test(sm.array_stream(X[:5]), np.array([0.5]), delta=0.1)
    assert short["decision"] == "inconclusive"
    assert short["truncated"] and not short["reject"] and not short["valid"]
    capped = sm.sprt_test(sm.bernoulli_stream(0.5, 2, default_rng(1)),
                          np.full(2, 0.5), delta=0.001, max_n=50)
    assert capped["n_used"] == 50 and capped["truncated"]


def test_sprt_validates_inputs():
    with pytest.raises(ValueError):
        sm.sprt_test(sm.array_stream(np.zeros((5, 2), int)), np.full(2, 0.5), delta=0.0)
    with pytest.raises(ValueError):
        sm.sprt_test(sm.array_stream(np.full((5, 2), 3)), np.full(2, 0.5), delta=0.1)



@pytest.mark.parametrize("p0,delta,anchor", [(0.5, 0.02, 4904.0), (0.1, 0.04, 484.0)])
def test_normal_approx_reproduces_plan_anchors(p0, delta, anchor):
    assert abs(ss.normal_approx_n(p0, delta, 0.05, 0.2) - anchor) <= 1.0


@pytest.mark.parametrize("p0,delta,anchor", [(0.5, 0.02, 4904), (0.1, 0.04, 484)])
def test_exact_binomial_n_near_plan_anchors(p0, delta, anchor):
    m = ss.exact_binomial_n(p0, delta, 0.05, 0.2)
    assert abs(m - anchor) / anchor < 0.02
    assert ss.exact_binomial_power(m, p0, delta, 0.05) >= 0.80
    assert ss.exact_binomial_power(m - 1, p0, delta, 0.05) < 0.80


def test_exact_binomial_n_stable_mode_is_larger_and_holds():
    m = ss.exact_binomial_n(0.1, 0.04, 0.05, 0.2, mode="stable", stable_window=25)
    assert m >= ss.exact_binomial_n(0.1, 0.04, 0.05, 0.2)
    assert all(ss.exact_binomial_power(m + j, 0.1, 0.04, 0.05) >= 0.80
               for j in range(25))


@pytest.mark.parametrize("p0,delta", [(0.3, 0.1), (0.5, 0.08), (0.2, 0.12)])
def test_exact_binomial_n_fast_search_equals_full_scan(p0, delta):
    assert (ss.exact_binomial_n(p0, delta, 0.05, 0.2)
            == ss.exact_binomial_n(p0, delta, 0.05, 0.2, full_scan=True))


def test_exact_binomial_n_monotonicity():
    sizes = [ss.exact_binomial_n(0.5, dl, 0.05, 0.2) for dl in (0.05, 0.1, 0.2)]
    assert sizes == sorted(sizes, reverse=True)
    assert ss.exact_binomial_n(0.5, 0.1, 0.01, 0.2) > ss.exact_binomial_n(0.5, 0.1, 0.05, 0.2)
    assert ss.exact_binomial_n(0.5, 0.1, 0.05, 0.1) > ss.exact_binomial_n(0.5, 0.1, 0.05, 0.2)


def test_two_sided_pvalues_match_the_test_implementation():
    for m, p0 in [(30, 0.5), (75, 0.1), (120, 0.83)]:
        vec = ss.two_sided_pvalues(m, p0)
        for k in (0, 1, m // 3, m // 2, m):
            assert vec[k] == pytest.approx(
                sm._binom_two_sided_p(k, m, p0, convention="production"), abs=1e-15)


def test_bonferroni_marginal_n_matches_plan_range():
    sizes = {d: ss.bonferroni_marginal_n(0.5, 0.02, 0.05, 0.2, d) for d in (2, 6, 12)}
    approx = {d: ss.bonferroni_marginal_n_normal(0.5, 0.02, 0.05, 0.2, d)
              for d in (2, 6, 12)}
    assert 5900 <= min(approx.values()) and max(approx.values()) <= 8600
    assert all(5900 <= n <= 8700 for n in sizes.values())
    assert sizes[2] < sizes[6] < sizes[12]
    assert sizes[2] > ss.exact_binomial_n(0.5, 0.02, 0.05, 0.2)


@pytest.mark.parametrize("w", [0.3, 0.5])
@pytest.mark.parametrize("df", [15, 63, 255])
def test_chisq_joint_n_in_plan_range(w, df):
    n = ss.chisq_joint_n(w, df, 0.05, 0.2)
    assert 65 <= n <= 900
    assert ss.chisq_joint_power(n, w, df, 0.05) >= 0.80
    assert ss.chisq_joint_power(n - 1, w, df, 0.05) < 0.80


def test_chisq_joint_n_matches_cohen_table():
    assert ss.chisq_joint_n(0.3, 1, 0.05, 0.2) == pytest.approx(88, abs=2)


def test_chisq_joint_n_monotonicity():
    assert ss.chisq_joint_n(0.5, 63) < ss.chisq_joint_n(0.3, 63)
    assert ss.chisq_joint_n(0.3, 15) < ss.chisq_joint_n(0.3, 255)


def test_cohen_w():
    p0 = np.array([0.25, 0.25, 0.25, 0.25])
    p1 = np.array([0.40, 0.20, 0.20, 0.20])
    expect = math.sqrt((0.15 ** 2 + 3 * 0.05 ** 2) / 0.25)
    assert ss.cohen_w(p1, p0) == pytest.approx(expect, rel=1e-12)
    assert ss.cohen_w(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == float("inf")


def test_sizing_rejects_bad_input():
    for args in [(0.0, 0.02), (1.0, 0.02), (0.5, 0.0)]:
        with pytest.raises(ValueError):
            ss.normal_approx_n(*args)
    with pytest.raises(ValueError):
        ss.chisq_joint_n(0.0, 5)
    with pytest.raises(ValueError):
        ss.chisq_joint_n(0.3, 0)



def test_every_test_returns_the_contract_keys():
    X, p0, _, _ = CELLS[3]
    results = [
        sm.exact_binomial_bonferroni(X, p0, alpha=0.05),
        sm.adaptive_test(X, p0, alpha=0.05),
        sm.g_test_marginal(X, p0, alpha=0.05),
        sm.bootstrap_test(X, p0, alpha=0.05, n_bootstrap=100, rng=default_rng(0)),
        sm.sequential_test(X, p0, alpha=0.05),
        sm.sprt_test(sm.array_stream(X), p0, delta=0.05),
    ]
    for res in results:
        assert isinstance(res["p_value"], float)
        assert isinstance(res["reject"], bool)


def test_safe_call_preserves_the_error():
    X = np.zeros((100, 2), dtype=int)
    res = sm.safe_call(sm.pearson_marginal_test, X, np.array([0.0, 0.5]), alpha=0.05)
    assert res["reject"] is False and math.isnan(res["p_value"])
    assert "ValueError" in res["error"]
    ok = sm.safe_call(sm.exact_binomial_bonferroni, X, np.array([0.0, 0.5]), alpha=0.05)
    assert "error" not in ok


def test_no_global_rng_use():
    X, p0, _, _ = CELLS[5]
    a = sm.bootstrap_test(X, p0, alpha=0.05, n_bootstrap=100, rng=default_rng(9))
    np.random.random(10_000)
    b = sm.bootstrap_test(X, p0, alpha=0.05, n_bootstrap=100, rng=default_rng(9))
    assert a["p_value"] == b["p_value"]


def test_sequential_exact_branch_uses_exact_binomial_at_every_look():
    import numpy as np
    from harness import stats_marginal as sm

    rng = np.random.default_rng(20260912)
    d, m = 4, 400
    p0 = np.full(d, 0.5)
    X = (rng.random((m, d)) < 0.5).astype(int)
    res = sm.sequential_test(X, p0, alpha=0.05, branch="exact")
    assert res["branch"] == "exact" and res["look_history"]
    for look in res["look_history"]:
        ref = sm.exact_binomial_bonferroni(X[:look["n"]], p0, alpha=look["threshold"])
        assert look["p_value"] == ref["p_value"] and look["test_used"] == ref["test_name"]
    assert sm.sequential_test(X, p0, alpha=0.05)["branch"] == "adaptive"
