
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness import i9_c3, r19_lumping_breakdown, sampling, stats_joint, stats_marginal

ALPHA = 0.05


def keys_of(d: int) -> list[str]:
    return ["".join(str((k >> j) & 1) for j in range(d)) for k in range(2 ** d)]


def uniform(d: int) -> dict[str, float]:
    return {k: 1.0 / 2 ** d for k in keys_of(d)}


def dirichlet_p0(d: int, rng: np.random.Generator, alpha: float = 1.0) -> dict[str, float]:
    probs = rng.dirichlet(np.full(2 ** d, alpha))
    return {k: float(p) for k, p in zip(keys_of(d), probs)}


def sparse_p0(d: int, n_support: int, rng: np.random.Generator) -> dict[str, float]:
    ks = keys_of(d)
    chosen = rng.choice(len(ks), size=n_support, replace=False)
    probs = rng.dirichlet(np.ones(n_support))
    out = {k: 0.0 for k in ks}
    for idx, p in zip(chosen, probs):
        out[ks[idx]] = float(p)
    return out




@pytest.mark.parametrize("style", ["comonotone", "independent", "pairwise", "auto"])
def test_marginals_preserved_exactly(style):
    rng = np.random.default_rng(11)
    cases = [uniform(2), uniform(3), uniform(4),
             dirichlet_p0(3, rng), dirichlet_p0(4, rng, alpha=0.3),
             sparse_p0(5, 6, rng), sparse_p0(6, 8, rng)]
    checked = 0
    for p0 in cases:
        mu0 = i9_c3.marginals(p0)
        for eps in (0.001, 0.01, 0.05, 0.2, 0.9):
            try:
                built = i9_c3.make_dependence_only(p0, eps, rng, style=style)
            except i9_c3.NotAdmissible:
                continue
            if built.epsilon_max_style <= 0:
                continue
            q = built.q
            assert abs(sum(q.values()) - 1.0) < 1e-12
            assert min(q.values()) >= 0.0
            assert float(np.max(np.abs(i9_c3.marginals(q) - mu0))) < 1e-12
            assert built.marginal_error < 1e-12
            checked += 1
    assert checked >= 10, "not enough admissible cases exercised"


def test_pairwise_move_is_marginal_preserving_by_construction():
    rng = np.random.default_rng(3)
    p0 = dirichlet_p0(4, rng)
    moves = i9_c3.candidate_moves(p0, rng, n_moves=3)
    assert moves
    for mv in moves:
        lose = np.array([[int(c) for c in s] for s in mv.lose]).sum(axis=0)
        gain = np.array([[int(c) for c in s] for s in mv.gain]).sum(axis=0)
        assert np.array_equal(lose, gain)
        assert set(mv.lose) != set(mv.gain)




@pytest.mark.parametrize("style", ["comonotone", "independent", "pairwise"])
def test_epsilon_monotone_and_exact(style):
    rng = np.random.default_rng(5)
    p0 = dirichlet_p0(4, rng)
    reach = i9_c3.constructible_epsilon(p0, rng, style=style)
    assert reach > 0
    grid = [0.0, 0.2 * reach, 0.5 * reach, reach, 2 * reach]
    realised = [i9_c3.make_dependence_only(p0, e, rng, style=style).epsilon for e in grid]
    assert all(b >= a - 1e-12 for a, b in zip(realised, realised[1:])), realised
    for target, got in zip(grid[:-1], realised[:-1]):
        assert abs(got - target) < 1e-10
    assert abs(realised[-1] - reach) < 1e-10
    assert i9_c3.make_dependence_only(p0, 2 * reach, rng, style=style).clipped


def test_tvd_matches_requested_epsilon_for_the_campaign_levels():
    rng = np.random.default_rng(7)
    p0 = uniform(3)
    for eps in i9_c3.EPSILON_LEVELS:
        built = i9_c3.make_dependence_only(p0, eps, rng)
        assert abs(i9_c3.tvd(p0, built.q) - eps) < 1e-12
        assert not built.clipped




def test_theorem_point_mass_and_distance_one_support():
    point = {k: (1.0 if k == "111" else 0.0) for k in keys_of(3)}
    verdict = i9_c3.admissibility(point)
    assert not verdict.admissible and verdict.n_free == 0
    assert i9_c3.max_epsilon(point) == 0.0
    with pytest.raises(i9_c3.NotAdmissible):
        i9_c3.make_dependence_only(point, 0.1, np.random.default_rng(0))

    near = {k: (0.5 if k in ("110", "111") else 0.0) for k in keys_of(3)}
    verdict = i9_c3.admissibility(near)
    assert not verdict.admissible and verdict.n_free == 1
    assert i9_c3.max_epsilon(near) == 0.0
    with pytest.raises(i9_c3.NotAdmissible):
        i9_c3.make_dependence_only(near, 0.1, np.random.default_rng(0))


def test_theorem_equivalences_on_random_nulls():
    rng = np.random.default_rng(19)
    for _ in range(40):
        d = int(rng.integers(2, 5))
        n_support = int(rng.integers(1, 2 ** d + 1))
        p0 = sparse_p0(d, n_support, rng)
        verdict = i9_c3.admissibility(p0)
        assert verdict.admissible == (verdict.n_free >= 2)
        assert verdict.admissible == (verdict.witness is not None)
        assert verdict.admissible == (i9_c3.max_epsilon(p0) > 1e-9)


def test_max_epsilon_exact_against_closed_forms():
    rng = np.random.default_rng(23)
    for _ in range(8):
        p0 = dirichlet_p0(2, rng)
        assert i9_c3.max_epsilon(p0) == pytest.approx(
            i9_c3.frechet_max_epsilon_d2(p0), abs=1e-9)
    for d in (2, 3, 4):
        info = i9_c3.max_epsilon_info(uniform(d))
        assert info["exact"] and info["method"] == "vertex-enumeration"
        assert info["value"] == pytest.approx(1.0 - 2.0 ** (1 - d), abs=1e-9)


def test_uniform_comonotone_target_is_ghz():
    q = i9_c3.comonotone_target(uniform(4))
    support = {k: v for k, v in q.items() if v > 1e-12}
    assert support == pytest.approx({"0000": 0.5, "1111": 0.5})


def test_max_epsilon_is_a_lower_bound_when_not_exact():
    p0 = uniform(6)
    info = i9_c3.max_epsilon_info(p0)
    assert not info["exact"] and "lower-bound" in info["method"]
    assert 0 < info["value"] <= 1.0 - 2.0 ** (1 - 6) + 1e-12


def test_support_preserving_option():
    rng = np.random.default_rng(29)
    p0 = sparse_p0(4, 8, rng)
    try:
        built = i9_c3.make_dependence_only(p0, 0.02, rng, style="pairwise",
                                           support_preserving=True)
    except i9_c3.NotAdmissible:
        pytest.skip("this null admits no support-preserving move")
    assert built.escaped_mass == 0.0 and not built.escapes_support
    assert built.marginal_error < 1e-12




def _rejection_rates(p_draw, p0, mu, m, reps, rng_seed):
    joint = {"chi_square": 0, "mmdagg": 0}
    marg = {"exact_binomial": 0, "adaptive": 0, "g_test_marginal": 0}
    for rep in range(reps):
        rng = np.random.default_rng([rng_seed, rep])
        states, counts = sampling.draw_counts(p_draw, m, rng)
        X = sampling.X_from_counts(states, counts, rng)
        joint["chi_square"] += stats_joint.chi_square_joint_counts(
            states, counts, p0, alpha=ALPHA)["reject"]
        joint["mmdagg"] += stats_joint.mmdagg_counts(
            states, counts, p0, alpha=ALPHA, rng=rng)["reject"]
        marg["exact_binomial"] += stats_marginal.exact_binomial_bonferroni(
            X, mu, alpha=ALPHA)["reject"]
        marg["adaptive"] += stats_marginal.adaptive_test(X, mu, alpha=ALPHA)["reject"]
        marg["g_test_marginal"] += stats_marginal.g_test_marginal(
            X, mu, alpha=ALPHA)["reject"]
    return ({k: v / reps for k, v in joint.items()},
            {k: v / reps for k, v in marg.items()})


def test_separation_joint_detects_marginal_blind():
    d, m, reps = 3, 200, 50
    p0 = uniform(d)
    mu = i9_c3.marginals(p0)
    built = i9_c3.make_dependence_only(p0, 0.75, np.random.default_rng(2))
    assert built.epsilon == pytest.approx(0.75)
    assert float(np.max(np.abs(i9_c3.marginals(built.q) - mu))) < 1e-12

    joint, marg = _rejection_rates(built.q, p0, mu, m, reps, rng_seed=12345)
    assert joint["chi_square"] >= 0.90, joint
    assert joint["mmdagg"] >= 0.90, joint
    assert marg["exact_binomial"] <= 0.16, marg
    assert marg["g_test_marginal"] <= 0.16, marg

    joint_h0, marg_h0 = _rejection_rates(p0, p0, mu, m, reps, rng_seed=999)
    assert all(r <= 0.16 for r in marg_h0.values()), marg_h0


def test_global_pearson_level_breaks_under_dependence():
    from scipy.stats import chi2

    d, m, reps = 3, 200, 400
    p0 = uniform(d)
    mu = i9_c3.marginals(p0)
    q = i9_c3.make_dependence_only(p0, 0.75, np.random.default_rng(2)).q
    theory = float(chi2.sf(chi2.ppf(1 - ALPHA, d) / d, 1))
    assert theory == pytest.approx(0.1065, abs=5e-4)

    rejects = 0
    for rep in range(reps):
        rng = np.random.default_rng([31337, rep])
        states, counts = sampling.draw_counts(q, m, rng)
        X = sampling.X_from_counts(states, counts, rng)
        res = stats_marginal.adaptive_test(X, mu, alpha=ALPHA)
        assert res["branch"] == "pearson"
        rejects += res["reject"]
    rate = rejects / reps
    assert 0.05 < rate < 0.20, rate




def test_classification_matches_the_theorem():
    pytest.importorskip("qiskit")
    rows = i9_c3.classify_benchmarks(dims=(2, 3, 4))
    by_key = {(r.circuit, r.d): r for r in rows}
    assert len(rows) == 18

    for circuit in ("BV", "QPE"):
        for d in (2, 3, 4):
            row = by_key[(circuit, d)]
            assert row.supported and row.n_support == 1
            assert not row.admissible and row.max_epsilon == 0.0
            assert "concentrated" in row.reason
    assert not by_key[("Adder", 4)].admissible
    assert by_key[("Adder", 4)].n_support == 1
    assert not by_key[("Adder", 2)].supported

    for circuit in ("Grover", "QWALK"):
        row = by_key[(circuit, 2)]
        assert not row.admissible and row.n_free == 1

    for d in (2, 3, 4):
        row = by_key[("QFT", d)]
        assert row.admissible and row.n_free == d
        assert row.max_epsilon == pytest.approx(1.0 - 2.0 ** (1 - d), abs=1e-9)
    for circuit, d in (("Grover", 3), ("Grover", 4), ("QWALK", 3), ("QWALK", 4)):
        assert by_key[(circuit, d)].admissible
        assert by_key[(circuit, d)].max_epsilon > 0

    for row in rows:
        if not row.supported:
            continue
        p0 = i9_c3.benchmark_null(row.circuit, row.d)
        assert i9_c3.admissibility(p0).admissible == row.admissible

    table = i9_c3.classification_table(rows)
    assert "admits C3" in table and len(table.splitlines()) == len(rows) + 2


def test_campaign_enumeration_and_manifest():
    pytest.importorskip("qiskit")
    from harness import runner

    dims = (2, 3, 4)
    grid = i9_c3.admissible_grid(dims)
    admitted = [e for e in grid if e.get("admissible")]
    assert {e["circuit"] for e in admitted} == {"QFT", "Grover", "QWALK"}
    for entry in admitted:
        assert entry["epsilons"], entry
        assert all(e <= entry["reach"] + 1e-12 for e in entry["epsilons"])

    cells = i9_c3.cells(dims=dims, m_points=(200, 1000), n_reps=25)
    assert cells and all(isinstance(c, runner.Cell) for c in cells)
    falsecase = [c for c in cells if c.case == "falsecase"]
    assert all(c.operator == i9_c3.C3_OPERATOR for c in falsecase)
    assert {c.campaign for c in cells} == {i9_c3.I9_CAMPAIGN}

    joint = {c for c in cells if c.paradigm == "joint"}
    for cell in joint:
        twin = next(c for c in cells if c.paradigm == "marginal"
                    and (c.circuit, c.d, c.m, c.operator, c.strength, c.case)
                    == (cell.circuit, cell.d, cell.m, cell.operator, cell.strength,
                        cell.case))
        assert runner.seed_master(twin) == runner.seed_master(cell)

    info = i9_c3.manifest(dims=dims, m_points=(200, 1000), n_reps=25)
    assert info["n_cells"] == len(cells)
    assert info["n_cells_plan_equivalent"] * 2 == info["n_cells"]
    assert info["n_excluded_circuit_widths"] > 0
    assert info["n_runs"] == 25 * len(cells)
    assert info["circuits_with_interior_marginals"] == ["QFT"]
    assert info["n_cells_marginal_restricted"] > 0


def test_marginal_method_set_avoids_the_strict_interior_crash():
    pytest.importorskip("qiskit")
    from harness import runner, sampling as _sampling, stats_marginal as _sm

    for circuit, expect_restricted in (("Grover", True), ("QFT", False)):
        cells = i9_c3.cells(dims=(3,), circuits=(circuit,), m_points=(200,), n_reps=5)
        marginal = [c for c in cells if c.paradigm == "marginal"]
        assert marginal
        for cell in marginal:
            expected = (i9_c3.MARGINAL_METHODS_INTERIOR_SAFE if expect_restricted
                        else runner.MARGINAL_METHODS)
            assert tuple(cell.methods) == tuple(expected)

        p0 = i9_c3.benchmark_null(circuit, 3)
        mu = i9_c3.marginals(p0)
        rng = np.random.default_rng(0)
        states, counts = _sampling.draw_counts(p0, 200, rng)
        X = _sampling.X_from_counts(states, counts, rng)
        for name in marginal[0].methods:
            runner._default_method(name)(
                runner.Draw(states=states, counts=counts, X=X, p0=p0, p0_marg=mu,
                            d=3, m=200), alpha=ALPHA, rng=rng)
        if expect_restricted:
            with pytest.raises(ValueError):
                _sm.bootstrap_test(X, mu, alpha=ALPHA, rng=rng)


def test_distributions_for_cell_round_trip():
    pytest.importorskip("qiskit")
    cells = i9_c3.cells(dims=(3,), circuits=("QFT",), m_points=(200,), n_reps=5)
    falsecase = [c for c in cells if c.case == "falsecase"][0]
    p0, q = i9_c3.distributions_for_cell(falsecase)
    assert i9_c3.tvd(p0, q) == pytest.approx(falsecase.strength, abs=1e-12)
    assert float(np.max(np.abs(i9_c3.marginals(q) - i9_c3.marginals(p0)))) < 1e-12
    p0b, qb = i9_c3.distributions_for_cell(falsecase)
    assert q == qb and p0 == p0b
    baseline = [c for c in cells if c.case == "baseline"][0]
    assert i9_c3.distributions_for_cell(baseline)[1] is None




def test_r19_runs_on_the_real_csv(tmp_path):
    csv = r19_lumping_breakdown.DEFAULT_CSV
    if not csv.exists():
        pytest.skip(f"{csv} not present")
    paths = r19_lumping_breakdown.run(csv, tmp_path)
    power = pd.read_csv(paths["power"])
    typei = pd.read_csv(paths["typei"])

    assert len(power) == 4860 and len(typei) == 1620
    assert set(power["fault_model"]) == {"bitflip", "asymmetric", "depolarize"}
    assert (power["successes"] == np.round(power["power"] * 1000)).all()
    assert (power["wilson_lo"] <= power["power"] + 1e-12).all()
    assert (power["wilson_hi"] >= power["power"] - 1e-12).all()
    assert power[["circuit", "d", "m", "test_method", "fault_model"]].duplicated().sum() == 0

    assert (typei["type_i"] == typei["false_positives"] / typei["n"]).all()
    assert (typei["wilson_lo"] <= typei["type_i"]).all()
    assert typei["conservative"].sum() > 0
    chi = typei[typei["test_method"] == "chi_square"]
    corner = chi[chi["conservative"]]
    assert corner["mmdagg_depolarize"].min() >= 0.68
    assert not (typei["conservative"] & typei["liberal"]).any()

    lines = r19_lumping_breakdown.summarise(power, typei)
    assert any("conservative" in line for line in lines)


def test_r19_count_reconstruction_guard():
    good = pd.Series([0.0, 0.123, 1.0])
    assert r19_lumping_breakdown.reconstruct_counts(good).tolist() == [0, 123, 1000]
    with pytest.raises(ValueError):
        r19_lumping_breakdown.reconstruct_counts(pd.Series([0.12345]))


def test_wilson_interval_known_values():
    lo, hi = r19_lumping_breakdown.wilson_interval(np.array([0.0]), np.array([1000.0]))
    assert lo[0] == 0.0 and 0.0 < hi[0] < 0.005
    lo, hi = r19_lumping_breakdown.wilson_interval(np.array([50.0]), np.array([1000.0]))
    assert lo[0] == pytest.approx(0.0381, abs=1e-3)
    assert hi[0] == pytest.approx(0.0654, abs=1e-3)
