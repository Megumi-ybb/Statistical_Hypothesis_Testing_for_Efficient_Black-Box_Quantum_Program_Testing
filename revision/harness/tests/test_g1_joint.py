from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.random import default_rng
from scipy.stats import chi2

_TESTS = Path(__file__).resolve().parent
_HARNESS = _TESTS.parent
_REVISION = _HARNESS.parent
_REPO = _REVISION.parent
for _p in (str(_HARNESS), str(_REVISION)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harness import stats_joint as sj

PRODUCTION = _REPO / "Joint" / "Experiment" / "joint_experiemnt_BVtestfalse.py"
BENCH = _HARNESS / "bench_runtime.py"


def _load_extract_defs():
    tree = ast.parse(BENCH.read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "extract_defs")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(BENCH), "exec"), ns)
    return ns["extract_defs"]


def _load_production():
    extract_defs = _load_extract_defs()
    ns = {"np": np, "default_rng": default_rng, "chi2": chi2, "math": math}
    exec(extract_defs(str(PRODUCTION),
                      ["bitstring", "chi_square_joint", "mmdagg_joint_test",
                       "simple_tolerance_joint_test"]), ns)
    return ns


PROD = _load_production()
prod_chi_square = PROD["chi_square_joint"]
prod_mmdagg = PROD["mmdagg_joint_test"]
prod_tolerance = PROD["simple_tolerance_joint_test"]
bitstring = PROD["bitstring"]



def _null(d, kind, rng):
    n_states = 2 ** d
    keys = [bitstring(k, d) for k in range(n_states)]
    if kind == "dense":
        probs = rng.dirichlet(np.ones(n_states))
    else:
        support = max(2, min(16, n_states // 2))
        probs = np.zeros(n_states)
        probs[rng.choice(n_states, support, replace=False)] = rng.dirichlet(np.ones(support))
    return keys, probs, dict(zip(keys, probs.tolist()))


def _draw(d, probs, m, case, rng):
    q = probs
    if case == "H1":
        q = 0.7 * probs + 0.3 * np.full(probs.size, 1.0 / probs.size)
        q = q / q.sum()
    idx = rng.choice(probs.size, size=m, p=q)
    X = np.array([[int(b) for b in bitstring(int(i), d)] for i in idx], dtype=int)
    uniq, cnt = np.unique(idx, return_counts=True)
    states = np.array([[int(b) for b in bitstring(int(i), d)] for i in uniq], dtype=int)
    return X, states, cnt.astype(np.int64)


def _cells():
    out = []
    for d in (2, 6, 10):
        for kind in ("dense", "sparse"):
            for m in (50, 500, 2000):
                for case in ("H0", "H1"):
                    for rep in range(3):
                        out.append((d, kind, m, case, rep))
    return out


CELLS = _cells()
assert len(CELLS) >= 100, f"G1 needs >= 100 cells, got {len(CELLS)}"


def _cell_rng(cell):
    d, kind, m, case, rep = cell
    key = (d, {"dense": 0, "sparse": 1}[kind], m, {"H0": 0, "H1": 1}[case], rep)
    return default_rng(np.random.SeedSequence(list(key)))



@pytest.mark.parametrize("cell", CELLS, ids=lambda c: "d{}_{}_m{}_{}_r{}".format(*c))
def test_chi_square_matches_production(cell):
    d, kind, m, case, _rep = cell
    rng = _cell_rng(cell)
    _keys, probs, p0 = _null(d, kind, rng)
    X, states, counts = _draw(d, probs, m, case, rng)

    ref = prod_chi_square(X, p0, alpha=0.05)
    got = sj.chi_square_joint_counts(states, counts, p0, alpha=0.05)

    assert got["statistic"] == ref["statistic"]
    assert got["p_value"] == ref["p_value"]
    assert got["df"] == ref["df"]
    assert got["cochran_satisfied"] == ref["cochran_satisfied"]
    assert got["observed_states"] == ref["observed_states"]
    assert got["reject"] == ("Fail to reject" not in ref["conclusion"])


@pytest.mark.parametrize("cell", CELLS, ids=lambda c: "d{}_{}_m{}_{}_r{}".format(*c))
def test_tolerance_matches_production(cell):
    d, kind, m, case, _rep = cell
    rng = _cell_rng(cell)
    _keys, probs, p0 = _null(d, kind, rng)
    X, states, counts = _draw(d, probs, m, case, rng)

    ref = prod_tolerance(X, p0, tolerance=0.05, alpha=0.05)
    got = sj.tolerance_counts(states, counts, p0, tolerance=0.05, alpha=0.05)

    assert got["statistic"] == ref["statistic"]
    assert got["worst_state"] == ref["worst_state"]
    assert got["worst_relative_error"] == ref["worst_relative_error"]
    assert got["worst_empirical_prob"] == ref["worst_empirical_prob"]
    assert got["n_violations"] == ref["n_violations"]
    assert got["n_observed_states"] == ref["n_observed_states"]
    assert got["reject"] == ("Fail to reject" not in ref["conclusion"])
    assert math.isnan(got["p_value"])


def test_impossible_state_flag_is_reported_separately():
    rng = default_rng(11)
    seen_true = seen_false = 0
    for d, kind, m, case in [(6, "sparse", 500, "H1"), (6, "sparse", 500, "H0"),
                             (10, "sparse", 2000, "H1"), (6, "dense", 500, "H1")]:
        _keys, probs, p0 = _null(d, kind, rng)
        X, states, counts = _draw(d, probs, m, case, rng)
        ref = prod_chi_square(X, p0)
        got = sj.chi_square_joint_counts(states, counts, p0)
        expected = ref["statistic"] == float("inf") and ref["df"] == 1
        assert got["impossible_state"] == expected
        if expected:
            assert got["reject"] and got["p_value"] == 0.0
            seen_true += 1
        else:
            seen_false += 1
    assert seen_true and seen_false, "both branches must be exercised"


def test_empty_null_matches_production():
    d = 3
    p0 = {bitstring(k, d): 0.0 for k in range(2 ** d)}
    states = np.array([[0, 0, 1], [1, 1, 0]], dtype=int)
    counts = np.array([7, 3], dtype=np.int64)
    X = np.repeat(states, counts, axis=0)
    ref = prod_chi_square(X, p0)
    got = sj.chi_square_joint_counts(states, counts, p0)
    assert (got["statistic"], got["p_value"], got["df"]) == (
        ref["statistic"], ref["p_value"], ref["df"])
    assert got["reject"] is False and got["impossible_state"] is False



@pytest.mark.parametrize("cell", CELLS[::7], ids=lambda c: "d{}_{}_m{}_{}_r{}".format(*c))
def test_g_test_uses_the_same_cells(cell):
    d, kind, m, case, _rep = cell
    rng = _cell_rng(cell)
    _keys, probs, p0 = _null(d, kind, rng)
    _X, states, counts = _draw(d, probs, m, case, rng)

    x2 = sj.chi_square_joint_counts(states, counts, p0)
    g = sj.g_test_counts(states, counts, p0)
    for key in ("df", "n_cells", "cochran_satisfied", "observed_states",
                "n_lumped_states", "impossible_state"):
        assert g[key] == x2[key], key
    if g["df"] > 0:
        assert g["statistic"] >= 0.0
        assert g["p_value"] == pytest.approx(1.0 - chi2.cdf(g["statistic"], g["df"]))


def test_g_test_zero_observed_cells_contribute_nothing():
    d = 4
    keys = [bitstring(k, d) for k in range(2 ** d)]
    p0 = dict(zip(keys, np.full(2 ** d, 1.0 / 2 ** d).tolist()))
    states = np.array([[int(b) for b in keys[i]] for i in range(4)], dtype=int)
    counts = np.array([40, 30, 20, 10], dtype=np.int64)
    got = sj.g_test_counts(states, counts, p0)
    assert np.isfinite(got["statistic"]) and got["statistic"] > 0.0
    assert 0.0 <= got["p_value"] <= 1.0



@pytest.mark.parametrize("seed", range(60))
def test_pair_median_matches_numpy_median_exactly(seed):
    rng = default_rng(1000 + seed)
    d = int(rng.integers(2, 7))
    n_states = int(rng.integers(1, min(2 ** d, 9) + 1))
    idx = rng.choice(2 ** d, size=n_states, replace=False)
    keys = [bitstring(int(i), d) for i in idx]
    cpool = rng.integers(1, 6, size=n_states).astype(float)

    M = sj._state_matrix(keys, d)
    D = sj._squared_distances(M)
    got = sj._pooled_median_distance(D, cpool, d)

    Z = np.repeat(M, cpool.astype(int), axis=0)
    N = Z.shape[0]
    if N < 2:
        assert math.isnan(got)
        return
    s = np.sum(Z * Z, axis=1)
    Dfull = s[:, None] + s[None, :] - 2.0 * (Z @ Z.T)
    Dfull[Dfull < 0.0] = 0.0
    iu = np.triu_indices(N, 1)
    expected = np.median(np.sqrt(Dfull[iu]))
    n_pairs = N * (N - 1) // 2
    assert got == expected, f"pairs={n_pairs} (even={n_pairs % 2 == 0})"


def test_pair_median_covers_both_parities():
    parities = set()
    for seed in range(60):
        rng = default_rng(1000 + seed)
        d = int(rng.integers(2, 7))
        n_states = int(rng.integers(1, min(2 ** d, 9) + 1))
        rng.choice(2 ** d, size=n_states, replace=False)
        cpool = rng.integers(1, 6, size=n_states)
        N = int(cpool.sum())
        if N >= 2:
            parities.add((N * (N - 1) // 2) % 2)
    assert parities == {0, 1}



def _dense_u2(K_dense, ix, iy):
    nx, ny = ix.size, iy.size
    Kxx = K_dense[np.ix_(ix, ix)]
    Kyy = K_dense[np.ix_(iy, iy)]
    Kxy = K_dense[np.ix_(ix, iy)]
    return ((np.sum(Kxx) - np.sum(np.diag(Kxx))) / (nx * (nx - 1))
            + (np.sum(Kyy) - np.sum(np.diag(Kyy))) / (ny * (ny - 1))
            - 2.0 * np.mean(Kxy))


@pytest.mark.parametrize("kernel", ["gaussian", "energy"])
@pytest.mark.parametrize("seed", range(12))
def test_permutation_engine_matches_dense_gram(kernel, seed):
    rng = default_rng(2000 + seed)
    d = int(rng.integers(2, 6))
    n_states = int(rng.integers(2, min(2 ** d, 8) + 1))
    idx = rng.choice(2 ** d, size=n_states, replace=False)
    keys = [bitstring(int(i), d) for i in idx]
    cx = rng.integers(0, 7, size=n_states).astype(float)
    cy = rng.integers(0, 7, size=n_states).astype(float)
    if cx.sum() < 2 or cy.sum() < 2:
        cx += 2.0
        cy += 2.0

    M = sj._state_matrix(keys, d)
    D = sj._squared_distances(M)
    if kernel == "gaussian":
        kernels = [lambda Dm: sj._gaussian_kernel(Dm, 1.3)]
        diag = 1.0
    else:
        kernels = [sj._negative_distance_kernel]
        diag = 0.0
    K = kernels[0](D)

    n_permute = 25
    stats, null = sj._permutation_stats(D, cx, cy, kernels, diag, n_permute,
                                        default_rng(seed))
    P = default_rng(seed).multivariate_hypergeometric(
        np.rint(cx + cy).astype(np.int64), int(cx.sum()), size=n_permute).astype(float)

    Z = np.repeat(M, (cx + cy).astype(int), axis=0)
    s = np.sum(Z * Z, axis=1)
    Dfull = s[:, None] + s[None, :] - 2.0 * (Z @ Z.T)
    Dfull[Dfull < 0.0] = 0.0
    K_dense = (sj._gaussian_kernel(Dfull, 1.3) if kernel == "gaussian"
               else sj._negative_distance_kernel(Dfull))
    starts = np.concatenate(([0], np.cumsum((cx + cy).astype(int))))

    def split_indices(counts_x):
        ix, iy = [], []
        for s_i, c in enumerate(counts_x.astype(int)):
            block = np.arange(starts[s_i], starts[s_i + 1])
            ix.extend(block[:c].tolist())
            iy.extend(block[c:].tolist())
        return np.array(ix, dtype=int), np.array(iy, dtype=int)

    ix, iy = split_indices(cx)
    assert stats[0] == pytest.approx(_dense_u2(K_dense, ix, iy), abs=1e-12)
    for b in range(n_permute):
        ixp, iyp = split_indices(P[b])
        assert null[b, 0] == pytest.approx(_dense_u2(K_dense, ixp, iyp), abs=1e-12)



def _production_reference_counts(p0, n_ref, seed):
    probs = np.array(list(p0.values()), dtype=float)
    probs = probs / probs.sum()
    ref_idx = default_rng(seed).choice(len(probs), size=n_ref, p=probs)
    return np.bincount(ref_idx, minlength=len(probs))


MMD_CELLS = [(d, kind, m, n_ref, rep)
             for d in (2, 6, 10)
             for kind in ("dense", "sparse")
             for m, n_ref in ((50, 200), (500, 200))
             for rep in range(2)]
MMD_CELLS += [(d, kind, 2000, 300, rep)
              for d in (6, 10) for kind in ("dense", "sparse") for rep in range(2)]


@pytest.mark.parametrize("cell", MMD_CELLS,
                         ids=lambda c: "d{}_{}_m{}_ref{}_r{}".format(*c))
def test_mmdagg_bandwidths_and_stats_match_production(cell):
    d, kind, m, n_ref, rep = cell
    rng = _cell_rng((d, kind, m, "H0", rep))
    _keys, probs, p0 = _null(d, kind, rng)
    X, states, counts = _draw(d, probs, m, "H0", rng)
    seed = 4000 + rep

    ref = prod_mmdagg(X.astype(float), p0, n_ref=n_ref, n_permute=1, seed=seed)
    ref_counts = _production_reference_counts(p0, n_ref, seed)
    got = sj.mmdagg_counts(states, counts, p0, n_ref=n_ref, n_permute=1,
                           rng=default_rng(seed), X_ref_counts=ref_counts)

    assert got["bandwidths"] == ref["bandwidths"], "bandwidth medians must be exact"
    assert got["n_test"] == ref["n_test"] and got["n_ref"] == ref["n_ref"]
    a = np.asarray(got["stats_per_kernel"])
    b = np.asarray(ref["stats_per_kernel"])
    assert np.max(np.abs(a - b)) <= 1e-10, f"max abs diff {np.max(np.abs(a - b)):.3e}"


def test_mmdagg_rejects_on_p_le_alpha():
    d = 4
    keys = [bitstring(k, d) for k in range(2 ** d)]
    p0 = dict(zip(keys, np.full(2 ** d, 1.0 / 2 ** d).tolist()))
    states = np.array([[int(b) for b in k] for k in keys], dtype=int)
    counts = np.full(2 ** d, 20, dtype=np.int64)
    res = sj.mmdagg_counts(states, counts, p0, alpha=1.0, n_ref=200, n_permute=20,
                           rng=default_rng(3))
    assert res["p_value"] <= 1.0 and res["reject"] is True
    res0 = sj.mmdagg_counts(states, counts, p0, alpha=0.0, n_ref=200, n_permute=20,
                            rng=default_rng(3))
    assert res0["reject"] is False


def test_mmdagg_weights_and_multipliers():
    d = 3
    keys = [bitstring(k, d) for k in range(2 ** d)]
    p0 = dict(zip(keys, np.full(8, 0.125).tolist()))
    states = np.array([[int(b) for b in k] for k in keys], dtype=int)
    counts = np.full(8, 10, dtype=np.int64)
    res = sj.mmdagg_counts(states, counts, p0, n_ref=100, n_permute=10,
                           rng=default_rng(5))
    w = np.asarray(res["weights"])
    expected = 2.0 ** -(np.arange(7) + 1.0)
    assert np.allclose(w, expected / expected.sum())
    bw = np.asarray(res["bandwidths"])
    assert np.allclose(bw / bw[2], sj.BANDWIDTH_MULTIPLIERS)



@pytest.mark.parametrize("fn,kw", [
    (sj.chi_square_joint_counts, {}),
    (sj.g_test_counts, {}),
    (sj.multinomial_mc_counts, {"n_boot": 50}),
    (sj.tv_bound_counts, {}),
    (sj.tolerance_counts, {}),
    (sj.mmdagg_counts, {"n_ref": 100, "n_permute": 20}),
    (sj.energy_distance_counts, {"n_ref": 100, "n_permute": 20}),
])
def test_data_contract(fn, kw):
    rng = default_rng(7)
    d = 5
    keys = [bitstring(k, d) for k in range(2 ** d)]
    probs = rng.dirichlet(np.ones(2 ** d))
    p0 = dict(zip(keys, probs.tolist()))
    _X, states, counts = _draw(d, probs, 400, "H0", rng)
    if "rng" in fn.__code__.co_varnames:
        kw = {**kw, "rng": default_rng(8)}
    res = fn(states, counts, p0, **kw)
    assert isinstance(res["reject"], bool)
    assert isinstance(res["p_value"], float)
    assert math.isnan(res["p_value"]) or 0.0 <= res["p_value"] <= 1.0


def test_multinomial_mc_uses_full_unlumped_table():
    rng = default_rng(9)
    d = 8
    keys = [bitstring(k, d) for k in range(2 ** d)]
    probs = rng.dirichlet(np.ones(2 ** d))
    p0 = dict(zip(keys, probs.tolist()))
    _X, states, counts = _draw(d, probs, 100, "H0", rng)
    res = sj.multinomial_mc_counts(states, counts, p0, n_boot=100, rng=default_rng(10))
    assert res["n_cells"] == 2 ** d
    obs = np.zeros(2 ** d)
    for row, c in zip(states, counts):
        obs[int("".join(map(str, row)), 2)] = c
    exp = 100 * probs / probs.sum()
    assert res["statistic"] == pytest.approx(float(np.sum((obs - exp) ** 2 / exp)))
    assert sj.chi_square_joint_counts(states, counts, p0)["df"] == 0


def test_tv_bound_epsilon_formula_and_large_support():
    rng = default_rng(12)
    d = 12
    keys = [bitstring(k, d) for k in range(2 ** d)]
    probs = rng.dirichlet(np.ones(2 ** d))
    p0 = dict(zip(keys, probs.tolist()))
    _X, states, counts = _draw(d, probs, 2000, "H0", rng)
    res = sj.tv_bound_counts(states, counts, p0, alpha=0.05)
    S, m = 2 ** d, 2000
    expected = math.sqrt(2.0 / m * (S * math.log(2.0)
                                    + math.log1p(-(2.0 ** (1 - S))) - math.log(0.05)))
    assert res["epsilon"] == pytest.approx(expected, rel=1e-12)
    assert np.isfinite(res["epsilon"]) and res["support_size"] == S
    keys3 = [bitstring(k, 3) for k in range(8)]
    p3 = dict(zip(keys3, [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]))
    st3 = np.array([[int(b) for b in k] for k in keys3[:4]], dtype=int)
    r3 = sj.tv_bound_counts(st3, np.array([25, 25, 25, 25], dtype=np.int64), p3)
    assert r3["support_size"] == 4
    assert r3["epsilon"] == pytest.approx(math.sqrt(2.0 / 100 * math.log(14 / 0.05)))


def test_tv_bound_reject_and_p_value_agree():
    rng = default_rng(13)
    d = 4
    keys = [bitstring(k, d) for k in range(16)]
    probs = rng.dirichlet(np.ones(16))
    p0 = dict(zip(keys, probs.tolist()))
    for case in ("H0", "H1"):
        _X, states, counts = _draw(d, probs, 300, case, rng)
        res = sj.tv_bound_counts(states, counts, p0, alpha=0.05)
        assert res["reject"] == (res["p_value"] < 0.05)


@pytest.mark.parametrize("kind", ["clean", "contaminated"])
def test_deterministic_null_is_handled_by_every_test(kind):
    d = 3
    keys = [bitstring(k, d) for k in range(2 ** d)]
    p0 = dict(zip(keys, [1.0] + [0.0] * 7))
    if kind == "clean":
        states = np.array([[0, 0, 0]], dtype=int)
        counts = np.array([500], dtype=np.int64)
    else:
        states = np.array([[0, 0, 0], [1, 1, 1]], dtype=int)
        counts = np.array([495, 5], dtype=np.int64)
    expect_reject = kind == "contaminated"

    calls = [
        ("chi2", sj.chi_square_joint_counts, {}),
        ("gtest", sj.g_test_counts, {}),
        ("mc", sj.multinomial_mc_counts, {"n_boot": 50, "rng": default_rng(1)}),
        ("tv", sj.tv_bound_counts, {}),
        ("tolerance", sj.tolerance_counts, {}),
        ("mmdagg", sj.mmdagg_counts,
         {"n_ref": 200, "n_permute": 50, "rng": default_rng(2)}),
        ("energy", sj.energy_distance_counts,
         {"n_ref": 200, "n_permute": 50, "rng": default_rng(3)}),
    ]
    for name, fn, kw in calls:
        res = fn(states, counts, p0, **kw)
        p = res["p_value"]
        assert math.isnan(p) or 0.0 <= p <= 1.0, name
        if name in ("chi2", "gtest", "mc", "tv", "tolerance"):
            assert res["reject"] == expect_reject, f"{name} on {kind} null"
        if not math.isnan(p) and name not in ("mmdagg", "energy"):
            assert res["reject"] == (p < 0.05) or name == "tv", name
    for name, fn, kw in calls[-2:]:
        res = fn(states, counts, p0, **kw)
        assert res["reject"] == (res["p_value"] <= 0.05), name


def test_tv_bound_deterministic_null_p_value_is_consistent():
    d = 3
    keys = [bitstring(k, d) for k in range(2 ** d)]
    p0 = dict(zip(keys, [1.0] + [0.0] * 7))
    clean = sj.tv_bound_counts(np.array([[0, 0, 0]], dtype=int),
                               np.array([500], dtype=np.int64), p0)
    assert clean["statistic"] == 0.0 and clean["epsilon"] == 0.0
    assert clean["reject"] is False and clean["p_value"] == 1.0
    dirty = sj.tv_bound_counts(np.array([[0, 0, 0], [1, 1, 1]], dtype=int),
                               np.array([495, 5], dtype=np.int64), p0)
    assert dirty["reject"] is True and dirty["p_value"] == 0.0


def test_energy_distance_is_szekely_rizzo():
    rng = default_rng(14)
    d = 5
    keys = [bitstring(k, d) for k in range(2 ** d)]
    probs = rng.dirichlet(np.ones(2 ** d))
    p0 = dict(zip(keys, probs.tolist()))
    _X, states, counts = _draw(d, probs, 150, "H0", rng)
    ref_counts = default_rng(15).multinomial(200, probs / probs.sum())
    res = sj.energy_distance_counts(states, counts, p0, n_ref=200, n_permute=20,
                                    rng=default_rng(16), X_ref_counts=ref_counts)

    Zx = np.repeat(states.astype(float), counts, axis=0)
    Zy = np.repeat(sj._state_matrix(keys, d), ref_counts, axis=0)

    def pdist(A, B):
        s1 = (A * A).sum(1)[:, None]
        s2 = (B * B).sum(1)[None, :]
        Dm = s1 + s2 - 2.0 * (A @ B.T)
        Dm[Dm < 0] = 0.0
        return np.sqrt(Dm)

    nx, ny = Zx.shape[0], Zy.shape[0]
    dxx = pdist(Zx, Zx)
    dyy = pdist(Zy, Zy)
    dxy = pdist(Zx, Zy)
    expected = (2.0 * dxy.mean()
                - (dxx.sum() - np.trace(dxx)) / (nx * (nx - 1))
                - (dyy.sum() - np.trace(dyy)) / (ny * (ny - 1)))
    assert res["statistic"] == pytest.approx(expected, abs=1e-10)
    assert res["reject"] == (res["p_value"] <= 0.05)


def test_rng_is_never_seeded_globally():
    rng = default_rng(17)
    d = 5
    keys = [bitstring(k, d) for k in range(2 ** d)]
    probs = rng.dirichlet(np.ones(2 ** d))
    p0 = dict(zip(keys, probs.tolist()))
    _X, states, counts = _draw(d, probs, 300, "H0", rng)

    before = np.random.get_state()[1][:5].copy()
    a = sj.mmdagg_counts(states, counts, p0, n_ref=200, n_permute=50, rng=default_rng(1))
    b = sj.mmdagg_counts(states, counts, p0, n_ref=200, n_permute=50, rng=default_rng(1))
    c = sj.mmdagg_counts(states, counts, p0, n_ref=200, n_permute=50, rng=default_rng(2))
    after = np.random.get_state()[1][:5]
    assert np.array_equal(before, after)
    assert a["p_value"] == b["p_value"]
    assert a["stats_per_kernel"] != c["stats_per_kernel"]
