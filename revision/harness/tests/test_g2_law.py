from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from numpy.random import default_rng
from scipy.stats import binomtest, ks_2samp

_TESTS = Path(__file__).resolve().parent
for _p in (str(_TESTS), str(_TESTS.parent), str(_TESTS.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harness import stats_joint as sj
from test_g1_joint import _draw, _null, bitstring, prod_mmdagg

D = 6
M = 200
N_REF = 400
N_PERMUTE = 500
ALPHA = 0.05
N_REPS = int(os.environ.get("G2_REPS", "200"))
KINDS = tuple(os.environ.get("G2_KINDS", "dense,sparse").split(","))


def _cell_null(kind):
    return _null(D, kind, default_rng(90210 + ("dense", "sparse").index(kind)))


def test_hypergeometric_split_matches_index_permutation_law():
    rng = default_rng(31)
    d = 4
    keys = [bitstring(k, d) for k in range(2 ** d)]
    cpool = rng.integers(1, 8, size=2 ** d).astype(float)
    nx = int(cpool.sum() // 3)

    M_states = sj._state_matrix(keys, d)
    Dsq = sj._squared_distances(M_states)
    K = sj._gaussian_kernel(Dsq, 1.7)

    n_draws = 3000
    P_hyper = default_rng(32).multivariate_hypergeometric(
        np.rint(cpool).astype(np.int64), nx, size=n_draws).astype(float)

    labels = np.repeat(np.arange(cpool.size), cpool.astype(int))
    perm_rng = default_rng(33)
    P_perm = np.empty_like(P_hyper)
    for b in range(n_draws):
        pick = perm_rng.permutation(labels.size)[:nx]
        P_perm[b] = np.bincount(labels[pick], minlength=cpool.size)

    def stats_from(P):
        Q = cpool[None, :] - P
        PK = P @ K
        nx_f, ny_f = float(nx), float(cpool.sum() - nx)
        pkp = np.einsum("bs,bs->b", PK, P)
        qx = pkp - nx_f
        qxy = np.einsum("bs,bs->b", PK, Q)
        qy = float(cpool @ K @ cpool) - 2.0 * (PK @ cpool) + pkp - ny_f
        return (qx / (nx_f * (nx_f - 1)) + qy / (ny_f * (ny_f - 1))
                - 2.0 * qxy / (nx_f * ny_f))

    ks = ks_2samp(stats_from(P_hyper), stats_from(P_perm))
    print(f"\n[G2 permutation law] KS D={ks.statistic:.4f} p={ks.pvalue:.4f} "
          f"(n={n_draws} per arm)")
    assert ks.pvalue >= 0.01, f"permutation-null laws differ: KS p={ks.pvalue:.4g}"


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_g2_rejection_rate_and_pvalue_law(kind):
    _keys, probs, p0 = _cell_null(kind)
    rejects_prod = np.zeros(N_REPS, dtype=bool)
    rejects_new = np.zeros(N_REPS, dtype=bool)
    p_prod = np.empty(N_REPS)
    p_new = np.empty(N_REPS)

    t0 = time.perf_counter()
    for rep in range(N_REPS):
        data_rng = default_rng(np.random.SeedSequence([7, rep]))
        X, states, counts = _draw(D, probs, M, "H0", data_rng)

        ref = prod_mmdagg(X.astype(float), p0, alpha=ALPHA, n_ref=N_REF,
                          n_permute=N_PERMUTE, seed=100_000 + rep)
        got = sj.mmdagg_counts(states, counts, p0, alpha=ALPHA, n_ref=N_REF,
                               n_permute=N_PERMUTE,
                               rng=default_rng(np.random.SeedSequence([9, rep])))
        rejects_prod[rep] = "Fail to reject" not in ref["conclusion"]
        rejects_new[rep] = got["reject"]
        p_prod[rep] = ref["p_value"]
        p_new[rep] = got["p_value"]
    elapsed = time.perf_counter() - t0

    k_prod, k_new = int(rejects_prod.sum()), int(rejects_new.sum())
    rate_prod, rate_new = k_prod / N_REPS, k_new / N_REPS
    ci_prod = binomtest(k_prod, N_REPS).proportion_ci(confidence_level=0.95,
                                                      method="exact")
    ci_new = binomtest(k_new, N_REPS).proportion_ci(confidence_level=0.95,
                                                    method="exact")
    ks = ks_2samp(p_prod, p_new)

    print(f"\n[G2 {kind}] reps={N_REPS} alpha={ALPHA} elapsed={elapsed:.1f}s\n"
          f"  production : reject {k_prod}/{N_REPS} = {rate_prod:.4f} "
          f"CI95 [{ci_prod.low:.4f}, {ci_prod.high:.4f}]  mean p={p_prod.mean():.4f}\n"
          f"  count kernel: reject {k_new}/{N_REPS} = {rate_new:.4f} "
          f"CI95 [{ci_new.low:.4f}, {ci_new.high:.4f}]  mean p={p_new.mean():.4f}\n"
          f"  KS on p-values: D={ks.statistic:.4f} p={ks.pvalue:.4f}")

    assert ci_prod.low <= rate_new <= ci_prod.high, (
        f"count-kernel rate {rate_new:.4f} outside production CI "
        f"[{ci_prod.low:.4f}, {ci_prod.high:.4f}]")
    assert ci_new.low <= rate_prod <= ci_new.high, (
        f"production rate {rate_prod:.4f} outside count-kernel CI "
        f"[{ci_new.low:.4f}, {ci_new.high:.4f}]")
    assert ks.pvalue >= 0.01, f"p-value distributions differ: KS p={ks.pvalue:.4g}"
