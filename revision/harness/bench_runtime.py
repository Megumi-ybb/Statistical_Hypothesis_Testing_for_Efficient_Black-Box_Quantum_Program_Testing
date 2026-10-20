#!/usr/bin/env python3
import time, math, sys
import numpy as np
from numpy.random import default_rng
from scipy.stats import chi2

REPO = str(__import__('pathlib').Path(__file__).resolve().parents[2])

def extract_defs(path, names):
    src = open(path).read()
    blocks = src.split("\ndef ")
    out = {}
    for b in blocks[1:]:
        name = b.split("(", 1)[0].strip()
        if name in names:
            out[name] = "def " + b
    missing = set(names) - set(out)
    assert not missing, f"missing {missing}"
    return "\n".join(out[n] for n in names)

g = {"np": np, "default_rng": default_rng, "chi2": chi2, "math": math}
exec(extract_defs(f"{REPO}/Joint/Experiment/joint_experiemnt_BVtestfalse.py",
                  ["bitstring", "chi_square_joint", "mmdagg_joint_test", "_sample_from_state",
                   "_normalize_state"]), g)
exec(extract_defs(f"{REPO}/Marginal/Experiment/marginal_QFTtestfalse.py",
                  ["_nearest_correlation", "sample_bernoulli_copula", "chi_square_bootstrap_test"]), g)
chi_square_joint = g["chi_square_joint"]
mmdagg_joint_test = g["mmdagg_joint_test"]
chi_square_bootstrap_test = g["chi_square_bootstrap_test"]
bitstring = g["bitstring"]

def mmdagg_counts(X_test, p_joint_true, alpha=0.05, n_ref=1000, n_permute=500, seed=None,
                  X_ref_counts=None):
    rng = default_rng(seed)
    X_test = np.asarray(X_test, dtype=int)
    n_test, d = X_test.shape
    spec_states = list(p_joint_true.keys())
    probs = np.array([p_joint_true[s] for s in spec_states], dtype=float)
    probs = probs / probs.sum()
    test_keys, test_cnt = np.unique(X_test, axis=0, return_counts=True)
    test_strs = ["".join(map(str, row)) for row in test_keys]
    if X_ref_counts is None:
        ref_cnt_spec = rng.multinomial(n_ref, probs)
    else:
        ref_cnt_spec = X_ref_counts
    support = {}
    for s, c in zip(spec_states, ref_cnt_spec):
        if c > 0: support[s] = [0, int(c)]
    for s, c in zip(test_strs, test_cnt):
        support.setdefault(s, [0, 0])[0] = int(c)
    S = sorted(support)
    cx = np.array([support[s][0] for s in S], dtype=float)
    cy = np.array([support[s][1] for s in S], dtype=float)
    M = np.array([[int(b) for b in s] for s in S], dtype=float)
    ssum = (M * M).sum(1)
    D = ssum[:, None] + ssum[None, :] - 2.0 * (M @ M.T)
    D[D < 0] = 0.0
    cpool = cx + cy
    iu = np.triu_indices(len(S), 0)
    ii, jj = iu
    dists = np.sqrt(D[iu])
    wts = np.where(ii == jj, cpool[ii] * (cpool[ii] - 1) / 2, cpool[ii] * cpool[jj])
    order = np.argsort(dists)
    cw = np.cumsum(wts[order]); tot = cw[-1]
    md = dists[order][np.searchsorted(cw, tot / 2)]
    if not np.isfinite(md) or md <= 0: md = np.sqrt(d)
    bws = md * np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    L = len(bws)
    w = np.array([2.0 ** (-(i + 1)) for i in range(L)]); w = w / w.sum()
    nx, ny = cx.sum(), cy.sum()
    Ks = [np.exp(-D / (2 * bw * bw)) for bw in bws]
    def stat_from_counts(K, ax, ay):
        qx = ax @ K @ ax - ax.sum()
        qy = ay @ K @ ay - ay.sum()
        qxy = ax @ K @ ay
        return qx / (nx * (nx - 1)) + qy / (ny * (ny - 1)) - 2 * qxy / (nx * ny)
    stats = np.array([stat_from_counts(K, cx, cy) for K in Ks])
    P = np.array([rng.multivariate_hypergeometric(cpool.astype(int), int(nx))
                  for _ in range(n_permute)], dtype=float)
    Q = cpool[None, :] - P
    null = np.empty((n_permute, L))
    for l, K in enumerate(Ks):
        PK = P @ K
        qx = (PK * P).sum(1) - P.sum(1)
        qxy = (PK * Q).sum(1)
        cKc = (cpool @ K @ cpool)
        qy = cKc - 2 * (PK @ cpool) + (PK * P).sum(1) + 0
        qy = qy - Q.sum(1)
        null[:, l] = qx / (nx * (nx - 1)) + qy / (ny * (ny - 1)) - 2 * qxy / (nx * ny)
    pvals = (1.0 + (null >= stats).sum(0)) / (n_permute + 1.0)
    p_agg = float(min(1.0, np.min(pvals / w)))
    return {"p_value": p_agg, "reject": p_agg <= alpha, "S": len(S), "stats": stats}

def timeit(f, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps): r = f()
    return (time.perf_counter() - t0) / reps, r

rng = default_rng(7)
results = []

d = 6
states6 = [bitstring(k, d) for k in range(2 ** d)]
pd6 = rng.dirichlet(np.ones(2 ** d))
p6 = dict(zip(states6, pd6))
Xt = rng.choice(2 ** d, size=200, p=pd6)
X_test6 = np.array([[int(b) for b in bitstring(int(i), d)] for i in Xt])
ref_counts = rng.multinomial(400, pd6)
X_ref = np.repeat(np.arange(2 ** d), ref_counts)
Z_ref = np.array([[int(b) for b in bitstring(int(i), d)] for i in X_ref], dtype=float)
res_c = mmdagg_counts(X_test6, p6, n_ref=400, n_permute=50, seed=1, X_ref_counts=ref_counts)
Zt = X_test6.astype(float); N = len(Zt) + len(Z_ref)
Z = np.vstack([Zt, Z_ref]); ssum = (Z * Z).sum(1)
Dfull = ssum[:, None] + ssum[None, :] - 2 * Z @ Z.T; Dfull[Dfull < 0] = 0
iu = np.triu_indices(N, 1); md = np.median(np.sqrt(Dfull[iu]))
bws = md * np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
ix = np.arange(len(Zt)); iy = np.arange(len(Zt), N)
def u2(K, ix, iy):
    Kxx = K[np.ix_(ix, ix)]; Kyy = K[np.ix_(iy, iy)]; Kxy = K[np.ix_(ix, iy)]
    nx, ny = ix.size, iy.size
    return ((Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1)) - 2 * Kxy.mean())
stats_naive = np.array([u2(np.exp(-Dfull / (2 * b * b)), ix, iy) for b in bws])
err = np.max(np.abs(stats_naive - res_c["stats"]))
print(f"[check] observed-stat max |naive - counts| = {err:.3e}  (bandwidth median match: "
      f"{np.isclose(md, md):})")
assert err < 1e-10, "count-based statistic does not match dense computation"

d = 10
states = [bitstring(k, d) for k in range(2 ** d)]
pdense = rng.dirichlet(np.ones(2 ** d))
p10 = dict(zip(states, pdense))
for m in (50, 500, 2000):
    Xi = rng.choice(2 ** d, size=m, p=pdense)
    X = np.array([[int(b) for b in bitstring(int(i), d)] for i in Xi])
    t_naive, _ = timeit(lambda: mmdagg_joint_test(X.astype(float), p10, n_ref=1000,
                                                  n_permute=500, seed=3), reps=1)
    t_opt, _ = timeit(lambda: mmdagg_counts(X, p10, n_ref=1000, n_permute=500, seed=3),
                      reps=5)
    t_chi, _ = timeit(lambda: chi_square_joint(X, p10), reps=20)
    results.append((f"joint d=10 m={m}", t_naive, t_opt, t_chi))
    print(f"[joint d=10 m={m:4d}] mmdagg naive {t_naive:7.2f}s | counts {t_opt*1000:8.1f}ms "
          f"| speedup x{t_naive/t_opt:6.0f} | chi2 {t_chi*1000:6.2f}ms")

psparse_v = np.zeros(2 ** d); idx = rng.choice(2 ** d, 16, replace=False)
psparse_v[idx] = rng.dirichlet(np.ones(16)); p10s = dict(zip(states, psparse_v))
Xi = rng.choice(2 ** d, size=2000, p=psparse_v)
X = np.array([[int(b) for b in bitstring(int(i), d)] for i in Xi])
t_naive, _ = timeit(lambda: mmdagg_joint_test(X.astype(float), p10s, n_ref=1000,
                                              n_permute=500, seed=3), reps=1)
t_opt, _ = timeit(lambda: mmdagg_counts(X, p10s, n_ref=1000, n_permute=500, seed=3), reps=5)
print(f"[joint d=10 m=2000 SPARSE] naive {t_naive:7.2f}s | counts {t_opt*1000:8.1f}ms | "
      f"x{t_naive/t_opt:6.0f}")

d = 12
p_true = np.clip(rng.uniform(0.2, 0.8, d), 0.05, 0.95)
for m in (500, 2000):
    Xm = (rng.uniform(size=(m, d)) < p_true).astype(int)
    R = np.corrcoef(Xm, rowvar=False)
    t_boot, _ = timeit(lambda: chi_square_bootstrap_test(Xm, p_true, R_assumed=R,
                                                         n_bootstrap=300, seed=5), reps=3)
    print(f"[marginal d=12 m={m:4d}] copula bootstrap B=300: {t_boot*1000:7.1f}ms")

print("\nDone. Apply cluster derating (grid Xeon core ~2-3x slower than M3 Max core) "
      "when converting to grid core-hours.")
