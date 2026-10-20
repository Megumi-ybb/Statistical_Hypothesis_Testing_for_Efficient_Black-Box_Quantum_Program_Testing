from __future__ import annotations

import math
from functools import partial
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import chi2

__all__ = [
    "mmdagg_counts",
    "chi_square_joint_counts",
    "g_test_counts",
    "multinomial_mc_counts",
    "energy_distance_counts",
    "tv_bound_counts",
    "tolerance_counts",
]

BANDWIDTH_MULTIPLIERS = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0], dtype=float)
LOG2 = math.log(2.0)



def _row_strings(states: np.ndarray) -> List[str]:
    arr = np.ascontiguousarray(states)
    if arr.ndim != 2:
        raise ValueError("states must be 2D (S, d)")
    if arr.size and not np.all((arr == 0) | (arr == 1)):
        raise ValueError("states must be binary")
    if arr.shape[0] == 0:
        return []
    buf = np.ascontiguousarray(arr, dtype=np.uint8) + np.uint8(ord("0"))
    return np.frombuffer(buf.tobytes(), dtype=f"S{arr.shape[1]}").astype(str).tolist()


def _observed_counts(states: np.ndarray, counts: np.ndarray) -> Tuple[Dict[str, int], int]:
    counts = np.asarray(counts)
    if counts.ndim != 1:
        raise ValueError("counts must be 1D")
    if counts.shape[0] != np.asarray(states).shape[0]:
        raise ValueError("states and counts must have the same length")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    obs: Dict[str, int] = {}
    for key, c in zip(_row_strings(states), counts.tolist()):
        c = int(c)
        if c > 0:
            obs[key] = obs.get(key, 0) + c
    return obs, int(counts.sum())


def _null_support(p0: Dict[str, float]) -> Tuple[List[str], np.ndarray]:
    keys = list(p0.keys())
    if not keys:
        raise ValueError("p0 is empty")
    probs = np.array([p0[k] for k in keys], dtype=float)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("p0 has non-positive total mass")
    return keys, probs / total


def _null_width(p0: Dict[str, float]) -> int:
    widths = {len(k) for k in p0}
    if len(widths) != 1:
        raise ValueError(f"p0 keys have inconsistent widths: {sorted(widths)}")
    return widths.pop()


def _state_matrix(keys: Sequence[str], d: int) -> np.ndarray:
    if not len(keys):
        return np.zeros((0, d), dtype=float)
    flat = np.frombuffer("".join(keys).encode("ascii"), dtype=np.uint8)
    return (flat.reshape(len(keys), d) - np.uint8(ord("0"))).astype(float)


def _squared_distances(M: np.ndarray) -> np.ndarray:
    s = np.sum(M * M, axis=1)
    D = s[:, None] + s[None, :] - 2.0 * (M @ M.T)
    D[D < 0.0] = 0.0
    return D



class _Lumping:

    __slots__ = ("status", "obs", "exp", "df", "cochran", "observed_states", "n_lumped")

    def __init__(self, status, obs, exp, df, cochran, observed_states, n_lumped):
        self.status = status
        self.obs = obs
        self.exp = exp
        self.df = df
        self.cochran = cochran
        self.observed_states = observed_states
        self.n_lumped = n_lumped


def _lump_cells(obs_counts: Dict[str, int], m: int, p0: Dict[str, float],
                min_count: float) -> _Lumping:
    observed_states = len(obs_counts)
    total = sum(v for v in p0.values() if v > 0.0)
    if total <= 0:
        return _Lumping("empty_null", None, None, 0, False, observed_states, 0)
    if any(p0.get(k, 0.0) == 0.0 and obs_counts.get(k, 0) > 0 for k in obs_counts):
        return _Lumping("impossible", None, None, 1, False, observed_states, 0)

    keys = [(k, v / total) for k, v in p0.items() if v > 0.0]
    keys.sort(key=lambda kv: kv[1])
    exp = m * np.array([v for _, v in keys], dtype=float)
    obs = np.array([obs_counts.get(k, 0) for k, _ in keys], dtype=float)

    i = 0
    while i < exp.size and exp[i] < min_count:
        i += 1
    if i > 0:
        exp = np.concatenate(([exp[:i].sum()], exp[i:]))
        obs = np.concatenate(([obs[:i].sum()], obs[i:]))

    if exp.size <= 1:
        cochran = bool(exp.size > 0 and exp.min() >= min_count)
        return _Lumping("degenerate", obs, exp, 0, cochran, observed_states, i)
    return _Lumping("ok", obs, exp, int(exp.size - 1), bool(exp.min() >= min_count),
                    observed_states, i)


def chi_square_joint_counts(states, counts, p0, *, alpha: float = 0.05,
                            min_count: float = 5) -> dict:
    obs_counts, m = _observed_counts(states, counts)
    lump = _lump_cells(obs_counts, m, p0, min_count)
    base = {"test_name": "Chi-square Multinomial (lumped)", "impossible_state": False,
            "cochran_satisfied": lump.cochran, "observed_states": lump.observed_states,
            "n_lumped_states": lump.n_lumped, "n_cells": 0}

    if lump.status == "empty_null":
        return {**base, "test_name": "Chi-square Multinomial", "statistic": 0.0,
                "p_value": 1.0, "df": 0, "reject": False,
                "conclusion": "Fail to reject H0"}
    if lump.status == "impossible":
        return {**base, "test_name": "Chi-square Multinomial", "statistic": float("inf"),
                "p_value": 0.0, "df": 1, "reject": True, "impossible_state": True,
                "conclusion": "Reject H0"}
    if lump.status == "degenerate":
        return {**base, "statistic": 0.0, "p_value": 1.0, "df": 0, "reject": False,
                "n_cells": int(lump.exp.size), "conclusion": "Fail to reject H0"}

    with np.errstate(divide="ignore", invalid="ignore"):
        stat = float(np.nansum((lump.obs - lump.exp) ** 2 / lump.exp))
    p_value = float(1.0 - chi2.cdf(stat, lump.df))
    reject = p_value < alpha
    return {**base, "statistic": stat, "p_value": p_value, "df": lump.df,
            "reject": bool(reject), "n_cells": int(lump.exp.size),
            "conclusion": "Reject H0" if reject else "Fail to reject H0"}


def g_test_counts(states, counts, p0, *, alpha: float = 0.05,
                  min_count: float = 5) -> dict:
    obs_counts, m = _observed_counts(states, counts)
    lump = _lump_cells(obs_counts, m, p0, min_count)
    base = {"test_name": "G-test Multinomial (lumped)", "impossible_state": False,
            "cochran_satisfied": lump.cochran, "observed_states": lump.observed_states,
            "n_lumped_states": lump.n_lumped, "n_cells": 0}

    if lump.status == "empty_null":
        return {**base, "test_name": "G-test Multinomial", "statistic": 0.0,
                "p_value": 1.0, "df": 0, "reject": False,
                "conclusion": "Fail to reject H0"}
    if lump.status == "impossible":
        return {**base, "test_name": "G-test Multinomial", "statistic": float("inf"),
                "p_value": 0.0, "df": 1, "reject": True, "impossible_state": True,
                "conclusion": "Reject H0"}
    if lump.status == "degenerate":
        return {**base, "statistic": 0.0, "p_value": 1.0, "df": 0, "reject": False,
                "n_cells": int(lump.exp.size), "conclusion": "Fail to reject H0"}

    obs, exp = lump.obs, lump.exp
    nz = obs > 0
    stat = float(2.0 * np.sum(obs[nz] * np.log(obs[nz] / exp[nz])))
    p_value = float(1.0 - chi2.cdf(stat, lump.df))
    reject = p_value < alpha
    return {**base, "statistic": stat, "p_value": p_value, "df": lump.df,
            "reject": bool(reject), "n_cells": int(lump.exp.size),
            "conclusion": "Reject H0" if reject else "Fail to reject H0"}



def _pair_level_counts(D: np.ndarray, cpool: np.ndarray, d: int) -> np.ndarray:
    S = cpool.size
    n = float(cpool.sum())
    ordered = np.zeros(d + 1, dtype=float)
    rows = max(1, int(4_000_000 // max(S, 1)))
    for start in range(0, S, rows):
        stop = min(start + rows, S)
        lev = np.rint(D[start:stop]).astype(np.int64).ravel()
        wts = (cpool[start:stop, None] * cpool[None, :]).ravel()
        ordered += np.bincount(lev, weights=wts, minlength=d + 1)[: d + 1]
    ordered[0] -= n
    return np.rint(ordered / 2.0).astype(np.int64)


def _median_from_pair_levels(pair_counts: np.ndarray) -> float:
    values = np.sqrt(np.arange(pair_counts.size, dtype=float))
    cw = np.cumsum(pair_counts)
    total = int(cw[-1]) if cw.size else 0
    if total <= 0:
        return float("nan")
    if total % 2 == 1:
        return float(values[np.searchsorted(cw, total // 2, side="right")])
    lo = values[np.searchsorted(cw, total // 2 - 1, side="right")]
    hi = values[np.searchsorted(cw, total // 2, side="right")]
    return float((lo + hi) / 2.0)


def _pooled_median_distance(D: np.ndarray, cpool: np.ndarray, d: int) -> float:
    return _median_from_pair_levels(_pair_level_counts(D, cpool, d))



def _gaussian_kernel(D: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-D / (2.0 * (bandwidth * bandwidth)))


def _negative_distance_kernel(D: np.ndarray) -> np.ndarray:
    return -np.sqrt(D)


def _mmd2u_counts(K, cx, cy, diag, nx, ny) -> float:
    qx = float(cx @ K @ cx) - diag * nx
    qy = float(cy @ K @ cy) - diag * ny
    qxy = float(cx @ K @ cy)
    return qx / (nx * (nx - 1.0)) + qy / (ny * (ny - 1.0)) - 2.0 * qxy / (nx * ny)


def _permutation_stats(D, cx, cy, kernels: Sequence[Callable[[np.ndarray], np.ndarray]],
                       diag: float, n_permute: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    nx = float(cx.sum())
    ny = float(cy.sum())
    if nx < 2 or ny < 2:
        raise ValueError("both samples need at least 2 points for a U-statistic")
    cpool = cx + cy
    colors = np.rint(cpool).astype(np.int64)
    P = rng.multivariate_hypergeometric(colors, int(round(nx)),
                                        size=n_permute).astype(float)
    Q = cpool[None, :] - P

    L = len(kernels)
    stats = np.empty(L, dtype=float)
    null = np.empty((n_permute, L), dtype=float)
    for l, kernel in enumerate(kernels):
        K = kernel(D)
        stats[l] = _mmd2u_counts(K, cx, cy, diag, nx, ny)
        PK = P @ K
        pkp = np.einsum("bs,bs->b", PK, P)
        qx = pkp - diag * nx
        qxy = np.einsum("bs,bs->b", PK, Q)
        qy = float(cpool @ K @ cpool) - 2.0 * (PK @ cpool) + pkp - diag * ny
        null[:, l] = (qx / (nx * (nx - 1.0)) + qy / (ny * (ny - 1.0))
                      - 2.0 * qxy / (nx * ny))
        del K, PK
    return stats, null


def _pooled_support(obs_counts: Dict[str, int], spec_keys: Sequence[str],
                    ref_counts: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
    support: Dict[str, List[int]] = {}
    for key, c in zip(spec_keys, ref_counts.tolist()):
        c = int(c)
        if c > 0:
            support[key] = [0, c]
    for key, c in obs_counts.items():
        support.setdefault(key, [0, 0])[0] = int(c)
    keys = sorted(support)
    cx = np.array([support[k][0] for k in keys], dtype=float)
    cy = np.array([support[k][1] for k in keys], dtype=float)
    return keys, cx, cy


def mmdagg_counts(states, counts, p0, *, alpha: float = 0.05, n_ref: int = 1000,
                  n_permute: int = 500, rng, X_ref_counts=None, ref_counts=None) -> dict:
    obs_counts, m = _observed_counts(states, counts)
    d = _null_width(p0)
    spec_keys, probs = _null_support(p0)
    if m < 2:
        raise ValueError("m must be at least 2")

    if X_ref_counts is not None and ref_counts is not None:
        raise ValueError("supply at most one of ref_counts / X_ref_counts")
    if X_ref_counts is None:
        X_ref_counts = ref_counts

    if X_ref_counts is None:
        ref_counts = rng.multinomial(n_ref, probs)
    else:
        ref_counts = np.asarray(X_ref_counts, dtype=np.int64)
        if ref_counts.shape != (len(spec_keys),):
            raise ValueError("X_ref_counts must align with p0 keys")
    n_ref_realized = int(ref_counts.sum())
    if n_ref_realized < 2:
        raise ValueError("n_ref must be at least 2")

    keys, cx, cy = _pooled_support(obs_counts, spec_keys, ref_counts)
    D = _squared_distances(_state_matrix(keys, d))

    md = _pooled_median_distance(D, cx + cy, d)
    if not np.isfinite(md) or md <= 0:
        md = math.sqrt(d)
    bandwidths = md * BANDWIDTH_MULTIPLIERS
    L = bandwidths.size
    w = np.array([2.0 ** (-(i + 1)) for i in range(L)], dtype=float)
    w = w / w.sum()

    kernels = [partial(_gaussian_kernel, bandwidth=bw) for bw in bandwidths]
    stats, null = _permutation_stats(D, cx, cy, kernels, 1.0, n_permute, rng)

    pvals = (1.0 + (null >= stats).sum(axis=0)) / (n_permute + 1.0)
    p_agg = float(np.minimum(1.0, np.min(pvals / w)))
    reject = p_agg <= alpha
    return {"test_name": "MMDAgg", "statistic": float(np.max(stats)),
            "p_value": p_agg, "reject": bool(reject),
            "conclusion": "Reject H0" if reject else "Fail to reject H0",
            "bandwidths": bandwidths.tolist(), "weights": w.tolist(),
            "stats_per_kernel": stats.tolist(), "pvals_per_kernel": pvals.tolist(),
            "median_distance": float(md), "n_test": int(m),
            "n_ref": n_ref_realized, "n_permute": int(n_permute),
            "n_support": int(len(keys))}


def energy_distance_counts(states, counts, p0, *, alpha: float = 0.05,
                           n_ref: int = 1000, n_permute: int = 500, rng,
                           X_ref_counts=None) -> dict:
    obs_counts, m = _observed_counts(states, counts)
    d = _null_width(p0)
    spec_keys, probs = _null_support(p0)
    if m < 2:
        raise ValueError("m must be at least 2")

    if X_ref_counts is None:
        ref_counts = rng.multinomial(n_ref, probs)
    else:
        ref_counts = np.asarray(X_ref_counts, dtype=np.int64)
        if ref_counts.shape != (len(spec_keys),):
            raise ValueError("X_ref_counts must align with p0 keys")
    n_ref_realized = int(ref_counts.sum())
    if n_ref_realized < 2:
        raise ValueError("n_ref must be at least 2")

    keys, cx, cy = _pooled_support(obs_counts, spec_keys, ref_counts)
    D = _squared_distances(_state_matrix(keys, d))
    stats, null = _permutation_stats(D, cx, cy, [_negative_distance_kernel], 0.0,
                                     n_permute, rng)
    stat = float(stats[0])
    p_value = float((1.0 + int((null[:, 0] >= stat).sum())) / (n_permute + 1.0))
    reject = p_value <= alpha
    return {"test_name": "Energy distance", "statistic": stat, "p_value": p_value,
            "reject": bool(reject),
            "conclusion": "Reject H0" if reject else "Fail to reject H0",
            "n_test": int(m), "n_ref": n_ref_realized, "n_permute": int(n_permute),
            "n_support": int(len(keys))}



def multinomial_mc_counts(states, counts, p0, *, alpha: float = 0.05,
                          n_boot: int = 300, rng) -> dict:
    obs_counts, m = _observed_counts(states, counts)
    observed_states = len(obs_counts)
    keys = [k for k, v in p0.items() if v > 0.0]
    total = sum(p0[k] for k in keys)
    base = {"test_name": "Multinomial Monte-Carlo GOF", "impossible_state": False,
            "observed_states": observed_states, "n_cells": len(keys),
            "n_boot": int(n_boot)}
    if total <= 0:
        return {**base, "statistic": 0.0, "p_value": 1.0, "reject": False,
                "conclusion": "Fail to reject H0"}
    if any(p0.get(k, 0.0) == 0.0 and c > 0 for k, c in obs_counts.items()):
        return {**base, "statistic": float("inf"), "p_value": 0.0, "reject": True,
                "impossible_state": True, "conclusion": "Reject H0"}

    p = np.array([p0[k] / total for k in keys], dtype=float)
    exp = m * p
    obs = np.array([obs_counts.get(k, 0) for k in keys], dtype=float)
    stat = float(np.sum((obs - exp) ** 2 / exp))

    boot = rng.multinomial(m, p, size=n_boot).astype(float)
    boot_stats = np.sum((boot - exp) ** 2 / exp, axis=1)
    p_value = float((1.0 + int((boot_stats >= stat).sum())) / (n_boot + 1.0))
    reject = p_value < alpha
    return {**base, "statistic": stat, "p_value": p_value, "reject": bool(reject),
            "conclusion": "Reject H0" if reject else "Fail to reject H0"}



def tv_bound_counts(states, counts, p0, *, alpha: float = 0.05) -> dict:
    obs_counts, m = _observed_counts(states, counts)
    keys = [k for k, v in p0.items() if v > 0.0]
    total = sum(p0[k] for k in keys)
    S = len(keys)
    base = {"test_name": "L1/TV confidence bound", "support_size": S,
            "observed_states": len(obs_counts), "n_samples": int(m)}
    if total <= 0 or m <= 0:
        return {**base, "statistic": 0.0, "epsilon": float("inf"), "p_value": 1.0,
                "reject": False, "conclusion": "Fail to reject H0"}

    p_hat = {k: c / m for k, c in obs_counts.items()}
    l1 = 0.0
    for k in keys:
        l1 += abs(p_hat.get(k, 0.0) - p0[k] / total)
    for k, v in p_hat.items():
        if p0.get(k, 0.0) <= 0.0:
            l1 += v

    if S <= 1:
        eps = 0.0
        reject = l1 > eps
        p_value = 0.0 if reject else 1.0
    else:
        log_prefactor = S * LOG2 + math.log1p(-(2.0 ** (1 - S)))
        eps = math.sqrt(2.0 / m * (log_prefactor - math.log(alpha)))
        reject = l1 > eps
        log_p = log_prefactor - m * l1 * l1 / 2.0
        p_value = 1.0 if log_p >= 0.0 else math.exp(log_p)
    return {**base, "statistic": float(l1), "epsilon": float(eps),
            "p_value": float(p_value), "reject": bool(reject),
            "conclusion": "Reject H0" if reject else "Fail to reject H0"}



def tolerance_counts(states, counts, p0, *, tolerance: float = 0.05,
                     alpha: float = 0.05) -> dict:
    _row_strings(states)
    if not (0 < tolerance < 1):
        raise ValueError(f"tolerance must be in (0, 1), got {tolerance}")
    obs_counts, m = _observed_counts(states, counts)
    p_hat = {s: c / m for s, c in obs_counts.items()} if m else {}

    states_list = list(p0.keys())
    abs_errors, rel_errors, within = {}, {}, {}
    for state in states_list:
        p_t = p0[state]
        p_e = p_hat.get(state, 0.0)
        ae = abs(p_e - p_t)
        re = ae / p_t if p_t > 0 else (float("inf") if p_e > 0 else 0.0)
        abs_errors[state] = float(ae)
        rel_errors[state] = float(re)
        within[state] = bool(re <= tolerance)

    worst_state = max(rel_errors.keys(), key=lambda s: rel_errors[s])
    worst_rel_error = rel_errors[worst_state]
    n_violations = sum(1 for v in within.values() if not v)
    reject = n_violations > 0
    max_rel_error = max(rel_errors.values())
    return {"test_name": f"Simple Tolerance Joint Test ({tolerance*100:.1f}% threshold)",
            "statistic": float(max_rel_error), "p_value": float("nan"),
            "reject": bool(reject),
            "conclusion": "Reject H0" if reject else "Fail to reject H0",
            "tolerance": float(tolerance), "n_samples": int(m),
            "n_states": len(states_list), "n_observed_states": len(obs_counts),
            "n_violations": int(n_violations), "worst_state": worst_state,
            "worst_relative_error": float(worst_rel_error),
            "worst_true_prob": float(p0[worst_state]),
            "worst_empirical_prob": float(p_hat.get(worst_state, 0.0))}
