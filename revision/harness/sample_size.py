
import numpy as np
from scipy.stats import binom, chi2, ncx2, norm

_DEFAULT_TIE_ATOL = 1e-16
_SCIPY_TIE_RTOL = 1e-7



def two_sided_pvalues(m, p0, *, convention="production"):
    k = np.arange(m + 1)
    pmf = binom.pmf(k, m, p0)
    order = np.argsort(pmf, kind="mergesort")
    ps = pmf[order]
    cum = np.cumsum(ps)
    if convention == "production":
        thresh = pmf + _DEFAULT_TIE_ATOL
    elif convention == "scipy":
        thresh = pmf * (1.0 + _SCIPY_TIE_RTOL)
    else:
        raise ValueError("convention must be 'production' or 'scipy'")
    idx = np.searchsorted(ps, thresh, side="right")
    pv = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)
    return np.minimum(pv, 1.0)


def exact_binomial_power(m, p0, delta, alpha, *, convention="production",
                         worst_case=True):
    pv = two_sided_pvalues(m, p0, convention=convention)
    reject = pv <= alpha
    if not reject.any():
        return 0.0
    k = np.arange(m + 1)
    powers = []
    for p1 in (p0 + delta, p0 - delta):
        if 0.0 < p1 < 1.0:
            powers.append(float(binom.pmf(k, m, p1)[reject].sum()))
    if not powers:
        raise ValueError(f"no admissible alternative for p0={p0}, delta={delta}")
    return min(powers) if worst_case else max(powers)


def normal_approx_n(p0, delta, alpha=0.05, beta=0.2, *, worst_case=True):
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be in (0, 1)")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    za = norm.ppf(1.0 - alpha / 2.0)
    zb = norm.ppf(1.0 - beta)
    vals = []
    for p1 in (p0 + delta, p0 - delta):
        if 0.0 < p1 < 1.0:
            vals.append(((za * np.sqrt(p0 * (1.0 - p0))
                          + zb * np.sqrt(p1 * (1.0 - p1))) / delta) ** 2)
    if not vals:
        raise ValueError(f"no admissible alternative for p0={p0}, delta={delta}")
    return float(max(vals) if worst_case else min(vals))


def exact_binomial_n(p0, delta, alpha=0.05, beta=0.2, *, mode="first",
                     stable_window=25, back_window=250, m_max=2_000_000,
                     convention="production", full_scan=False):
    if mode not in ("first", "stable"):
        raise ValueError("mode must be 'first' or 'stable'")
    target = 1.0 - beta

    def power(m):
        return exact_binomial_power(m, p0, delta, alpha, convention=convention)

    def stable_ok(m):
        return all(power(m + j) >= target for j in range(stable_window))

    if full_scan:
        m = 2
        while m <= m_max:
            if power(m) >= target:
                if mode == "first":
                    return int(m)
                if stable_ok(m):
                    return int(m)
            m += 1
        raise RuntimeError(f"no m <= {m_max} reaches power {target}")

    hi = max(5, int(np.ceil(normal_approx_n(p0, delta, alpha, beta))))
    while not stable_ok(hi):
        hi *= 2
        if hi > m_max:
            raise RuntimeError(f"no m <= {m_max} reaches power {target}")
    lo = max(2, hi // 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if stable_ok(mid):
            hi = mid
        else:
            lo = mid + 1
    m_stable = lo
    if mode == "stable":
        return int(m_stable)

    best = m_stable
    misses = 0
    m = m_stable - 1
    while m >= 2 and misses < back_window:
        if power(m) >= target:
            best = m
            misses = 0
        else:
            misses += 1
        m -= 1
    return int(best)


def bonferroni_marginal_n(p0, delta, alpha=0.05, beta=0.2, d=1, **kwargs):
    if d < 1:
        raise ValueError("d must be >= 1")
    return exact_binomial_n(p0, delta, alpha / d, beta, **kwargs)


def bonferroni_marginal_n_normal(p0, delta, alpha=0.05, beta=0.2, d=1):
    return normal_approx_n(p0, delta, alpha / d, beta)



def chisq_joint_n(w, df, alpha=0.05, beta=0.2, *, n_max=1e9):
    if w <= 0:
        raise ValueError("w must be > 0")
    if df < 1:
        raise ValueError("df must be >= 1")
    crit = chi2.ppf(1.0 - alpha, df)
    target = 1.0 - beta

    def power(n):
        return float(ncx2.sf(crit, df, n * w * w))

    lo, hi = 1.0, 16.0
    while power(hi) < target:
        hi *= 2.0
        if hi > n_max:
            raise RuntimeError(f"no n <= {n_max:g} reaches power {target}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if power(mid) < target:
            lo = mid
        else:
            hi = mid
    return int(np.ceil(hi))


def chisq_joint_power(n, w, df, alpha=0.05):
    return float(ncx2.sf(chi2.ppf(1.0 - alpha, df), df, n * w * w))


def cohen_w(p1, p0):
    p1 = np.asarray(p1, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    keep = p0 > 0
    if np.any(p1[~keep] > 0):
        return float("inf")
    return float(np.sqrt(np.sum((p1[keep] - p0[keep]) ** 2 / p0[keep])))
