
import numpy as np
from scipy.stats import binom, chi2
from scipy.special import erf

_PRODUCTION_TIE_ATOL = 1e-16
_SCIPY_TIE_RTOL = 1e-7

_TWO_SIDED_CONVENTIONS = ("production", "scipy")



def _validate(X, p0_marg, *, strict_interior):
    X = np.asarray(X)
    p0 = np.asarray(p0_marg, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if p0.ndim != 1:
        raise ValueError(f"p0_marg must be 1D, got {p0.shape}")
    m, d = X.shape
    if len(p0) != d:
        raise ValueError(f"Dimension mismatch: X has {d} columns, p0_marg has {len(p0)}")
    if not np.all((X == 0) | (X == 1)):
        raise ValueError("X must be binary (0 or 1)")
    if strict_interior:
        if not np.all((p0 > 0) & (p0 < 1)):
            raise ValueError("p0_marg must be in (0, 1)")
    elif not np.all((p0 >= 0) & (p0 <= 1)):
        raise ValueError("p0_marg must be in [0, 1]")
    return X, p0, m, d


def cochran_gate(m, p0_marg):
    p0 = np.asarray(p0_marg, dtype=float)
    expected_ones = m * p0
    expected_zeros = m * (1.0 - p0)
    all_expected = np.concatenate([expected_ones, expected_zeros])
    total_cells = 2 * len(p0)
    min_expected = float(np.min(all_expected))
    counts_gte_5 = int(np.sum(all_expected >= 5.0))
    percent_gte_5 = counts_gte_5 / total_cells
    return {
        "condition_i": bool(min_expected >= 1.0),
        "condition_ii": bool(percent_gte_5 >= 0.80),
        "min_expected": min_expected,
        "percent_gte_5": float(percent_gte_5),
        "n_cells": total_cells,
    }


def _binom_two_sided_p(k, n, p, *, convention="production", midp=False):
    pmf_obs = binom.pmf(k, n, p)
    ks = np.arange(n + 1)
    pmf = binom.pmf(ks, n, p)
    if convention == "production":
        pv = pmf[pmf <= pmf_obs + _PRODUCTION_TIE_ATOL].sum()
    elif convention == "scipy":
        pv = pmf[pmf <= pmf_obs * (1.0 + _SCIPY_TIE_RTOL)].sum()
    else:
        raise ValueError(f"two_sided_convention must be one of {_TWO_SIDED_CONVENTIONS}")
    if midp:
        pv -= 0.5 * pmf_obs
    pv = min(1.0, pv)
    return max(0.0, min(1.0, pv))



def exact_binomial_bonferroni(X, p0_marg, *, alpha=0.05, alternative="two-sided",
                              midp=False, two_sided_convention="production"):
    X, p0, m, d = _validate(X, p0_marg, strict_interior=False)
    counts = X.sum(axis=0)
    pvals = np.zeros(d, dtype=float)
    for i in range(d):
        k = int(counts[i])
        p = float(p0[i])
        if alternative == "less":
            pv = binom.cdf(k, m, p)
            if midp:
                pv -= 0.5 * binom.pmf(k, m, p)
            pv = max(0.0, min(1.0, pv))
        elif alternative == "greater":
            pv = 1.0 - binom.cdf(k - 1, m, p)
            if midp:
                pv -= 0.5 * binom.pmf(k, m, p)
            pv = max(0.0, min(1.0, pv))
        elif alternative == "two-sided":
            pv = _binom_two_sided_p(k, m, p, convention=two_sided_convention, midp=midp)
        else:
            raise ValueError("alternative must be 'less', 'greater', or 'two-sided'")
        pvals[i] = pv

    p_family = float(min(1.0, d * np.min(pvals)))
    return {
        "test_name": "Exact Binomial Test (Bonferroni)",
        "p_value": p_family,
        "reject": bool(p_family < alpha),
        "valid": True,
        "p_values_individual": pvals,
        "worst_qubit": int(np.argmin(pvals)),
        "alpha_per_qubit": float(alpha / d),
        "counts": counts.astype(np.int64),
        "m": int(m),
        "alternative": alternative,
        "midp": bool(midp),
        "two_sided_convention": two_sided_convention,
    }



def pearson_marginal_test(X, p0_marg, *, alpha=0.05, force=False):
    X, p0, m, d = _validate(X, p0_marg, strict_interior=True)
    gate = cochran_gate(m, p0)
    is_valid = gate["condition_i"] and gate["condition_ii"]
    if not is_valid and not force:
        raise ValueError("Chi-square test INVALID by Cochran's rule.")

    p_hat = X.mean(axis=0)
    var_null = p0 * (1.0 - p0)
    if np.any(var_null < 1e-10):
        raise ValueError("p0_marg too close to 0 or 1")

    chi2_stat = m * np.sum((p_hat - p0) ** 2 / var_null)
    p_value = 1.0 - chi2.cdf(chi2_stat, df=d)
    return {
        "test_name": "Chi-square Test",
        "p_value": float(p_value),
        "reject": bool(p_value < alpha),
        "valid": bool(is_valid),
        "statistic": float(chi2_stat),
        "df": int(d),
        "cochran_condition_i": gate["condition_i"],
        "cochran_condition_ii": gate["condition_ii"],
    }


def adaptive_test(X, p0_marg, *, alpha=0.05, two_sided_convention="production"):
    X, p0, m, d = _validate(X, p0_marg, strict_interior=False)
    gate = cochran_gate(m, p0)
    use_chi_square = gate["condition_i"] and gate["condition_ii"]
    if use_chi_square:
        res = pearson_marginal_test(X, p0, alpha=alpha, force=False)
        branch = "pearson"
    else:
        res = exact_binomial_bonferroni(X, p0, alpha=alpha,
                                        two_sided_convention=two_sided_convention)
        branch = "exact_binomial"
    out = dict(res)
    out["branch"] = branch
    out["cochran_condition_i"] = gate["condition_i"]
    out["cochran_condition_ii"] = gate["condition_ii"]
    out["cochran_satisfied"] = bool(use_chi_square)
    return out



def _sanitize_R(R, eps=1e-8):
    R = np.asarray(R, dtype=float)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    R = np.clip(R, -1 + 1e-6, 1 - 1e-6)
    return R


def _nearest_correlation(R, eps=1e-8, iters=5):
    R = _sanitize_R(R, eps)
    for _ in range(iters):
        try:
            w, V = np.linalg.eigh(R)
        except np.linalg.LinAlgError:
            R = R + eps * np.eye(R.shape[0])
            continue
        w = np.maximum(w, eps)
        A = (V * w) @ V.T
        d = np.sqrt(np.clip(np.diag(A), eps, None))
        Dinv = np.diag(1.0 / d)
        R = Dinv @ A @ Dinv
        R = 0.5 * (R + R.T)
        np.fill_diagonal(R, 1.0)
    return R


def sample_bernoulli_copula(n, p, R, rng):
    m = len(p)
    R_work = np.array(R, copy=True, dtype=float)
    if not np.allclose(R_work, R_work.T):
        R_work = 0.5 * (R_work + R_work.T)
    np.fill_diagonal(R_work, 1.0)
    w, V = np.linalg.eigh(R_work)
    w = np.maximum(w, 0.0)
    R_psd = (V * w) @ V.T
    try:
        L = np.linalg.cholesky(R_psd)
    except np.linalg.LinAlgError:
        w = np.maximum(w, 1e-10)
        L = V @ np.diag(np.sqrt(w))
    Z_uncorr = rng.standard_normal(size=(n, m))
    Z = Z_uncorr @ L.T
    U = 0.5 * (1.0 + erf(Z / np.sqrt(2.0)))
    X = (U <= p[None, :]).astype(int)
    return X


def estimate_R(X):
    X = np.asarray(X)
    col_stds = np.std(X, axis=0)
    if np.any(col_stds < 1e-10):
        return np.eye(X.shape[1])
    R = np.corrcoef(X, rowvar=False)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def bootstrap_test(X, p0_marg, *, alpha=0.05, n_bootstrap=300, rng, R_assumed=None):
    X, p0, m, d = _validate(X, p0_marg, strict_interior=True)
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap >= 100")
    if R_assumed is None:
        R_assumed = estimate_R(X)
    R_assumed = np.asarray(R_assumed, dtype=float)
    if R_assumed.shape != (d, d):
        raise ValueError(f"R_assumed must be {d}x{d}")
    R_assumed = _nearest_correlation(R_assumed)

    p_hat_obs = X.mean(axis=0)
    var_null = p0 * (1.0 - p0)
    if np.any(var_null < 1e-10):
        raise ValueError("p0_marg too close to 0 or 1")
    chi2_obs = m * np.sum((p_hat_obs - p0) ** 2 / var_null)

    chi2_bootstrap = np.zeros(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        X_boot = sample_bernoulli_copula(m, p0, R_assumed, rng)
        p_hat_boot = X_boot.mean(axis=0)
        chi2_bootstrap[b] = m * np.sum((p_hat_boot - p0) ** 2 / var_null)

    p_value = (1.0 + np.sum(chi2_bootstrap >= chi2_obs)) / (n_bootstrap + 1.0)
    return {
        "test_name": "Chi-square Bootstrap Test",
        "p_value": float(p_value),
        "reject": bool(p_value < alpha),
        "valid": True,
        "statistic": float(chi2_obs),
        "n_bootstrap": int(n_bootstrap),
        "assumed_independence": bool(np.allclose(R_assumed, np.eye(d))),
    }



def sequential_test(X, p0_marg, *, alpha=0.05, min_n=30, check_interval=50,
                    two_sided_convention="production", branch="adaptive"):
    if branch not in ("adaptive", "exact"):
        raise ValueError(f"branch must be 'adaptive' or 'exact', got {branch!r}")
    X, p0, m, d = _validate(X, p0_marg, strict_interior=True)
    if min_n < 10:
        raise ValueError(f"min_n must be at least 10, got {min_n}")
    if check_interval < 1:
        raise ValueError(f"check_interval must be at least 1, got {check_interval}")

    if m < min_n:
        looks = [m]
    else:
        looks = list(range(min_n, m + 1, check_interval))
        if looks[-1] != m:
            looks.append(m)

    n_looks = len(looks)
    alpha_per_look = alpha / n_looks
    look_history = []
    decision = None
    final_result = None
    n_used = looks[-1]

    for i, n_cur in enumerate(looks, 1):
        if branch == "exact":
            res = exact_binomial_bonferroni(X[:n_cur, :], p0, alpha=alpha_per_look,
                                            two_sided_convention=two_sided_convention)
        else:
            res = adaptive_test(X[:n_cur, :], p0, alpha=alpha_per_look,
                                two_sided_convention=two_sided_convention)
        rej = res["p_value"] <= alpha_per_look
        look_history.append({
            "look": i,
            "n": int(n_cur),
            "test_used": res["test_name"],
            "p_value": float(res["p_value"]),
            "threshold": float(alpha_per_look),
            "reject": bool(rej),
            "valid": bool(res.get("valid", True)),
        })
        final_result = res
        if rej:
            decision = "reject"
            n_used = n_cur
            break

    if decision is None:
        decision = "fail_to_reject"

    return {
        "test_name": ("Sequential Exact Binomial Test (Bonferroni over looks)" if branch == "exact"
                      else "Sequential Chi-square Test (Bonferroni over looks)"),
        "branch": branch,
        "p_value": float(final_result["p_value"]),
        "reject": bool(decision == "reject"),
        "valid": bool(final_result.get("valid", True)),
        "n_used": int(n_used),
        "stopped_early": bool(n_used < m),
        "n_looks_planned": int(n_looks),
        "n_looks_used": int(len(look_history)),
        "alpha_per_look": float(alpha_per_look),
        "decision": decision,
        "final_test": final_result["test_name"],
        "look_history": look_history,
    }



def array_stream(X):
    X = np.asarray(X)
    state = {"i": 0}

    def stream(batch):
        i = state["i"]
        j = min(i + int(batch), X.shape[0])
        state["i"] = j
        return X[i:j]

    return stream


def bernoulli_stream(p, d, rng):
    p = np.broadcast_to(np.asarray(p, dtype=float), (d,))

    def stream(batch):
        return (rng.random((int(batch), d)) < p[None, :]).astype(np.int8)

    return stream


def _llr_increments(p0_i, p1_i):
    eps = 1e-12
    a = min(max(p0_i, eps), 1.0 - eps)
    b = min(max(p1_i, eps), 1.0 - eps)
    return float(np.log((1.0 - b) / (1.0 - a))), float(np.log(b / a))


def sprt_expected_n(p0_i, delta, alpha_dir, beta, *, direction="upper", under="H0"):
    p1_i = p0_i + delta if direction == "upper" else p0_i - delta
    p1_i = min(max(p1_i, 1e-12), 1.0 - 1e-12)
    z0, z1 = _llr_increments(p0_i, p1_i)
    logA = np.log((1.0 - beta) / alpha_dir)
    logB = np.log(beta / (1.0 - alpha_dir))
    p_gen = p0_i if under == "H0" else p1_i
    ez = p_gen * z1 + (1.0 - p_gen) * z0
    if under == "H0":
        num = (1.0 - alpha_dir) * logB + alpha_dir * logA
    else:
        num = beta * logB + (1.0 - beta) * logA
    return float(num / ez)


def sprt_test(stream, p0_marg, *, delta, alpha=0.05, beta=0.2, batch=1,
              max_n=100_000, directions=("upper", "lower"), direction_split=True):
    p0 = np.asarray(p0_marg, dtype=float)
    d = len(p0)
    if delta <= 0:
        raise ValueError("delta must be > 0")
    for direction in directions:
        if direction not in ("upper", "lower"):
            raise ValueError("directions entries must be 'upper' or 'lower'")
    n_dir = len(directions)
    alpha_i = alpha / d
    alpha_dir = alpha_i / n_dir if (direction_split and n_dir > 1) else alpha_i
    logA = np.log((1.0 - beta) / alpha_dir)
    logB = np.log(beta / (1.0 - alpha_dir))

    z0 = np.zeros((n_dir, d))
    z1 = np.zeros((n_dir, d))
    for j, direction in enumerate(directions):
        for i in range(d):
            p1_i = p0[i] + delta if direction == "upper" else p0[i] - delta
            z0[j, i], z1[j, i] = _llr_increments(float(p0[i]), float(p1_i))

    llr = np.zeros((n_dir, d))
    accepted = np.zeros((n_dir, d), dtype=bool)
    rejected = np.zeros((n_dir, d), dtype=bool)
    qubit_n = np.full(d, -1, dtype=np.int64)

    n_used = 0
    decision = None
    truncated = False
    while decision is None:
        take = min(int(batch), max_n - n_used)
        if take <= 0:
            truncated = True
            break
        chunk = np.asarray(stream(take))
        if chunk.size == 0:
            truncated = True
            break
        if chunk.ndim != 2 or chunk.shape[1] != d:
            raise ValueError(f"stream must yield (k, {d}) arrays, got {chunk.shape}")
        if not np.all((chunk == 0) | (chunk == 1)):
            raise ValueError("stream must yield binary values")

        ones = chunk.sum(axis=0).astype(float)
        zeros = float(chunk.shape[0]) - ones
        n_used += chunk.shape[0]
        live = ~(accepted | rejected)
        llr += live * (z1 * ones[None, :] + z0 * zeros[None, :])

        rejected |= live & (llr >= logA)
        accepted |= live & (llr <= logB)
        decided = (rejected | accepted).all(axis=0)
        newly = decided & (qubit_n < 0)
        qubit_n[newly] = n_used

        if rejected.any():
            decision = "reject"
        elif decided.all():
            decision = "accept"
        elif chunk.shape[0] < take:
            truncated = True
            break

    if decision is None:
        decision = "inconclusive"

    qubit_n = np.where(qubit_n < 0, n_used, qubit_n)
    return {
        "test_name": "Two-sided SPRT (Bonferroni over qubits)",
        "p_value": float("nan"),
        "reject": bool(decision == "reject"),
        "valid": bool(not truncated),
        "decision": decision,
        "n_used": int(n_used),
        "truncated": bool(truncated),
        "alpha_dir": float(alpha_dir),
        "log_A": float(logA),
        "log_B": float(logB),
        "llr": llr,
        "qubit_n_used": qubit_n,
        "rejected_qubits": np.where(rejected.any(axis=0))[0],
    }



def g_test_marginal(X, p0_marg, *, alpha=0.05):
    X, p0, m, d = _validate(X, p0_marg, strict_interior=False)
    ones = X.sum(axis=0).astype(float)
    zeros = float(m) - ones
    e1 = m * p0
    e0 = m * (1.0 - p0)

    def term(o, e):
        out = np.zeros(d, dtype=float)
        pos = o > 0
        safe_e = np.where(e > 0, e, 1.0)
        ratio = np.where(e > 0, o / safe_e, np.inf)
        out[pos] = o[pos] * np.log(ratio[pos])
        return out

    g_stat = 2.0 * (term(ones, e1) + term(zeros, e0))
    pvals = chi2.sf(g_stat, 1)
    p_family = float(min(1.0, d * np.min(pvals)))
    return {
        "test_name": "G-test (Bonferroni)",
        "p_value": p_family,
        "reject": bool(p_family < alpha),
        "valid": True,
        "statistics": g_stat,
        "p_values_individual": pvals,
        "worst_qubit": int(np.argmin(pvals)),
        "df": 1,
    }



def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"p_value": float("nan"), "reject": False, "valid": False,
                "error": f"{type(exc).__name__}: {exc}"}
