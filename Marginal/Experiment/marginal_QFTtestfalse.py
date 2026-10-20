#!/usr/bin/env python
from scipy.stats import chi2, binom
import numpy as np
from typing import Dict, Tuple, List
import os
from datetime import datetime
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import warnings
from numpy.random import default_rng
from scipy.spatial.distance import cdist, pdist, squareform
from itertools import combinations
from numpy.linalg import cholesky, eigh, inv
import math
import pandas as pd
from itertools import product
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm.auto import tqdm
from fuzzstorm import produce_state
warnings.filterwarnings('ignore')
APPLY_QFT_TO_EXTERNAL = False

def bitstring(k: int, n: int) -> str:
    return format(k, f'0{n}b')

def qft_matrix(n: int) -> np.ndarray:
    N = 1 << n
    j = np.arange(N).reshape(-1, 1)
    k = np.arange(N).reshape(1, -1)
    omega = np.exp(2j * np.pi * (j @ k) / N)
    return omega / np.sqrt(N)

def phase_kicked_superposition(n: int, phi: float) -> np.ndarray:
    N = 1 << n
    x = np.arange(N, dtype=float)
    state = np.exp(2j * np.pi * phi * x) / np.sqrt(N)
    return state.astype(complex)

def qft_of_phase_state(n: int, phi: float) -> np.ndarray:
    F = qft_matrix(n)
    psi_in = phase_kicked_superposition(n, phi)
    return F @ psi_in

def probability_distribution(state: np.ndarray) -> np.ndarray:
    probs = np.abs(state)**2
    return probs / probs.sum()

def state_to_marginals(state: np.ndarray, n_qubits: int) -> np.ndarray:
    state = np.asarray(state, dtype=complex).reshape(-1)
    N = 1 << n_qubits
    if state.size != N:
        raise ValueError(f"state 长度 {state.size} 与 2^{n_qubits} 不一致")

    probs = probability_distribution(state)
    p_true = np.zeros(n_qubits, dtype=float)

    for j in range(n_qubits):
        p1 = 0.0
        for idx, prob in enumerate(probs):
            if bitstring(idx, n_qubits)[j] == '1':
                p1 += prob
        p_true[j] = p1

    return p_true




def probs_from_state(psi: np.ndarray) -> np.ndarray:
    p = np.abs(np.asarray(psi, dtype=complex))**2
    s = p.sum()
    return p / s if s > 0 else p

def marginals_from_probs(n: int, probs: np.ndarray) -> np.ndarray:
    N = 1 << n
    m = np.zeros(n, dtype=float)
    for idx in range(N):
        bs = bitstring(idx, n)
        for i, ch in enumerate(bs):
            if ch == '1':
                m[i] += probs[idx]
    return m

def measurements_from_state(psi: np.ndarray, n: int, shots: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = probs_from_state(psi)
    p = p / p.sum()
    N = 1 << n
    idxs = rng.choice(N, size=shots, p=p)
    X = np.zeros((shots, n), dtype=int)
    for i, idx in enumerate(idxs):
        bs = bitstring(int(idx), n)
        X[i, :] = [1 if ch == '1' else 0 for ch in bs]
    return X

def single_qubit_marginals(n: int, probs: np.ndarray) -> List[Tuple[float, float, float]]:
    N = 1 << n
    out = []
    for i in range(n):
        p0 = 0.0; p1 = 0.0
        for idx in range(N):
            if bitstring(idx, n)[i] == '0':
                p0 += probs[idx]
            else:
                p1 += probs[idx]
        out.append((p0, p1, p0 - p1))
    return out

def _draw_phi(rng: np.random.Generator, guard: float = 1e-4) -> float:
    while True:
        phi = rng.random()
        if guard < phi < 1.0 - guard:
            return float(phi)

def run_once(n: int, seed: int = None, verbose: bool = False, jitter_ratio=None) -> Dict:
    rng = np.random.default_rng(seed)
    if n <= 0:
        raise ValueError("n 必须为正整数")
    phi = _draw_phi(rng, guard=1e-4)
    final_state = qft_of_phase_state(n, phi)
    probs = probability_distribution(final_state)

    order = np.argsort(-probs)
    marginals = single_qubit_marginals(n, probs)
    ordered_probs = {bitstring(idx, n): float(probs[idx]) for idx in order}
    return {'marginal': np.array(marginals)[:, 1], 'joint': ordered_probs}

def qft_nonuniform_measurements(n: int, shots: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = _draw_phi(rng, guard=1e-4)
    psi = qft_of_phase_state(n, phi)
    probs = probability_distribution(psi)
    probs = probs / probs.sum()

    N = 1 << n
    sampled = rng.choice(N, size=shots, p=probs)
    X = np.zeros((shots, n), dtype=int)
    for i, idx in enumerate(sampled):
        bs = bitstring(int(idx), n)
        X[i, :] = [1 if ch == '1' else 0 for ch in bs]
    return X


def ensure_binary_matrix(samples_in, n_expected=None, m_take=None):
    if isinstance(samples_in, dict):
        if 'joint' in samples_in:
            X = np.asarray(samples_in['joint'], dtype=int)
        elif 'marginal' in samples_in:
            X = np.asarray(samples_in['marginal'], dtype=int)
        else:
            raise ValueError("samples dict 必须包含 'joint' 或 'marginal'")
    elif isinstance(samples_in, (list, tuple)) and len(samples_in) > 0 and isinstance(samples_in[0], (str, bytes)):
        rows = []
        for s in samples_in:
            bs = s.decode() if isinstance(s, bytes) else s
            rows.append([1 if ch == '1' else 0 for ch in bs.strip()])
        X = np.asarray(rows, dtype=int)
    else:
        X = np.asarray(samples_in, dtype=int)

    if X.ndim != 2:
        raise ValueError(f"samples 必须是二维，得到维度 {X.ndim}")

    if m_take is not None:
        X = X[:m_take, :]

    if n_expected is not None:
        if X.shape[1] < n_expected:
            raise ValueError(f"samples 列数({X.shape[1]}) < 期望列数({n_expected})")
        elif X.shape[1] > n_expected:
            X = X[:, :n_expected]

    if not np.all((X == 0) | (X == 1)):
        raise ValueError("samples 中存在非 0/1 值")
    return X

def apply_random_errors(X: np.ndarray,
                        mode: str = 'bitflip',
                        p: float = 0.05,
                        p01: float = None,
                        p10: float = None,
                        row_mix: float = 0.0,
                        seed: int = None,
                        ensure_change: bool = False,
                        force_flip_cols: Tuple[int, ...] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=int)
    m, n = X.shape
    X_noisy = X.copy()

    if force_flip_cols is not None and len(force_flip_cols) > 0:
        cols = [c for c in force_flip_cols if 0 <= c < n]
        if cols:
            X_noisy[:, cols] ^= 1

    if mode == 'bitflip':
        if p > 0.0:
            mask = rng.random(size=X_noisy.shape) < float(p)
            X_noisy ^= mask.astype(int)
    elif mode == 'asymmetric':
        if p01 is None or p10 is None:
            raise ValueError("asymmetric 模式需要提供 p01 与 p10")
        if p01 > 0.0:
            flip01 = (X_noisy == 0) & (rng.random(size=X_noisy.shape) < float(p01))
            X_noisy[flip01] = 1
        if p10 > 0.0:
            flip10 = (X_noisy == 1) & (rng.random(size=X_noisy.shape) < float(p10))
            X_noisy[flip10] = 0
    elif mode == 'depolarize':
        if p > 0.0:
            row_mask = rng.random(size=m) < float(p)
            num_rows = int(np.sum(row_mask))
            if num_rows > 0:
                X_noisy[row_mask, :] = rng.integers(0, 2, size=(num_rows, n))
    else:
        raise ValueError(f"未知错误模式: {mode}")

    if row_mix and row_mix > 0.0:
        row_mask2 = rng.random(size=m) < float(row_mix)
        num_rows2 = int(np.sum(row_mask2))
        if num_rows2 > 0:
            X_noisy[row_mask2, :] = rng.integers(0, 2, size=(num_rows2, n))

    if ensure_change and np.array_equal(X_noisy, X):
        r = rng.integers(0, m)
        c = rng.integers(0, n)
        X_noisy[r, c] ^= 1

    return X_noisy


def chi_square_marginal_test(samples, p_true, alpha=0.05, force=False):
    samples = np.asarray(samples); p_true = np.asarray(p_true)
    if samples.ndim != 2: raise ValueError(f"samples must be 2D, got {samples.shape}")
    if p_true.ndim != 1: raise ValueError(f"p_true must be 1D, got {p_true.shape}")
    n, m = samples.shape
    if len(p_true) != m: raise ValueError(f"Dimension mismatch: {m} vs {len(p_true)}")
    if not np.all((samples == 0) | (samples == 1)): raise ValueError("samples must be binary")
    if not np.all((p_true > 0) & (p_true < 1)): raise ValueError("p_true in (0,1)")

    expected_ones = n * p_true
    expected_zeros = n * (1 - p_true)
    all_expected = np.concatenate([expected_ones, expected_zeros])
    total_cells = 2 * m

    condition_i = np.min(all_expected) >= 1.0
    percent_gte_5 = np.sum(all_expected >= 5.0) / total_cells
    condition_ii = percent_gte_5 >= 0.80
    is_valid = condition_i and condition_ii

    warnings = []
    if not condition_i: warnings.append("COCHRAN(i) violated: min expected < 1.")
    if not condition_ii: warnings.append("COCHRAN(ii) violated: <80% expected >= 5.")
    if not is_valid: warnings.append("Use exact_binomial_test() instead.")
    if n < 30: warnings.append(f"NOTE: n={n} < 30; asymptotic approx less accurate.")

    if not is_valid and not force:
        raise ValueError("Chi-square invalid by Cochran rule.\n" + "\n".join(warnings))

    p_hat = samples.mean(axis=0)
    var_null = p_true * (1.0 - p_true)
    if np.any(var_null < 1e-10): raise ValueError("p_true too close to 0 or 1")

    chi2_stat = n * np.sum((p_hat - p_true)**2 / var_null)
    df = m
    p_value = 1.0 - chi2.cdf(chi2_stat, df=df)
    reject = p_value < alpha

    return {
        'test_name': 'Chi-square Test',
        'valid': is_valid,
        'p_value': float(p_value),
        'conclusion': ("Reject H0: samples NOT from p_true (at alpha={})".format(alpha)
                       if reject else
                       "Fail to reject H0: consistent with p_true (at alpha={})".format(alpha)),
        'warnings': warnings
    }

def exact_binomial_test(samples, p_true, alpha=0.05, method='bonferroni', alternative='two-sided', midp=False):
    samples = np.asarray(samples); p_true = np.asarray(p_true, dtype=float)
    if samples.ndim != 2: raise ValueError("samples must be 2D")
    if p_true.ndim != 1: raise ValueError("p_true must be 1D")
    n, m = samples.shape
    if len(p_true) != m: raise ValueError("Dimension mismatch")
    if not np.all((samples == 0) | (samples == 1)): raise ValueError("samples must be binary")
    if not np.all((p_true >= 0) & (p_true <= 1)): raise ValueError("p_true in [0,1]")

    counts = samples.sum(axis=0)
    pvals = np.zeros(m, dtype=float)
    for i in range(m):
        k = int(counts[i]); p = float(p_true[i])
        if alternative == 'less':
            pv = binom.cdf(k, n, p);  pv -= 0.5*binom.pmf(k, n, p) if midp else 0.0
        elif alternative == 'greater':
            pv = 1.0 - binom.cdf(k-1, n, p); pv -= 0.5*binom.pmf(k, n, p) if midp else 0.0
        elif alternative == 'two-sided':
            pmf_obs = binom.pmf(k, n, p)
            ks = np.arange(n+1); pmf = binom.pmf(ks, n, p)
            pv = pmf[pmf <= pmf_obs + 1e-16].sum()
            if midp: pv -= 0.5*pmf_obs
            pv = min(1.0, pv)
        else:
            raise ValueError("alternative must be 'less', 'greater', or 'two-sided'")
        pvals[i] = max(0.0, min(1.0, pv))

    if method == 'bonferroni':
        p_value_family = float(min(1.0, m * np.min(pvals)))
        reject = bool(p_value_family < alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        'test_name': 'Exact Binomial Test (Bonferroni)',
        'valid': True,
        'p_value': p_value_family,
        'conclusion': ("Reject H0: samples NOT from p_true (at alpha={})".format(alpha)
                       if reject else
                       "Fail to reject H0: consistent with p_true (at alpha={})".format(alpha)),
        'p_values_individual': pvals.tolist(),
        'alternative': alternative,
        'midp': bool(midp),
        'warnings': []
    }

def chi_square_marginal_auto_test(samples, p_true, alpha=0.05, verbose=True):
    samples = np.asarray(samples); p_true = np.asarray(p_true)
    n, m = samples.shape
    expected_ones = n * p_true
    expected_zeros = n * (1 - p_true)
    all_expected = np.concatenate([expected_ones, expected_zeros])
    total_cells = 2 * m
    min_expected = np.min(all_expected)
    counts_gte_5 = np.sum(all_expected >= 5.0)
    percent_gte_5 = counts_gte_5 / total_cells
    condition_i = min_expected >= 1.0
    condition_ii = percent_gte_5 >= 0.80
    use_chi_square = condition_i and condition_ii

    if verbose:
        print("Cochran Check: (i) min>=1:", condition_i, " (ii) >=80%>=5:", condition_ii)

    if use_chi_square:
        return chi_square_marginal_test(samples, p_true, alpha, force=False)
    else:
        return exact_binomial_test(samples, p_true, alpha)

def sequential_chi_square_test(samples, p_true, alpha=0.05, min_n=30, check_interval=10, verbose=False):
    samples = np.asarray(samples); p_true = np.asarray(p_true, dtype=float)
    if samples.ndim != 2: raise ValueError(f"samples must be 2D, got {samples.shape}")
    if p_true.ndim != 1: raise ValueError(f"p_true must be 1D, got {p_true.shape}")
    n_total, m = samples.shape
    if len(p_true) != m: raise ValueError(f"Dimension mismatch: samples {m} vs p_true {len(p_true)}")
    if not np.all((samples == 0) | (samples == 1)): raise ValueError("samples must be binary")
    if not np.all((p_true > 0) & (p_true < 1)): raise ValueError("p_true in (0,1)")
    if min_n < 10: raise ValueError("min_n >= 10")
    if check_interval < 1: raise ValueError("check_interval >= 1")

    looks = [n_total] if n_total < min_n else list(range(min_n, n_total + 1, check_interval))
    if looks[-1] != n_total: looks.append(n_total)

    n_looks = len(looks)
    alpha_per_look = alpha / n_looks
    look_history = []
    decision = None
    final_result = None
    n_used = looks[-1]

    for i, n_cur in enumerate(looks, 1):
        res = chi_square_marginal_auto_test(samples[:n_cur, :], p_true, alpha=alpha_per_look, verbose=False)
        rej = res['p_value'] <= alpha_per_look
        look_history.append({
            'look': i, 'n': int(n_cur),
            'test_used': res['test_name'],
            'p_value': float(res['p_value']),
            'threshold': float(alpha_per_look),
            'reject': bool(rej),
            'valid': bool(res.get('valid', True))
        })
        final_result = res
        if rej:
            decision = 'reject'
            n_used = n_cur
            break

    if decision is None:
        decision = 'fail_to_reject'

    return {
        'test_name': 'Sequential Chi-square Test (Bonferroni over looks)',
        'valid': bool(final_result.get('valid', True)),
        'p_value': float(final_result['p_value']),
        'conclusion': (f"Reject H0: samples NOT from p_true (at alpha={alpha})" if decision == 'reject'
                       else f"Fail to reject H0: consistent with p_true (at alpha={alpha})"),
        'stopped_early': bool(n_used < n_total),
        'n_looks_planned': int(n_looks),
        'n_looks_used': int(len(look_history)),
        'alpha': float(alpha),
        'alpha_per_look': float(alpha_per_look),
        'decision': decision,
        'look_history': look_history,
        'final_test': final_result['test_name'],
        'warnings': final_result.get('warnings', [])
    }

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
            R = R + eps * np.eye(R.shape[0]); continue
        w = np.maximum(w, eps)
        A = (V * w) @ V.T
        d = np.sqrt(np.clip(np.diag(A), eps, None))
        Dinv = np.diag(1.0 / d)
        R = Dinv @ A @ Dinv
        R = 0.5 * (R + R.T)
        np.fill_diagonal(R, 1.0)
    return R

def chi_square_bootstrap_test(samples, p_true, alpha=0.05, R_assumed=None,
                              n_bootstrap=500, seed=None, verbose=False):
    samples = np.asarray(samples); p_true = np.asarray(p_true, dtype=float)
    if samples.ndim != 2: raise ValueError("samples must be 2D")
    if p_true.ndim != 1: raise ValueError("p_true must be 1D")
    n, m = samples.shape
    if len(p_true) != m: raise ValueError("Dimension mismatch")
    if not np.all((samples == 0) | (samples == 1)): raise ValueError("samples must be binary")
    if not np.all((p_true > 0) & (p_true < 1)): raise ValueError("p_true in (0,1)")
    if n_bootstrap < 100: raise ValueError("n_bootstrap >= 100")

    rng = np.random.default_rng(seed)
    if R_assumed is None:
        R_assumed = np.eye(m)
    R_assumed = np.asarray(R_assumed, dtype=float)
    if R_assumed.shape != (m, m): raise ValueError(f"R_assumed must be {m}x{m}")
    R_assumed = _nearest_correlation(R_assumed)

    p_hat_obs = samples.mean(axis=0)
    var_null = p_true * (1.0 - p_true)
    if np.any(var_null < 1e-10): raise ValueError("p_true too close to 0 or 1")
    chi2_obs = n * np.sum((p_hat_obs - p_true)**2 / var_null)

    chi2_bootstrap = np.zeros(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        X_boot = sample_bernoulli_copula(n, p_true, R_assumed, rng)
        p_hat_boot = X_boot.mean(axis=0)
        chi2_bootstrap[b] = n * np.sum((p_hat_boot - p_true)**2 / var_null)

    p_value = (1.0 + np.sum(chi2_bootstrap >= chi2_obs)) / (n_bootstrap + 1.0)
    reject = p_value < alpha
    return {
        'test_name': 'Chi-square Bootstrap Test',
        'valid': True,
        'p_value': float(p_value),
        'conclusion': ("Reject H0: samples NOT from p_true (at alpha={})".format(alpha)
                       if reject else
                       "Fail to reject H0: consistent with p_true (at alpha={})".format(alpha)),
        'assumed_independence': bool(np.allclose(R_assumed, np.eye(m))),
        'warnings': []
    }

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
    from scipy.special import erf
    U = 0.5 * (1.0 + erf(Z / np.sqrt(2.0)))
    X = (U <= p[None, :]).astype(int)
    return X

def simple_tolerance_test(samples, p_true, tolerance=0.05, alpha=0.05):
    samples = np.asarray(samples); p_true = np.asarray(p_true, dtype=float)
    if samples.ndim != 2: raise ValueError("samples must be 2D")
    if p_true.ndim != 1: raise ValueError("p_true must be 1D")
    n, m = samples.shape
    if len(p_true) != m: raise ValueError("Dimension mismatch")
    if not np.all((samples == 0) | (samples == 1)): raise ValueError("samples must be binary")
    if not np.all((p_true > 0) & (p_true < 1)): raise ValueError("p_true in (0,1)")
    if not (0 < tolerance < 1): raise ValueError(f"tolerance must be in (0,1), got {tolerance}")

    p_hat = samples.mean(axis=0)
    abs_errors = np.abs(p_hat - p_true)
    rel_errors = abs_errors / p_true
    within_tolerance = rel_errors <= tolerance
    reject = bool(not np.all(within_tolerance))
    worst_idx = int(np.argmax(rel_errors))
    return {
        'test_name': f'Simple Tolerance Test ({tolerance*100:.1f}% threshold)',
        'valid': True,
        'conclusion': (f"Reject H0: at least one dimension exceeds {tolerance*100:.1f}% tolerance"
                       if reject else
                       f"Fail to reject H0: all dimensions within {tolerance*100:.1f}% tolerance"),
        'tolerance': float(tolerance),
        'p_hat': p_hat.tolist(),
        'p_true': p_true.tolist(),
        'absolute_errors': abs_errors.tolist(),
        'relative_errors': rel_errors.tolist(),
        'within_tolerance': within_tolerance.tolist(),
        'worst_dimension': worst_idx,
        'worst_relative_error': float(rel_errors[worst_idx]),
        'n_violations': int(np.sum(~within_tolerance))
    }

def _get_real_samples_for_run(run_idx: int,
                              n_qubits: int,
                              m_samples: int,
                              rng: np.random.Generator,
                              seed_base: int):
    try:
        sr = samples_runs
        if isinstance(sr, (list, tuple)) and run_idx < len(sr):
            X = ensure_binary_matrix(sr[run_idx], n_expected=n_qubits, m_take=None)
            if X.shape[0] < m_samples:
                raise ValueError(f"samples_runs[{run_idx}] 行数不足 {m_samples}")
            return X[:m_samples, :]
    except NameError:
        pass

    try:
        ps = provide_samples
        if callable(ps):
            X = ps(run_idx, n_qubits, m_samples, seed_base)
            X = ensure_binary_matrix(X, n_expected=n_qubits, m_take=m_samples)
            return X
    except NameError:
        pass

    try:
        Xall = ensure_binary_matrix(samples, n_expected=n_qubits, m_take=None)
        if Xall.shape[0] < m_samples:
            raise ValueError(f"`samples` 行数不足 {m_samples}")
        return Xall[:m_samples, :]
    except NameError:
        raise RuntimeError(
            "未找到当轮真实测量矩阵。请提供下列任一：\n"
            "  - samples_runs: list/tuple，每轮一个 (m_samples, n_qubits) 的 0/1 矩阵；\n"
            "  - provide_samples(run_idx, n_qubits, m_samples, seed_base): 返回当轮矩阵；\n"
            "  - samples: 单个 (>=m_samples, n_qubits) 的 0/1 矩阵（每轮复用）。"
        )

def provide_samples(run_idx: int,
                    n_qubits: int,
                    m_samples: int,
                    seed_base: int) -> np.ndarray:
    seed = seed_base + run_idx

    psi0 = produce_state(n_qubits, seed=seed)

    if APPLY_QFT_TO_EXTERNAL:
        F = qft_matrix(n_qubits)
        psi0 = F @ psi0

    return measurements_from_state(psi0, n_qubits, m_samples, seed=seed)



def run_single_test(X, p_true, method, alpha, tolerance, seed):
    try:
        if method == 'tolerance':
            res = simple_tolerance_test(X, p_true, tolerance=tolerance, alpha=alpha)
        elif method == 'bootstrap':
            col_stds = np.std(X, axis=0)
            min_variance_threshold = 1e-10
            
            if np.any(col_stds < min_variance_threshold):
                R_assumed = np.eye(X.shape[1])
            else:
                R_assumed = np.corrcoef(X, rowvar=False)
                R_assumed = 0.5 * (R_assumed + R_assumed.T)  
                np.fill_diagonal(R_assumed, 1.0) 
            
            res = chi_square_bootstrap_test(
                X, p_true, alpha=alpha, R_assumed=R_assumed,
                n_bootstrap=300, seed=seed, verbose=False
            )
        elif method == 'sequential':
            res = sequential_chi_square_test(
                X, p_true, alpha=alpha, min_n=30, check_interval=50, verbose=False
            )
        elif method == 'auto':
            res = chi_square_marginal_auto_test(X, p_true, alpha=alpha, verbose=False)
        else:
            return {'reject': None, 'p_value': None, 'error': f'unknown method: {method}'}
        return {'reject': ("Fail to reject" not in res["conclusion"]),
                'p_value': res.get('p_value', None), 'error': None}
    except Exception as e:
        return {'reject': None, 'p_value': None, 'error': str(e)}

def get_p_true_for_run(run_idx: int, n_qubits: int, seed_base: int) -> np.ndarray:
    seed = seed_base + run_idx

    psi0 = produce_state(n_qubits, seed=seed)

    if APPLY_QFT_TO_EXTERNAL:
        F = qft_matrix(n_qubits)
        psi0 = F @ psi0

    probs = probs_from_state(psi0)
    p_true = marginals_from_probs(n_qubits, probs)
    return p_true.astype(float)

def run_single_experiment_optimized(
    n_qubits,
    m_samples,
    alpha,
    tolerance,
    seed_base,
    run_idx
):
    seed = seed_base + run_idx
    rng = np.random.default_rng(seed)

    psi0 = produce_state(n_qubits, seed=seed)
    
    probs = probs_from_state(psi0)
    p_true = marginals_from_probs(n_qubits, probs)
    X = measurements_from_state(psi0, n=n_qubits, shots=m_samples, seed=seed)

    methods = ['tolerance', 'bootstrap', 'sequential', 'auto']
    results = {}
    for method in methods:
        results[f'{method}_correct'] = run_single_test(
            X, p_true, method, alpha, tolerance, seed
        )
    return results




def process_single_configuration(config, inner_n_jobs):
    n_qubits, m_samples, alpha, tolerance, seed_base, N_runs = config

    experiment_results = Parallel(n_jobs=inner_n_jobs, backend='loky', verbose=0)(
        delayed(run_single_experiment_optimized)(
            n_qubits, m_samples, alpha, tolerance, seed_base, run_idx
        ) for run_idx in range(N_runs)
    )

    methods = ['tolerance', 'bootstrap', 'sequential', 'auto']
    metrics = {}
    for method in methods:
        correct_rejects = []
        for result in experiment_results:
            r = result[f'{method}_correct']['reject']
            if r is not None:
                correct_rejects.append(r)
        TN = sum((r is False) for r in correct_rejects)
        FP = sum((r is True)  for r in correct_rejects)
        total = TN + FP
        type_i_error = FP / total if total > 0 else 0.0
        accuracy = TN / total if total > 0 else 0.0
        metrics[method] = {
            'type_i_error': type_i_error,
            'accuracy': accuracy,
            'TP': 0, 'TN': TN, 'FP': FP, 'FN': 0
        }

    measurement_reduction_ratio = m_samples / float(2 ** n_qubits)
    results = []
    for method, method_metrics in metrics.items():
        results.append({
            'n_qubits': n_qubits,
            'm_samples': m_samples,
            'alpha': alpha,
            'test_method': method,
            'measurement_reduction_ratio': measurement_reduction_ratio,
            **method_metrics
        })
    return results


def run_comprehensive_simulation_nested_parallel(
    n_qubits_range=range(2, 13),
    m_samples_range=None,
    N_runs=5000,
    alpha_values=[0.05],
    tolerance=0.05,
    seed_base=801,
    n_jobs=-1,
    output_dir='marginal_experiment_results'
):
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if m_samples_range is None:
        m_samples_range = np.unique(np.logspace(np.log10(30), np.log10(2000), 25).astype(int))
    
    total_cores = cpu_count()
    if n_jobs == -1:
        n_jobs = total_cores
    
    configurations = []
    config_idx = 0
    for n_qubits in n_qubits_range:
        for m_samples in m_samples_range:
            for alpha in alpha_values:
                unique_seed_base = seed_base + config_idx * N_runs
                configurations.append((n_qubits, m_samples, alpha, tolerance, unique_seed_base, N_runs))
                config_idx += 1 
    
    total_configs = len(configurations)
    total_experiments = total_configs * N_runs
    
    outer_jobs = min(total_cores, total_configs)
    inner_jobs = 1

    print("=" * 80)
    print("NESTED PARALLEL COMPREHENSIVE SIMULATION")
    print("=" * 80)
    print(f"n_qubits range: {list(n_qubits_range)}")
    print(f"m_samples values: {len(m_samples_range)} points from {min(m_samples_range)} to {max(m_samples_range)}")
    print(f"alpha values: {alpha_values}")
    print(f"N_runs per configuration: {N_runs}")
    print(f"Total configurations: {total_configs}")
    print(f"Total experiments: {total_experiments:,}")
    print(f"\nParallelization strategy:")
    print(f"  Total CPU cores: {total_cores}")
    print(f"  Outer parallelism (configs): {outer_jobs} processes")
    print(f"  Inner parallelism (runs): {inner_jobs} threads per process")
    print(f"  Effective parallelism: {outer_jobs * inner_jobs} concurrent experiments")
    print("=" * 80)
    print("\nStarting nested parallel computation...\n")

    all_results = Parallel(n_jobs=outer_jobs, backend='loky', verbose=0)(
        delayed(process_single_configuration)(config, inner_jobs) 
        for config in tqdm(configurations, desc="Processing configurations", ncols=100)
    )
    
    flattened_results = []
    for config_results in all_results:
        flattened_results.extend(config_results)
    
    results_df = pd.DataFrame(flattened_results)
    
    csv_path = os.path.join(output_dir, f'simulation_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Results saved to: {csv_path}")
    print(f"✓ Total data points: {len(results_df)}")
    
    return results_df

if __name__ == "__main__":
    import time

    n_qubits_range = range(2, 13)
    m_samples_range = np.array([50, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000])
    N_runs = 1000
    alpha_values = [0.05]
    tolerance = 0.05
    seed_base = 801

    start_time = time.time()

    results_df = run_comprehensive_simulation_nested_parallel(
        n_qubits_range=n_qubits_range,
        m_samples_range=m_samples_range,
        N_runs=N_runs,
        alpha_values=alpha_values,
        tolerance=tolerance,
        seed_base=seed_base,
        n_jobs=-1,
        output_dir='marginal_experiment_nested_parallel_results_QFT'
    )

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"\n{'=' * 80}")
    print(f"COMPUTATION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Total runtime: {hours}h {minutes}m {seconds}s")
    print(f"Configurations processed: {len(results_df) // 4}")
    print(f"Total experiments: {(len(results_df) // 4) * N_runs:,}")


    print(f"\n{'=' * 80}")
    print("ALL TASKS COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\nResults directory: marginal_experiment_nested_parallel_results_QFT/")
    print(f"Runtime: {hours}h {minutes}m {seconds}s")
    print(f"{'=' * 80}\n")