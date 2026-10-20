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


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    norm = np.linalg.norm(psi)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Statevector has invalid norm (0 or non-finite).")
    if not np.isclose(norm, 1.0, atol=1e-9):
        psi = psi / norm
    return psi

def _state_to_joint(psi: np.ndarray, n: int) -> dict:
    psi = _normalize_state(psi)
    probs = np.abs(psi) ** 2
    if probs.size != (1 << n):
        raise ValueError(f"State size {probs.size} doesn't match n={n}")
    probs = probs / probs.sum()
    return {bitstring(k, n): float(probs[k]) for k in range(probs.size)}

def _sample_from_state(psi: np.ndarray, n: int, shots: int, rng: np.random.Generator) -> np.ndarray:
    psi = _normalize_state(psi)
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    N = 1 << n
    idxs = rng.choice(N, size=shots, p=probs)
    X = np.zeros((shots, n), dtype=int)
    for i, idx in enumerate(idxs):
        bs = bitstring(int(idx), n)
        X[i, :] = [1 if ch == '1' else 0 for ch in bs]
    return X

def chi_square_joint(X_joint, p_joint_true, alpha=0.05, min_count=5, verbose=False):
    m, n = X_joint.shape
    obs_counts = {}
    for row in X_joint:
        s = ''.join(map(str, row.astype(int)))
        obs_counts[s] = obs_counts.get(s, 0) + 1
    total = sum(v for v in p_joint_true.values() if v > 0.0)
    if total <= 0:
        return {'test_name':'Chi-square Multinomial','statistic':0.0,'p_value':1.0,'df':0,'conclusion':'Fail to reject H0','cochran_satisfied':False,'observed_states':int(len(obs_counts))}
    if any(p_joint_true.get(k, 0.0) == 0.0 and obs_counts.get(k, 0) > 0 for k in obs_counts):
        return {'test_name':'Chi-square Multinomial','statistic':float('inf'),'p_value':0.0,'df':1,'conclusion':'Reject H0','cochran_satisfied':False,'observed_states':int(len(obs_counts))}
    keys = [(k, v/total) for k, v in p_joint_true.items() if v > 0.0]
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
        return {'test_name':'Chi-square Multinomial (lumped)','statistic':0.0,'p_value':1.0,'df':0,'conclusion':'Fail to reject H0','cochran_satisfied':bool(exp.size>0 and exp.min()>=min_count),'observed_states':int(len(obs_counts))}
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_stat = float(np.nansum((obs - exp)**2 / exp))
    df = int(exp.size - 1)
    p_value = float(1.0 - chi2.cdf(chi2_stat, df))
    cochran = bool(exp.min() >= min_count)
    reject = p_value < alpha
    return {'test_name':'Chi-square Multinomial (lumped)','statistic':chi2_stat,'p_value':p_value,'df':df,'conclusion':'Reject H0' if reject else 'Fail to reject H0','cochran_satisfied':cochran,'observed_states':int(len(obs_counts))}

def mmdagg_joint_test(X_test: np.ndarray, p_joint_true: dict, alpha: float = 0.05, n_ref: int = 5000, n_permute: int = 500, bandwidths: list = None, weights: np.ndarray = None, seed: int = None, verbose: bool = False) -> dict:
    rng = default_rng(seed)
    X_test = np.asarray(X_test, dtype=float)
    n_test, d = X_test.shape
    states = np.array(list(p_joint_true.keys()))
    probs = np.array(list(p_joint_true.values()), dtype=float)
    probs = probs / probs.sum()
    ref_idx = rng.choice(len(states), size=n_ref, p=probs)
    X_ref = np.array([[int(b) for b in s] for s in states[ref_idx]], dtype=float)
    Z = np.vstack([X_test, X_ref])
    N = Z.shape[0]
    G = Z @ Z.T
    s = np.sum(Z * Z, axis=1)
    D = s[:, None] + s[None, :] - 2.0 * G
    D[D < 0.0] = 0.0
    if bandwidths is None:
        iu = np.triu_indices(N, 1)
        md = np.median(np.sqrt(D[iu]))
        if not np.isfinite(md) or md <= 0:
            md = np.sqrt(d)
        bandwidths = (md * np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])).tolist()
    L = len(bandwidths)
    if weights is None:
        w = np.array([2.0**(-(i+1)) for i in range(L)], dtype=float); w = w / w.sum()
    else:
        w = np.asarray(weights, dtype=float); w = w / w.sum()
    Ks = [np.exp(-D / (2.0 * (bw * bw))) for bw in bandwidths]
    ix = np.arange(n_test); iy = np.arange(n_test, N)
    def mmd_u2_from_gram(K, ix, iy):
        nx = ix.size; ny = iy.size
        Kxx = K[np.ix_(ix, ix)]; Kyy = K[np.ix_(iy, iy)]; Kxy = K[np.ix_(ix, iy)]
        sx = (np.sum(Kxx) - np.sum(np.diag(Kxx))) / (nx * (nx - 1))
        sy = (np.sum(Kyy) - np.sum(np.diag(Kyy))) / (ny * (ny - 1))
        sxy = np.mean(Kxy)
        return sx + sy - 2.0 * sxy
    stats = np.array([mmd_u2_from_gram(K, ix, iy) for K in Ks])
    null_stats = np.empty((n_permute, L), dtype=float)
    for b in range(n_permute):
        p = rng.permutation(N); ixp = p[:n_test]; iyp = p[n_test:]
        null_stats[b] = [mmd_u2_from_gram(K, ixp, iyp) for K in Ks]
    B = n_permute
    pvals = (1.0 + (null_stats >= stats).sum(axis=0)) / (B + 1.0)
    p_agg = float(np.minimum(1.0, np.min(pvals / w)))
    reject = p_agg <= alpha
    if verbose:
        print(f"Aggregated p-value: {p_agg:.6f}", flush=True)
    return {'test_name': 'MMDAgg', 'statistic': float(np.max(stats)), 'p_value': float(p_agg), 'conclusion': ('Reject H0' if reject else 'Fail to reject H0'), 'bandwidths': list(map(float, bandwidths)), 'weights': list(map(float, w)), 'stats_per_kernel': list(map(float, stats)), 'pvals_per_kernel': list(map(float, pvals)), 'n_test': int(n_test), 'n_ref': int(n_ref), 'n_permute': int(n_permute)}

def sequential_mmdagg_joint_test(p_joint_true, sample_generator, max_n=150, batch_size=20, alpha=0.05, n_ref=5000, n_permute=500, seed=None, verbose=False, scheme="obf", min_n_first_test=30):
    if seed is not None:
        np.random.seed(seed)
    states = np.array(list(p_joint_true.keys()))
    probs = np.array(list(p_joint_true.values()), dtype=float)
    probs /= probs.sum()
    n_looks = int(math.ceil(max_n / batch_size))
    alphas = _alpha_schedule(n_looks, alpha, scheme)
    X_accum = []; n_current = 0; history = []; look = 0; last_res = None
    while n_current < max_n:
        X_batch = sample_generator(batch_size)
        X_accum.append(X_batch); n_current += batch_size; look += 1
        if n_current < min_n_first_test: continue
        res = mmdagg_joint_test(np.vstack(X_accum).astype(float), p_joint_true, alpha=float(alphas[look-1]), n_ref=n_ref, n_permute=n_permute, seed=None, verbose=False)
        last_res = res
        history.append({'look':int(look),'n':int(n_current),'alpha_look':float(alphas[look-1]),'p_value':float(res['p_value']),'statistic':float(res['statistic']),'reject':bool(res['p_value'] <= alphas[look-1])})
        if res['p_value'] <= alphas[look-1]:
            return {'test_name':'Sequential MMDAgg','statistic':float(res['statistic']),'p_value':float(res['p_value']),'conclusion':'Reject H0','scheme':scheme,'alpha_total':float(alpha),'alphas':alphas.tolist(),'n_used':int(n_current),'n_looks':int(look),'early_stop':True,'history':history,'bandwidths':list(map(float, res.get('bandwidths', []))),'weights':list(map(float, res.get('weights', []))),'stats_per_kernel':list(map(float, res.get('stats_per_kernel', []))),'pvals_per_kernel':list(map(float, res.get('pvals_per_kernel', []))),'n_test':int(res.get('n_test', n_current)),'n_ref':int(res.get('n_ref', n_ref)),'n_permute':int(res.get('n_permute', n_permute))}
    if last_res:
        return {'test_name':'Sequential MMDAgg','statistic':float(last_res['statistic']),'p_value':float(last_res['p_value']),'conclusion':'Fail to reject H0','scheme':scheme,'alpha_total':float(alpha),'alphas':alphas.tolist(),'n_used':int(n_current),'n_looks':int(look),'early_stop':False,'history':history,'bandwidths':list(map(float, last_res.get('bandwidths', []))),'weights':list(map(float, last_res.get('weights', []))),'stats_per_kernel':list(map(float, last_res.get('stats_per_kernel', []))),'pvals_per_kernel':list(map(float, last_res.get('pvals_per_kernel', []))),'n_test':int(last_res.get('n_test', n_current)),'n_ref':int(last_res.get('n_ref', n_ref)),'n_permute':int(last_res.get('n_permute', n_permute))}
    return {'test_name':'Sequential MMDAgg','statistic':float('nan'),'p_value':float('nan'),'conclusion':'Fail to reject H0','scheme':scheme,'alpha_total':float(alpha),'alphas':alphas.tolist(),'n_used':int(n_current),'n_looks':0,'early_stop':False,'history':[]}

def simple_tolerance_joint_test(X_joint, p_joint_true, tolerance=0.05, alpha=0.05, verbose=False):
    X_joint = np.asarray(X_joint)
    if X_joint.ndim != 2: raise ValueError("X_joint must be 2D")
    m, n = X_joint.shape
    if not np.all((X_joint == 0) | (X_joint == 1)): raise ValueError("X_joint must be binary")
    if not (0 < tolerance < 1): raise ValueError(f"tolerance must be in (0, 1), got {tolerance}")
    obs_counts = {}
    for row in X_joint:
        s = ''.join(map(str, row.astype(int)))
        obs_counts[s] = obs_counts.get(s, 0) + 1
    p_hat = {s: count / m for s, count in obs_counts.items()}
    states = list(p_joint_true.keys())
    abs_errors = {}; rel_errors = {}; within = {}
    for state in states:
        p_t = p_joint_true[state]; p_e = p_hat.get(state, 0.0)
        ae = abs(p_e - p_t)
        re = ae / p_t if p_t > 0 else (float('inf') if p_e > 0 else 0.0)
        abs_errors[state] = float(ae); rel_errors[state] = float(re); within[state] = bool(re <= tolerance)
    worst_state = max(rel_errors.keys(), key=lambda s: rel_errors[s])
    worst_rel_error = rel_errors[worst_state]
    n_violations = sum(1 for v in within.values() if not v)
    reject = n_violations > 0
    max_rel_error = max(rel_errors.values())
    return {'test_name': f'Simple Tolerance Joint Test ({tolerance*100:.1f}% threshold)', 
            'statistic': float(max_rel_error), 
            'p_value': float('nan'), 'conclusion': ('Reject H0' if reject else 'Fail to reject H0'), 
            'tolerance': float(tolerance), 'n_samples': int(m), 'n_states': len(states), 
            'n_observed_states': len(obs_counts), 'n_violations': int(n_violations), 'worst_state': worst_state, 
            'worst_relative_error': float(worst_rel_error), 'worst_true_prob': float(p_joint_true[worst_state]), 
            'worst_empirical_prob': float(p_hat.get(worst_state, 0.0)), 'absolute_errors': abs_errors, 
            'relative_errors': rel_errors, 'within_tolerance': within}

def _get_real_samples_for_run(run_idx: int, n_qubits: int, m_samples: int, rng: np.random.Generator):
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
            X = ps(run_idx, n_qubits, m_samples)
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
            "  - provide_samples(run_idx, n_qubits, m_samples): 返回当轮矩阵；\n"
            "  - samples: 单个 (>=m_samples, n_qubits) 的 0/1 矩阵（每轮复用）。"
        )

def provide_samples(run_idx: int, n_qubits: int, m_samples: int) -> np.ndarray:
    try:
        sb = seed_base
    except NameError:
        sb = 12345
    seed = sb + run_idx

    psi0 = produce_state(n_qubits, seed=seed)

    if APPLY_QFT_TO_EXTERNAL:
        F = qft_matrix(n_qubits)
        psi0 = F @ psi0

    return measurements_from_state(psi0, n_qubits, m_samples, seed=seed)


def run_single_test(X, p_true, method, alpha, tolerance, seed):
    try:
        if method == 'tolerance':
            res = simple_tolerance_joint_test(X, p_true, tolerance=tolerance, alpha=alpha, verbose=False)
        elif method == 'chi_square':
            res = chi_square_joint(X, p_true, alpha=alpha, verbose=False)
        elif method == 'mmdagg':
            res = mmdagg_joint_test(X, p_true, alpha=alpha, n_ref=1000, n_permute=500, seed=seed, verbose=False)
        else:
            return {'reject': None, 'p_value': None, 'error': f'unknown method: {method}'}
        return {'reject': ("Fail to reject" not in res["conclusion"]),
                'p_value': res.get('p_value', None), 'error': None}
    except Exception as e:
        return {'reject': None, 'p_value': None, 'error': str(e)}



def run_single_experiment_optimized(n_qubits, m_samples, alpha, tolerance, seed_base, run_idx):
    run_seed = seed_base + run_idx * 10
    rng = np.random.default_rng(run_seed)

    psi = produce_state(num_qubits=n_qubits, seed=run_seed)
    if APPLY_QFT_TO_EXTERNAL:
        F = qft_matrix(n_qubits)
        psi = F @ psi

    p_joint_true = _state_to_joint(psi, n_qubits)
    X_correct = _sample_from_state(psi, n_qubits, m_samples, rng)

    X_wrong_bitflip = apply_random_errors(
        X_correct, mode='bitflip', p=0.25, seed=seed_base + 10000 + run_idx,
        ensure_change=True, force_flip_cols=(0,)
    )
    X_wrong_asym = apply_random_errors(
        X_correct, mode='asymmetric', p01=0.30, p10=0.15, seed=seed_base + 20000 + run_idx,
        ensure_change=True, force_flip_cols=(0,)
    )
    X_wrong_depol = apply_random_errors(
        X_correct, mode='depolarize', p=0.30, row_mix=0.05, seed=seed_base + 30000 + run_idx,
        ensure_change=True, force_flip_cols=(0,)
    )

    methods = ['tolerance', 'chi_square', 'mmdagg']
    results = {}
    for method in methods:
        method_offset = {"tolerance": 11, "chi_square": 17, "mmdagg": 23}[method]
        test_seed = run_seed + method_offset
        results[f'{method}_wrong_bitflip'] = run_single_test(
            X_wrong_bitflip, p_joint_true, method, alpha, tolerance, test_seed
        )
        results[f'{method}_wrong_asymmetric'] = run_single_test(
            X_wrong_asym, p_joint_true, method, alpha, tolerance, test_seed
        )
        results[f'{method}_wrong_depolarize'] = run_single_test(
            X_wrong_depol, p_joint_true, method, alpha, tolerance, test_seed
        )
    return results





def aggregate_experiment_results(experiment_results, config_key):
    n_qubits, m_samples, alpha, tolerance = config_key
    methods = ['tolerance', 'chi_square', 'mmdagg']
    measurement_reduction_ratio = m_samples / float(2 ** n_qubits)

    results = []
    for method in methods:
        wrong_modes = {'bitflip': [], 'asymmetric': [], 'depolarize': []}

        for result in experiment_results:
            rb = result.get(f'{method}_wrong_bitflip', {}).get('reject')
            ra = result.get(f'{method}_wrong_asymmetric', {}).get('reject')
            rd = result.get(f'{method}_wrong_depolarize', {}).get('reject')
            if rb is not None:
                wrong_modes['bitflip'].append(rb)
            if ra is not None:
                wrong_modes['asymmetric'].append(ra)
            if rd is not None:
                wrong_modes['depolarize'].append(rd)

        power_bitflip    = (sum(wrong_modes['bitflip'])    / len(wrong_modes['bitflip']))    if wrong_modes['bitflip']    else 0.0
        power_asymmetric = (sum(wrong_modes['asymmetric']) / len(wrong_modes['asymmetric'])) if wrong_modes['asymmetric'] else 0.0
        power_depolarize = (sum(wrong_modes['depolarize']) / len(wrong_modes['depolarize'])) if wrong_modes['depolarize'] else 0.0

        results.append({
            'n_qubits': n_qubits,
            'm_samples': m_samples,
            'alpha': alpha,
            'test_method': method,
            'measurement_reduction_ratio': measurement_reduction_ratio,
            'power_bitflip': float(power_bitflip),
            'power_asymmetric': float(power_asymmetric),
            'power_depolarize': float(power_depolarize),
        })
    return results


def run_comprehensive_simulation_parallel(
    n_qubits_range=range(2, 13),
    m_samples_range=None,
    N_runs=5000,
    alpha_values=[0.05],
    tolerance=0.05,
    seed_base=801,
    n_jobs=-1,
    output_dir='joint_experiment_QFT_falsecase_results'
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if m_samples_range is None:
        m_samples_range = np.array([50, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000])

    total_cores = cpu_count()
    if n_jobs == -1:
        n_jobs = total_cores

    all_experiments = []
    config_map = {}
    exp_idx = 0
    for n_qubits in n_qubits_range:
        for m_samples in m_samples_range:
            for alpha in alpha_values:
                config_key = (n_qubits, m_samples, alpha, tolerance)
                config_map[config_key] = []
                for run_idx in range(N_runs):
                    all_experiments.append((n_qubits, m_samples, alpha, tolerance, seed_base, run_idx))
                    config_map[config_key].append(exp_idx)
                    exp_idx += 1

    total_configs = len(config_map)
    total_experiments = len(all_experiments)
    n_qubits_list = list(n_qubits_range)
    n_qubits_label = str(n_qubits_list[0]) if len(n_qubits_list) == 1 else f'{n_qubits_list[0]}-{n_qubits_list[-1]}'

    print("=" * 80)
    print("QFT POWER SIMULATION (FLAT PARALLEL)")
    print("=" * 80)
    print(f"n_qubits range: {list(n_qubits_range)}")
    print(f"m_samples values: {len(m_samples_range)} points from {min(m_samples_range)} to {max(m_samples_range)}")
    print(f"alpha values: {alpha_values}")
    print(f"N_runs per configuration: {N_runs}")
    print(f"Total configurations: {total_configs}")
    print(f"Total experiments: {total_experiments:,}")
    print(f"\nParallelization strategy:")
    print(f"  Total CPU cores: {total_cores}")
    print(f"  Parallel jobs: {n_jobs} processes")
    print("=" * 80)
    print("\nStarting computation...\n")

    all_experiment_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(run_single_experiment_optimized)(n_qubits, m_samples, alpha, tolerance, seed_base, run_idx)
        for n_qubits, m_samples, alpha, tolerance, seed_base, run_idx in all_experiments
    )

    print("\nAggregating results by configuration...")
    flattened_results = []
    for config_key, exp_indices in tqdm(config_map.items(), desc="Aggregating configs"):
        config_exp_results = [all_experiment_results[i] for i in exp_indices]
        config_summary = aggregate_experiment_results(config_exp_results, config_key)
        flattened_results.extend(config_summary)

    results_df = pd.DataFrame(flattened_results)
    csv_path = os.path.join(output_dir, f'joint_simulation_results_{n_qubits_label}_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)

    print(f"\n✓ Results saved to: {csv_path}")
    print(f"✓ Total data points: {len(results_df)}")
    return results_df

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) >= 2:
        try:
            nq = int(sys.argv[1])
            if nq < 1:
                raise ValueError("n_qubits must be positive")
            n_qubits_range = range(nq, nq + 1)
        except ValueError as e:
            print(f"Invalid n_qubits: {e}")
            sys.exit(1)
    else:
        n_qubits_range = range(2, 13)

    m_samples_range = np.array([50, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000])
    N_runs = 1000
    alpha_values = [0.05]
    tolerance = 0.05
    seed_base = 801
    output_dir = 'joint_experiment_QFT_falsecase_results'

    print(f"Running power experiment for n_qubits in {list(n_qubits_range)}")
    print(f"Output directory: {output_dir}/")
    print("=" * 80)

    start_time = time.time()

    results_df = run_comprehensive_simulation_parallel(
        n_qubits_range=n_qubits_range,
        m_samples_range=m_samples_range,
        N_runs=N_runs,
        alpha_values=alpha_values,
        tolerance=tolerance,
        seed_base=seed_base,
        n_jobs=-1,
        output_dir=output_dir
    )

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"\n{'=' * 80}")
    print("COMPUTATION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Total runtime: {hours}h {minutes}m {seconds}s")
    print(f"Configurations processed: {len(results_df) // 3}")
    print(f"Total experiments: {(len(results_df) // 3) * N_runs:,}")

    print(f"\n{'=' * 80}")
    print("Results directory:", output_dir + "/")
    print(f"{'=' * 80}\n")
