#!/usr/bin/env python
import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist, pdist, squareform
from itertools import combinations
from typing import Dict, Tuple
from scipy.stats import chi2, binom, multivariate_normal, combine_pvalues
from numpy.linalg import cholesky, eigh, inv
from typing import Dict, Tuple, List
import math
import pandas as pd
from itertools import product
import os
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')



def bitstring(k: int, n: int) -> str:
    return format(k, f'0{n}b')

def uniform_superposition(n: int) -> np.ndarray:
    N = 1 << n
    return np.ones(N, dtype=complex) / np.sqrt(N)

def oracle_phase_flip(n: int, target_index: int) -> np.ndarray:
    N = 1 << n
    O = np.ones(N, dtype=complex)
    O[target_index] = -1.0
    return np.diag(O)

def diffusion_operator(n: int) -> np.ndarray:
    s = uniform_superposition(n).reshape(-1, 1)
    N = 1 << n
    I = np.eye(N, dtype=complex)
    return 2.0 * (s @ s.conj().T) - I

def optimal_iterations(n: int) -> int:
    N = 1 << n
    r = int(np.floor(np.pi / 4.0 * np.sqrt(N)))
    return max(1, r)

def grover_state(n: int, target_index: int, r: int = None) -> np.ndarray:
    N = 1 << n
    psi0 = np.zeros(N, dtype=complex)
    psi0[0] = 1.0
    psi = uniform_superposition(n)

    O = oracle_phase_flip(n, target_index)
    D = diffusion_operator(n)

    if r is None:
        r = optimal_iterations(n)

    G = D @ O
    for _ in range(r):
        psi = G @ psi
    return psi

def probability_distribution(state: np.ndarray) -> np.ndarray:
    return np.abs(state)**2

def single_qubit_marginals(n: int, probs: np.ndarray) -> List[Tuple[float, float, float]]:
    N = 1 << n
    results = []
    for i in range(n):
        p0 = 0.0
        p1 = 0.0
        for idx in range(N):
            b = bitstring(idx, n)[i]
            if b == '0':
                p0 += probs[idx]
            else:
                p1 += probs[idx]
        results.append((p0, p1, p0 - p1))
    return results

def run_once(n: int, seed: int = None, verbose = False) -> None:
    if seed is not None:
        np.random.seed(seed)

    if n <= 0:
        raise ValueError("n 必须为正整数")

    N = 1 << n
    target_index = np.random.randint(0, N)
    target_bits = bitstring(target_index, n)

    r = optimal_iterations(n)
    final_state = grover_state(n, target_index, r=r)
    probs = probability_distribution(final_state)

    topk = min(16, N)
    order = np.argsort(-probs)
    if verbose:
        print(f"\n=== Grover 设置 ===")
        print(f"n = {n}, N = 2^{n} = {N}")
        print(f"随机目标态（被标记解）: |{target_bits}>  (index={target_index})")
        print(f"Grover 迭代次数 r = {r}")
    
        print(f"\n=== 理论分布（按概率降序，显示前 {topk} 项） ===")
        for rank, idx in enumerate(order[:topk], 1):
            print(f"#{rank:>2}  |{bitstring(idx, n)}>: {probs[idx]:.6f}")
    
        print("\n目标态的成功概率：", probs[target_index])

    marginals = single_qubit_marginals(n, probs)
    if verbose:
        print("\n=== 每个量子比特的理论边际与期望（大端：第0位为最高位） ===")
        print("位次\tP(0)\t\tP(1)\t\t<E(Z)>=P(0)-P(1)")
        for i, (p0, p1, ez) in enumerate(marginals):
            print(f"{i}\t{p0:.6f}\t{p1:.6f}\t{ez:.6f}")
    ordered_probs = {bitstring(idx, n): probs[idx] for idx in order}
    return {
        'marginal': np.array(marginals)[:,1],
        'joint': ordered_probs
    }


def generate_samples(result: Dict, m: int, seed: int = None) -> Dict[str, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    
    marginal_probs = result['marginal']
    joint_dist = result['joint']
    
    n = len(marginal_probs) 
    
    marginal_samples = np.zeros((m, n), dtype=int)
    for i in range(n):
        marginal_samples[:, i] = np.random.binomial(1, marginal_probs[i], size=m)
    
    bitstrings = list(joint_dist.keys())
    probabilities = np.array(list(joint_dist.values()))
    probabilities = probabilities / probabilities.sum()
    
    sampled_indices = np.random.choice(len(bitstrings), size=m, p=probabilities)
    
    joint_samples = np.zeros((m, n), dtype=int)
    for idx, sample_idx in enumerate(sampled_indices):
        bitstr = bitstrings[sample_idx]
        joint_samples[idx] = [int(bit) for bit in bitstr]
    
    return {
        'marginal': marginal_samples,
        'joint': joint_samples
    }



def chi_square_marginal_test(samples, p_true, alpha=0.05, force=False):    
    samples = np.asarray(samples)
    p_true = np.asarray(p_true)
    
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {samples.shape}")
    
    if p_true.ndim != 1:
        raise ValueError(f"p_true must be 1D, got shape {p_true.shape}")
    
    n, m = samples.shape
    
    if len(p_true) != m:
        raise ValueError(
            f"Dimension mismatch: samples has {m} columns, "
            f"p_true has {len(p_true)} elements"
        )
    
    if not np.all((samples == 0) | (samples == 1)):
        raise ValueError("samples must be binary (0 or 1)")
    
    if not np.all((p_true > 0) & (p_true < 1)):
        raise ValueError("p_true must be in (0, 1)")
    
    expected_ones = n * p_true       
    expected_zeros = n * (1 - p_true)
    
    all_expected_counts = np.concatenate([expected_ones, expected_zeros])
    
    total_cells = 2 * m 

    min_expected = np.min(all_expected_counts)
    condition_i = min_expected >= 1.0
    
    counts_gte_5 = np.sum(all_expected_counts >= 5.0)
    percent_gte_5 = counts_gte_5 / total_cells
    condition_ii = percent_gte_5 >= 0.80
    
    is_valid = condition_i and condition_ii
    
    warnings = []
    
    if not condition_i:
        warnings.append(
            f"COCHRAN RULE VIOLATED (i): Minimum expected count is "
            f"{min_expected:.2f} < 1. Chi-square test INVALID."
        )
    
    if not condition_ii:
        warnings.append(
            f"COCHRAN RULE VIOLATED (ii): Only {counts_gte_5}/{total_cells} "
            f"({percent_gte_5:.1%}) of expected counts >= 5. "
            f"Need at least 80%. Chi-square test INVALID."
        )
    
    if not is_valid:
        warnings.append(
            "RECOMMENDATION: Use exact_binomial_test() instead."
        )
    
    if n < 30:
        warnings.append(
            f"NOTE: Sample size n={n} < 30. "
            f"Asymptotic approximation may be less accurate."
        )
    
    if not is_valid and not force:
        error_msg = (
            "Chi-square test INVALID by Cochran's rule.\n\n" +
            "\n".join(warnings) +
            "\n\nSet force=True to run anyway (not recommended)."
        )
        raise ValueError(error_msg)
    
    p_hat = samples.mean(axis=0)
    var_null = p_true * (1.0 - p_true)
    
    if np.any(var_null < 1e-10):
        raise ValueError("p_true too close to 0 or 1")
    
    chi2_stat = n * np.sum((p_hat - p_true)**2 / var_null)
    df = m
    p_value = 1.0 - chi2.cdf(chi2_stat, df=df)
    
    reject = p_value < alpha
    
    result = {
        'test_name': 'Chi-square Test',
        'valid': is_valid,
        'cochran_condition_i': condition_i,
        'cochran_condition_ii': condition_ii,
        'p_value': float(p_value),
        'conclusion': (
            f"Reject H0: samples NOT from p_true (at alpha={alpha})" if reject 
            else f"Fail to reject H0: consistent with p_true (at alpha={alpha})"
        ),
        'warnings': warnings
    }
    
    return result


def exact_binomial_test(samples, p_true, alpha=0.05, method='bonferroni', alternative='two-sided', midp=False):
    
    samples = np.asarray(samples)
    p_true = np.asarray(p_true, dtype=float)
    
    if samples.ndim != 2:
        raise ValueError("samples must be 2D")
    if p_true.ndim != 1:
        raise ValueError("p_true must be 1D")
    n, m = samples.shape
    if len(p_true) != m:
        raise ValueError("Dimension mismatch")
    if not np.all((samples == 0) | (samples == 1)):
        raise ValueError("samples must be binary")
    if not np.all((p_true >= 0) & (p_true <= 1)):
        raise ValueError("p_true must be in [0, 1]")
        
    counts = samples.sum(axis=0)
    pvals = np.zeros(m, dtype=float)
    
    for i in range(m):
        k = int(counts[i])
        p = float(p_true[i])
        if alternative == 'less':
            pv = binom.cdf(k, n, p)
            if midp:
                pv -= 0.5*binom.pmf(k, n, p)
        elif alternative == 'greater':
            pv = 1.0 - binom.cdf(k-1, n, p)
            if midp:
                pv -= 0.5*binom.pmf(k, n, p)
        elif alternative == 'two-sided':
            pmf_obs = binom.pmf(k, n, p)
            ks = np.arange(n+1)
            pmf = binom.pmf(ks, n, p)
            mask = pmf <= pmf_obs + 1e-16
            pv = pmf[mask].sum()
            if midp:
                pv -= 0.5*pmf_obs
            pv = min(1.0, pv)
        else:
            raise ValueError("alternative must be 'less', 'greater', or 'two-sided'")
        pvals[i] = max(0.0, min(1.0, pv))
        
    if method == 'bonferroni':
        p_value_family = float(min(1.0, m * np.min(pvals)))
        reject = bool(p_value_family < alpha)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    worst_idx = int(np.argmin(pvals))
    result = {
        'test_name': 'Exact Binomial Test (Bonferroni)',
        'valid': True,
        'p_value': p_value_family,
        'conclusion': (
            f"Reject H0: samples NOT from p_true (at alpha={alpha})" if reject
            else f"Fail to reject H0: consistent with p_true (at alpha={alpha})"
        ),
        'p_values_individual': pvals.tolist(),
        'alternative': alternative,
        'midp': bool(midp),
        'warnings': []
    }
    return result


def chi_square_marginal_auto_test(samples, p_true, alpha=0.05, verbose=True):
    
    samples = np.asarray(samples)
    p_true = np.asarray(p_true)
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
        print("Cochran's Rule Check:")
        print(f"  (i) Min expected count >= 1: {min_expected:.2f} >= 1 ? {condition_i}")
        print(f"  (ii) >= 80% of counts >= 5: {counts_gte_5}/{total_cells} "
              f"({percent_gte_5:.1%}) >= 80% ? {condition_ii}")
        print()
    
    if use_chi_square:
        if verbose:
            print("Decision: Using CHI-SQUARE TEST (Cochran's rule satisfied)")
            print()
        result = chi_square_marginal_test(samples, p_true, alpha, force=False)
    else:
        if verbose:
            print("Decision: Using EXACT BINOMIAL TEST (Cochran's rule violated)")
            print("Chi-square would be unreliable for this data.")
            print()
        result = exact_binomial_test(samples, p_true, alpha)
    
    return result


def sequential_chi_square_test(samples, p_true, alpha=0.05, min_n=30, check_interval=10, verbose=False):
    
    samples = np.asarray(samples)
    p_true = np.asarray(p_true, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {samples.shape}")
    if p_true.ndim != 1:
        raise ValueError(f"p_true must be 1D, got shape {p_true.shape}")
        
    n_total, m = samples.shape
    
    if len(p_true) != m:
        raise ValueError(f"Dimension mismatch: samples has {m} columns, p_true has {len(p_true)} elements")
    if not np.all((samples == 0) | (samples == 1)):
        raise ValueError("samples must be binary (0 or 1)")
    if not np.all((p_true > 0) & (p_true < 1)):
        raise ValueError("p_true must be in (0, 1)")
    if min_n < 10:
        raise ValueError(f"min_n must be at least 10, got {min_n}")
    if check_interval < 1:
        raise ValueError(f"check_interval must be at least 1, got {check_interval}")

    if n_total < min_n:
        looks = [n_total]
    else:
        looks = list(range(min_n, n_total + 1, check_interval))
        if looks[-1] != n_total:
            looks.append(n_total)

    n_looks = len(looks)
    alpha_per_look = alpha / n_looks

    look_history = []
    decision = None
    final_result = None
    n_used = looks[-1]

    if verbose:
        print("="*70)
        print("SEQUENTIAL CHI-SQUARE TEST")
        print("="*70)
        print(f"Total samples available: {n_total}")
        print(f"Looks (n): {looks}")
        print(f"Overall alpha: {alpha}")
        print(f"Alpha per look (Bonferroni): {alpha_per_look:.6f}")
        print("="*70)
        print()

    for i, n_cur in enumerate(looks, 1):
        if verbose:
            print(f"Look {i}/{n_looks}: Testing with n={n_cur}")
        res = chi_square_marginal_auto_test(samples[:n_cur, :], p_true, alpha=alpha_per_look, verbose=False)
        rej = res['p_value'] <= alpha_per_look
        look_history.append({
            'look': i,
            'n': int(n_cur),
            'test_used': res['test_name'],
            'p_value': float(res['p_value']),
            'threshold': float(alpha_per_look),
            'reject': bool(rej),
            'valid': bool(res.get('valid', True))
        })
        if verbose:
            print(f"  Test: {res['test_name']}")
            print(f"  P-value: {res['p_value']:.6f}")
            print(f"  Threshold: {alpha_per_look:.6f}")
            print(f"  Reject: {rej}")
            print()
        final_result = res
        if rej:
            decision = 'reject'
            n_used = n_cur
            break

    if decision is None:
        decision = 'fail_to_reject'

    if verbose:
        print("="*70)
        print("SEQUENTIAL TEST COMPLETE")
        print("="*70)
        print(f"Decision: {decision}")
        print(f"Samples used: {n_used}/{n_total}")
        print(f"Stopped early: {n_used < n_total}")
        print(f"Looks performed: {len(look_history)}/{n_looks}")
        print(f"Final p-value: {final_result['p_value']:.6f}")
        print(f"Conclusion: {('Reject H0' if decision=='reject' else 'Fail to reject H0')} at alpha={alpha}")
        print("="*70)

    return {
        'test_name': 'Sequential Chi-square Test (Bonferroni over looks)',
        'valid': bool(final_result.get('valid', True)),
        'p_value': float(final_result['p_value']),
        'conclusion': (f"Reject H0: samples NOT from p_true (at alpha={alpha})" if decision == 'reject'
                       else f"Fail to reject H0: consistent with p_true (at alpha={alpha})"),
        'stopped_early': bool(n_used < n_total),
        'n_used': int(n_used),
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

def chi_square_bootstrap_test(samples, p_true, alpha=0.05, R_assumed=None, 
                              n_bootstrap=500, seed=None, verbose=False):
    samples = np.asarray(samples)
    p_true = np.asarray(p_true, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be 2D")
    if p_true.ndim != 1:
        raise ValueError("p_true must be 1D")
    n, m = samples.shape
    if len(p_true) != m:
        raise ValueError("Dimension mismatch")
    if not np.all((samples == 0) | (samples == 1)):
        raise ValueError("samples must be binary")
    if not np.all((p_true > 0) & (p_true < 1)):
        raise ValueError("p_true must be in (0, 1)")
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap must be at least 100, got {n_bootstrap}")
    rng = np.random.default_rng(seed)
    if R_assumed is None:
        R_assumed = np.eye(m)
    R_assumed = np.asarray(R_assumed, dtype=float)
    if R_assumed.shape != (m, m):
        raise ValueError(f"R_assumed must be {m}x{m}, got {R_assumed.shape}")
    R_assumed = _nearest_correlation(R_assumed)
    expected_ones = n * p_true
    expected_zeros = n * (1 - p_true)
    all_expected = np.concatenate([expected_ones, expected_zeros])
    total_cells = 2 * m
    min_expected = np.min(all_expected)
    counts_gte_5 = np.sum(all_expected >= 5.0)
    percent_gte_5 = counts_gte_5 / total_cells
    condition_i = min_expected >= 1.0
    condition_ii = percent_gte_5 >= 0.80
    is_valid = condition_i and condition_ii
    warnings = []
    if not is_valid:
        warnings.append("WARNING: Cochran's rule violated. Chi-square statistic may be unreliable even with bootstrap. Consider using exact_binomial_test instead.")
    p_hat_obs = samples.mean(axis=0)
    var_null = p_true * (1.0 - p_true)
    if np.any(var_null < 1e-10):
        raise ValueError("p_true too close to 0 or 1")
    chi2_obs = n * np.sum((p_hat_obs - p_true)**2 / var_null)
    chi2_bootstrap = np.zeros(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        X_boot = sample_bernoulli_copula(n, p_true, R_assumed, rng)
        p_hat_boot = X_boot.mean(axis=0)
        chi2_bootstrap[b] = n * np.sum((p_hat_boot - p_true)**2 / var_null)
    p_value = (1.0 + np.sum(chi2_bootstrap >= chi2_obs)) / (n_bootstrap + 1.0)
    reject = p_value < alpha
    result = {
        'test_name': 'Chi-square Bootstrap Test',
        'valid': is_valid,
        'p_value': float(p_value),
        'conclusion': (
            f"Reject H0: samples NOT from p_true (at alpha={alpha})" if reject
            else f"Fail to reject H0: consistent with p_true (at alpha={alpha})"
        ),
        'assumed_independence': bool(np.allclose(R_assumed, np.eye(m))),
        'warnings': warnings
    }
    return result

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
    
    samples = np.asarray(samples)
    p_true = np.asarray(p_true, dtype=float)
    
    if samples.ndim != 2:
        raise ValueError("samples must be 2D")
    if p_true.ndim != 1:
        raise ValueError("p_true must be 1D")
    
    n, m = samples.shape
    
    if len(p_true) != m:
        raise ValueError("Dimension mismatch")
    if not np.all((samples == 0) | (samples == 1)):
        raise ValueError("samples must be binary")
    if not np.all((p_true > 0) & (p_true < 1)):
        raise ValueError("p_true must be in (0, 1)")
    
    if not (0 < tolerance < 1):
        raise ValueError(f"tolerance must be in (0, 1), got {tolerance}")
    
    p_hat = samples.mean(axis=0)
    abs_errors = np.abs(p_hat - p_true)
    
    rel_errors = abs_errors / p_true
    
    within_tolerance = rel_errors <= tolerance
    
    reject = bool(not np.all(within_tolerance))

    worst_idx = int(np.argmax(rel_errors))
    
    result = {
        'test_name': f'Simple Tolerance Test ({tolerance*100:.1f}% threshold)',
        'valid': True,
        'conclusion': (
            f"Reject H0: at least one dimension exceeds {tolerance*100:.1f}% tolerance" if reject
            else f"Fail to reject H0: all dimensions within {tolerance*100:.1f}% tolerance"
        ),
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
    
    return result


def stats_prob_generate(n: int, seed: int = None, verbose: bool = False, independence: bool = True) -> dict:
    if seed is not None:
        np.random.seed(seed)
    
    if n <= 0:
        raise ValueError("n 必须为正整数")
    
    N = 1 << n 

    marginals = np.random.uniform(1e-9, 1 - 1e-9, size=n)
    
    if independence:
        ordered_probs = {}
        for combo in product([0, 1], repeat=n):
            binary_str = ''.join(map(str, combo))
            
            prob = 1.0
            for j, bit in enumerate(combo):
                prob *= marginals[j] if bit == 1 else (1 - marginals[j])
            
            ordered_probs[binary_str] = prob
        
        if verbose:
            print(f"Independent case: {n} binary variables")
            print(f"Marginals P(X_i=1): {marginals}")
            print(f"Sum of joint probs: {sum(ordered_probs.values()):.6f}")
        
        return {
            'marginal': marginals,
            'joint': ordered_probs
        }
    
    else:
        joint_probs = np.random.exponential(scale=1.0, size=N)
        max_iterations = 5000
        tolerance = 1e-9
        
        for iteration in range(max_iterations):
            joint_probs /= joint_probs.sum()
            
            current_marginals = np.zeros(n)
            for i in range(N):
                binary_str = format(i, f'0{n}b')
                for j, bit in enumerate(binary_str):
                    if bit == '1':
                        current_marginals[j] += joint_probs[i]
                        
            if np.allclose(current_marginals, marginals, atol=tolerance):
                break

            for i in range(N):
                binary_str = format(i, f'0{n}b')
                adjustment = 1.0
                for j, bit in enumerate(binary_str):
                    if bit == '1':
                        if current_marginals[j] > 0:
                            adjustment *= marginals[j] / current_marginals[j]
                    else:
                        if current_marginals[j] < 1:
                            adjustment *= (1 - marginals[j]) / (1 - current_marginals[j])
                joint_probs[i] *= adjustment
        
        joint_probs /= joint_probs.sum()
        
        ordered_probs = {}
        for i in range(N):
            binary_str = format(i, f'0{n}b')
            ordered_probs[binary_str] = joint_probs[i]
        
        covariance = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                expectation_product = 0.0
                for binary_str, prob in ordered_probs.items():
                    if binary_str[i] == '1' and binary_str[j] == '1':
                        expectation_product += prob
                
                covariance[i, j] = expectation_product - marginals[i] * marginals[j]
        
        computed_marginals = np.zeros(n)
        for binary_str, prob in ordered_probs.items():
            for i, bit in enumerate(binary_str):
                if bit == '1':
                    computed_marginals[i] += prob
        
        if verbose:
            print(f"Dependent case: {n} binary variables")
            print(f"Target marginals: {marginals}")
            print(f"Computed marginals: {computed_marginals}")
            print(f"Sum of joint probs: {sum(ordered_probs.values()):.6f}")
            print(f"\nCovariance matrix:")
            print(covariance)
            print(f"\nConverged in {iteration+1} iterations")
        
        return {
            'marginal': marginals,
            'joint': ordered_probs,
            'covariance': covariance
        }



def run_single_test(X, p_true, method, alpha, tolerance, seed):
    try:
        if method == 'tolerance':
            res = simple_tolerance_test(X, p_true, tolerance=tolerance, alpha=alpha)
        elif method == 'bootstrap':
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
        
        result = {
            'reject': "Fail to reject" not in res["conclusion"],
            'p_value': res.get('p_value', None)
        }
        if method == 'sequential':
            result['n_used'] = res.get('n_used', None)
        return result
    except:
        result = {
            'reject': None,
            'p_value': None
        }
        if method == 'sequential':
            result['n_used'] = None
        return result


def run_single_experiment_optimized(n_qubits, m_samples, alpha, tolerance, seed_base, run_idx):
    seed = seed_base + run_idx
    wrong_seed = seed + 80213
    
    probs = stats_prob_generate(n_qubits, seed=seed, verbose=False, independence=True)
    p_true = np.asarray(probs['marginal'], dtype=float)
    samples = generate_samples(probs, m=m_samples, seed=seed)
    X = np.asarray(samples['marginal'], dtype=int)
    
    probs_wrong = stats_prob_generate(n_qubits, seed=wrong_seed, verbose=False, independence=True)
    wrong_samples = generate_samples(probs_wrong, m=m_samples, seed=wrong_seed)
    X_wrong = np.asarray(wrong_samples['marginal'], dtype=int)
    
    methods = ['tolerance', 'bootstrap', 'sequential', 'auto']
    results = {}
    
    for method in methods:
        results[f'{method}_correct'] = run_single_test(X, p_true, method, alpha, tolerance, seed)
        results[f'{method}_wrong'] = run_single_test(X_wrong, p_true, method, alpha, tolerance, seed)
    
    return results


def process_single_configuration(config, inner_n_jobs):
    n_qubits, m_samples, alpha, tolerance, seed_base, N_runs = config
    
    experiment_results = Parallel(n_jobs=inner_n_jobs, backend='threading', verbose=0)(
        delayed(run_single_experiment_optimized)(
            n_qubits, m_samples, alpha, tolerance, seed_base, run_idx
        ) for run_idx in range(N_runs)
    )
    
    methods = ['tolerance', 'bootstrap', 'sequential', 'auto']
    metrics = {}
    
    for method in methods:
        correct_rejects = []
        wrong_rejects = []
        correct_pvalues = []
        wrong_pvalues = []
        n_used_correct_list = []
        n_used_wrong_list = []

        for result in experiment_results:
            correct_res = result[f'{method}_correct']
            wrong_res = result[f'{method}_wrong']

            if correct_res['reject'] is not None:
                correct_rejects.append(correct_res['reject'])

            if wrong_res['reject'] is not None:
                wrong_rejects.append(wrong_res['reject'])

            if method == 'sequential':
                if correct_res.get('n_used') is not None:
                    n_used_correct_list.append(correct_res['n_used'])
                if wrong_res.get('n_used') is not None:
                    n_used_wrong_list.append(wrong_res['n_used'])

        TN = sum([not r for r in correct_rejects])
        FP = sum(correct_rejects)
        TP = sum(wrong_rejects)
        FN = sum([not r for r in wrong_rejects])

        total = TN + FP + TP + FN

        type_i_error = FP / (FP + TN) if (FP + TN) > 0 else 0
        power = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        f1_score = 2 * (precision * power) / (precision + power) if (precision + power) > 0 else 0

        method_metrics = {
            'type_i_error': type_i_error,
            'type_ii_error': 1 - power,
            'power': power,
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1_score,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
        }

        if method == 'sequential' and n_used_wrong_list:
            method_metrics['n_used_h1_median'] = float(np.median(n_used_wrong_list))
            method_metrics['n_used_h1_mean'] = float(np.mean(n_used_wrong_list))
            method_metrics['n_used_h1_q25'] = float(np.percentile(n_used_wrong_list, 25))
            method_metrics['n_used_h1_q75'] = float(np.percentile(n_used_wrong_list, 75))
        if method == 'sequential' and n_used_correct_list:
            method_metrics['n_used_h0_median'] = float(np.median(n_used_correct_list))
            method_metrics['n_used_h0_mean'] = float(np.mean(n_used_correct_list))

        metrics[method] = method_metrics
    
    max_measurements = 2 ** n_qubits
    measurement_reduction_ratio = m_samples / max_measurements
    
    results = []
    for method, method_metrics in metrics.items():
        result_row = {
            'n_qubits': n_qubits,
            'm_samples': m_samples,
            'alpha': alpha,
            'test_method': method,
            'measurement_reduction_ratio': measurement_reduction_ratio,
            **{k: v for k, v in method_metrics.items()}
        }
        results.append(result_row)
    
    return results


def run_comprehensive_simulation_nested_parallel(
    n_qubits_range=range(2, 13),
    m_samples_range=None,
    N_runs=5000,
    alpha_values=[0.05, 0.01],
    tolerance=0.05,
    seed_base=801,
    n_jobs=-1,
    output_dir='simulation_results'
):
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if m_samples_range is None:
        m_samples_range = np.unique(np.logspace(np.log10(30), np.log10(2000), 25).astype(int))
    
    total_cores = cpu_count()
    if n_jobs == -1:
        n_jobs = total_cores
    
    configurations = []
    for n_qubits in n_qubits_range:
        for m_samples in m_samples_range:
            for alpha in alpha_values:
                configurations.append((n_qubits, m_samples, alpha, tolerance, seed_base, N_runs))
    
    total_configs = len(configurations)
    total_experiments = total_configs * N_runs
    
    outer_jobs = min(int(math.sqrt(total_cores)), total_configs)
    inner_jobs = max(1, total_cores // outer_jobs)
    
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
    m_samples_range = np.unique(np.logspace(np.log10(30), np.log10(2000), 25).astype(int))
    N_runs = 5000
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
        output_dir='nested_parallel_results'
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

    generate_quick_summary(results_df)
    optimal_df = generate_optimal_sample_table(results_df, output_dir='nested_parallel_results')
    viz_dir = create_compact_visualizations(results_df, output_dir='nested_parallel_results')

    print(f"\n{'=' * 80}")
    print("ALL TASKS COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\nResults directory: nested_parallel_results/")
    print(f"Runtime: {hours}h {minutes}m {seconds}s")
    print(f"{'=' * 80}\n")