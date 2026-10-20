#!/usr/bin/env python
import math
import os
from datetime import datetime
from multiprocessing import cpu_count
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import default_rng
from tqdm.auto import tqdm
from scipy.stats import chi2
from fuzzstorm import produce_state

import warnings
warnings.filterwarnings('ignore')


def bitstring(k: int, n: int) -> str:
    return format(k, f'0{n}b')


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

def _alpha_schedule(L, alpha, scheme):
    if scheme in ("bonferroni","pocock"):
        return np.full(L, alpha / L, dtype=float)
    if scheme == "obf":
        v = 1.0 / np.sqrt(np.arange(1, L + 1, dtype=float))
        v = v / v.sum()
        return alpha * v
    raise ValueError("unknown scheme")

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


def run_single_test(X, p_joint_true, method, alpha, tolerance, seed):
    try:
        if method == 'tolerance':
            res = simple_tolerance_joint_test(X, p_joint_true, tolerance=tolerance, alpha=alpha, verbose=False)
        elif method == 'chi_square':
            res = chi_square_joint(X, p_joint_true, alpha=alpha, verbose=False)
        elif method == 'mmdagg':
            res = mmdagg_joint_test(X, p_joint_true, alpha=alpha, n_ref=1000, n_permute=500, seed=seed, verbose=False)
        else:
            return {'reject': None, 'p_value': None, 'error': f'unknown method: {method}'}
        return {'reject': ("Fail to reject" not in res["conclusion"]),
                'p_value': res.get('p_value', None), 'error': None}
    except Exception as e:
        return {'reject': None, 'p_value': None, 'error': str(e)}


def run_single_experiment_optimized(n_qubits, m_samples, alpha, tolerance, seed_base, run_idx):
    run_seed = seed_base + run_idx * 10
    rng = np.random.default_rng(run_seed)

    psi0 = produce_state(num_qubits=n_qubits, seed=run_seed)
    p_joint_true = _state_to_joint(psi0, n_qubits)
    X_correct = _sample_from_state(psi0, n_qubits, m_samples, rng)

    methods = ['tolerance', 'chi_square', 'mmdagg']
    results = {}
    for method in methods:
        method_offset = {"tolerance": 11, "chi_square": 17, "mmdagg": 23}[method]
        test_seed = run_seed + method_offset
        results[f'{method}_correct'] = run_single_test(X_correct, p_joint_true, method, alpha, tolerance, test_seed)
    return results


def aggregate_experiment_results(experiment_results, config_key):
    n_qubits, m_samples, alpha, tolerance = config_key
    methods = ['tolerance', 'chi_square', 'mmdagg']
    measurement_reduction_ratio = m_samples / float(2 ** n_qubits)

    results = []
    for method in methods:
        correct_rejects = []
        for result in experiment_results:
            r = result.get(f'{method}_correct', {}).get('reject')
            if r is not None:
                correct_rejects.append(r)
        TN = sum((r is False) for r in correct_rejects)
        FP = sum((r is True) for r in correct_rejects)
        total = TN + FP
        type_i_error = FP / total if total > 0 else 0.0
        accuracy = TN / total if total > 0 else 0.0

        results.append({
            'n_qubits': n_qubits,
            'm_samples': m_samples,
            'alpha': alpha,
            'test_method': method,
            'measurement_reduction_ratio': measurement_reduction_ratio,
            'type_i_error': type_i_error,
            'type_ii_error': 1.0,
            'power': 0.0,
            'accuracy': accuracy,
            'precision': 0.0,
            'f1_score': 0.0,
            'TP': 0, 'TN': TN, 'FP': FP, 'FN': 0
        })
    return results


def run_comprehensive_simulation_parallel(
    n_qubits_range=range(2, 13),
    m_samples_range=None,
    N_runs=1000,
    alpha_values=[0.05],
    tolerance=0.05,
    seed_base=801,
    n_jobs=-1,
    output_dir='joint_experiment_QWALK_results'
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
    print("QWALK TYPE-I SIMULATION (FLAT PARALLEL)")
    print("=" * 80)
    print(f"n_qubits range: {list(n_qubits_range)}")
    print(f"m_samples values: {len(m_samples_range)} points from {int(np.min(m_samples_range))} to {int(np.max(m_samples_range))}")
    print(f"alpha values: {alpha_values}")
    print(f"N_runs per configuration: {N_runs}")
    print(f"Total configurations: {total_configs}")
    print(f"Total experiments: {total_experiments:,}")
    print(f"Total CPU cores: {total_cores}")
    print(f"Parallel jobs: {n_jobs}")
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
    output_dir = 'joint_experiment_QWALK_results'

    print(f"Running type-I experiment for n_qubits in {list(n_qubits_range)}")
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
    print(f"\nResults directory: {output_dir}/")
    print(f"{'=' * 80}\n")
