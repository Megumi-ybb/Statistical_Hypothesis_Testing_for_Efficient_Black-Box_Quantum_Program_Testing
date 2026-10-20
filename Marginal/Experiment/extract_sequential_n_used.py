#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import chi2, binom
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

try:
    from fuzzstorm import produce_state
except ImportError:
    print("ERROR: fuzzstorm not installed. Run: pip install fuzzstorm")
    sys.exit(1)


def probs_from_state(psi):
    p = np.abs(psi) ** 2
    s = p.sum()
    return p / s if s > 0 else p


def marginals_from_probs(n, probs):
    N = 1 << n
    out = np.zeros(n, dtype=float)
    for i in range(n):
        for idx in range(N):
            if ((idx >> (n - 1 - i)) & 1) == 1:
                out[i] += probs[idx]
    return out


def measurements_from_state(psi, n, shots, seed=None):
    rng = np.random.default_rng(seed)
    p = probs_from_state(psi)
    p = p / p.sum()
    N = 1 << n
    idxs = rng.choice(N, size=shots, p=p)
    X = np.zeros((shots, n), dtype=int)
    for r, idx in enumerate(idxs):
        bs = format(idx, f'0{n}b')
        X[r, :] = [1 if ch == '1' else 0 for ch in bs]
    return X


def apply_random_errors(X, mode='bitflip', p=0.05, p01=None, p10=None,
                        row_mix=0.0, seed=None, ensure_change=False,
                        force_flip_cols=None):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=int)
    m, n = X.shape
    X_noisy = X.copy()

    if force_flip_cols is not None and hasattr(force_flip_cols, "__len__") and len(force_flip_cols) > 0:
        cols = [c for c in force_flip_cols if 0 <= c < n]
        if cols:
            X_noisy[:, cols] ^= 1

    if mode == 'bitflip':
        if p > 0.0:
            mask = rng.random(size=X_noisy.shape) < float(p)
            X_noisy ^= mask.astype(int)
    elif mode == 'asymmetric':
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



def chi_square_marginal_auto_test(samples, p_true, alpha=0.05, verbose=False):
    samples = np.asarray(samples)
    p_true = np.asarray(p_true, dtype=float)
    n_total, d = samples.shape

    O = samples.sum(axis=0)
    E1 = n_total * p_true
    E0 = n_total * (1 - p_true)

    all_E = np.concatenate([E1, E0])
    cochran_ok = (all_E.min() >= 1) and (np.mean(all_E >= 5) >= 0.8)

    if cochran_ok:
        chi2_stat = np.sum((O - E1) ** 2 / (E1 * (1 - p_true / 1.0)))
        chi2_stat = 0.0
        for j in range(d):
            chi2_stat += (O[j] - n_total * p_true[j]) ** 2 / (n_total * p_true[j] * (1 - p_true[j]))
        p_value = 1 - chi2.cdf(chi2_stat, df=d)
        return {
            'test_name': 'Pearson Chi-square',
            'p_value': float(p_value),
            'valid': True,
            'conclusion': (f"Reject H0: samples NOT from p_true (at alpha={alpha})"
                           if p_value < alpha
                           else f"Fail to reject H0: consistent with p_true (at alpha={alpha})")
        }
    else:
        p_values = []
        for j in range(d):
            p_val = 2 * min(
                binom.cdf(O[j], n_total, p_true[j]),
                1 - binom.cdf(O[j] - 1, n_total, p_true[j])
            )
            p_val = min(p_val, 1.0)
            p_values.append(p_val)
        p_global = min(1.0, d * min(p_values))
        return {
            'test_name': 'Exact Binomial (Bonferroni)',
            'p_value': float(p_global),
            'valid': True,
            'conclusion': (f"Reject H0: samples NOT from p_true (at alpha={alpha})"
                           if p_global < alpha
                           else f"Fail to reject H0: consistent with p_true (at alpha={alpha})")
        }


def sequential_chi_square_test(samples, p_true, alpha=0.05, min_n=30,
                               check_interval=50, verbose=False):
    samples = np.asarray(samples)
    p_true = np.asarray(p_true, dtype=float)
    n_total = samples.shape[0]

    looks = [n_total] if n_total < min_n else list(range(min_n, n_total + 1, check_interval))
    if looks[-1] != n_total:
        looks.append(n_total)

    n_looks = len(looks)
    alpha_per_look = alpha / n_looks
    decision = None
    final_result = None
    n_used = looks[-1]

    for i, n_cur in enumerate(looks, 1):
        res = chi_square_marginal_auto_test(samples[:n_cur, :], p_true,
                                            alpha=alpha_per_look, verbose=False)
        rej = res['p_value'] <= alpha_per_look
        final_result = res
        if rej:
            decision = 'reject'
            n_used = n_cur
            break

    if decision is None:
        decision = 'fail_to_reject'

    return {
        'decision': decision,
        'n_used': int(n_used),
        'stopped_early': bool(n_used < n_total),
        'p_value': float(final_result['p_value']),
    }



def generate_p_true_and_samples(n_qubits, m_samples, seed):
    psi = produce_state(num_qubits=n_qubits, seed=seed)
    probs = probs_from_state(psi)
    p_true = marginals_from_probs(n_qubits, probs)
    X = measurements_from_state(psi, n=n_qubits, shots=m_samples, seed=seed)
    return p_true, X



def extract_one_run(n_qubits, m_samples, seed_base, run_idx, alpha=0.05):
    seed = seed_base + run_idx
    p_true, X_correct = generate_p_true_and_samples(n_qubits, m_samples, seed)

    res_h0 = sequential_chi_square_test(X_correct, p_true, alpha=alpha,
                                        min_n=30, check_interval=50)

    X_bf = apply_random_errors(X_correct, mode='bitflip', p=0.05,
                               seed=seed_base + 10000 + run_idx)
    X_asym = apply_random_errors(X_correct, mode='asymmetric', p01=0.08, p10=0.02,
                                  seed=seed_base + 20000 + run_idx)
    X_depol = apply_random_errors(X_correct, mode='depolarize', p=0.10,
                                   seed=seed_base + 30000 + run_idx)

    res_bf = sequential_chi_square_test(X_bf, p_true, alpha=alpha,
                                        min_n=30, check_interval=50)
    res_asym = sequential_chi_square_test(X_asym, p_true, alpha=alpha,
                                           min_n=30, check_interval=50)
    res_depol = sequential_chi_square_test(X_depol, p_true, alpha=alpha,
                                            min_n=30, check_interval=50)

    return {
        'h0_n_used': res_h0['n_used'],
        'h0_stopped_early': res_h0['stopped_early'],
        'bitflip_n_used': res_bf['n_used'],
        'bitflip_stopped_early': res_bf['stopped_early'],
        'asymmetric_n_used': res_asym['n_used'],
        'asymmetric_stopped_early': res_asym['stopped_early'],
        'depolarize_n_used': res_depol['n_used'],
        'depolarize_stopped_early': res_depol['stopped_early'],
    }



def run_extraction(
    n_qubits_range=range(2, 13),
    m_samples_range=None,
    N_runs=1000,
    alpha=0.05,
    seed_base=801,
    n_jobs=-1,
    output_path='sequential_n_used_results.csv'
):
    if m_samples_range is None:
        m_samples_range = [50, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000]

    configs = [(nq, ms) for nq in n_qubits_range for ms in m_samples_range]
    all_rows = []

    print(f"Extracting sequential n_used across {len(configs)} configurations x {N_runs} runs")
    print(f"Seed base: {seed_base}, alpha: {alpha}")

    for nq, ms in tqdm(configs, desc="Configurations"):
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(extract_one_run)(nq, ms, seed_base, ri, alpha)
            for ri in range(N_runs)
        )

        h0_n_used = [r['h0_n_used'] for r in results]

        for noise_type in ['bitflip', 'asymmetric', 'depolarize']:
            h1_n_used = [r[f'{noise_type}_n_used'] for r in results]
            h1_early = [r[f'{noise_type}_stopped_early'] for r in results]

            row = {
                'n_qubits': nq,
                'm_samples': ms,
                'noise_type': noise_type,
                'n_used_h1_median': float(np.median(h1_n_used)),
                'n_used_h1_mean': float(np.mean(h1_n_used)),
                'n_used_h1_q25': float(np.percentile(h1_n_used, 25)),
                'n_used_h1_q75': float(np.percentile(h1_n_used, 75)),
                'early_stop_rate_h1': float(np.mean(h1_early)),
                'n_used_h0_median': float(np.median(h0_n_used)),
                'n_used_h0_mean': float(np.mean(h0_n_used)),
                'early_stop_rate_h0': float(np.mean([r['h0_stopped_early'] for r in results])),
                'n_runs': N_runs,
                'reduction_vs_m': 1.0 - float(np.median(h1_n_used)) / ms,
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"\nSummary (H1 median n_used by m_samples, bitflip noise):")
    bf = df[df['noise_type'] == 'bitflip']
    pivot = bf.pivot_table(index='n_qubits', columns='m_samples',
                           values='n_used_h1_median', aggfunc='first')
    print(pivot.to_string(float_format='%.0f'))

    return df


if __name__ == '__main__':
    df = run_extraction(
        n_qubits_range=range(2, 13),
        m_samples_range=[50, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000],
        N_runs=1000,
        alpha=0.05,
        seed_base=801,
        n_jobs=-1,
        output_path=os.path.join(os.path.dirname(__file__),
                                 'sequential_n_used_results.csv')
    )
