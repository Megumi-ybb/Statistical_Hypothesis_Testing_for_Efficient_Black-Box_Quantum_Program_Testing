# Error Tracking Guide: RuntimeWarning in divide

## Problem Description

When running simulations, you may encounter this warning:
```
RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
```

This warning occurs in `numpy.corrcoef()` when computing correlation matrices from data that has **columns with zero variance** (i.e., all values in a column are the same).

## Root Cause

In your code, this happens at [line 687](marginal_experiemnt_grovertestfalse.py#L687):
```python
R_assumed = np.corrcoef(X, rowvar=False)
```

When a quantum state measurement produces samples where all measurements for a particular qubit are identical (all 0s or all 1s), the standard deviation for that column is 0, causing division by zero in the correlation calculation.

## Solution: The `check_error` Parameter

A new `check_error` parameter has been added throughout the code to help diagnose this issue.

### Usage

#### 1. Basic Testing (Single Experiment)

```python
from marginal_experiemnt_grovertestfalse import run_single_experiment_optimized

result = run_single_experiment_optimized(
    n_qubits=3,
    m_samples=100,
    alpha=0.05,
    tolerance=0.05,
    seed_base=1000,
    run_idx=0,
    check_error=True  # Enable diagnostic output
)
```

#### 2. Full Simulation

```python
from marginal_experiemnt_grovertestfalse import run_comprehensive_simulation_nested_parallel

results_df = run_comprehensive_simulation_nested_parallel(
    n_qubits_range=range(2, 8),
    m_samples_range=[30, 50, 100],
    N_runs=100,
    alpha_values=[0.05],
    tolerance=0.05,
    seed_base=802,
    n_jobs=-1,
    output_dir='diagnostic_results',
    check_error=True  # Enable diagnostic output for the entire simulation
)
```

#### 3. Testing with Specific Data

```python
from marginal_experiemnt_grovertestfalse import run_single_test
import numpy as np

# Your measurement data
X = np.array([...])  # shape (m_samples, n_qubits)
p_true = np.array([...])  # shape (n_qubits,)

result = run_single_test(
    X, p_true,
    method='bootstrap',
    alpha=0.05,
    tolerance=0.05,
    seed=42,
    check_error=True  # Enable diagnostic output
)
```

### What `check_error=True` Does

When enabled, it will print detailed diagnostic information:

```
⚠️  Warning: Columns with near-zero variance detected: [2, 5]
   Standard deviations: [0. 0.]
   Column values: [array([0]), array([1])]
   This causes 'invalid value encountered in divide' warning in np.corrcoef
   Called from:
   [Full stack trace showing where the issue occurred]
```

This tells you:
- **Which columns** (qubits) have zero variance
- **What their standard deviations** are
- **What values** appear in those columns (all 0s, all 1s, etc.)
- **The full stack trace** to understand the execution path

### When to Use `check_error=True`

✅ **Use when:**
- You're seeing the RuntimeWarning and want to know which qubits are causing it
- You're debugging unexpected behavior in the bootstrap test
- You're analyzing specific runs that failed
- You're developing or modifying the code

❌ **Don't use when:**
- Running large production simulations (adds overhead and verbose output)
- The warning is expected and doesn't affect your results
- You've already identified the issue and are just collecting results

## Understanding the Issue

### Why does this happen?

In quantum computing experiments, sometimes:
1. A qubit is measured in a near-deterministic state (p ≈ 0 or p ≈ 1)
2. With limited samples, you might get all 0s or all 1s
3. This results in zero variance for that qubit's measurements

### Is this a problem?

**Not necessarily!** The warning occurs because:
- `np.corrcoef` internally divides by standard deviation
- Division by zero creates NaN values
- These NaNs are handled by `np.nan_to_num()` on line 688

The code already handles this gracefully by converting NaN to 0. However, understanding *when* and *why* it happens can be valuable for:
- Validating your quantum states
- Understanding sample size requirements
- Debugging unexpected test failures

## Example Output

Running the test script:

```bash
python test_error_tracking.py
```

You'll see:
```
Testing Error Tracking Feature
================================================================================

1. Running with check_error=False (default behavior):
----------------------------------------
✓ Test completed without diagnostic output

2. Running with check_error=True (diagnostic mode):
----------------------------------------
✓ Test completed with diagnostic output (if any issues detected)

3. Testing with artificially created zero-variance data:
----------------------------------------
Created test matrix with zero-variance column...
Column standard deviations: [0.49539081 0.49789989 0.         0.49939949]

Without error checking:
Result: False, p-value: 0.553

With error checking:
⚠️  Warning: Columns with near-zero variance detected: [2]
   Standard deviations: [0.]
   Column values: [array([0])]
   This causes 'invalid value encountered in divide' warning in np.corrcoef
   Called from:
   [stack trace...]
Result: False, p-value: 0.553
```

## Summary

| Parameter | Default | Effect |
|-----------|---------|--------|
| `check_error=False` | Yes | Normal operation, no diagnostic output |
| `check_error=True` | No | Prints detailed diagnostics when zero-variance detected |

The feature is **non-intrusive**: by default, everything works as before. Enable it only when you need to diagnose issues.
