#!/usr/bin/env python

import numpy as np
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marginal_experiemnt_grovertestfalse import (
    run_single_experiment_optimized,
    run_single_test
)

print("=" * 80)
print("Testing Error Tracking Feature")
print("=" * 80)

print("\n1. Running with check_error=False (default behavior):")
print("-" * 40)
result = run_single_experiment_optimized(
    n_qubits=3,
    m_samples=100,
    alpha=0.05,
    tolerance=0.05,
    seed_base=1000,
    run_idx=0,
    check_error=False
)
print("✓ Test completed without diagnostic output")

print("\n2. Running with check_error=True (diagnostic mode):")
print("-" * 40)
result = run_single_experiment_optimized(
    n_qubits=3,
    m_samples=100,
    alpha=0.05,
    tolerance=0.05,
    seed_base=1000,
    run_idx=0,
    check_error=True
)
print("✓ Test completed with diagnostic output (if any issues detected)")

print("\n3. Testing with artificially created zero-variance data:")
print("-" * 40)

X = np.random.randint(0, 2, size=(100, 4))
X[:, 2] = 0

p_true = np.array([0.5, 0.5, 0.5, 0.5])

print("Created test matrix with zero-variance column...")
print(f"Column standard deviations: {np.std(X, axis=0)}")

print("\nWithout error checking:")
result_no_check = run_single_test(X, p_true, 'bootstrap', 0.05, 0.05, 42, check_error=False)
print(f"Result: {result_no_check['reject']}, p-value: {result_no_check['p_value']}")

print("\nWith error checking:")
result_with_check = run_single_test(X, p_true, 'bootstrap', 0.05, 0.05, 42, check_error=True)
print(f"Result: {result_with_check['reject']}, p-value: {result_with_check['p_value']}")

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)
print("\nUsage in your main script:")
print("  results_df = run_comprehensive_simulation_nested_parallel(")
print("      ...,")
print("      check_error=True  # Enable diagnostic output")
print("  )")
