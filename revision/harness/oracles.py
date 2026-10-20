
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity

from .base_circuits import CircuitSpec

EQUIVALENT_FIDELITY: float = 0.999
UNDETECTABLE_TVD: float = 1e-12


def state_keys(width: int) -> list[str]:
    return ["".join(str((k >> j) & 1) for j in range(width)) for k in range(1 << width)]


def _probabilities(spec: CircuitSpec) -> np.ndarray:
    sv = Statevector.from_instruction(spec.circuit)
    return np.asarray(sv.probabilities(list(spec.measured)), dtype=float)


@lru_cache(maxsize=512)
def _p0_cached(spec: CircuitSpec) -> dict[str, float]:
    probs = _probabilities(spec)
    keys = state_keys(len(spec.measured))
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{spec.name}: measured-register probabilities do not normalise")
    return {k: float(p / total) for k, p in zip(keys, probs)}


def p0_from_spec(spec: CircuitSpec) -> dict[str, float]:
    return dict(_p0_cached(spec))


def _reduced_state(spec: CircuitSpec):
    sv = Statevector.from_instruction(spec.circuit)
    traced = [q for q in range(spec.circuit.num_qubits) if q not in set(spec.measured)]
    if not traced:
        return sv
    return partial_trace(sv, traced)


def fidelity(spec_a: CircuitSpec, spec_b: CircuitSpec) -> float:
    if len(spec_a.measured) != len(spec_b.measured):
        raise ValueError("fidelity requires the same measured-register width")
    return float(state_fidelity(_reduced_state(spec_a), _reduced_state(spec_b),
                                validate=False))


def fidelity_full(spec_a: CircuitSpec, spec_b: CircuitSpec) -> float:
    if spec_a.circuit.num_qubits != spec_b.circuit.num_qubits:
        raise ValueError("fidelity_full requires the same qubit count")
    return float(state_fidelity(Statevector.from_instruction(spec_a.circuit),
                                Statevector.from_instruction(spec_b.circuit),
                                validate=False))


def tvd(spec_a: CircuitSpec, spec_b: CircuitSpec) -> float:
    pa, pb = _p0_cached(spec_a), _p0_cached(spec_b)
    keys = set(pa) | set(pb)
    return 0.5 * float(sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys))


total_variation = tvd


def is_undetectable(spec_a: CircuitSpec, spec_b: CircuitSpec,
                    *, atol: float = UNDETECTABLE_TVD) -> bool:
    return tvd(spec_a, spec_b) <= atol


def is_equivalent(spec_a: CircuitSpec, spec_b: CircuitSpec,
                  *, threshold: float = EQUIVALENT_FIDELITY) -> bool:
    return fidelity(spec_a, spec_b) >= threshold


def p0_marginals(p0: dict[str, float], d: int) -> np.ndarray:
    out = np.zeros(d, dtype=float)
    for key, prob in p0.items():
        if prob:
            for j in range(d):
                if key[j] == "1":
                    out[j] += prob
    return out
