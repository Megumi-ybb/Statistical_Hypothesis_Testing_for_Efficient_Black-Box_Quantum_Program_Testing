
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ControlledGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from .base_circuits import CircuitSpec, assemble

OPERATORS: tuple[str, ...] = ("O1", "O2", "O3", "O4", "O5", "O6", "O7")

DEFAULT_STRENGTH: float = 0.1

_SKIP = {"barrier", "measure", "delay", "reset"}


class OperatorNotApplicable(RuntimeError):
    pass


def _rebuild(spec: CircuitSpec, edit) -> QuantumCircuit:
    out = QuantumCircuit(*spec.circuit.qregs, *spec.circuit.cregs,
                         name=spec.circuit.name + "_mut")
    for idx, inst in enumerate(spec.circuit.data):
        edit(idx, inst, out)
    return out


def _copy_through(inst, out: QuantumCircuit) -> None:
    out.append(inst.operation, inst.qubits, inst.clbits)


def _numeric_param_sites(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    sites: list[tuple[int, int]] = []
    for i, inst in enumerate(circuit.data):
        if inst.operation.name in _SKIP:
            continue
        for j, p in enumerate(inst.operation.params):
            try:
                float(p)
            except (TypeError, ValueError):
                continue
            sites.append((i, j))
    return sites


def _mutable_sites(circuit: QuantumCircuit) -> list[int]:
    return [i for i, inst in enumerate(circuit.data) if inst.operation.name not in _SKIP]


def _o1_angle(spec: CircuitSpec, strength: float, rng: np.random.Generator) -> CircuitSpec:
    sites = _numeric_param_sites(spec.circuit)
    if not sites:
        raise OperatorNotApplicable(f"{spec.name}: no parameterised gate to perturb")
    gate_idx, par_idx = sites[int(rng.integers(len(sites)))]
    sign = 1.0 if rng.integers(2) else -1.0
    delta = sign * float(strength)

    def edit(idx, inst, out):
        if idx != gate_idx:
            _copy_through(inst, out)
            return
        op = inst.operation.copy()
        params = list(op.params)
        params[par_idx] = float(params[par_idx]) + delta
        op.params = params
        out.append(op, inst.qubits, inst.clbits)

    qc = _rebuild(spec, edit)
    name = spec.circuit.data[gate_idx].operation.name
    return spec.replace(qc, meta={"mutation": {
        "operator": "O1", "gate_index": gate_idx, "gate": name,
        "param_index": par_idx, "delta": delta, "strength": float(strength)}})


def _o2_missing(spec: CircuitSpec, strength: float, rng: np.random.Generator) -> CircuitSpec:
    sites = _mutable_sites(spec.circuit)
    if not sites:
        raise OperatorNotApplicable(f"{spec.name}: empty circuit")
    gate_idx = sites[int(rng.integers(len(sites)))]

    def edit(idx, inst, out):
        if idx != gate_idx:
            _copy_through(inst, out)

    qc = _rebuild(spec, edit)
    return spec.replace(qc, meta={"mutation": {
        "operator": "O2", "gate_index": gate_idx,
        "gate": spec.circuit.data[gate_idx].operation.name}})


def _insert_rotation(spec: CircuitSpec, kinds: Sequence[str], strength: float,
                     rng: np.random.Generator, operator: str) -> CircuitSpec:
    n = spec.circuit.num_qubits
    pos = int(rng.integers(len(spec.circuit.data) + 1))
    qubit = int(rng.integers(n))
    kind = kinds[int(rng.integers(len(kinds)))]
    theta = float(strength)

    def add(out: QuantumCircuit) -> None:
        getattr(out, kind)(theta, out.qubits[qubit])

    def edit(idx, inst, out):
        if idx == pos:
            add(out)
        _copy_through(inst, out)

    qc = _rebuild(spec, edit)
    if pos == len(spec.circuit.data):
        add(qc)
    return spec.replace(qc, meta={"mutation": {
        "operator": operator, "position": pos, "gate": kind,
        "qubit": qubit, "theta": theta, "strength": float(strength)}})


_SWAP_TRIVIAL: dict[tuple, bool] = {}


def _swap_is_identity(op, ctrl_pick: int) -> bool:
    n = op.num_qubits
    if n > 4:
        return False
    key = (op.name, tuple(float(p) for p in op.params) if op.params else (), n, ctrl_pick)
    if key not in _SWAP_TRIVIAL:
        order = list(range(n))
        order[ctrl_pick], order[-1] = order[-1], order[ctrl_pick]
        straight = QuantumCircuit(n)
        straight.append(op, range(n))
        swapped = QuantumCircuit(n)
        swapped.append(op, order)
        _SWAP_TRIVIAL[key] = bool(
            np.allclose(Operator(straight).data, Operator(swapped).data))
    return _SWAP_TRIVIAL[key]


def _o4_control_target(spec: CircuitSpec, strength: float, rng: np.random.Generator) -> CircuitSpec:
    sites: list[tuple[int, int]] = []
    for i, inst in enumerate(spec.circuit.data):
        op = inst.operation
        if not isinstance(op, ControlledGate) or len(inst.qubits) < 2:
            continue
        for c in range(op.num_ctrl_qubits):
            if not _swap_is_identity(op, c):
                sites.append((i, c))
    if not sites:
        raise OperatorNotApplicable(
            f"{spec.name}: no controlled gate whose control/target exchange changes it")
    gate_idx, ctrl_pick = sites[int(rng.integers(len(sites)))]
    inst0 = spec.circuit.data[gate_idx]

    def edit(idx, inst, out):
        if idx != gate_idx:
            _copy_through(inst, out)
            return
        qubits = list(inst.qubits)
        qubits[ctrl_pick], qubits[-1] = qubits[-1], qubits[ctrl_pick]
        out.append(inst.operation, qubits, inst.clbits)

    qc = _rebuild(spec, edit)
    return spec.replace(qc, meta={"mutation": {
        "operator": "O4", "gate_index": gate_idx, "gate": inst0.operation.name,
        "control_index": ctrl_pick}})


_O6_FIELD = {
    "BV": "secret",
    "Grover": "marked",
    "Adder": None,
    "QPE": "phase_bits",
    "QWALK": "start",
    "QFT": "input",
}


def _o6_oracle(spec: CircuitSpec, strength: float, rng: np.random.Generator) -> CircuitSpec:
    params: dict[str, Any] = {k: (list(v) if isinstance(v, list) else v)
                              for k, v in spec.meta["params"].items()}
    if spec.name == "Adder":
        words = ("a", "b") if spec.variant == "benchmark" else ("b",)
        word = words[int(rng.integers(len(words)))]
        bits = params[word]
        if not bits:
            raise OperatorNotApplicable("Adder: no input bits to flip")
        pos = int(rng.integers(len(bits)))
        bits[pos] ^= 1
        site = {"word": word, "bit": pos}
    else:
        field = _O6_FIELD[spec.name]
        bits = params[field]
        if not bits:
            raise OperatorNotApplicable(f"{spec.name}: instance word is empty")
        pos = int(rng.integers(len(bits)))
        bits[pos] ^= 1
        site = {"word": field, "bit": pos}

    mutant = assemble(spec.name, spec.d, params, variant=spec.variant)
    if "seed" in spec.meta:
        mutant.meta["seed"] = spec.meta["seed"]
    mutant.meta["parent_params"] = spec.meta["params"]
    mutant.meta["mutation"] = {"operator": "O6", **site}
    return mutant


def _o7_routing(spec: CircuitSpec, strength: float, rng: np.random.Generator) -> CircuitSpec:
    n = spec.circuit.num_qubits
    if n < 3:
        raise OperatorNotApplicable(f"{spec.name}: line routing needs >= 3 qubits, got {n}")
    seed_t = int(rng.integers(0, 2**31 - 1))
    routed = transpile(spec.circuit, coupling_map=CouplingMap.from_line(n),
                       basis_gates=["u", "cx", "swap"],
                       optimization_level=1, seed_transpiler=seed_t)
    swaps = [i for i, inst in enumerate(routed.data) if inst.operation.name == "swap"]
    if not swaps:
        raise OperatorNotApplicable(f"{spec.name}: router inserted no SWAP at d={spec.d}")
    site = swaps[int(rng.integers(len(swaps)))]

    broken = QuantumCircuit(*routed.qregs, *routed.cregs, name=routed.name + "_o7")
    for idx, inst in enumerate(routed.data):
        if idx == site:
            broken.cx(inst.qubits[0], inst.qubits[1])
        else:
            broken.append(inst.operation, inst.qubits, inst.clbits)

    layout = routed.layout.final_index_layout()
    measured = tuple(int(layout[v]) for v in spec.measured)
    return spec.replace(broken, measured=measured, meta={"mutation": {
        "operator": "O7", "swap_index": site, "n_swaps": len(swaps),
        "seed_transpiler": seed_t, "coupling": f"line({n})"}})


def mutate(spec: CircuitSpec, operator: str, *, strength: float | None = None,
           rng: np.random.Generator) -> CircuitSpec:
    s = DEFAULT_STRENGTH if strength is None else float(strength)
    if operator == "O1":
        return _o1_angle(spec, s, rng)
    if operator == "O2":
        return _o2_missing(spec, s, rng)
    if operator == "O3":
        return _insert_rotation(spec, ("rx", "ry"), s, rng, "O3")
    if operator == "O4":
        return _o4_control_target(spec, s, rng)
    if operator == "O5":
        return _insert_rotation(spec, ("rz",), s, rng, "O5")
    if operator == "O6":
        return _o6_oracle(spec, s, rng)
    if operator == "O7":
        return _o7_routing(spec, s, rng)
    raise ValueError(f"unknown operator {operator!r}; expected one of {OPERATORS}")
