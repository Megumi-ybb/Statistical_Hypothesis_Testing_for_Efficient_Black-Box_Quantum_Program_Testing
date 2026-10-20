
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit, qasm2

CIRCUITS: tuple[str, ...] = ("BV", "Grover", "QFT", "QPE", "QWALK", "Adder")
VARIANTS: tuple[str, ...] = ("benchmark", "textbook")

QPE_BASE_ANGLE_DENOM: int = 512
QPE_PHASE_BITS: int = 10


class UnsupportedWidth(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class CircuitSpec:

    name: str
    d: int
    circuit: QuantumCircuit
    measured: tuple[int, ...]
    variant: str = "benchmark"
    meta: dict[str, Any] = field(default_factory=dict)
    sha: str = ""

    def __post_init__(self) -> None:
        if not self.sha:
            object.__setattr__(self, "sha", _sha_of(self.circuit, self.measured))
        object.__setattr__(self, "measured", tuple(int(q) for q in self.measured))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CircuitSpec) and self.sha == other.sha

    def __hash__(self) -> int:
        return hash(self.sha)

    @property
    def qasm(self) -> str:
        return _serialize(self.circuit)

    def replace(self, circuit: QuantumCircuit, *, measured: Sequence[int] | None = None,
                meta: Mapping[str, Any] | None = None) -> "CircuitSpec":
        new_meta = dict(self.meta)
        if meta:
            new_meta.update(meta)
        return CircuitSpec(
            name=self.name,
            d=self.d,
            circuit=circuit,
            measured=tuple(self.measured if measured is None else measured),
            variant=self.variant,
            meta=new_meta,
        )


def _serialize(circuit: QuantumCircuit) -> str:
    try:
        return qasm2.dumps(circuit)
    except Exception:
        from qiskit import qasm3

        return qasm3.dumps(circuit)


_ADDRESSED_NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)_(\d{6,})\b")


def _canonical_qasm(qasm: str) -> str:
    seen: dict[str, str] = {}

    def rename(match: re.Match) -> str:
        token = match.group(0)
        if token not in seen:
            seen[token] = f"{match.group(1)}_g{len(seen)}"
        return seen[token]

    return _ADDRESSED_NAME.sub(rename, qasm)


def _sha_of(circuit: QuantumCircuit, measured: Sequence[int]) -> str:
    payload = (_canonical_qasm(_serialize(circuit))
               + "\nmeasured=" + ",".join(str(int(q)) for q in measured))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()




def instance_params(name: str, d: int, rng: np.random.Generator) -> dict[str, Any]:
    if name == "BV":
        return {"secret": rng.integers(0, 2, size=max(d - 1, 0)).astype(int).tolist()}
    if name == "Grover":
        k = _grover_search_width(d)
        return {"marked": rng.integers(0, 2, size=k).astype(int).tolist()}
    if name == "QFT":
        return {"input": rng.integers(0, 2, size=d).astype(int).tolist()}
    if name == "QPE":
        return {"phase_bits": [0] * (QPE_PHASE_BITS - 1) + [1]}
    if name == "QWALK":
        p = max(d - 1, 1)
        return {
            "steps": int(rng.integers(2, max(3, d + 1))),
            "coin_theta": float(rng.uniform(np.pi / 4, 3 * np.pi / 4)),
            "start": rng.integers(0, 2, size=p).astype(int).tolist(),
        }
    if name == "Adder":
        k = _adder_blocks(d)
        return {
            "a": rng.integers(0, 2, size=k).astype(int).tolist(),
            "b": rng.integers(0, 2, size=k).astype(int).tolist(),
        }
    raise ValueError(f"unknown circuit {name!r}; expected one of {CIRCUITS}")


def _grover_search_width(d: int) -> int:
    if d < 2:
        raise UnsupportedWidth(f"Grover needs d >= 2, got {d}")
    return (d + 1) // 2


def _adder_blocks(d: int) -> int:
    if d < 4:
        raise UnsupportedWidth(
            f"Adder needs d >= 4 (benchmark width is 3k+1 for k>=1 input bits), got {d}"
        )
    return (d - 1) // 3




def _build_bv(d: int, params: Mapping[str, Any], variant: str) -> tuple[QuantumCircuit, tuple[int, ...]]:
    if d < 2:
        raise UnsupportedWidth(f"BV needs d >= 2, got {d}")
    secret = list(params["secret"])
    anc = d - 1
    qc = QuantumCircuit(d, name=f"bv_{d}")
    for q in range(anc):
        qc.h(q)
    qc.x(anc)
    qc.h(anc)
    for q, bit in enumerate(secret):
        if bit:
            qc.cx(q, anc)
    for q in range(d):
        qc.h(q)
    return qc, tuple(range(d))


def _mcx_vchain(qc: QuantumCircuit, controls: Sequence[int], target: int,
                ancillas: Sequence[int]) -> None:
    controls = list(controls)
    if len(controls) == 0:
        qc.x(target)
        return
    if len(controls) == 1:
        qc.cx(controls[0], target)
        return
    if len(controls) == 2:
        qc.ccx(controls[0], controls[1], target)
        return
    chain = list(ancillas)[: len(controls) - 2]
    forward: list[tuple[int, int, int]] = [(controls[0], controls[1], chain[0])]
    for i, ctrl in enumerate(controls[2:-1]):
        forward.append((ctrl, chain[i], chain[i + 1]))
    for a, b, t in forward:
        qc.ccx(a, b, t)
    qc.ccx(controls[-1], chain[-1], target)
    for a, b, t in reversed(forward):
        qc.ccx(a, b, t)


def _build_grover(d: int, params: Mapping[str, Any], variant: str) -> tuple[QuantumCircuit, tuple[int, ...]]:
    k = _grover_search_width(d)
    marked = list(params["marked"])
    target = d - 1
    ancillas = list(range(k, target))
    qc = QuantumCircuit(d, name=f"grover_{d}")

    for q in range(k):
        qc.h(q)
    if variant == "textbook":
        qc.x(target)
    qc.h(target)

    flips = [q for q in range(k) if not marked[q]]
    for q in flips:
        qc.x(q)
    _mcx_vchain(qc, list(range(k)), target, ancillas)
    for q in flips:
        qc.x(q)

    for q in range(k):
        qc.h(q)
    for q in range(k):
        qc.x(q)
    qc.h(k - 1)
    _mcx_vchain(qc, list(range(k - 1)), k - 1, ancillas)
    qc.h(k - 1)
    for q in range(k):
        qc.x(q)
    for q in range(k):
        qc.h(q)
    qc.h(target)
    return qc, tuple(range(d))


def _qft_block(qc: QuantumCircuit, qubits: Sequence[int], sign: float = 1.0) -> None:
    qubits = list(qubits)
    for i, qi in enumerate(qubits):
        qc.h(qi)
        for j in range(i + 1, len(qubits)):
            qc.cp(sign * np.pi / (2 ** (j - i)), qi, qubits[j])


def _build_qft(d: int, params: Mapping[str, Any], variant: str) -> tuple[QuantumCircuit, tuple[int, ...]]:
    if d < 1:
        raise UnsupportedWidth(f"QFT needs d >= 1, got {d}")
    qc = QuantumCircuit(d, name=f"qft_{d}")
    for q, bit in enumerate(params["input"]):
        if bit:
            qc.x(q)
    _qft_block(qc, range(d))
    return qc, tuple(range(d))


def _build_qpe(d: int, params: Mapping[str, Any], variant: str) -> tuple[QuantumCircuit, tuple[int, ...]]:
    if d < 1:
        raise UnsupportedWidth(f"QPE needs d >= 1, got {d}")
    eig = d
    bits = list(params["phase_bits"])
    theta = sum(b * 2.0 ** -(i + 1) for i, b in enumerate(bits))
    base = 2 * np.pi * theta

    qc = QuantumCircuit(d + 1, name=f"pe_{d}")
    for q in range(d):
        qc.h(q)
    if variant == "textbook":
        qc.x(eig)
    for i in range(d - 1, -1, -1):
        qc.cp(base * (2 ** (d - 1 - i)), i, eig)
    _qft_block(qc, range(d), sign=-1.0)
    return qc, tuple(range(d))


def _build_qwalk(d: int, params: Mapping[str, Any], variant: str) -> tuple[QuantumCircuit, tuple[int, ...]]:
    if d < 2:
        raise UnsupportedWidth(f"QWALK needs d >= 2, got {d}")
    p = d - 1
    coin = d - 1
    pos = list(range(p))
    qc = QuantumCircuit(d, name=f"qwalk_{d}")
    for q, bit in enumerate(params["start"]):
        if bit:
            qc.x(pos[q])

    theta = float(params["coin_theta"])
    for _ in range(int(params["steps"])):
        qc.ry(theta, coin)
        for i in range(p - 1, 0, -1):
            qc.mcx([coin] + pos[:i], pos[i])
        qc.cx(coin, pos[0])
        qc.x(coin)
        qc.cx(coin, pos[0])
        for i in range(1, p):
            qc.mcx([coin] + pos[:i], pos[i])
        qc.x(coin)
    return qc, tuple(range(d))


def _build_adder(d: int, params: Mapping[str, Any], variant: str) -> tuple[QuantumCircuit, tuple[int, ...]]:
    k = _adder_blocks(d)
    qc = QuantumCircuit(d, name=f"adder_{d}")
    a = [3 * i + 1 for i in range(k)]
    b = [3 * i + 2 for i in range(k)]
    c = [0] + [3 * i + 3 for i in range(k)]

    for i in range(k):
        if variant == "textbook":
            qc.h(a[i])
        elif params["a"][i]:
            qc.x(a[i])
        if params["b"][i]:
            qc.x(b[i])

    for i in range(k):
        qc.ccx(a[i], b[i], c[i + 1])
        qc.cx(a[i], b[i])
    for i in range(k):
        qc.ccx(c[i], b[i], c[i + 1])
    for i in range(k):
        qc.cx(c[i], b[i])
    return qc, tuple(range(d))


_BUILDERS = {
    "BV": _build_bv,
    "Grover": _build_grover,
    "QFT": _build_qft,
    "QPE": _build_qpe,
    "QWALK": _build_qwalk,
    "Adder": _build_adder,
}


def assemble(name: str, d: int, params: Mapping[str, Any], *,
             variant: str = "benchmark") -> CircuitSpec:
    if name not in _BUILDERS:
        raise ValueError(f"unknown circuit {name!r}; expected one of {CIRCUITS}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    circuit, measured = _BUILDERS[name](int(d), params, variant)
    return CircuitSpec(
        name=name,
        d=int(d),
        circuit=circuit,
        measured=measured,
        variant=variant,
        meta={"params": dict(params), "n_qubits": circuit.num_qubits},
    )


def build(name: str, d: int, *, seed: int, variant: str = "benchmark") -> CircuitSpec:
    rng = np.random.default_rng(int(seed))
    params = instance_params(name, int(d), rng)
    spec = assemble(name, d, params, variant=variant)
    spec.meta["seed"] = int(seed)
    return spec
