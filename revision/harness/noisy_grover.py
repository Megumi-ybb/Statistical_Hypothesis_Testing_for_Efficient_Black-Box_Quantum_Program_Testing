
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import runner


CAMPAIGN: str = "I5"
CIRCUIT: str = "Grover"
D: int = 5
VARIANT: str = "textbook"

M_POINTS: tuple[int, ...] = (50, 100, 150, 200, 250, 300)
M_POINTS_EXTENDED: tuple[int, ...] = runner.M_POINTS

BASE_DEPOL_1Q: float = 0.001
BASE_DEPOL_2Q: float = 0.01
BASE_READOUT: float = 0.02

NOISE_SCALES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)

FAKE_BACKEND: str = "FakeManilaV2"

N_REF: int = 1000
N_PERMUTE: int = 500
ALPHA: float = 0.05

BUG_OPERATOR: str = "O6"
BUG_STRENGTH: float | None = None

BASIS_GATES: tuple[str, ...] = ("rz", "sx", "x", "cx")
OPT_LEVEL: int = 1
TRANSPILE_SEED: int = 7
NOISY_1Q_GATES: tuple[str, ...] = ("sx", "x", "id")
NOISY_2Q_GATES: tuple[str, ...] = ("cx",)

ARMS: tuple[str, ...] = ("A", "B", "P")
SAMPLERS: tuple[str, ...] = ("exact", "aer")
REF_MODES: tuple[str, ...] = ("per-rep", "per-config")

METHODS: dict[str, tuple[str, ...]] = {
    "A": ("regimeA_mmdagg_1s", "regimeA_chi2_1s"),
    "B": ("regimeB_mmdagg_2s", "regimeB_chi2_2s"),
    "P": ("regimeB_mmdagg_2s", "regimeB_chi2_2s"),
}


class FakeBackendUnavailable(RuntimeError):
    pass




@dataclass(frozen=True)
class NoiseConfig:

    label: str
    kind: str
    scale: float | None = None
    backend: str | None = None

    def describe(self) -> str:
        if self.kind == "fake":
            return f"{self.label} ({self.backend})"
        return (f"{self.label} (x{self.scale:g}: 1q {BASE_DEPOL_1Q * self.scale:.4g}, "
                f"2q {BASE_DEPOL_2Q * self.scale:.4g}, ro {BASE_READOUT * self.scale:.4g})")


def noise_configs(levels: Sequence[str] | None = None) -> list[NoiseConfig]:
    out = [NoiseConfig(label=f"aer{i}", kind="aer", scale=s)
           for i, s in enumerate(NOISE_SCALES)]
    out.append(NoiseConfig(label="fake", kind="fake", backend=FAKE_BACKEND))
    if levels is None:
        return out
    want = {str(x) for x in levels}
    keep = [c for c in out if c.label in want or (c.kind == "aer" and c.label[3:] in want)]
    if not keep:
        raise ValueError(f"no noise configuration matches {sorted(want)}")
    return keep


def parametric_noise_model(scale: float):
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    scale = float(scale)
    if scale < 0:
        raise ValueError(f"noise scale must be >= 0, got {scale}")
    nm = NoiseModel()
    p1 = min(BASE_DEPOL_1Q * scale, 1.0)
    p2 = min(BASE_DEPOL_2Q * scale, 1.0)
    ro = min(BASE_READOUT * scale, 0.5)
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), list(NOISY_1Q_GATES))
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), list(NOISY_2Q_GATES))
    if ro > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
    return nm


def fake_backend():
    try:
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    except Exception as exc:
        raise FakeBackendUnavailable(
            "FakeManilaV2 needs qiskit-ibm-runtime "
            "(harness/.venv/bin/pip install qiskit-ibm-runtime)"
        ) from exc
    return FakeManilaV2()




def readout_confusion(noise_model, phys: Sequence[int]) -> np.ndarray:
    entries: dict[int | None, np.ndarray] = {}
    for err in noise_model.to_dict().get("errors", []):
        if err.get("type") != "roerror":
            continue
        probs = np.asarray(err["probabilities"], dtype=float)
        qubits = err.get("gate_qubits")
        if not qubits:
            entries[None] = probs
        else:
            for group in qubits:
                if len(group) == 1:
                    entries[int(group[0])] = probs
    out = np.array([[1.0]])
    for q in phys:
        table = entries.get(int(q), entries.get(None))
        A = np.eye(2) if table is None else np.asarray(table, dtype=float).T
        out = np.kron(A, out)
    return out


def _physical_measured(circuit, measured: Sequence[int]) -> tuple[int, ...]:
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return tuple(int(q) for q in measured)
    index = layout.final_index_layout()
    return tuple(int(index[int(v)]) for v in measured)


@dataclass(frozen=True, eq=False)
class NoisyCircuit:

    label: str
    probs: np.ndarray
    phys: tuple[int, ...]
    keys: tuple[str, ...]
    circuit: Any = field(repr=False, default=None)
    noise_model: Any = field(repr=False, default=None)
    _measured: Any = field(repr=False, default=None)

    @property
    def d(self) -> int:
        return len(self.phys)


_COMPILED: dict[tuple, NoisyCircuit] = {}


def compile_noisy(spec, cfg: NoiseConfig) -> NoisyCircuit:
    from qiskit import transpile
    from qiskit.quantum_info import DensityMatrix
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    from harness import oracles

    key = (spec.sha, cfg.label, cfg.kind, cfg.scale, cfg.backend)
    hit = _COMPILED.get(key)
    if hit is not None:
        return hit

    if cfg.kind == "fake":
        backend = fake_backend()
        noise_model = NoiseModel.from_backend(backend)
        circuit = transpile(spec.circuit, backend=backend, optimization_level=OPT_LEVEL,
                            seed_transpiler=TRANSPILE_SEED)
    elif cfg.kind == "aer":
        noise_model = parametric_noise_model(cfg.scale)
        circuit = transpile(spec.circuit, basis_gates=list(BASIS_GATES),
                            optimization_level=OPT_LEVEL, seed_transpiler=TRANSPILE_SEED)
    else:
        raise ValueError(f"unknown noise kind {cfg.kind!r}")

    phys = _physical_measured(circuit, spec.measured)
    probe = circuit.copy()
    probe.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    rho = sim.run(probe, shots=1).result().data()["density_matrix"]
    probs = np.asarray(DensityMatrix(rho).probabilities(list(phys)), dtype=float)
    probs = readout_confusion(noise_model, phys) @ probs
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError(f"{spec.name}/{cfg.label}: noisy probabilities do not normalise")
    probs = probs / total

    out = NoisyCircuit(label=cfg.label, probs=probs, phys=phys,
                       keys=tuple(oracles.state_keys(len(phys))),
                       circuit=circuit, noise_model=noise_model)
    _COMPILED[key] = out
    return out


def _aer_measured_circuit(noisy: NoisyCircuit):
    from qiskit import ClassicalRegister

    if noisy._measured is not None:
        return noisy._measured
    qc = noisy.circuit.copy()
    qc.add_register(ClassicalRegister(len(noisy.phys), "c_i5"))
    for j, q in enumerate(noisy.phys):
        qc.measure(q, j)
    object.__setattr__(noisy, "_measured", qc)
    return qc




def sample_exact(noisy: NoisyCircuit, m: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(int(m), noisy.probs).astype(np.int64)


def sample_aer(noisy: NoisyCircuit, m: int, rng: np.random.Generator) -> np.ndarray:
    from qiskit_aer import AerSimulator

    qc = _aer_measured_circuit(noisy)
    seed = int(rng.integers(0, 2**31 - 1))
    result = AerSimulator(noise_model=noisy.noise_model).run(
        qc, shots=int(m), seed_simulator=seed).result()
    index = {k: i for i, k in enumerate(noisy.keys)}
    out = np.zeros(len(noisy.keys), dtype=np.int64)
    for bits, count in result.get_counts().items():
        out[index[bits.replace(" ", "")[::-1]]] += int(count)
    return out


_SAMPLERS = {"exact": sample_exact, "aer": sample_aer}


def _state_matrix(keys: Sequence[str]) -> np.ndarray:
    return np.array([[int(c) for c in k] for k in keys], dtype=np.int64)


def states_counts(full_counts: np.ndarray, keys: Sequence[str],
                  state_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nz = np.flatnonzero(full_counts)
    return state_matrix[nz], np.asarray(full_counts, dtype=np.int64)[nz]




def chi_square_homogeneity(test_counts, ref_counts, *, alpha: float = ALPHA,
                           min_count: float = 5.0) -> dict:
    from scipy.stats import chi2

    x = np.asarray(test_counts, dtype=float).ravel()
    y = np.asarray(ref_counts, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"count vectors disagree: {x.shape} vs {y.shape}")
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("counts must be non-negative")
    n1, n2 = float(x.sum()), float(y.sum())
    base = {"test_name": "Chi-square homogeneity (2-sample)", "n_test": int(n1),
            "n_ref": int(n2), "n_cells": 0, "n_lumped_states": 0,
            "cochran_satisfied": False, "observed_states": int(np.count_nonzero(x))}
    if n1 <= 0 or n2 <= 0:
        return {**base, "statistic": 0.0, "p_value": 1.0, "df": 0, "reject": False,
                "conclusion": "Fail to reject H0"}

    total = n1 + n2
    keep = (x + y) > 0
    x, y = x[keep], y[keep]
    order = np.argsort(x + y, kind="stable")
    x, y = x[order], y[order]

    threshold = min_count * total / min(n1, n2)
    rows_total = x + y
    i = int(np.searchsorted(rows_total, threshold, side="left"))
    if i > 0:
        obs = np.vstack([[x[:i].sum(), y[:i].sum()], np.column_stack([x[i:], y[i:]])])
    else:
        obs = np.column_stack([x, y])
    n_merged = int(keep.sum() - obs.shape[0])
    obs = obs.astype(float)
    rows = obs.sum(axis=1)
    exp = np.outer(rows, np.array([n1, n2])) / total
    cochran = bool(exp.size and exp.min() >= min_count)
    if obs.shape[0] <= 1:
        return {**base, "statistic": 0.0, "p_value": 1.0, "df": 0, "reject": False,
                "n_cells": int(obs.shape[0]), "n_lumped_states": n_merged,
                "cochran_satisfied": cochran, "conclusion": "Fail to reject H0"}

    stat = float(np.sum((obs - exp) ** 2 / exp))
    df = int(obs.shape[0] - 1)
    p_value = float(chi2.sf(stat, df))
    reject = p_value < alpha
    return {**base, "statistic": stat, "p_value": p_value, "df": df,
            "reject": bool(reject), "n_cells": int(obs.shape[0]),
            "n_lumped_states": n_merged, "cochran_satisfied": cochran,
            "conclusion": "Reject H0" if reject else "Fail to reject H0"}




@dataclass(frozen=True)
class I5Cell:

    arm: str
    noise: NoiseConfig
    m: int
    n_reps: int = runner.DEFAULT_N_REPS
    operator: str = runner.BASELINE_OPERATOR
    strength: float | None = None

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")

    @property
    def case(self) -> str:
        return "falsecase" if self.arm == "P" else "baseline"

    @property
    def methods(self) -> tuple[str, ...]:
        return METHODS[self.arm]

    def as_runner_cell(self) -> runner.Cell:
        return runner.Cell(
            campaign=CAMPAIGN, circuit=CIRCUIT, d=D, m=self.m, paradigm="joint",
            operator=self.operator, strength=self.strength, case=self.case,
            methods=self.methods, n_reps=self.n_reps,
        )

    def label(self) -> str:
        return (f"{CAMPAIGN} arm={self.arm} noise={self.noise.label:<5} m={self.m:<5} "
                f"op={self.operator:<8} case={self.case}")


def seed_key(cell: I5Cell) -> str:
    return f"{runner.seed_key(cell.as_runner_cell())}|{cell.arm}|{cell.noise.label}"


def seed_master(cell: I5Cell) -> int:
    return int.from_bytes(hashlib.sha256(seed_key(cell).encode("utf-8")).digest()[:8], "big")


def rep_seed_sequences(cell: I5Cell) -> list[np.random.SeedSequence]:
    return np.random.SeedSequence(seed_master(cell)).spawn(cell.n_reps)


def calibration_seed(cfg: NoiseConfig, n_ref: int) -> int:
    key = f"{CAMPAIGN}|{CIRCUIT}|{D}|{cfg.label}|{n_ref}|calibration"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def enumerate_cells(*, arms: Sequence[str] = ARMS, configs: Sequence[NoiseConfig] | None = None,
                    m_points: Sequence[int] = M_POINTS, n_reps: int = runner.DEFAULT_N_REPS,
                    bug_operator: str = BUG_OPERATOR,
                    bug_strength: float | None = BUG_STRENGTH) -> list[I5Cell]:
    configs = list(noise_configs() if configs is None else configs)
    out: list[I5Cell] = []
    for arm in arms:
        for cfg in configs:
            for m in m_points:
                out.append(I5Cell(
                    arm=arm, noise=cfg, m=int(m), n_reps=int(n_reps),
                    operator=bug_operator if arm == "P" else runner.BASELINE_OPERATOR,
                    strength=bug_strength if arm == "P" else None,
                ))
    return out




def _uniform_placeholder(keys: Sequence[str]) -> dict[str, float]:
    return {k: 1.0 / len(keys) for k in keys}


@dataclass
class I5CellResult:
    cell: I5Cell
    result: runner.CellResult
    extras: dict[str, dict]
    ref_draws: int = 0
    tvd_vs_null: float = float("nan")


def run_i5_cell(cell: I5Cell, *, alpha: float = ALPHA, n_ref: int = N_REF,
                n_permute: int = N_PERMUTE, sampler: str = "exact",
                ref_mode: str = "per-rep", variant: str = VARIANT) -> I5CellResult:
    from harness import base_circuits, mutants, oracles, stats_joint

    if sampler not in _SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
    if ref_mode not in REF_MODES:
        raise ValueError(f"ref_mode must be one of {REF_MODES}, got {ref_mode!r}")
    draw = _SAMPLERS[sampler]
    rcell = cell.as_runner_cell()

    spec = base_circuits.build(CIRCUIT, D, seed=runner.circuit_seed(rcell), variant=variant)
    p0_ideal = oracles.p0_from_spec(spec)
    correct = compile_noisy(spec, cell.noise)
    keys = correct.keys
    smat = _state_matrix(keys)
    placeholder = _uniform_placeholder(keys)
    p_ideal = np.array([p0_ideal[k] for k in keys], dtype=float)

    if cell.arm == "P":
        mutant = mutants.mutate(spec, cell.operator, strength=cell.strength,
                                rng=np.random.default_rng(runner.mutant_seed(rcell)))
        data_source = compile_noisy(mutant, cell.noise)
        tvd_vs_null = 0.5 * float(np.abs(data_source.probs - correct.probs).sum())
    else:
        data_source = correct
        tvd_vs_null = (0.5 * float(np.abs(correct.probs - p_ideal).sum())
                       if cell.arm == "A" else 0.0)

    shared_ref = None
    if cell.arm != "A" and ref_mode == "per-config":
        shared_ref = draw(correct, n_ref,
                          np.random.default_rng(calibration_seed(cell.noise, n_ref)))

    master = seed_master(cell)
    methods = cell.methods
    results = {name: runner.MethodResult(method=name) for name in methods}
    pvals = {name: np.full(cell.n_reps, np.nan) for name in methods}
    impossible = {name: 0 for name in methods}
    cochran = {name: 0 for name in methods}
    ref_draws = 0

    t_cell = time.perf_counter()
    for rep, child in enumerate(rep_seed_sequences(cell)):
        rng = np.random.default_rng(child)
        full = draw(data_source, cell.m, rng)
        states, counts = states_counts(full, keys, smat)

        ref_full = None
        if cell.arm != "A":
            if shared_ref is not None:
                ref_full = shared_ref
            else:
                ref_full = draw(correct, n_ref, rng)
                ref_draws += int(ref_full.sum())

        for name in methods:
            mrng = runner._method_rng(master, rep, name)
            t0 = time.perf_counter()
            if name == "regimeA_mmdagg_1s":
                out = stats_joint.mmdagg_counts(states, counts, p0_ideal, alpha=alpha,
                                                n_ref=n_ref, n_permute=n_permute, rng=mrng)
            elif name == "regimeA_chi2_1s":
                out = stats_joint.chi_square_joint_counts(states, counts, p0_ideal,
                                                          alpha=alpha)
            elif name == "regimeB_mmdagg_2s":
                out = stats_joint.mmdagg_counts(states, counts, placeholder, alpha=alpha,
                                                n_permute=n_permute, rng=mrng,
                                                ref_counts=ref_full)
            elif name == "regimeB_chi2_2s":
                out = chi_square_homogeneity(full, ref_full, alpha=alpha)
            else:
                raise KeyError(f"unknown I5 method {name!r}")
            res = results[name]
            res.wall_s += time.perf_counter() - t0
            res.n_evaluated += 1
            reject = bool(out["reject"])
            if cell.case == "baseline":
                if reject:
                    res.FP += 1
                else:
                    res.TN += 1
            else:
                if reject:
                    res.TP += 1
                else:
                    res.FN += 1
            pvals[name][rep] = float(out.get("p_value", np.nan))
            impossible[name] += int(bool(out.get("impossible_state", False)))
            cochran[name] += int(bool(out.get("cochran_satisfied", False)))

    for name, res in results.items():
        col = pvals[name]
        res.mean_p = float(np.nanmean(col)) if np.any(np.isfinite(col)) else float("nan")

    cell_result = runner.CellResult(cell=rcell, seed_master=master, methods=results,
                                    wall_s=time.perf_counter() - t_cell)
    extras = {}
    for name, res in results.items():
        n = max(res.n_evaluated, 1)
        k = res.TP + res.FP
        lo, hi = wilson(k, res.n_evaluated, alpha=0.05)
        extras[name] = {
            "arm": cell.arm,
            "noise_label": cell.noise.label,
            "noise_kind": cell.noise.kind,
            "noise_scale": float("nan") if cell.noise.scale is None else float(cell.noise.scale),
            "backend": cell.noise.backend or "",
            "variant": variant,
            "sampler": sampler,
            "ref_mode": ref_mode if cell.arm != "A" else "",
            "n_ref": int(n_ref),
            "n_permute": int(n_permute),
            "ref_draws": int(ref_draws),
            "rejection_rate": k / n,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "impossible_frac": impossible[name] / n,
            "cochran_frac": cochran[name] / n,
            "tvd_vs_null": float(tvd_vs_null),
            "n_support_null": int(np.count_nonzero(p_ideal > 0)),
        }
    return I5CellResult(cell=cell, result=cell_result, extras=extras,
                        ref_draws=ref_draws, tvd_vs_null=float(tvd_vs_null))


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import norm

    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(norm.isf(alpha / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))




def write_results(path: str | Path, results: Sequence[I5CellResult]):
    import pandas as pd
    import pyarrow as pa

    from harness import io_store

    backend = pa.default_memory_pool().backend_name
    if backend == "mimalloc":
        raise RuntimeError(
            "pyarrow is using the mimalloc pool, which corrupts parquet output once "
            "qiskit is imported (see harness/noisy_grover.py header).  pyarrow was "
            "imported before ARROW_DEFAULT_MEMORY_POOL could be set; re-run with "
            "ARROW_DEFAULT_MEMORY_POOL=system in the environment."
        )

    rows: list[dict] = []
    extras: list[dict] = []
    for item in results:
        flat = io_store.rows_from_cell_result(item.result)
        rows.extend(flat)
        extras.extend(item.extras[r["method"]] for r in flat)
    df = io_store.to_frame(rows)
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame(extras).reset_index(drop=True)], axis=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    return path


def _summarise(results: Sequence[I5CellResult]) -> list[str]:
    lines = []
    for item in results:
        for name, res in item.result.methods.items():
            e = item.extras[name]
            rate = e["rejection_rate"]
            kind = "power" if item.cell.arm == "P" else "type I"
            lines.append(
                f"  arm {item.cell.arm} {item.cell.noise.label:<5} m={item.cell.m:<5} "
                f"{name:<19} {kind:<6} {rate:.3f} "
                f"[{e['wilson_lo']:.3f}, {e['wilson_hi']:.3f}]  "
                f"impossible={e['impossible_frac']:.3f}  TVD={e['tvd_vs_null']:.4f}"
            )
    return lines


def run_campaign(cells: Sequence[I5Cell], *, alpha: float = ALPHA, n_ref: int = N_REF,
                 n_permute: int = N_PERMUTE, sampler: str = "exact",
                 ref_mode: str = "per-rep", variant: str = VARIANT,
                 verbose: bool = True) -> list[I5CellResult]:
    out: list[I5CellResult] = []
    skipped = 0
    for cell in cells:
        try:
            out.append(run_i5_cell(cell, alpha=alpha, n_ref=n_ref, n_permute=n_permute,
                                   sampler=sampler, ref_mode=ref_mode, variant=variant))
        except FakeBackendUnavailable as exc:
            skipped += 1
            if skipped == 1:
                print(f"SKIP device arm: {exc}", file=sys.stderr)
            continue
        if verbose:
            for line in _summarise(out[-1:]):
                print(line, flush=True)
    if skipped:
        print(f"SKIPPED {skipped} device-arm cell(s)", file=sys.stderr)
    return out




def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python harness/noisy_grover.py",
        description="I5 -- noisy-simulator validation on d=5 Grover (two regimes).")
    parser.add_argument("--arm", default="all", choices=("A", "B", "P", "all"))
    parser.add_argument("--strength-level", default="all",
                        help="0..4 (aer scale index), 'fake', or 'all'")
    parser.add_argument("--m", type=int, nargs="*", help="override the m-grid")
    parser.add_argument("--extended-m", action="store_true",
                        help=f"use the canonical 10-point grid {M_POINTS_EXTENDED}")
    parser.add_argument("--n-reps", type=int, default=runner.DEFAULT_N_REPS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--n-ref", type=int, default=N_REF)
    parser.add_argument("--n-permute", type=int, default=N_PERMUTE)
    parser.add_argument("--sampler", default="exact", choices=SAMPLERS)
    parser.add_argument("--ref-mode", default="per-rep", choices=REF_MODES,
                        help="per-rep: fresh calibration sample each replication "
                             "(unconditional Type I); per-config: one shared sample "
                             "(conditional -- replications are then dependent and the "
                             "Wilson CI understates the Monte-Carlo error)")
    parser.add_argument("--bug-operator", default=BUG_OPERATOR)
    parser.add_argument("--bug-strength", type=float, default=BUG_STRENGTH)
    parser.add_argument("--variant", default=VARIANT, help="Grover variant")
    parser.add_argument("--out", type=Path, help="parquet output path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run: 100 reps, m=200, one noise level, all arms")
    args = parser.parse_args(argv)

    arms = ARMS if args.arm == "all" else (args.arm,)
    levels = None if args.strength_level == "all" else [args.strength_level]
    m_points = M_POINTS_EXTENDED if args.extended_m else M_POINTS
    if args.m:
        m_points = tuple(int(x) for x in args.m)
    n_reps = args.n_reps
    if args.smoke:
        arms, levels, m_points, n_reps = ARMS, ["2"], (200,), 100

    cells = enumerate_cells(arms=arms, configs=noise_configs(levels), m_points=m_points,
                            n_reps=n_reps, bug_operator=args.bug_operator,
                            bug_strength=args.bug_strength)

    n_ref_cells = sum(1 for c in cells if c.arm != "A")
    print(f"campaign {CAMPAIGN}: {len(cells)} cells, {sum(c.n_reps for c in cells):,d} runs "
          f"(n_reps={n_reps}, sampler={args.sampler}, ref-mode={args.ref_mode})")
    if args.ref_mode == "per-rep":
        print(f"  reference (calibration) shots, disclosed separately: "
              f"{n_ref_cells * n_reps * args.n_ref:,d}")
    else:
        print(f"  reference (calibration) shots, disclosed separately: "
              f"{len({c.noise.label for c in cells if c.arm != 'A'}) * args.n_ref:,d}")
    for cfg in noise_configs(levels):
        print(f"  {cfg.describe()}")

    if args.dry_run:
        for cell in cells:
            print(f"{cell.label()}  master={seed_master(cell)}")
        return 0

    t0 = time.perf_counter()
    results = run_campaign(cells, alpha=args.alpha, n_ref=args.n_ref,
                           n_permute=args.n_permute, sampler=args.sampler,
                           ref_mode=args.ref_mode, variant=args.variant)
    print(f"{len(results)} cell(s) in {time.perf_counter() - t0:.1f} s")
    if not results:
        return 1
    if args.out is not None:
        print(f"wrote {write_results(args.out, results)}")
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
