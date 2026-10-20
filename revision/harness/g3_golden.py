#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import binom, binomtest, fisher_exact


HARNESS_DIR = Path(__file__).resolve().parent
REVISION_DIR = HARNESS_DIR.parent
REPO_ROOT = REVISION_DIR.parent
if str(REVISION_DIR) not in sys.path:
    sys.path.insert(0, str(REVISION_DIR))

from harness import sampling, stats_joint

PRODUCTION_DIR = REPO_ROOT / "Joint" / "Experiment"
BENCH_DIR = PRODUCTION_DIR / "benchmark" / "combinational"
COMBINED_CSV = REPO_ROOT / "Joint" / "combined_results.csv"
FUZZSTORM_PY = PRODUCTION_DIR / "fuzzstorm.py"

ALPHA = 0.05
TOLERANCE = 0.05
N_REF = 1000
N_PERMUTE = 500
MIN_COUNT = 5

DEFAULT_METHODS: tuple[str, ...] = ("tolerance", "chi_square", "mmdagg")
P0_SOURCES: tuple[str, ...] = ("campaign", "builder")
ORACLE_ATOL = 1e-9

LOCAL_BUDGET_REPLICATIONS = 5_000
QUICK_CELLS, QUICK_REPS = 3, 200
FULL_CELLS, FULL_REPS = 20, 1000

EXPERIMENT_TO_CIRCUIT = {
    "BV": "BV", "GROVER": "Grover", "QFT": "QFT",
    "QPE": "QPE", "QWALK": "QWALK", "add": "Adder",
}

EXPERIMENT_SCRIPTS = {
    "BV": "joint_experiemnt_BVtestfalse.py",
    "GROVER": "joint_experiemnt_grovertestfalse.py",
    "QFT": "joint_QFTtestfalse.py",
    "QPE": "joint_QPEtestfalse.py",
    "QWALK": "joint_QWALKtestfalse.py",
    "add": "joint_addtestfalse.py",
}



REFERENCE_WIDTHS: dict[str, tuple[int, ...]] = {
    "BV": (4, 6, 9),
    "Grover": (3, 5, 7),
    "QFT": (4, 6, 9),
    "QPE": (4, 6, 9),
    "Adder": (4, 7, 13),
    "QWALK": (),
}

NO_REFERENCE = {
    "QWALK": "benchmark/combinational/ ships adder, bv, grover, pe, qft, rand_cliff "
             "only -- there is no QWALK instance anywhere in the repo",
}

_REFERENCE_FILES = {
    "BV": ("bv", "bv_{d:02d}.qasm"),
    "Grover": ("grover", "grover_{d:02d}.qasm"),
    "QFT": ("qft", "qft_{d:02d}.qasm"),
    "QPE": ("pe", "pe_{d:02d}.qasm"),
    "Adder": ("adder", "adder_{d:02d}.qasm"),
}


def reference_qasm_path(name: str, d: int) -> Path:
    folder, pattern = _REFERENCE_FILES[name]
    return BENCH_DIR / folder / pattern.format(d=d)


def reference_params(name: str, d: int) -> dict[str, Any]:
    from harness import base_circuits

    if name == "BV":
        return {"secret": [1] * (d - 1)}
    if name == "Grover":
        return {"marked": [1] * base_circuits._grover_search_width(d)}
    if name == "QFT":
        return {"input": [0] * d}
    if name == "QPE":
        bits = [0] * (base_circuits.QPE_PHASE_BITS - 1) + [1]
        return {"phase_bits": bits}
    if name == "Adder":
        k = base_circuits._adder_blocks(d)
        return {"a": [0] * k, "b": [0] * k}
    raise KeyError(f"no reference instance parameters for {name!r}")


def load_qasm_p0(path: Path, measured: Sequence[int]) -> tuple[dict[str, float], int]:
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector

    from harness.oracles import state_keys

    circuit = qasm2.load(
        str(path),
        include_path=(str(path.parent),) + qasm2.LEGACY_INCLUDE_PATH,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
        strict=False,
    )
    probs = np.asarray(Statevector.from_instruction(circuit).probabilities(list(measured)))
    keys = state_keys(len(measured))
    return {k: float(p) for k, p in zip(keys, probs)}, circuit.num_qubits


def max_abs_dp(p_a: Mapping[str, float], p_b: Mapping[str, float]) -> float:
    keys = set(p_a) | set(p_b)
    return max((abs(p_a.get(k, 0.0) - p_b.get(k, 0.0)) for k in keys), default=0.0)


def oracle_equality_rows(widths: Mapping[str, Sequence[int]] | None = None,
                         atol: float = ORACLE_ATOL) -> list[dict[str, Any]]:
    from harness import base_circuits, oracles

    widths = REFERENCE_WIDTHS if widths is None else widths
    rows: list[dict[str, Any]] = []
    for name in base_circuits.CIRCUITS:
        dims = tuple(widths.get(name, ()))
        if not dims:
            rows.append({
                "part": "a1_oracle", "circuit": name, "d": None, "qasm": "",
                "n_states": None, "max_abs_dp": float("nan"),
                "status": "NO_REFERENCE" if name in NO_REFERENCE else "NOT_REQUESTED",
                "detail": NO_REFERENCE.get(name, ""),
            })
            continue
        for d in dims:
            path = reference_qasm_path(name, d)
            row: dict[str, Any] = {
                "part": "a1_oracle", "circuit": name, "d": int(d),
                "qasm": str(path.relative_to(REPO_ROOT)) if path.exists() else str(path),
                "n_states": None, "max_abs_dp": float("nan"),
                "status": "", "detail": "",
            }
            if not path.exists():
                row["status"] = "MISSING_FILE"
                rows.append(row)
                continue
            spec = base_circuits.assemble(name, d, reference_params(name, d))
            p_new = oracles.p0_from_spec(spec)
            p_ref, n_qubits_file = load_qasm_p0(path, spec.measured)
            delta = max_abs_dp(p_new, p_ref)
            row["n_states"] = len(p_new)
            row["max_abs_dp"] = float(delta)
            row["status"] = "PASS" if delta <= atol else "FAIL"
            row["detail"] = f"qasm qubits={n_qubits_file}, spec qubits={spec.circuit.num_qubits}"
            rows.append(row)
    return rows




def _qft_inverse(n: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from qiskit.circuit.library import QFT

            return QFT(num_qubits=n, approximation_degree=0, do_swaps=True,
                       inverse=True, insert_barriers=False, name="qft")
        except ImportError:
            from qiskit.synthesis import synth_qft_full

            return synth_qft_full(n, approximation_degree=0, do_swaps=True, inverse=True)


def campaign_state(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import Statevector

    mean = rng.uniform(-15, 15)
    var = rng.uniform(0.1, 30)
    sample = rng.normal(mean, var, n_qubits * 3)

    q = QuantumRegister(n_qubits)
    c = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(q, c)
    for i in range(n_qubits):
        qc.u(sample[3 * i], sample[3 * i + 1], sample[3 * i + 2], i)
    qc.append(_qft_inverse(n_qubits), list(range(n_qubits)))
    return np.asarray(Statevector.from_instruction(qc).data)


def campaign_p0(n_qubits: int, rng: np.random.Generator) -> dict[str, float]:
    from harness.oracles import state_keys

    psi = campaign_state(n_qubits, rng)
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    return {k: float(p) for k, p in zip(state_keys(n_qubits), probs)}


def _produce_state_uses_seed() -> bool | None:
    if not FUZZSTORM_PY.exists():
        return None
    tree = ast.parse(FUZZSTORM_PY.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "produce_state":
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            return "seed" in names
    return None


def _script_state_source(script: Path) -> str:
    if not script.exists():
        return "MISSING"
    tree = ast.parse(script.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "run_single_experiment_optimized":
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id not in {"default_rng", "_state_to_joint",
                                        "_sample_from_state", "run_single_test", "print"}:
                    return call.func.id
    return "UNKNOWN"


def campaign_provenance_rows(d: int = 6, n_draws: int = 3,
                             seed: int = 801) -> list[dict[str, Any]]:
    from harness import base_circuits, oracles

    uses_seed = _produce_state_uses_seed()
    rng = np.random.default_rng(seed)
    ensemble = [campaign_p0(d, rng) for _ in range(n_draws)]

    rows: list[dict[str, Any]] = []
    for experiment, script_name in EXPERIMENT_SCRIPTS.items():
        circuit = EXPERIMENT_TO_CIRCUIT[experiment]
        source = _script_state_source(PRODUCTION_DIR / script_name)
        try:
            spec = base_circuits.build(circuit, d, seed=seed)
            p_builder = oracles.p0_from_spec(spec)
            tvds = [0.5 * sum(abs(p_builder.get(k, 0.0) - p.get(k, 0.0))
                              for k in set(p_builder) | set(p))
                    for p in ensemble]
            tvd_mean = float(np.mean(tvds))
            support = int(sum(1 for v in p_builder.values() if v > 1e-12))
        except Exception as exc:
            tvd_mean = float("nan")
            support = -1
            circuit = f"{circuit} ({type(exc).__name__})"
        rows.append({
            "part": "a2_provenance", "experiment": experiment, "circuit": circuit,
            "d": int(d), "script": script_name, "state_source": source,
            "produce_state_reads_seed": uses_seed,
            "builder_support": support,
            "campaign_support": int(sum(1 for v in ensemble[0].values() if v > 1e-12)),
            "tvd_builder_vs_campaign": tvd_mean,
        })
    return rows




def load_disk_cells(csv_path: Path | None = None) -> list[dict[str, Any]]:
    path = COMBINED_CSV if csv_path is None else csv_path
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("case") != "baseline":
                continue
            reject = int(float(raw["FP"]))
            n_disk = reject + int(float(raw["TN"]))
            rows.append({
                "experiment": raw["experiment"],
                "n_qubits": int(raw["n_qubits"]),
                "m_samples": int(raw["m_samples"]),
                "test_method": raw["test_method"],
                "reject_disk": reject,
                "n_disk": n_disk,
                "p_disk": reject / n_disk if n_disk else float("nan"),
            })
    return rows


def disk_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, Mapping[str, Any]]:
    return {(r["experiment"], r["n_qubits"], r["m_samples"], r["test_method"]): r
            for r in rows}


def _stratum_order_key(seed: int, *parts: Any) -> str:
    payload = "|".join(str(p) for p in (seed, *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_cells(rows: Sequence[Mapping[str, Any]], n_cells: int,
                 seed: int = 801) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    cells: list[dict[str, Any]] = []
    for row in rows:
        key = (row["experiment"], row["n_qubits"], row["m_samples"])
        if key not in seen:
            seen.add(key)
            cells.append({"experiment": key[0], "n_qubits": key[1], "m_samples": key[2]})

    strata: dict[tuple, list[dict[str, Any]]] = {}
    for cell in cells:
        key = (cell["experiment"], bool(cell["n_qubits"] >= 8), bool(cell["m_samples"] >= 1000))
        strata.setdefault(key, []).append(cell)

    for key, members in strata.items():
        members.sort(key=lambda c: _stratum_order_key(
            seed, c["experiment"], c["n_qubits"], c["m_samples"]))
    order = sorted(strata, key=lambda k: _stratum_order_key(seed, *k))

    chosen: list[dict[str, Any]] = []
    depth = 0
    while len(chosen) < n_cells:
        progressed = False
        for key in order:
            members = strata[key]
            if depth >= len(members):
                continue
            chosen.append(members[depth])
            progressed = True
            if len(chosen) == n_cells:
                break
        if not progressed:
            break
        depth += 1
    return [{"experiment": c["experiment"], "d": int(c["n_qubits"]), "m": int(c["m_samples"])}
            for c in chosen]




def acceptance_band(n: int, p: float, confidence: float = 0.95) -> tuple[int, int]:
    lo, hi = binom.interval(confidence, int(n), float(p))
    lo = 0 if not np.isfinite(lo) else int(lo)
    hi = int(n) if not np.isfinite(hi) else int(hi)
    return lo, hi


def band_verdict(reject_new: int, n_new: int, reject_disk: int, n_disk: int,
                 confidence: float = 0.95) -> dict[str, Any]:
    p_disk = reject_disk / n_disk if n_disk else float("nan")
    lo, hi = acceptance_band(n_new, p_disk, confidence)
    binom_p = float(binomtest(int(reject_new), int(n_new), float(p_disk)).pvalue)
    table = [[int(reject_new), int(n_new) - int(reject_new)],
             [int(reject_disk), int(n_disk) - int(reject_disk)]]
    fisher_p = float(fisher_exact(table)[1])
    return {
        "p_disk": p_disk,
        "p_new": reject_new / n_new if n_new else float("nan"),
        "band_lo": lo, "band_hi": hi,
        "in_band": bool(lo <= reject_new <= hi),
        "binom_p": binom_p,
        "fisher_p": fisher_p,
        "in_band_2sample": bool(fisher_p > 1.0 - confidence),
    }




def g3_seed_master(source: str, experiment: str, d: int, m: int, seed: int) -> int:
    key = f"G3|{seed}|{source}|{experiment}|{d}|{m}|baseline"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _run_method(method: str, states: np.ndarray, counts: np.ndarray,
                p0: Mapping[str, float], rng: np.random.Generator) -> dict:
    if method == "tolerance":
        return stats_joint.tolerance_counts(states, counts, p0,
                                            tolerance=TOLERANCE, alpha=ALPHA)
    if method == "chi_square":
        return stats_joint.chi_square_joint_counts(states, counts, p0,
                                                   alpha=ALPHA, min_count=MIN_COUNT)
    if method == "mmdagg":
        return stats_joint.mmdagg_counts(states, counts, p0, alpha=ALPHA,
                                         n_ref=N_REF, n_permute=N_PERMUTE, rng=rng)
    raise ValueError(f"unknown method {method!r}")


def _p0_provider(source: str, experiment: str, d: int,
                 seed: int) -> Callable[[np.random.Generator], Mapping[str, float]]:
    if source == "campaign":
        return lambda rng: campaign_p0(d, rng)
    if source == "builder":
        from harness import base_circuits, oracles

        circuit = EXPERIMENT_TO_CIRCUIT[experiment]
        spec = base_circuits.build(circuit, d, seed=seed)
        p0 = oracles.p0_from_spec(spec)
        return lambda rng, _p0=p0: _p0
    raise ValueError(f"unknown p0 source {source!r}")


def run_cell(experiment: str, d: int, m: int, *, source: str, n_reps: int,
             methods: Sequence[str] = DEFAULT_METHODS, seed: int = 801) -> dict[str, Any]:
    master = g3_seed_master(source, experiment, d, m, seed)
    sequences = np.random.SeedSequence(master).spawn(n_reps)

    rejects = {mth: 0 for mth in methods}
    errors = {mth: 0 for mth in methods}
    p_sum = {mth: 0.0 for mth in methods}
    p_n = {mth: 0 for mth in methods}
    first_error = {mth: "" for mth in methods}

    started = time.time()
    try:
        provider = _p0_provider(source, experiment, d, seed)
    except Exception as exc:
        return {"experiment": experiment, "d": d, "m": m, "source": source,
                "n_reps": 0, "seed_master": master, "seconds": 0.0,
                "rejects": {mth: 0 for mth in methods},
                "errors": {mth: n_reps for mth in methods},
                "mean_p": {mth: float("nan") for mth in methods},
                "first_error": {mth: f"{type(exc).__name__}: {exc}" for mth in methods},
                "support": -1}

    support = -1
    for sequence in sequences:
        rng = np.random.default_rng(sequence)
        p0 = provider(rng)
        if support < 0:
            support = int(sum(1 for v in p0.values() if v > 1e-12))
        states, counts = sampling.draw_counts(p0, m, rng)
        for mth in methods:
            try:
                res = _run_method(mth, states, counts, p0, rng)
            except Exception as exc:
                errors[mth] += 1
                if not first_error[mth]:
                    first_error[mth] = f"{type(exc).__name__}: {exc}"
                continue
            rejects[mth] += int(bool(res["reject"]))
            pv = res.get("p_value", float("nan"))
            if pv is not None and np.isfinite(pv):
                p_sum[mth] += float(pv)
                p_n[mth] += 1

    return {
        "experiment": experiment, "d": d, "m": m, "source": source,
        "n_reps": n_reps, "seed_master": master, "seconds": time.time() - started,
        "rejects": rejects, "errors": errors,
        "mean_p": {mth: (p_sum[mth] / p_n[mth] if p_n[mth] else float("nan"))
                   for mth in methods},
        "first_error": first_error, "support": support,
    }


def _cell_rows(result: dict[str, Any], index: Mapping[tuple, Mapping[str, Any]],
               methods: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mth in methods:
        row = index.get((result["experiment"], result["d"], result["m"], mth))
        if row is None:
            continue
        n_new = result["n_reps"] - result["errors"][mth]
        base = {
            "part": "b_regression", "p0_source": result["source"],
            "experiment": result["experiment"], "d": result["d"], "m": result["m"],
            "method": mth,
            "reject_disk": int(row["reject_disk"]), "n_disk": int(row["n_disk"]),
            "reject_new": int(result["rejects"][mth]), "n_new": int(n_new),
            "n_errors": int(result["errors"][mth]),
            "mean_p_new": result["mean_p"][mth],
            "p0_support": result["support"],
            "seed_master": result["seed_master"], "seconds": result["seconds"],
            "first_error": result["first_error"][mth],
        }
        if n_new > 0:
            base.update(band_verdict(result["rejects"][mth], n_new,
                                     int(row["reject_disk"]), int(row["n_disk"])))
        else:
            base.update({"p_disk": float(row["p_disk"]), "p_new": float("nan"),
                         "band_lo": -1, "band_hi": -1, "in_band": False,
                         "binom_p": float("nan"), "fisher_p": float("nan"),
                         "in_band_2sample": False})
        rows.append(base)
    return rows




def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _print_oracle(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    print("\n" + "=" * 92)
    print("PART (a1) ORACLE EQUALITY -- new builder p0 vs shipped OPENQASM instance")
    print("=" * 92)
    print(f"{'circuit':<8}{'d':>4}  {'states':>7}  {'max|dp|':>12}  {'status':<14}reference")
    n_pass = n_total = 0
    for row in rows:
        d = "" if row["d"] is None else int(row["d"])
        states = "" if row["n_states"] is None else int(row["n_states"])
        delta = row["max_abs_dp"]
        delta_s = "        n/a" if not np.isfinite(delta) else f"{delta:12.3e}"
        ref = row["qasm"] or row["detail"]
        print(f"{row['circuit']:<8}{d:>4}  {states:>7}  {delta_s}  {row['status']:<14}{ref}")
        if row["status"] in {"PASS", "FAIL"}:
            n_total += 1
            n_pass += row["status"] == "PASS"
    print(f"\n(a1) {n_pass}/{n_total} compared instances match at atol {ORACLE_ATOL:g}")
    return n_pass, n_total


def _print_provenance(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print("PART (a2) CAMPAIGN PROVENANCE -- what the published joint runs sampled from")
    print("=" * 92)
    print(f"{'experiment':<11}{'builder':<9}{'d':>3}  {'state source':<16}"
          f"{'supp(bld)':>10}{'supp(camp)':>11}  {'TVD builder-vs-campaign':>24}")
    for row in rows:
        print(f"{row['experiment']:<11}{str(row['circuit'])[:8]:<9}{row['d']:>3}  "
              f"{row['state_source']:<16}{row['builder_support']:>10}"
              f"{row['campaign_support']:>11}  {row['tvd_builder_vs_campaign']:>24.6f}")
    seed_read = rows[0]["produce_state_reads_seed"] if rows else None
    print(f"\nfuzzstorm.produce_state reads its `seed` argument: {seed_read}")
    if seed_read is False:
        print("  -> published campaign runs are NOT reproducible from their recorded seed,")
        print("     and every 'circuit' label indexes the SAME random-state ensemble.")


def _print_regression(rows: Sequence[Mapping[str, Any]], sources: Sequence[str]) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for source in sources:
        subset = [r for r in rows if r["p0_source"] == source]
        if not subset:
            continue
        print("\n" + "=" * 100)
        print(f"PART (b) MONTE CARLO REGRESSION -- p0 source = {source}")
        print("=" * 100)
        print(f"{'exp':<7}{'d':>3}{'m':>6}  {'method':<11}{'disk':>6}{'new':>6}  "
              f"{'band':>12}  {'binom p':>10}{'fisher p':>10}  {'verdict':<8}err")
        for r in sorted(subset, key=lambda x: (x["experiment"], x["d"], x["m"], x["method"])):
            band = f"[{r['band_lo']},{r['band_hi']}]"
            verdict = "PASS" if r["in_band"] else "FAIL"
            print(f"{r['experiment']:<7}{r['d']:>3}{r['m']:>6}  {r['method']:<11}"
                  f"{r['reject_disk']:>6}{r['reject_new']:>6}  {band:>12}  "
                  f"{r['binom_p']:>10.3g}{r['fisher_p']:>10.3g}  {verdict:<8}{r['n_errors']}")
        n_pass = sum(1 for r in subset if r["in_band"])
        n_pass2 = sum(1 for r in subset if r["in_band_2sample"])
        print(f"\n({source}) band PASS {n_pass}/{len(subset)}   "
              f"two-sample (Fisher) PASS {n_pass2}/{len(subset)}")
        summary[source] = {"pass": n_pass, "total": len(subset), "pass_2sample": n_pass2}
    return summary




def budget_guard(n_cells: int, n_sources: int, n_reps: int, *, full: bool,
                 budget: int = LOCAL_BUDGET_REPLICATIONS) -> str | None:
    planned = n_cells * n_sources * n_reps
    if full or planned <= budget:
        return None
    return (
        f"REFUSING part (b): {planned:,} replications "
        f"({n_cells} cells x {n_sources} p0 sources x {n_reps} reps) exceeds the local "
        f"budget of {budget:,} (~{planned * 0.0085:.0f} s CPU at the measured "
        f"8.5 ms/replication).\n"
        "  --quick  3 cells x 200 reps, runs here in seconds\n"
        "  --part a oracle equality only, ~1 s\n"
        "  --full   the full gate configuration -- CLUSTER ONLY, do not run locally"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Golden-cell regression of the harness against the original campaign summaries.')
    parser.add_argument("--cells", type=int, default=FULL_CELLS,
                        help=f"number of baseline cells for part (b) (default {FULL_CELLS})")
    parser.add_argument("--reps", type=int, default=FULL_REPS,
                        help=f"replications per cell (default {FULL_REPS})")
    parser.add_argument("--jobs", type=int, default=-1, help="joblib n_jobs (default -1)")
    parser.add_argument("--seed", type=int, default=801,
                        help="master seed for cell selection and streams (default 801)")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--quick", action="store_true",
                      help=f"{QUICK_CELLS} cells x {QUICK_REPS} reps (local smoke run)")
    size.add_argument("--full", action="store_true",
                      help="CLUSTER ONLY: run the full gate configuration "
                           f"({FULL_CELLS} cells x {FULL_REPS} reps, ~340 s CPU) and "
                           "lift the local budget guard")
    parser.add_argument("--part", choices=("a", "b", "both"), default="both")
    parser.add_argument("--sources", default=",".join(P0_SOURCES),
                        help="comma-separated p0 sources for part (b)")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--out", default=str(HARNESS_DIR / "g3_report.csv"))
    args = parser.parse_args(argv)

    n_cells, n_reps = (QUICK_CELLS, QUICK_REPS) if args.quick else (args.cells, args.reps)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    out_path = Path(args.out)

    if args.part in {"b", "both"}:
        refusal = budget_guard(n_cells, len(sources), n_reps, full=args.full)
        if refusal:
            print(refusal)
            return 2
        if args.full:
            print(f"--full: gate configuration, {n_cells * len(sources) * n_reps:,} "
                  "replications (cluster job)")

    started = time.time()
    oracle_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    a_pass = a_total = 0

    if args.part in {"a", "both"}:
        widths = REFERENCE_WIDTHS
        if args.quick:
            widths = {k: v[:2] for k, v in REFERENCE_WIDTHS.items()}
        oracle_rows = oracle_equality_rows(widths)
        a_pass, a_total = _print_oracle(oracle_rows)
        provenance = campaign_provenance_rows(d=6, n_draws=2 if args.quick else 3,
                                              seed=args.seed)
        _print_provenance(provenance)
        oracle_rows = oracle_rows + provenance

    b_summary: dict[str, dict[str, int]] = {}
    if args.part in {"b", "both"}:
        disk = load_disk_cells()
        bad = [r for r in disk if r["n_disk"] != 1000]
        if bad:
            print(f"\nWARNING: {len(bad)} baseline rows do not sum to 1000 replications")
        index = disk_index(disk)
        cells = select_cells(disk, n_cells, args.seed)
        tasks = [(c, s) for c in cells for s in sources]
        print(f"\nPart (b): {len(cells)} cells x {len(sources)} p0 sources x "
              f"{n_reps} reps x {len(methods)} methods "
              f"= {len(tasks) * n_reps:,} replications")
        for c in cells:
            print(f"  cell {c['experiment']:<7} d={c['d']:<3} m={c['m']}")

        from joblib import Parallel, delayed

        results = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(run_cell)(cell["experiment"], cell["d"], cell["m"],
                              source=source, n_reps=n_reps, methods=methods,
                              seed=args.seed)
            for cell, source in tasks)
        for result in results:
            regression_rows.extend(_cell_rows(result, index, methods))
        b_summary = _print_regression(regression_rows, sources)

    if regression_rows:
        write_csv(out_path, regression_rows)
        print(f"\nwrote {out_path}")
    if oracle_rows:
        oracle_path = out_path.with_name(out_path.stem + "_oracle.csv")
        write_csv(oracle_path, oracle_rows)
        print(f"wrote {oracle_path}")

    elapsed = time.time() - started
    print("\n" + "=" * 92)
    print("G3 SUMMARY")
    print("=" * 92)
    if a_total:
        print(f"  (a1) oracle equality      : {a_pass}/{a_total} PASS at atol {ORACLE_ATOL:g}")
    for source, s in b_summary.items():
        print(f"  (b) regression [{source:<9}]: {s['pass']}/{s['total']} in band, "
              f"{s['pass_2sample']}/{s['total']} by two-sample Fisher")
    print(f"  wall time                 : {elapsed:.1f} s")
    failed = any(r["status"] == "FAIL" for r in oracle_rows if "status" in r)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
