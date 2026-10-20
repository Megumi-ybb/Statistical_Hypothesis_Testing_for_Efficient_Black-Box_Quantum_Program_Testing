
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qiskit")

from harness import base_circuits as bc
from harness import mutants, oracles, sampling
from harness.base_circuits import UnsupportedWidth
from harness.mutants import OperatorNotApplicable

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "Joint/Experiment/benchmark/combinational"

SUPPORTED = {
    "BV": (2, 3, 5, 6),
    "Grover": (2, 3, 5, 6, 7),
    "QFT": (2, 4, 6),
    "QPE": (2, 3, 5),
    "QWALK": (2, 3, 5),
    "Adder": (4, 7, 10),
}


def _qasm_body(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg", "//")):
            continue
        line = line.replace("cu1", "cp").replace(" ", "")
        m = re.match(r"(\w+)\(([^)]*)\)(.*)", line)
        if m:
            angle = eval(m.group(2), {"pi": np.pi})
            line = f"{m.group(1)}({round(float(angle), 12)}){m.group(3)}"
        out.append(line)
    return out




@pytest.mark.parametrize(
    "name,d,ref,params",
    [
        ("BV", 5, "bv/bv_05.qasm", {"secret": [1, 1, 1, 1]}),
        ("BV", 6, "bv/bv_06.qasm", {"secret": [1, 1, 1, 1, 1]}),
        ("Grover", 3, "grover/grover_03.qasm", {"marked": [1, 1]}),
        ("Grover", 5, "grover/grover_05.qasm", {"marked": [1, 1, 1]}),
        ("Grover", 7, "grover/grover_07.qasm", {"marked": [1, 1, 1, 1]}),
        ("Grover", 9, "grover/grover_09.qasm", {"marked": [1, 1, 1, 1, 1]}),
        ("QFT", 4, "qft/qft_04.qasm", {"input": [0, 0, 0, 0]}),
        ("QPE", 3, "pe/pe_03.qasm", {"phase_bits": [0] * 9 + [1]}),
        ("QPE", 5, "pe/pe_05.qasm", {"phase_bits": [0] * 9 + [1]}),
        ("Adder", 7, "adder/adder_07.qasm", {"a": [0, 0], "b": [0, 0]}),
        ("Adder", 10, "adder/adder_01.qasm", {"a": [0, 0, 0], "b": [0, 0, 0]}),
    ],
)
def test_builders_reproduce_reference_qasm(name, d, ref, params):
    path = BENCH / ref
    if not path.exists():
        pytest.skip(f"reference {path} not present")
    spec = bc.assemble(name, d, params)
    assert _qasm_body(spec.qasm) == _qasm_body(path.read_text())


@pytest.mark.parametrize("name", bc.CIRCUITS)
def test_build_deterministic_and_d_parametric(name):
    shas = {}
    for d in SUPPORTED[name]:
        a = bc.build(name, d, seed=1234)
        b = bc.build(name, d, seed=1234)
        assert a.sha == b.sha, f"{name} d={d} not deterministic under a fixed seed"
        assert a.d == d and len(a.measured) == d
        shas[d] = a.sha
    assert len(set(shas.values())) == len(shas), f"{name} sha does not vary with d"


@pytest.mark.parametrize("name", bc.CIRCUITS)
def test_seed_changes_instance(name):
    d = SUPPORTED[name][-1]
    shas = {bc.build(name, d, seed=s).sha for s in range(12)}
    if name in ("QFT", "QPE"):
        pytest.skip(f"{name} has no seed-dependent instance parameter that changes p0")
    assert len(shas) > 1, f"{name} ignores its seed (the produce_state bug)"


def test_qpe_excludes_ancilla_and_p0_normalises():
    for d in (2, 3, 5):
        spec = bc.build("QPE", d, seed=7)
        assert spec.circuit.num_qubits == d + 1, "QPE carries one eigenvector qubit"
        assert spec.measured == tuple(range(d)), "QPE measures only the counting register"
        p0 = oracles.p0_from_spec(spec)
        assert len(p0) == 2**d
        assert abs(sum(p0.values()) - 1.0) < 1e-12


@pytest.mark.parametrize("name", bc.CIRCUITS)
def test_p0_normalises_everywhere(name):
    for d in SUPPORTED[name]:
        p0 = oracles.p0_from_spec(bc.build(name, d, seed=3))
        assert abs(sum(p0.values()) - 1.0) < 1e-12
        assert min(p0.values()) >= -1e-15


def test_adder_rejects_widths_below_the_benchmark_layout():
    for d in (2, 3):
        with pytest.raises(UnsupportedWidth):
            bc.build("Adder", d, seed=0)


def test_bitorder_matches_measured_register():
    secret = [1, 0, 1, 1]
    spec = bc.assemble("BV", 5, {"secret": secret})
    p0 = oracles.p0_from_spec(spec)
    top = max(p0, key=p0.get)
    assert p0[top] > 0.999
    assert top == "".join(map(str, secret)) + "1", f"unexpected BV outcome {top}"




UNDETECTABLE = {
    ("QFT", "O1"): ("phase-only",
                    "after its H, each QFT qubit is only ever a control of a diagonal CP, "
                    "so every marginal is pinned at 1/2 whatever the angles are"),
    ("QFT", "O5"): ("phase-only", "same: an inserted RZ is diagonal and cannot move p0"),
    ("QFT", "O6"): ("phase-only",
                    "|<y|QFT|x>|^2 is flat for every basis input x, so an input-word "
                    "flip sends the state somewhere orthogonal without touching p0"),
    ("QWALK", "O5"): ("phase-only",
                      "the coin/shift walk leaves the RZ phase in a relative phase the "
                      "computational-basis measurement never resolves"),
    ("Grover", "O6"): ("inert",
                       "grover_*.qasm puts the oracle target in |+>, so the oracle is a no-op"),
    ("QPE", "O6"): ("inert",
                    "pe_*.qasm leaves the eigenvector register in |0>, so no phase is kicked back"),
    ("QPE", "O1"): ("inert",
                    "every gate in the benchmark QPE acts on a |0> control -- each H returns "
                    "its qubit to |0> before the next CP reads it"),
    ("Adder", "O5"): ("inert",
                      "a permutation network on a basis input; an RZ is only a global phase"),
}

KICKBACK_DEFECTS = [("Grover", "O6"), ("QPE", "O6"), ("QPE", "O1")]

TVD_ATOL = 1e-12
N_SEEDS_SCAN = 150
N_SEEDS_WHITELIST = 100


@pytest.mark.parametrize("operator", mutants.OPERATORS)
@pytest.mark.parametrize("name", bc.CIRCUITS)
def test_operator_moves_the_distribution_for_at_least_one_seed(operator, name):
    d = 5 if name != "Adder" else 7
    spec = bc.build(name, d, seed=11)
    moved, applicable = False, False
    for s in range(N_SEEDS_SCAN):
        try:
            mutant = mutants.mutate(spec, operator, strength=0.4,
                                    rng=np.random.default_rng(s))
        except OperatorNotApplicable:
            continue
        applicable = True
        if oracles.tvd(spec, mutant) > TVD_ATOL:
            moved = True
            break
    if not applicable:
        pytest.skip(f"{operator} has no site in {name} d={d}")
    if (name, operator) in UNDETECTABLE:
        kind, why = UNDETECTABLE[(name, operator)]
        assert not moved, f"({name}, {operator}) is listed {kind} but moved p0: {why}"
        return
    assert moved, f"{operator} never moved p0 for {name} d={d} over {N_SEEDS_SCAN} seeds"


@pytest.mark.parametrize("name,operator", sorted(UNDETECTABLE))
def test_undetectable_pairs_have_zero_tvd_at_every_width(name, operator):
    kind, why = UNDETECTABLE[(name, operator)]
    widths = (7, 10) if name == "Adder" else (4, 5, 6)
    fidelities = []
    for d in widths:
        spec = bc.build(name, d, seed=11)
        for s in range(N_SEEDS_WHITELIST):
            try:
                mutant = mutants.mutate(spec, operator, strength=0.4,
                                        rng=np.random.default_rng(s))
            except OperatorNotApplicable:
                continue
            assert oracles.tvd(spec, mutant) <= TVD_ATOL, (
                f"({name}, {operator}) d={d} seed={s} moved p0 -- {why}")
            fidelities.append(oracles.fidelity(spec, mutant))
    assert fidelities, f"({name}, {operator}) never applied at any width"
    if kind == "phase-only":
        assert min(fidelities) < 1 - 1e-9, (
            f"({name}, {operator}) is listed phase-only but the state never moved")
    else:
        assert min(fidelities) > 1 - 1e-9, (
            f"({name}, {operator}) is listed inert but the state moved")


@pytest.mark.parametrize("name,operator", KICKBACK_DEFECTS)
def test_textbook_variant_repairs_the_kickback_defects(name, operator):
    spec = bc.build(name, 5, seed=11, variant="textbook")
    for s in range(12):
        try:
            mutant = mutants.mutate(spec, operator, strength=0.4,
                                    rng=np.random.default_rng(1000 + s))
        except OperatorNotApplicable:
            continue
        if oracles.tvd(spec, mutant) > TVD_ATOL:
            return
    pytest.fail(f"textbook {name} is still immune to {operator}")


def test_qft_o6_is_orthogonal_yet_undetectable():
    spec = bc.build("QFT", 5, seed=11)
    for s in range(6):
        mutant = mutants.mutate(spec, "O6", rng=np.random.default_rng(1000 + s))
        assert oracles.fidelity(spec, mutant) < 0.01, "expected near-orthogonal states"
        assert oracles.tvd(spec, mutant) <= TVD_ATOL, "expected identical p0"
        assert oracles.is_undetectable(spec, mutant)
        assert not oracles.is_equivalent(spec, mutant), "fidelity screen misses this"


def test_phase_error_never_moves_a_permutation_circuit():
    for variant in ("benchmark", "textbook"):
        spec = bc.build("Adder", 7, seed=11, variant=variant)
        for s in range(8):
            mutant = mutants.mutate(spec, "O5", strength=0.4,
                                    rng=np.random.default_rng(1000 + s))
            assert oracles.tvd(spec, mutant) <= TVD_ATOL


def test_fidelity_full_sees_the_traced_out_ancilla():
    spec = bc.build("QPE", 4, seed=11)
    mutant = mutants.mutate(spec, "O3", strength=0.6, rng=np.random.default_rng(5))
    assert spec.circuit.num_qubits == 5 and len(spec.measured) == 4
    f_reg = oracles.fidelity(spec, mutant)
    f_all = oracles.fidelity_full(spec, mutant)
    assert 0.0 <= f_all <= 1.0 and 0.0 <= f_reg <= 1.0
    assert f_reg >= f_all - 1e-9

    routed = mutants.mutate(spec, "O7", rng=np.random.default_rng(2))
    assert oracles.fidelity(spec, routed) >= 0.0



@pytest.mark.parametrize("operator,delta", [("O1", 0), ("O2", -1), ("O3", 1), ("O4", 0), ("O5", 1)])
def test_gate_count_within_one_of_intent(operator, delta):
    for name in ("BV", "Grover", "QFT", "QPE", "Adder"):
        d = 5 if name != "Adder" else 7
        spec = bc.build(name, d, seed=5)
        base = len(spec.circuit.data)
        for s in range(4):
            try:
                mutant = mutants.mutate(spec, operator, strength=0.2,
                                        rng=np.random.default_rng(s))
            except OperatorNotApplicable:
                continue
            got = len(mutant.circuit.data) - base
            assert abs(got - delta) <= 1, f"{operator} on {name}: {got} vs intended {delta}"


def test_o1_zero_strength_is_a_no_op():
    for name in bc.CIRCUITS:
        d = 5 if name != "Adder" else 7
        spec = bc.build(name, d, seed=2)
        try:
            mutant = mutants.mutate(spec, "O1", strength=0.0,
                                    rng=np.random.default_rng(99))
        except OperatorNotApplicable:
            continue
        assert mutant.sha == spec.sha, f"{name}: O1 at strength 0 changed the circuit"
        assert abs(oracles.fidelity(spec, mutant) - 1.0) < 1e-12


def test_mutation_site_is_recorded_and_rng_driven():
    spec = bc.build("QFT", 5, seed=4)
    sites = set()
    for s in range(16):
        mutant = mutants.mutate(spec, "O1", strength=0.3, rng=np.random.default_rng(s))
        site = mutant.meta["mutation"]
        assert site["operator"] == "O1"
        sites.add((site["gate_index"], site["param_index"], site["delta"]))
    assert len(sites) > 1, "O1 always picked the same site -- rng is not driving it"


def test_o7_remaps_measured_register_through_the_final_layout():
    spec = bc.build("QFT", 5, seed=8)
    mutant = mutants.mutate(spec, "O7", rng=np.random.default_rng(3))
    assert mutant.meta["mutation"]["operator"] == "O7"
    assert mutant.meta["mutation"]["n_swaps"] >= 1
    assert len(mutant.measured) == len(spec.measured)
    assert set(mutant.measured) <= set(range(mutant.circuit.num_qubits))


def test_o6_flips_exactly_one_instance_bit():
    spec = bc.build("BV", 6, seed=13)
    mutant = mutants.mutate(spec, "O6", rng=np.random.default_rng(1))
    before = spec.meta["params"]["secret"]
    after = mutant.meta["params"]["secret"]
    assert sum(x != y for x, y in zip(before, after)) == 1


def test_mutate_rejects_unknown_operator():
    spec = bc.build("QFT", 3, seed=0)
    with pytest.raises(ValueError):
        mutants.mutate(spec, "O9", rng=np.random.default_rng(0))




def test_fidelity_and_tv_agree_on_equivalence():
    spec = bc.build("Grover", 5, seed=6)
    assert abs(oracles.fidelity(spec, spec) - 1.0) < 1e-12
    assert oracles.tvd(spec, spec) <= TVD_ATOL
    assert oracles.total_variation is oracles.tvd
    assert oracles.is_equivalent(spec, spec)


def test_p0_cache_is_keyed_on_sha():
    a = bc.build("QFT", 4, seed=1)
    b = bc.build("QFT", 4, seed=1)
    assert a is not b and a.sha == b.sha
    assert oracles.p0_from_spec(a) == oracles.p0_from_spec(b)


def test_p0_marginals_match_a_direct_sum():
    spec = bc.build("QWALK", 5, seed=21)
    p0 = oracles.p0_from_spec(spec)
    marg = oracles.p0_marginals(p0, 5)
    direct = np.zeros(5)
    for key, prob in p0.items():
        direct += prob * np.array([int(c) for c in key])
    assert np.allclose(marg, direct)


def test_sampling_round_trip():
    rng = np.random.default_rng(0)
    spec = bc.build("QWALK", 4, seed=17)
    p0 = oracles.p0_from_spec(spec)
    for m in (0, 1, 50, 2000):
        states, counts = sampling.draw_counts(p0, m, rng)
        assert counts.sum() == m
        assert states.shape[0] == counts.shape[0]
        if m:
            assert states.shape[1] == 4
            assert len(np.unique(states, axis=0)) == states.shape[0], "rows must be unique"
            assert set(np.unique(states).tolist()) <= {0, 1}
        X = sampling.X_from_counts(states, counts, rng)
        assert X.shape == (m, 4) if m else X.shape[0] == 0
        back_states, back_counts = sampling.counts_from_X(X)
        if m:
            lhs = dict(zip(sampling.states_to_keys(states), counts.tolist()))
            rhs = dict(zip(sampling.states_to_keys(back_states), back_counts.tolist()))
            assert lhs == rhs, "counts did not survive the matrix round-trip"


def test_draw_counts_columns_follow_p0_keys():
    p0 = {"0110": 1.0, "1001": 0.0}
    states, counts = sampling.draw_counts(p0, 32, np.random.default_rng(2))
    assert counts.tolist() == [32]
    assert states.tolist() == [[0, 1, 1, 0]]


def test_X_from_counts_shuffles():
    states = np.array([[0, 0], [1, 1]], dtype=np.int64)
    counts = np.array([50, 50], dtype=np.int64)
    X = sampling.X_from_counts(states, counts, np.random.default_rng(0))
    assert X.shape == (100, 2)
    assert X[:, 0].sum() == 50
    ordered = np.repeat(states, counts, axis=0)
    assert not np.array_equal(X, ordered), "rows were not shuffled"
