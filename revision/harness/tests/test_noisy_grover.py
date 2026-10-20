
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parent
_HARNESS = _TESTS.parent
_REVISION = _HARNESS.parent
for _p in (str(_HARNESS), str(_REVISION)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from harness import base_circuits, mutants, oracles, runner, stats_joint
from harness import noisy_grover as ng

ALPHA = 0.05


@pytest.fixture(scope="module")
def spec():
    cell = ng.I5Cell(arm="A", noise=ng.noise_configs(["2"])[0], m=200)
    return base_circuits.build(ng.CIRCUIT, ng.D, seed=runner.circuit_seed(cell.as_runner_cell()),
                               variant=ng.VARIANT)


@pytest.fixture(scope="module")
def compiled(spec):
    return ng.compile_noisy(spec, ng.noise_configs(["2"])[0])




def test_noise_configs_are_the_committed_six():
    cfgs = ng.noise_configs()
    assert [c.label for c in cfgs] == ["aer0", "aer1", "aer2", "aer3", "aer4", "fake"]
    assert [c.scale for c in cfgs[:5]] == list(ng.NOISE_SCALES)
    assert ng.NOISE_SCALES[2] == 1.0
    assert ng.NOISE_SCALES[0] == 0.25 and ng.NOISE_SCALES[-1] == 4.0


def test_parametric_noise_model_rates_scale():
    for scale in (0.25, 1.0, 4.0):
        nm = ng.parametric_noise_model(scale)
        assert set(nm.noise_instructions) >= {"cx", "sx", "x", "measure"}
        ro = [e for e in nm.to_dict()["errors"] if e["type"] == "roerror"]
        assert len(ro) == 1
        p = ng.BASE_READOUT * scale
        assert np.allclose(ro[0]["probabilities"], [[1 - p, p], [p, 1 - p]])
        depol = [e for e in nm.to_dict()["errors"]
                 if e["type"] == "qerror" and e["operations"] == ["cx"]]
        assert depol, "no 2-qubit depolarizing channel attached to cx"


def test_zero_noise_reproduces_the_ideal_distribution(spec):
    noisy = ng.compile_noisy(spec, ng.NoiseConfig(label="aerzero", kind="aer", scale=0.0))
    p0 = oracles.p0_from_spec(spec)
    ideal = np.array([p0[k] for k in noisy.keys])
    assert 0.5 * np.abs(noisy.probs - ideal).sum() < 1e-9


def test_noise_strength_monotonically_degrades_the_marked_state(spec):
    marked = max(oracles.p0_from_spec(spec).items(), key=lambda kv: kv[1])[0]
    keys = oracles.state_keys(ng.D)
    idx = keys.index(marked)
    probs = [ng.compile_noisy(spec, c).probs[idx] for c in ng.noise_configs()[:5]]
    assert probs == sorted(probs, reverse=True), probs
    assert probs[0] > 0.7 > probs[-1]




@pytest.mark.parametrize("sampler", ["exact", "aer"])
@pytest.mark.parametrize("m", [50, 300])
def test_sample_shapes_and_totals(compiled, sampler, m):
    rng = np.random.default_rng(0)
    full = ng._SAMPLERS[sampler](compiled, m, rng)
    assert full.shape == (2 ** ng.D,)
    assert full.dtype == np.int64
    assert int(full.sum()) == m
    assert np.all(full >= 0)

    states, counts = ng.states_counts(full, compiled.keys, ng._state_matrix(compiled.keys))
    assert states.shape == (counts.size, ng.D)
    assert set(np.unique(states).tolist()) <= {0, 1}
    assert int(counts.sum()) == m
    assert len(np.unique(states, axis=0)) == counts.size


def test_sampling_is_deterministic_given_the_seed(compiled):
    a = ng.sample_exact(compiled, 200, np.random.default_rng(7))
    b = ng.sample_exact(compiled, 200, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_exact_matches_aer_shot_sampling(compiled):
    shots = 200_000
    emp = ng.sample_aer(compiled, shots, np.random.default_rng(3)).astype(float) / shots
    tvd = 0.5 * np.abs(emp - compiled.probs).sum()
    floor = 0.5 * np.sqrt(2 / np.pi) * np.sqrt(
        compiled.probs * (1 - compiled.probs) / shots).sum()
    assert tvd < 5 * floor, f"TVD {tvd:.5f} vs sampling floor {floor:.5f}"


def test_readout_confusion_orientation_is_p_observed_given_true():
    from qiskit import QuantumCircuit
    from qiskit_aer.noise import NoiseModel, ReadoutError

    nm = NoiseModel()
    nm.add_readout_error(ReadoutError([[0.9, 0.1], [0.3, 0.7]]), [0])
    A = ng.readout_confusion(nm, [0])
    assert np.allclose(A, [[0.9, 0.3], [0.1, 0.7]])

    qc = QuantumCircuit(1)
    qc.x(0)
    spec = base_circuits.CircuitSpec(name="x1", d=1, circuit=qc, measured=(0,))
    noisy = ng.compile_noisy(spec, ng.NoiseConfig(label="asym", kind="aer", scale=0.0))
    p = ng.readout_confusion(nm, [0]) @ np.array([0.0, 1.0])
    assert np.allclose(p, [0.3, 0.7])
    assert np.allclose(noisy.probs, [0.0, 1.0], atol=1e-9)




def test_homogeneity_hand_checked_2x2():
    from scipy.stats import chi2

    out = ng.chi_square_homogeneity(np.array([60, 40]), np.array([50, 50]), alpha=ALPHA)
    expected = 200 * (60 * 50 - 50 * 40) ** 2 / (110 * 90 * 100 * 100)
    assert expected == pytest.approx(2.0202020202020203)
    assert out["statistic"] == pytest.approx(expected, rel=1e-12)
    assert out["df"] == 1
    assert out["n_cells"] == 2
    assert out["n_lumped_states"] == 0
    assert out["cochran_satisfied"] is True
    assert out["p_value"] == pytest.approx(float(chi2.sf(expected, 1)), rel=1e-12)
    assert out["reject"] is False


def test_homogeneity_identical_samples_have_zero_statistic():
    out = ng.chi_square_homogeneity([30, 70], [30, 70])
    assert out["statistic"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["reject"] is False


def test_homogeneity_rejects_a_gross_difference():
    out = ng.chi_square_homogeneity([200, 0], [0, 200])
    assert out["reject"] is True
    assert out["p_value"] < 1e-10


def test_homogeneity_lumps_rare_states_and_drops_empty_rows():
    out = ng.chi_square_homogeneity(np.array([50, 1, 1, 1, 0]),
                                    np.array([50, 1, 1, 1, 0]), min_count=5)
    assert out["n_cells"] == 2
    assert out["n_lumped_states"] == 2
    assert out["df"] == 1
    assert out["cochran_satisfied"] is False
    assert out["statistic"] == pytest.approx(0.0)


def test_homogeneity_degenerate_table_fails_to_reject():
    out = ng.chi_square_homogeneity([10, 0], [10, 0], min_count=5)
    assert out["n_cells"] == 1 and out["df"] == 0
    assert out["reject"] is False and out["p_value"] == 1.0


def test_homogeneity_is_permutation_invariant_in_the_state_order():
    rng = np.random.default_rng(1)
    x = rng.multinomial(300, np.full(8, 1 / 8))
    y = rng.multinomial(300, np.full(8, 1 / 8))
    perm = rng.permutation(8)
    a = ng.chi_square_homogeneity(x, y)
    b = ng.chi_square_homogeneity(x[perm], y[perm])
    assert a["statistic"] == pytest.approx(b["statistic"], rel=1e-12)




def test_mmdagg_ref_counts_alias_is_backward_compatible():
    keys = oracles.state_keys(4)
    p0 = {k: 1 / 16 for k in keys}
    rng = np.random.default_rng(0)
    ref = rng.multinomial(500, np.full(16, 1 / 16))
    counts_full = rng.multinomial(200, np.full(16, 1 / 16))
    smat = ng._state_matrix(keys)
    states, counts = ng.states_counts(counts_full, keys, smat)

    a = stats_joint.mmdagg_counts(states, counts, p0, alpha=ALPHA, n_permute=50,
                                  rng=np.random.default_rng(5), ref_counts=ref)
    b = stats_joint.mmdagg_counts(states, counts, p0, alpha=ALPHA, n_permute=50,
                                  rng=np.random.default_rng(5), X_ref_counts=ref)
    assert a["p_value"] == b["p_value"]
    assert a["stats_per_kernel"] == b["stats_per_kernel"]
    with pytest.raises(ValueError):
        stats_joint.mmdagg_counts(states, counts, p0, alpha=ALPHA, n_permute=50,
                                  rng=np.random.default_rng(5), ref_counts=ref,
                                  X_ref_counts=ref)


@pytest.mark.slow
def test_two_sample_mmdagg_is_calibrated_on_the_same_distribution(compiled):
    reps, m, n_ref = 200, 100, 500
    keys = compiled.keys
    smat = ng._state_matrix(keys)
    placeholder = ng._uniform_placeholder(keys)
    rng = np.random.default_rng(20260726)

    mmd_rejects = chi_rejects = 0
    for _ in range(reps):
        full = ng.sample_exact(compiled, m, rng)
        ref = ng.sample_exact(compiled, n_ref, rng)
        states, counts = ng.states_counts(full, keys, smat)
        mmd_rejects += stats_joint.mmdagg_counts(
            states, counts, placeholder, alpha=ALPHA, n_permute=ng.N_PERMUTE,
            rng=rng, ref_counts=ref)["reject"]
        chi_rejects += ng.chi_square_homogeneity(full, ref, alpha=ALPHA)["reject"]

    lo = ng.wilson(mmd_rejects, reps, alpha=0.05)[0]
    assert lo <= ALPHA, (
        f"two-sample MMDAgg over-rejects under H0: {mmd_rejects}/{reps} "
        f"(Wilson lower bound {lo:.3f} > {ALPHA})")
    chi_lo, chi_hi = ng.wilson(chi_rejects, reps, alpha=0.05)
    assert chi_lo <= ALPHA <= chi_hi, (
        f"homogeneity chi-square is not calibrated: {chi_rejects}/{reps}, "
        f"Wilson [{chi_lo:.3f}, {chi_hi:.3f}] misses {ALPHA}")


@pytest.mark.slow
def test_regime_a_inflates_far_above_alpha(compiled, spec):
    reps, m = 100, 200
    keys = compiled.keys
    smat = ng._state_matrix(keys)
    p0 = oracles.p0_from_spec(spec)
    rng = np.random.default_rng(31337)

    mmd = chi = impossible = 0
    for _ in range(reps):
        full = ng.sample_exact(compiled, m, rng)
        states, counts = ng.states_counts(full, keys, smat)
        mmd += stats_joint.mmdagg_counts(states, counts, p0, alpha=ALPHA,
                                         n_ref=ng.N_REF, n_permute=ng.N_PERMUTE,
                                         rng=rng)["reject"]
        out = stats_joint.chi_square_joint_counts(states, counts, p0, alpha=ALPHA)
        chi += out["reject"]
        impossible += out["impossible_state"]

    assert mmd / reps > 0.5, f"Regime A MMDAgg did not inflate: {mmd}/{reps}"
    assert chi / reps > 0.9, f"Regime A chi-square did not inflate: {chi}/{reps}"
    assert impossible / reps > 0.9




def test_campaign_enumeration_matches_the_committed_arithmetic():
    cells = ng.enumerate_cells()
    assert len(cells) == 108
    assert sum(c.n_reps for c in cells) == 108_000
    aer = [c for c in cells if c.noise.kind == "aer"]
    fake = [c for c in cells if c.noise.kind == "fake"]
    assert len(aer) == 90 and len(fake) == 18
    assert len({(c.arm, c.noise.label, c.m) for c in cells}) == 108
    extended = ng.enumerate_cells(m_points=ng.M_POINTS_EXTENDED)
    assert len(extended) == 180
    assert sum(c.n_reps for c in extended) - 108_000 == 72_000


def test_cell_seeds_are_distinct_across_arms_and_noise_levels():
    cells = ng.enumerate_cells(n_reps=4)
    masters = [ng.seed_master(c) for c in cells]
    assert len(set(masters)) == len(cells)
    a = ng.I5Cell(arm="A", noise=ng.noise_configs(["2"])[0], m=200, n_reps=4)
    b = ng.I5Cell(arm="B", noise=ng.noise_configs(["2"])[0], m=200, n_reps=4)
    assert runner.seed_key(a.as_runner_cell()) == runner.seed_key(b.as_runner_cell())
    assert ng.seed_key(a) != ng.seed_key(b)
    assert ng.seed_key(a).startswith(runner.seed_key(a.as_runner_cell()))
    assert ng.seed_master(a) == ng.seed_master(
        ng.I5Cell(arm="A", noise=ng.noise_configs(["2"])[0], m=200, n_reps=4))


def test_power_arm_uses_a_real_structural_mutant(spec):
    cell = ng.I5Cell(arm="P", noise=ng.noise_configs(["2"])[0], m=200,
                     operator=ng.BUG_OPERATOR)
    mutant = mutants.mutate(spec, ng.BUG_OPERATOR, strength=None,
                            rng=np.random.default_rng(
                                runner.mutant_seed(cell.as_runner_cell())))
    assert oracles.tvd(spec, mutant) > 0.1
    correct = ng.compile_noisy(spec, cell.noise)
    buggy = ng.compile_noisy(mutant, cell.noise)
    assert 0.5 * np.abs(correct.probs - buggy.probs).sum() > 0.1

    benchmark = base_circuits.build(ng.CIRCUIT, ng.D, seed=1234, variant="benchmark")
    equivalent = mutants.mutate(benchmark, "O6", rng=np.random.default_rng(11))
    assert oracles.tvd(benchmark, equivalent) == pytest.approx(0.0, abs=1e-12)


def test_run_i5_cell_produces_a_writable_frame(tmp_path):
    from harness import io_store

    cells = [ng.I5Cell(arm=arm, noise=ng.noise_configs(["2"])[0], m=60, n_reps=6,
                       operator=ng.BUG_OPERATOR if arm == "P" else runner.BASELINE_OPERATOR)
             for arm in ng.ARMS]
    results = [ng.run_i5_cell(c, n_ref=200, n_permute=50) for c in cells]

    for item, cell in zip(results, cells):
        assert set(item.result.methods) == set(cell.methods)
        for name, res in item.result.methods.items():
            assert res.n_evaluated == 6
            assert res.TP + res.FP + res.TN + res.FN == 6
            if cell.case == "baseline":
                assert res.TP == 0 and res.FN == 0
            else:
                assert res.FP == 0 and res.TN == 0
            assert item.extras[name]["arm"] == cell.arm
        assert (item.ref_draws == 0) if cell.arm == "A" else (item.ref_draws == 6 * 200)

    path = ng.write_results(tmp_path / "i5.parquet", results)
    df = io_store.read_grid(path)
    assert len(df) == sum(len(c.methods) for c in cells)
    assert set(df["campaign"]) == {"I5"} and set(df["circuit"]) == {"Grover"}
    assert set(df["arm"]) == set(ng.ARMS)
    assert (df["TP"] + df["FP"] + df["TN"] + df["FN"] == 6).all()
    assert df["rejection_rate"].between(0, 1).all()
    assert (df["wilson_lo"] <= df["rejection_rate"]).all()
    assert (df["rejection_rate"] <= df["wilson_hi"]).all()


def test_per_config_ref_mode_shares_one_calibration_sample():
    cell = ng.I5Cell(arm="B", noise=ng.noise_configs(["2"])[0], m=60, n_reps=5)
    item = ng.run_i5_cell(cell, n_ref=200, n_permute=50, ref_mode="per-config")
    assert item.ref_draws == 0
    assert item.extras["regimeB_mmdagg_2s"]["ref_mode"] == "per-config"


def test_device_arm_is_optional_not_fatal(monkeypatch):
    def boom():
        raise ng.FakeBackendUnavailable("qiskit-ibm-runtime is not installed")

    monkeypatch.setattr(ng, "fake_backend", boom)
    monkeypatch.setattr(ng, "_COMPILED", {})
    cell = ng.I5Cell(arm="B", noise=ng.noise_configs(["fake"])[0], m=60, n_reps=2)
    out = ng.run_campaign([cell], n_ref=100, n_permute=20, verbose=False)
    assert out == []


def test_fake_backend_arm_runs_if_available():
    try:
        ng.fake_backend()
    except ng.FakeBackendUnavailable as exc:
        pytest.skip(str(exc))
    cell = ng.I5Cell(arm="B", noise=ng.noise_configs(["fake"])[0], m=60, n_reps=3)
    item = ng.run_i5_cell(cell, n_ref=200, n_permute=50)
    assert item.result.methods["regimeB_mmdagg_2s"].n_evaluated == 3
    assert item.extras["regimeB_mmdagg_2s"]["backend"] == ng.FAKE_BACKEND


def test_cli_dry_run_prints_the_grid(capsys):
    assert ng.main(["--dry-run", "--n-reps", "1000"]) == 0
    out = capsys.readouterr().out
    assert "108 cells, 108,000 runs" in out
    assert out.count("arm=") == 108
    assert "FakeManilaV2" in out
