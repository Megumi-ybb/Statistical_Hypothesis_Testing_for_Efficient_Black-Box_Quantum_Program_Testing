
from __future__ import annotations

import csv

import numpy as np
import pytest
from scipy.stats import binom

from harness import g3_golden as g3




def test_acceptance_band_matches_binom_interval():
    lo, hi = g3.acceptance_band(1000, 0.05)
    expected = binom.interval(0.95, 1000, 0.05)
    assert (lo, hi) == (int(expected[0]), int(expected[1]))
    assert lo < 50 < hi


@pytest.mark.parametrize("p, n", [(0.05, 1000), (0.5, 1000), (0.02, 200)])
def test_acceptance_band_covers_about_95_percent(p, n):
    lo, hi = g3.acceptance_band(n, p)
    coverage = binom.cdf(hi, n, p) - binom.cdf(lo - 1, n, p)
    assert 0.95 <= coverage <= 0.99


def test_band_verdict_passes_a_close_count():
    v = g3.band_verdict(reject_new=52, n_new=1000, reject_disk=50, n_disk=1000)
    assert v["in_band"] is True
    assert v["in_band_2sample"] is True
    assert v["binom_p"] > 0.05
    assert v["band_lo"] <= 52 <= v["band_hi"]
    assert v["p_disk"] == pytest.approx(0.05)
    assert v["p_new"] == pytest.approx(0.052)


def test_band_verdict_fails_a_distant_count():
    v = g3.band_verdict(reject_new=200, n_new=1000, reject_disk=50, n_disk=1000)
    assert v["in_band"] is False
    assert v["in_band_2sample"] is False
    assert v["binom_p"] < 1e-6
    assert v["fisher_p"] < 1e-6


def test_band_verdict_is_exactly_on_the_edge():
    lo, hi = g3.acceptance_band(1000, 0.05)
    assert g3.band_verdict(hi, 1000, 50, 1000)["in_band"] is True
    assert g3.band_verdict(hi + 1, 1000, 50, 1000)["in_band"] is False
    assert g3.band_verdict(lo, 1000, 50, 1000)["in_band"] is True
    assert g3.band_verdict(lo - 1, 1000, 50, 1000)["in_band"] is False


def test_band_verdict_degenerate_disk_rates():
    zero = g3.band_verdict(reject_new=0, n_new=500, reject_disk=0, n_disk=1000)
    assert (zero["band_lo"], zero["band_hi"]) == (0, 0)
    assert zero["in_band"] is True
    assert g3.band_verdict(1, 500, 0, 1000)["in_band"] is False

    one = g3.band_verdict(reject_new=500, n_new=500, reject_disk=1000, n_disk=1000)
    assert (one["band_lo"], one["band_hi"]) == (500, 500)
    assert one["in_band"] is True
    assert g3.band_verdict(499, 500, 1000, 1000)["in_band"] is False


def test_band_verdict_handles_unequal_replication_counts():
    v = g3.band_verdict(reject_new=10, n_new=200, reject_disk=50, n_disk=1000)
    assert v["band_hi"] < 50
    assert v["in_band"] is True




def _synthetic_disk() -> list[dict]:
    rows = []
    for experiment in g3.EXPERIMENT_TO_CIRCUIT:
        for d in range(2, 11):
            for m in (50, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000):
                for method in g3.DEFAULT_METHODS:
                    rows.append({"experiment": experiment, "n_qubits": d,
                                 "m_samples": m, "test_method": method,
                                 "reject_disk": 50, "n_disk": 1000, "p_disk": 0.05})
    return rows


def test_select_cells_is_deterministic():
    df = _synthetic_disk()
    assert g3.select_cells(df, 20, seed=801) == g3.select_cells(df, 20, seed=801)


def test_select_cells_depends_on_seed():
    df = _synthetic_disk()
    assert g3.select_cells(df, 20, seed=801) != g3.select_cells(df, 20, seed=802)


def test_select_cells_spans_circuits_and_the_hard_corners():
    df = _synthetic_disk()
    cells = g3.select_cells(df, 20, seed=801)
    assert len(cells) == 20
    assert len({(c["experiment"], c["d"], c["m"]) for c in cells}) == 20
    assert {c["experiment"] for c in cells} == set(g3.EXPERIMENT_TO_CIRCUIT)
    assert sum(c["d"] >= 8 for c in cells) >= 6
    assert sum(c["m"] >= 1000 for c in cells) >= 6
    assert sum(c["d"] >= 8 and c["m"] >= 1000 for c in cells) >= 1


def test_select_cells_prefix_is_stable_under_growth():
    df = _synthetic_disk()
    assert g3.select_cells(df, 12, seed=801) == g3.select_cells(df, 24, seed=801)[:12]


def test_select_cells_returns_only_existing_cells():
    rows = _synthetic_disk()
    available = {(r["experiment"], r["n_qubits"], r["m_samples"]) for r in rows}
    for cell in g3.select_cells(rows, 20, seed=801):
        assert (cell["experiment"], cell["d"], cell["m"]) in available


def test_select_cells_cannot_exceed_the_grid():
    rows = _synthetic_disk()[:3 * len(g3.DEFAULT_METHODS)]
    distinct = {(r["experiment"], r["n_qubits"], r["m_samples"]) for r in rows}
    assert len(g3.select_cells(rows, 50, seed=801)) == len(distinct)




def test_budget_guard_allows_the_quick_configuration():
    assert g3.budget_guard(g3.QUICK_CELLS, 2, g3.QUICK_REPS, full=False) is None


def test_budget_guard_refuses_the_gate_configuration_without_full():
    refusal = g3.budget_guard(g3.FULL_CELLS, 2, g3.FULL_REPS, full=False)
    assert refusal is not None
    assert "40,000" in refusal and "CLUSTER ONLY" in refusal


def test_budget_guard_lets_full_through():
    assert g3.budget_guard(g3.FULL_CELLS, 2, g3.FULL_REPS, full=True) is None


def test_budget_guard_is_exact_at_the_boundary():
    assert g3.budget_guard(1, 1, g3.LOCAL_BUDGET_REPLICATIONS, full=False) is None
    assert g3.budget_guard(1, 1, g3.LOCAL_BUDGET_REPLICATIONS + 1, full=False) is not None


def test_cli_refuses_an_oversized_part_b_run_before_doing_any_work(capsys):
    assert g3.main(["--part", "b"]) == 2
    out = capsys.readouterr().out
    assert "REFUSING part (b)" in out
    assert "PART (b) MONTE CARLO REGRESSION" not in out


def test_cli_part_a_is_never_blocked(tmp_path, capsys):
    pytest.importorskip("qiskit")
    out = tmp_path / "g3.csv"
    assert g3.main(["--part", "a", "--quick", "--out", str(out)]) == 0
    assert "REFUSING" not in capsys.readouterr().out
    assert (tmp_path / "g3_oracle.csv").exists()




def test_disk_baseline_cells_sum_to_1000_replications():
    rows = g3.load_disk_cells()
    assert len(rows) == 1620
    assert all(r["n_disk"] == 1000 for r in rows)
    assert {r["experiment"] for r in rows} == set(g3.EXPERIMENT_TO_CIRCUIT)
    assert {r["test_method"] for r in rows} == set(g3.DEFAULT_METHODS)


def test_disk_index_is_keyed_by_cell_and_method():
    rows = g3.load_disk_cells()
    index = g3.disk_index(rows)
    assert len(index) == len(rows)
    assert index[("BV", 10, 50, "mmdagg")]["reject_disk"] == 24


def test_seed_master_is_deterministic_and_source_specific():
    a = g3.g3_seed_master("campaign", "BV", 6, 500, 801)
    assert a == g3.g3_seed_master("campaign", "BV", 6, 500, 801)
    assert a != g3.g3_seed_master("builder", "BV", 6, 500, 801)
    assert a != g3.g3_seed_master("campaign", "BV", 6, 500, 802)
    assert 0 <= a < 2 ** 64


def test_write_csv_round_trips_ragged_rows(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "c": 3.5}]
    path = tmp_path / "out.csv"
    g3.write_csv(path, rows)
    with path.open(newline="") as handle:
        back = list(csv.DictReader(handle))
    assert [r["a"] for r in back] == ["1", "2"]
    assert back[0]["c"] == "" and back[1]["b"] == ""




def test_production_scripts_all_source_their_state_from_produce_state():
    for experiment, script in g3.EXPERIMENT_SCRIPTS.items():
        assert g3._script_state_source(g3.PRODUCTION_DIR / script) == "produce_state", (
            f"{experiment}: {script}")


def test_produce_state_ignores_its_seed_argument():
    assert g3._produce_state_uses_seed() is False




def test_bv_builder_reproduces_the_shipped_instance():
    pytest.importorskip("qiskit")
    rows = g3.oracle_equality_rows({"BV": (4,)})
    compared = [r for r in rows if r["status"] in {"PASS", "FAIL"}]
    assert len(compared) == 1
    row = compared[0]
    assert row["circuit"] == "BV" and row["d"] == 4
    assert row["status"] == "PASS"
    assert row["max_abs_dp"] <= g3.ORACLE_ATOL
    assert row["n_states"] == 16
    assert {r["status"] for r in rows if r is not row} <= {"NOT_REQUESTED", "NO_REFERENCE"}
    assert next(r for r in rows if r["circuit"] == "QWALK")["status"] == "NO_REFERENCE"


def test_bv_null_is_the_secret_string_and_matches_the_qasm_file():
    pytest.importorskip("qiskit")
    from harness import base_circuits, oracles

    spec = base_circuits.assemble("BV", 4, g3.reference_params("BV", 4))
    p_new = oracles.p0_from_spec(spec)
    p_ref, n_qubits = g3.load_qasm_p0(g3.reference_qasm_path("BV", 4), spec.measured)

    assert n_qubits == 4
    assert g3.max_abs_dp(p_new, p_ref) <= g3.ORACLE_ATOL
    assert max(p_new, key=p_new.get) == "1111"
    assert p_new["1111"] == pytest.approx(1.0, abs=1e-12)


def test_campaign_state_is_normalised_and_rng_driven():
    pytest.importorskip("qiskit")
    a = g3.campaign_state(3, np.random.default_rng(7))
    b = g3.campaign_state(3, np.random.default_rng(7))
    c = g3.campaign_state(3, np.random.default_rng(8))
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert np.linalg.norm(a) == pytest.approx(1.0)


def test_run_cell_is_reproducible_from_its_seed():
    pytest.importorskip("qiskit")
    kwargs = dict(source="campaign", n_reps=4, methods=("chi_square",), seed=801)
    first = g3.run_cell("BV", 3, 50, **kwargs)
    second = g3.run_cell("BV", 3, 50, **kwargs)
    assert first["rejects"] == second["rejects"]
    assert first["seed_master"] == second["seed_master"]
    assert first["errors"]["chi_square"] == 0


def test_run_cell_records_unsupported_builder_widths_as_errors():
    pytest.importorskip("qiskit")
    result = g3.run_cell("add", 2, 50, source="builder", n_reps=3,
                         methods=("chi_square",), seed=801)
    assert result["errors"]["chi_square"] == 3
    assert "UnsupportedWidth" in result["first_error"]["chi_square"]
