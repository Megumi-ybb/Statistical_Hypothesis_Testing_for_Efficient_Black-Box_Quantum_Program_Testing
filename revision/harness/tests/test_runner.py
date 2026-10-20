
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from harness import io_store, runner
from harness.runner import Cell

REVISION_DIR = Path(__file__).resolve().parents[2]




def uniform_p0(d: int) -> dict[str, float]:
    keys = [format(i, f"0{d}b") for i in range(2 ** d)]
    return {k: 1.0 / len(keys) for k in keys}


def skewed_p0(d: int) -> dict[str, float]:
    keys = [format(i, f"0{d}b") for i in range(2 ** d)]
    w = np.arange(1, len(keys) + 1, dtype=float)
    w /= w.sum()
    return dict(zip(keys, w))


def stub_draw(p_draw, m, rng):
    keys = list(p_draw)
    probs = np.fromiter((p_draw[k] for k in keys), dtype=float, count=len(keys))
    probs = probs / probs.sum()
    full = rng.multinomial(m, probs)
    nz = np.flatnonzero(full)
    states = np.array([[int(c) for c in keys[i]] for i in nz], dtype=np.uint8)
    counts = full[nz].astype(np.int64)
    X = np.repeat(states, counts, axis=0)
    rng.shuffle(X, axis=0)
    return states, counts, X


def stub_stat(draw, *, alpha, rng):
    p = float(rng.random()) * float(draw.counts.max()) / draw.m * len(draw.counts)
    p = min(1.0, p)
    return {"p_value": p, "reject": p <= alpha}


def stub_stat_sequential(draw, *, alpha, rng):
    out = stub_stat(draw, alpha=alpha, rng=rng)
    out["n_used"] = int(rng.integers(10, draw.m + 1))
    return out


def make_recorder():
    seen: list[tuple[str, bytes, bytes, bytes]] = []

    def factory(name):
        def impl(draw, *, alpha, rng):
            seen.append((name, draw.states.tobytes(), draw.counts.tobytes(), draw.X.tobytes()))
            return {"p_value": 0.5, "reject": False}

        return impl

    return seen, factory


STUBS = {
    "mmdagg": stub_stat,
    "chi_square": stub_stat,
    "g_test": stub_stat,
    "sequential": stub_stat_sequential,
    "exact_binomial": stub_stat,
}




def test_grid_constants_match_plan():
    assert len(runner.CIRCUITS) == 6
    assert len(runner.JOINT_D) == 9 and runner.JOINT_D[0] == 2 and runner.JOINT_D[-1] == 10
    assert len(runner.MARGINAL_D) == 11 and runner.MARGINAL_D[-1] == 12
    assert len(runner.M_POINTS) == 10 and runner.M_POINTS[0] == 50 and runner.M_POINTS[-1] == 2000
    assert len(runner.OPERATORS) == 7


def test_i4e1_plan_grid_arithmetic():
    cells = runner.enumerate_cells("I4E1", n_reps=1000)
    joint = [c for c in cells if c.paradigm == "joint"]
    marginal = [c for c in cells if c.paradigm == "marginal"]

    assert len(joint) == 6 * 9 * 10 * 7 == 3_780
    assert len(marginal) == 6 * 11 * 10 * 7 == 4_620
    assert len(cells) == 8_400
    assert runner.total_runs(cells) == 8_400_000
    assert runner.total_runs(joint) == 3_780_000
    assert runner.total_runs(marginal) == 4_620_000

    assert {c.case for c in cells} == {"falsecase"}
    assert {c.operator for c in cells} == set(runner.OPERATORS)
    assert len(set(cells)) == len(cells)


def test_i4e1_realized_is_bounded_by_the_plan():
    from harness import campaigns

    enum = campaigns.enumerate_campaign("I4E1", n_reps=1000)
    assert enum.candidates == 8_400
    assert enum.n_cells <= 8_400
    assert enum.n_runs == enum.n_cells * 1000
    dropped = [e for e in enum.exclusions if e["level"] == "cell"]
    assert enum.n_cells + len(dropped) == 8_400
    assert all(e["reason"] for e in dropped)

    campaigns.register_with_runner()
    try:
        assert len(runner.enumerate_cells("I4E1", n_reps=1000)) == enum.n_cells
    finally:
        campaigns.unregister_from_runner()
    assert len(runner.enumerate_cells("I4E1", n_reps=1000)) == 8_400


def test_i4e1_cells_are_uniquely_seeded_per_paradigm():
    cells = runner.enumerate_cells("I4E1", n_reps=10)
    masters = [runner.seed_master(c) for c in cells]
    for paradigm in runner.PARADIGMS:
        sub = [runner.seed_master(c) for c in cells if c.paradigm == paradigm]
        assert len(set(sub)) == len(sub)
    assert len(set(masters)) == 4_620


def test_i4e1_job_slices():
    cells = runner.enumerate_cells("I4E1", n_reps=1000)
    jobs = list(runner.iter_jobs(cells))
    assert len(jobs) == 6 * 11 * 7 == 462
    assert sum(len(g) for _, g in jobs) == 8_400
    key, group = jobs[0]
    assert {(c.campaign, c.circuit, c.d, c.operator) for c in group} == {
        (key.campaign, key.circuit, key.d, key.operator)
    }
    assert key.filename().endswith(".parquet")


def test_other_campaign_arithmetic():
    assert len(runner.enumerate_cells("I4E1B", n_reps=1000)) == 6 * 9 * 10 + 6 * 11 * 10 == 1_200
    assert len(runner.enumerate_cells("I4E2", n_reps=1000)) == 6 * 3 * 5 * 3 * 5 == 1_350
    assert len(runner.enumerate_cells("I4E2b", n_reps=1000)) == 1_350
    assert len(runner.enumerate_cells("I4E3", n_reps=1000)) == 6 * 9 * 7 == 378
    assert {c.case for c in runner.enumerate_cells("I4E1B", n_reps=1)} == {"baseline"}
    assert {c.m for c in runner.enumerate_cells("I4E3", n_reps=1)} == {2000}


def test_enumeration_filters():
    cells = runner.enumerate_cells("I4E1", circuit="BV", d=5, operator="O1", n_reps=1000)
    assert len(cells) == 2 * 10
    assert len(list(runner.iter_jobs(cells))) == 1
    joint_only = runner.enumerate_cells("I4E1", circuit="BV", d=5, operator="O1",
                                        paradigm="joint", n_reps=1000)
    assert len(joint_only) == 10
    assert runner.enumerate_cells("I4E1", d=11, paradigm="joint", n_reps=1000) == []
    with pytest.raises(KeyError):
        runner.enumerate_cells("nope")




def test_seed_master_is_sha256_prefix():
    cell = Cell(campaign="I4E1", circuit="BV", d=5, m=500, paradigm="joint",
                operator="O1", case="falsecase", n_reps=4)
    key = "I4E1|BV|5|500|O1|None|falsecase"
    assert runner.seed_key(cell) == key
    expected = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    assert runner.seed_master(cell) == expected
    assert 0 <= runner.seed_master(cell) < 2 ** 64


def test_rep_streams_are_spawned_from_the_master():
    cell = Cell(campaign="I4E1", circuit="QFT", d=3, m=50, operator="O2",
                case="falsecase", n_reps=7)
    children = runner.rep_seed_sequences(cell)
    assert len(children) == 7
    expected = np.random.SeedSequence(runner.seed_master(cell)).spawn(7)
    for got, want in zip(children, expected):
        assert np.random.default_rng(got).integers(2 ** 62) == \
               np.random.default_rng(want).integers(2 ** 62)


def test_paradigm_is_not_in_the_seed_key():
    common = dict(campaign="I4E1", circuit="BV", d=5, m=200, operator="O1",
                  case="falsecase", n_reps=6)
    j = Cell(paradigm="joint", methods=("chi_square",), **common)
    mgl = Cell(paradigm="marginal", methods=("exact_binomial",), **common)
    assert runner.seed_master(j) == runner.seed_master(mgl)

    p0 = uniform_p0(5)
    p0_alt = skewed_p0(5)
    seen_j, factory_j = make_recorder()
    seen_m, factory_m = make_recorder()
    runner.run_cell(j, p0, p0_alt, method_impls={"chi_square": factory_j("chi_square")},
                    draw_fn=stub_draw)
    runner.run_cell(mgl, p0, p0_alt,
                    method_impls={"exact_binomial": factory_m("exact_binomial")},
                    draw_fn=stub_draw)
    assert [r[1:] for r in seen_j] == [r[1:] for r in seen_m]


def test_mutant_seed_ignores_m():
    a = Cell(campaign="I4E1", circuit="BV", d=4, m=50, operator="O1", case="falsecase")
    b = Cell(campaign="I4E1", circuit="BV", d=4, m=2000, operator="O1", case="falsecase")
    c = Cell(campaign="I4E1", circuit="BV", d=4, m=50, operator="O2", case="falsecase")
    assert runner.mutant_seed(a) == runner.mutant_seed(b)
    assert runner.mutant_seed(a) != runner.mutant_seed(c)
    assert runner.circuit_seed(a) == runner.circuit_seed(c)




def _run(case="falsecase", methods=("mmdagg", "chi_square", "g_test"), n_reps=60, d=4, m=120,
         alpha=0.25, impls=None):
    operator = "O1" if case == "falsecase" else runner.BASELINE_OPERATOR
    cell = Cell(campaign="I4E1", circuit="BV", d=d, m=m, paradigm="joint",
                operator=operator, case=case, methods=methods, n_reps=n_reps)
    return runner.run_cell(cell, uniform_p0(d), skewed_p0(d), alpha=alpha,
                           method_impls=impls or STUBS, draw_fn=stub_draw)


def test_seed_determinism_byte_identical():
    a = _run()
    b = _run()
    for name in a.methods:
        ra, rb = a.methods[name], b.methods[name]
        assert (ra.TP, ra.FP, ra.TN, ra.FN) == (rb.TP, rb.FP, rb.TN, rb.FN)
        assert ra.mean_p == rb.mean_p
    assert a.seed_master == b.seed_master
    c = _run(m=140)
    assert any(
        (a.methods[n].TP, a.methods[n].FN) != (c.methods[n].TP, c.methods[n].FN)
        for n in a.methods
    )


def test_counts_are_complete_and_case_mapped():
    false = _run(case="falsecase", n_reps=40)
    for res in false.methods.values():
        assert res.FP == 0 and res.TN == 0
        assert res.TP + res.FN == 40 == res.n_evaluated
    base = _run(case="baseline", n_reps=40)
    for res in base.methods.values():
        assert res.TP == 0 and res.FN == 0
        assert res.FP + res.TN == 40
    assert 0.0 <= min(r.mean_p for r in base.methods.values()) <= 1.0
    assert all(r.wall_s >= 0.0 for r in base.methods.values())


def test_every_method_sees_the_same_draw():
    seen, factory = make_recorder()
    impls = {n: factory(n) for n in ("mmdagg", "chi_square", "g_test")}
    _run(methods=tuple(impls), n_reps=5, impls=impls)
    assert len(seen) == 15
    for rep in range(5):
        block = seen[rep * 3:(rep + 1) * 3]
        assert len({r[1:] for r in block}) == 1, "methods disagreed on the realised draw"
    assert len({r[1:] for r in seen}) == 5, "draws must differ across reps"


def test_method_streams_are_independent_of_the_method_subset():
    full = _run(methods=("mmdagg", "chi_square", "g_test"))
    subset = _run(methods=("mmdagg",))
    a, b = full.methods["mmdagg"], subset.methods["mmdagg"]
    assert (a.TP, a.FP, a.TN, a.FN) == (b.TP, b.FP, b.TN, b.FN)
    assert a.mean_p == b.mean_p


def test_n_used_reported_for_sequential_methods():
    res = _run(methods=("chi_square", "sequential"), n_reps=30)
    assert res.methods["chi_square"].n_used_mean is None
    seq = res.methods["sequential"]
    assert seq.n_used_mean is not None and 10 <= seq.n_used_mean <= 120
    assert seq.n_used_median is not None


def test_falsecase_requires_p0_alt():
    cell = Cell(campaign="I4E1", circuit="BV", d=3, m=50, operator="O1",
                case="falsecase", methods=("chi_square",), n_reps=2)
    with pytest.raises(ValueError, match="p0_alt"):
        runner.run_cell(cell, uniform_p0(3), None, method_impls=STUBS, draw_fn=stub_draw)


def test_cell_validation():
    with pytest.raises(ValueError):
        Cell(campaign="X", circuit="BV", d=2, m=50, case="baseline", operator="O1")
    with pytest.raises(ValueError):
        Cell(campaign="X", circuit="BV", d=2, m=50, case="falsecase")
    with pytest.raises(ValueError):
        Cell(campaign="X", circuit="BV", d=2, m=50, paradigm="mixed", operator="O1")
    assert Cell(campaign="X", circuit="BV", d=2, m=50, operator="O1").methods == \
        runner.JOINT_METHODS
    assert Cell(campaign="X", circuit="BV", d=2, m=50, paradigm="marginal",
                operator="O1").methods == runner.MARGINAL_METHODS


def test_marginal_null_matches_the_joint_null():
    p0 = skewed_p0(3)
    got = runner.marginal_null(p0, 3)
    want = np.zeros(3)
    for k, p in p0.items():
        want += np.array([int(c) for c in k]) * p
    assert np.allclose(got, want)
    with pytest.raises(ValueError):
        runner.marginal_null(p0, 4)




def test_parquet_round_trip_preserves_every_column(tmp_path):
    import pandas as pd

    results = [_run(case="falsecase", n_reps=12), _run(case="baseline", n_reps=12)]
    rows = [r for res in results for r in io_store.rows_from_cell_result(res)]
    before = io_store.to_frame(rows)
    written = io_store.write_job(tmp_path / "job.parquet", rows)
    back = io_store.read_grid(tmp_path / "*.parquet")

    assert written.exists()
    assert list(back.columns) == list(io_store.COLUMNS)
    assert set(io_store.COUNT_COLUMNS) <= set(back.columns)
    assert len(back) == len(before) == 2 * 3
    pd.testing.assert_frame_equal(
        back.reset_index(drop=True), before.reset_index(drop=True),
        check_dtype=False, check_exact=True,
    )
    assert back["TP"].sum() + back["FN"].sum() == 12 * 3
    assert back["FP"].sum() + back["TN"].sum() == 12 * 3
    assert back["seed_master"].dtype == np.uint64
    assert (back["seed_master"] == results[0].seed_master).any()
    assert back["n_reps"].tolist() == [12] * 6
    assert back["harness_version"].iloc[0] == io_store.HARNESS_VERSION
    assert back["git_describe"].iloc[0]
    assert back["numpy_version"].iloc[0] == np.__version__


def test_large_seed_master_survives_parquet(tmp_path):
    res = _run(n_reps=4)
    res.seed_master = 2 ** 64 - 1
    io_store.write_job(tmp_path / "big.parquet", [res])
    back = io_store.read_grid(tmp_path)
    assert int(back["seed_master"].iloc[0]) == 2 ** 64 - 1


def test_write_job_rejects_missing_counts(tmp_path):
    row = io_store.rows_from_cell_result(_run(n_reps=2))[0]
    row["TP"] = None
    with pytest.raises(ValueError, match="confusion counts"):
        io_store.write_job(tmp_path / "bad.parquet", [row])
    with pytest.raises(ValueError, match="empty job file"):
        io_store.write_job(tmp_path / "empty.parquet", [])
    with pytest.raises(FileNotFoundError):
        io_store.read_grid(tmp_path / "does_not_exist_*.parquet")


def test_run_job_writes_one_file_per_slice(tmp_path):
    cells = [
        Cell(campaign="I4E1", circuit="BV", d=3, m=m, paradigm="joint", operator="O1",
             case="falsecase", methods=("chi_square", "mmdagg"), n_reps=5)
        for m in (50, 100)
    ]
    p0, p0_alt = uniform_p0(3), skewed_p0(3)
    results, path = runner.run_job(
        cells, tmp_path, alpha=0.25, n_jobs=1,
        dist_provider=lambda cell: (p0, p0_alt),
        method_impls=STUBS, draw_fn=stub_draw,
    )
    assert len(results) == 2
    assert path.name == "I4E1__BV__d03__O1.parquet"
    df = io_store.read_grid(tmp_path)
    assert len(df) == 4
    assert set(df["m"]) == {50, 100}
    assert (df["TP"] + df["FN"] == 5).all()

    mixed = cells + [Cell(campaign="I4E1", circuit="QFT", d=3, m=50, operator="O1",
                          case="falsecase", n_reps=1)]
    with pytest.raises(ValueError, match="one \\(campaign, circuit, d, operator\\) slice"):
        runner.run_job(mixed, tmp_path, n_jobs=1, dist_provider=lambda cell: (p0, p0_alt),
                       method_impls=STUBS, draw_fn=stub_draw)

    shard_dir = tmp_path / "shards"
    _, sharded = runner.run_job(
        cells[:1], shard_dir, alpha=0.25, n_jobs=1, filename_suffix="__m50__joint",
        dist_provider=lambda cell: (p0, p0_alt), method_impls=STUBS, draw_fn=stub_draw,
    )
    assert sharded.name == "I4E1__BV__d03__O1__m50__joint.parquet"




@pytest.fixture(autouse=True)
def _plain_runner_campaigns():
    yield
    from harness import campaigns

    campaigns.unregister_from_runner()


def test_dry_run_prints_the_realized_cell_and_run_counts(capsys):
    from harness import campaigns

    enum = campaigns.enumerate_campaign("I4E1", n_reps=1000)
    rc = runner.main(["--campaign", "I4E1", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert (f"campaign I4E1: {enum.n_cells:,d} cells, {enum.n_runs:,d} runs "
            f"(n_reps=1000)") in out
    assert enum.n_cells < 8_400
    for paradigm in ("joint", "marginal"):
        n = sum(1 for c in enum.cells if c.paradigm == paradigm)
        assert f"{paradigm:<8} {n:>7,d} cells" in out
    assert "more cells" in out


def test_dry_run_lists_a_single_job(capsys):
    rc = runner.main(["--campaign", "I4E1", "--circuit", "BV", "--d", "5",
                      "--operator", "O2", "--paradigm", "joint", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("master=") == 10
    assert "campaign I4E1: 10 cells, 10,000 runs" in out
    assert "more cells" not in out


def test_dry_run_refuses_an_inapplicable_slice(capsys):
    rc = runner.main(["--campaign", "I4E1", "--circuit", "BV", "--d", "5",
                      "--operator", "O1", "--dry-run"])
    assert rc == 0
    assert "campaign I4E1: 0 cells" in capsys.readouterr().out


def test_cli_requires_out_unless_dry_run():
    with pytest.raises(SystemExit):
        runner.main(["--campaign", "I4E1", "--circuit", "BV", "--d", "5",
                     "--operator", "O2", "--m", "50"])


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "harness.runner", *args],
        cwd=REVISION_DIR, capture_output=True, text=True,
    )


@pytest.mark.skipif(
    not (REVISION_DIR / "harness" / "campaigns.py").exists(), reason="campaigns.py absent"
)
class TestCliUnderDashM:

    def test_phase3_campaign_is_registered(self):
        got = _cli("--campaign", "I8", "--dry-run", "--list-limit", "0")
        assert got.returncode == 0, got.stderr
        assert "campaign I8: 36 cells, 36,000 runs" in got.stdout

    def test_excluded_slice_enumerates_no_cells(self):
        got = _cli("--campaign", "I4E1", "--circuit", "BV", "--d", "3",
                   "--operator", "O1", "--dry-run")
        assert got.returncode == 0, got.stderr
        assert "campaign I4E1: 0 cells, 0 runs" in got.stdout
        applicable = _cli("--campaign", "I4E1", "--circuit", "BV", "--d", "3",
                          "--operator", "O2", "--dry-run")
        assert "campaign I4E1: 20 cells, 20,000 runs" in applicable.stdout

    def test_filtered_grid_is_the_campaigns_grid(self):
        got = _cli("--campaign", "I4E1", "--dry-run")
        assert "campaign I4E1: 6,400 cells, 6,400,000 runs" in got.stdout, (
            "CLI enumerated the unfiltered plan grid instead of the applicability-"
            "filtered campaigns.py grid"
        )

    def test_i8_shape_arm_is_selectable(self):
        got = _cli("--campaign", "I8", "--d", "3", "--circuit", "shapeA_dense", "--dry-run")
        assert got.returncode == 0, got.stderr
        assert "campaign I8: 4 cells, 4,000 runs" in got.stdout

    def test_unknown_circuit_is_rejected_not_silently_empty(self):
        got = _cli("--campaign", "I8", "--circuit", "BV", "--dry-run")
        assert got.returncode != 0
        assert "does not occur in campaign I8" in got.stderr
        assert "shapeA_dense" in got.stderr


def test_importing_runner_does_not_require_qiskit():
    code = "import sys; import harness.runner; print('qiskit' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=REVISION_DIR,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
