
import json

import numpy as np
import pytest

from harness import campaigns, runner
from harness.runner import Cell

APP = campaigns.load_applicability()




def test_candidate_grids_reproduce_the_plan_arithmetic():
    man = campaigns.manifest(app=APP)
    for entry in man["campaigns"]:
        if entry["plan_cells"] is None:
            continue
        assert entry["candidates"] == entry["plan_cells"], entry["campaign"]


def test_manifest_realized_within_bounds_and_self_consistent():
    man = campaigns.manifest(app=APP)
    for entry in man["campaigns"]:
        assert entry["cells"] + entry["excluded_cells"] == entry["candidates"]
        assert entry["runs"] == entry["cells"] * man["n_reps"]
        if entry["plan_cells"] is not None:
            assert entry["cells"] <= entry["plan_cells"]
            assert entry["runs"] <= entry["plan_runs"]
    assert man["campaigns"][0]["campaign"] == "I4E1"
    text = campaigns.format_manifest(man)
    assert "TOTAL" in text and "I7seq" in text
    assert json.loads(json.dumps(man))["totals"]["cells"] == man["totals"]["cells"]
    rows = campaigns.manifest_rows()
    assert [r["campaign"] for r in rows] == list(campaigns.CAMPAIGN_ORDER)
    assert all({"cells", "runs", "plan_cells"} <= set(r) for r in rows)


def test_i4e1_is_the_plan_bound_minus_documented_exclusions():
    enum = campaigns.enumerate_campaign("I4E1", app=APP)
    assert enum.candidates == 8_400
    assert enum.n_cells <= 8_400
    assert enum.n_cells + len(
        [e for e in enum.exclusions if e["level"] == "cell"]
    ) == 8_400
    assert set(enum.by_reason()) <= {"unsupported_width", "operator_not_applicable",
                                     "no_runnable_method"}
    joint = [c for c in enum.cells if c.paradigm == "joint"]
    marginal = [c for c in enum.cells if c.paradigm == "marginal"]
    assert len(joint) + len(marginal) == enum.n_cells
    assert {c.d for c in joint} <= set(runner.JOINT_D)
    assert {c.d for c in marginal} <= set(runner.MARGINAL_D)


def test_adder_only_exists_at_widths_3k_plus_1():
    for campaign in ("I4E1", "I4E1B", "I4E3", "I7size"):
        widths = {c.d for c in campaigns.cells(campaign, app=APP) if c.circuit == "Adder"}
        assert widths <= {4, 7, 10}, campaign
    assert {c.d for c in campaigns.cells("I4E1", app=APP) if c.circuit == "Adder"} == {4, 7, 10}
    enum = campaigns.enumerate_campaign("I4E1", app=APP)
    dropped = {(e["circuit"], e["d"]) for e in enum.exclusions
               if e["reason"] == "unsupported_width"}
    assert all(c == "Adder" for c, _ in dropped)
    assert {d for _, d in dropped} == {2, 3, 5, 6, 8, 9, 11, 12}


def test_operator_applicability_exclusions_are_real():
    enum = campaigns.enumerate_campaign("I4E1", app=APP)
    live = {(c.circuit, c.d, c.operator) for c in enum.cells}
    for circuit in ("BV", "Grover", "Adder"):
        assert not any(c == circuit and o == "O1" for c, _, o in live), circuit
    for circuit in ("QFT", "QPE"):
        assert not any(c == circuit and o == "O4" for c, _, o in live), circuit
    assert any(o == "O1" for c, _, o in live if c in ("QFT", "QPE", "QWALK"))
    assert enum.by_reason()["operator_not_applicable"] > 0


def test_qwalk_cells_are_flagged_newly_constructed():
    qwalk = next(c for c in campaigns.cells("I4E1", app=APP) if c.circuit == "QWALK")
    bv = next(c for c in campaigns.cells("I4E1", app=APP) if c.circuit == "BV")
    assert campaigns.cell_provenance(qwalk, APP)["newly_constructed"] is True
    assert campaigns.cell_provenance(bv, APP)["newly_constructed"] is False
    man = campaigns.manifest(app=APP)
    e1 = next(e for e in man["campaigns"] if e["campaign"] == "I4E1")
    assert e1["newly_constructed_cells"] > 0
    assert any("QWALK" in note for note in e1["notes"])


def test_structural_fallback_is_an_upper_bound_and_says_so():
    structural = campaigns.Applicability(source="structural")
    loose = campaigns.enumerate_campaign("I4E1", app=structural)
    tight = campaigns.enumerate_campaign("I4E1", app=APP)
    assert loose.n_cells >= tight.n_cells
    assert set(loose.by_reason()) == {"unsupported_width"}
    assert any("UPPER BOUND" in note for note in loose.notes)


def test_interior_only_methods_are_dropped_not_the_cell():
    cells = campaigns.cells("I4E1", app=APP, paradigm="marginal")
    bv = [c for c in cells if c.circuit == "BV"]
    qft = [c for c in cells if c.circuit == "QFT"]
    assert bv and qft
    assert all(set(c.methods) & campaigns.INTERIOR_METHODS == set() for c in bv)
    assert all("exact_binomial" in c.methods for c in bv)
    assert all(set(runner.MARGINAL_METHODS) == set(c.methods) for c in qft)
    drops = campaigns.enumerate_campaign("I4E1", app=APP).method_drops()
    assert drops["bootstrap:degenerate_marginal_null"] > 0




def test_i4_frontier_cells_carry_strengths_and_angle_operators():
    for campaign, paradigm in (("I4E2", "joint"), ("I4E2b", "marginal")):
        enum = campaigns.enumerate_campaign(campaign, app=APP)
        assert enum.candidates == 1_350
        assert {c.paradigm for c in enum.cells} == {paradigm}
        assert {c.d for c in enum.cells} <= set(runner.FRONTIER_D)
        assert {c.m for c in enum.cells} <= set(runner.FRONTIER_M)
        assert {c.strength for c in enum.cells} == set(runner.FRONTIER_STRENGTHS)
        assert {c.operator for c in enum.cells} <= set(runner.ANGLE_OPERATORS)


def test_i4e3_detector_cells_and_shot_free_screen_agree():
    enum = campaigns.enumerate_campaign("I4E3", app=APP)
    assert enum.candidates == 378
    assert {c.m for c in enum.cells} == {runner.E3_M}
    assert {c.methods for c in enum.cells} == {("chi_square",)}
    pairs = campaigns.screen_pairs(app=APP)
    assert len(pairs) == enum.n_cells
    assert {(p["circuit"], p["d"], p["operator"]) for p in pairs} == {
        (c.circuit, c.d, c.operator) for c in enum.cells
    }


def test_i7size_extends_the_m_ladder_with_the_matched_alternative():
    enum = campaigns.enumerate_campaign("I7size", app=APP)
    assert enum.candidates == 528
    assert {c.m for c in enum.cells} == set(campaigns.I7SIZE_M)
    assert {c.paradigm for c in enum.cells} == {"marginal"}
    assert {c.case for c in enum.cells} == {"baseline", "falsecase"}
    false = [c for c in enum.cells if c.case == "falsecase"]
    assert {c.operator for c in false} == {campaigns.DELTA_OPERATOR}
    assert {c.strength for c in false} == {campaigns.MATCHED_DELTA}
    assert all("exact_binomial" in c.methods for c in enum.cells)


def test_i7seq_cells_are_sequential_and_degenerate_nulls_are_excluded():
    enum = campaigns.enumerate_campaign("I7seq", app=APP)
    assert enum.candidates == 132
    assert all(campaigns.is_sequential_cell(c) for c in enum.cells)
    assert {c.m for c in enum.cells} == {campaigns.SEQ_HORIZON}
    assert {c.methods for c in enum.cells} == {campaigns.SEQ_METHODS}
    assert enum.by_reason().get("degenerate_marginal_null", 0) > 0
    assert all(not APP.marginal_degenerate(c.circuit, c.d) for c in enum.cells)


def test_i8_sweeps_the_permutation_budget_under_common_random_numbers():
    enum = campaigns.enumerate_campaign("I8", app=APP)
    assert enum.n_cells == enum.candidates == 36
    assert {c.circuit for c in enum.cells} == set(campaigns.SHAPE_ARMS)
    assert {c.case for c in enum.cells} == {"baseline"}
    names = {m for c in enum.cells for m in c.methods}
    assert names == {campaigns.mmdagg_method_name(B) for B in campaigns.I8_B}
    by_key = {}
    for c in enum.cells:
        by_key.setdefault((c.circuit, c.d, c.m), set()).add(runner.seed_master(c))
    assert all(len(v) == 1 for v in by_key.values())


def test_i9_enumeration_applies_the_c3_applicability_theorem():
    enum = campaigns.enumerate_campaign(campaigns.I9_CAMPAIGN, app=APP)
    assert enum.candidates == 540
    assert enum.by_reason().get("no_c3_alternative", 0) > 0
    for cell in enum.cells:
        assert APP.admits_c3(cell.circuit, cell.d) is True
        assert cell.strength == campaigns.I9_EPSILON
    assert all(c.circuit not in ("BV", "QPE") for c in enum.cells)
    assert all(set(runner.JOINT_METHODS) <= set(c.methods) for c in enum.cells)
    assert all(set(runner.MARGINAL_METHODS) & set(c.methods) for c in enum.cells)


def test_i9_drops_cells_whose_epsilon_cannot_be_built():
    app = campaigns.Applicability(source="probe")
    for circuit in campaigns.CIRCUITS:
        for d in runner.JOINT_D:
            app.widths[(circuit, d)] = APP.width_ok(circuit, d)
            info = APP.null(circuit, d)
            if info is None:
                continue
            reach = 0.5 * campaigns.I9_EPSILON if circuit == "QWALK" else 1.0
            app.nulls[(circuit, d)] = campaigns.NullInfo(
                support=info.support, max_prob=info.max_prob,
                n_interior=info.n_interior, d=info.d, c3_epsilon_max=reach,
            )
    enum = campaigns.enumerate_campaign(campaigns.I9_CAMPAIGN, app=app)
    assert enum.by_reason()["epsilon_above_constructible_max"] > 0
    assert not any(c.circuit == "QWALK" for c in enum.cells)


def test_i9_execution_hook_is_lazy_and_uses_the_real_constructor():
    i9_c3 = pytest.importorskip("harness.i9_c3")
    pytest.importorskip("qiskit")
    cell = next(c for c in campaigns.cells(campaigns.I9_CAMPAIGN, app=APP)
                if c.circuit == "QFT" and c.d == 3)
    p0, q = campaigns.distributions(cell)
    assert np.allclose(i9_c3.marginals(q), i9_c3.marginals(p0), atol=1e-12)
    assert i9_c3.tvd(p0, q) == pytest.approx(campaigns.I9_EPSILON, abs=1e-9)
    other_m = next(c for c in campaigns.cells(campaigns.I9_CAMPAIGN, app=APP)
                   if c.circuit == "QFT" and c.d == 3 and c.m != cell.m)
    assert campaigns.distributions(other_m)[1] == q




@pytest.mark.parametrize("arm", campaigns.SHAPE_ARMS)
def test_shape_arms_are_seeded_and_deterministic(arm):
    a = campaigns.shape_distribution(arm, 4, seed=7)
    b = campaigns.shape_distribution(arm, 4, seed=7)
    c = campaigns.shape_distribution(arm, 4, seed=8)
    assert a == b
    assert a != c
    assert abs(sum(a.values()) - 1.0) < 1e-12
    assert all(v >= 0 for v in a.values())
    assert all(len(k) == 4 for k in a)


def test_shape_arm_support_structure():
    dense = campaigns.shape_distribution("shapeA_dense", 5, seed=1)
    sparse = campaigns.shape_distribution("shapeB_sparse", 5, seed=1)
    ghz = campaigns.shape_distribution("shapeC_ghz", 5, seed=1)

    assert len(dense) == 2 ** 5 and min(dense.values()) > 0
    assert max(dense.values()) < 0.5

    assert len(sparse) == min(campaigns.I8_SPARSE_SUPPORT, 2 ** 5)
    assert max(sparse.values()) == pytest.approx(1 - campaigns.I8_SPARSE_TAIL)

    assert len(ghz) == 2
    (ka, va), (kb, vb) = ghz.items()
    assert va == vb == 0.5
    assert all(x != y for x, y in zip(ka, kb))


def test_bitflip_alternative_moves_every_marginal_by_exactly_delta():
    rng = np.random.default_rng(0)
    d, delta = 4, 0.02
    probs = rng.dirichlet(np.ones(2 ** d))
    p0 = {format(i, f"0{d}b"): float(p) for i, p in enumerate(probs)}
    q = campaigns.bitflip_shift_alternative(p0, d, delta)

    assert abs(sum(q.values()) - 1.0) < 1e-12
    assert min(q.values()) >= 0
    m0 = runner.marginal_null(p0, d)
    m1 = runner.marginal_null(q, d)
    assert np.allclose(np.abs(m1 - m0), delta, atol=1e-12)
    assert np.all((m1 > 0) & (m1 < 1))


def test_bitflip_alternative_is_defined_on_a_concentrated_null():
    d, delta = 3, 0.05
    p0 = {"000": 1.0}
    q = campaigns.bitflip_shift_alternative(p0, d, delta)
    assert q["000"] == pytest.approx((1 - delta) ** d)
    assert np.allclose(runner.marginal_null(q, d), delta)




def _uniform_p0(d):
    keys = [format(i, f"0{d}b") for i in range(1 << d)]
    return {k: 1.0 / len(keys) for k in keys}


def test_shot_source_replays_one_shot_sequence_for_every_method():
    p0 = _uniform_p0(3)
    src = campaigns.ShotSource(p0, 3, np.random.default_rng(0), block=8)
    first = src.cursor()(37).copy()
    assert first.shape == (37, 3)
    assert np.array_equal(src.prefix(37), first)
    stream = src.cursor()
    assert np.array_equal(np.vstack([stream(10), stream(27)]), first)
    assert src.n_drawn >= 37

    other = campaigns.ShotSource(p0, 3, np.random.default_rng(0), block=8)
    assert np.array_equal(other.prefix(37), first)
    assert not np.array_equal(
        campaigns.ShotSource(p0, 3, np.random.default_rng(1)).prefix(37), first
    )


def test_shot_source_draws_nothing_beyond_what_is_consumed():
    src = campaigns.ShotSource(_uniform_p0(2), 2, np.random.default_rng(0), block=16)
    src.cursor()(5)
    assert src.n_drawn == 16


def _stub_stop_at(n_stop, reject):

    def impl(source, *, p0_marg, alpha, horizon, delta, rng):
        used = 0
        stream = source.cursor()
        while used < n_stop:
            chunk = stream(10)
            if not len(chunk):
                break
            used += len(chunk)
        return {"p_value": 0.01 if reject else 0.5, "reject": reject, "n_used": used}

    return impl


def test_sequential_path_records_n_used_and_confusion_counts():
    cell = Cell(campaign="I7seq", circuit="QFT", d=3, m=200, paradigm="marginal",
                operator=campaigns.DELTA_OPERATOR, strength=campaigns.MATCHED_DELTA,
                case="falsecase", methods=("fast", "slow"), n_reps=25)
    p0 = _uniform_p0(3)
    impls = {"fast": _stub_stop_at(40, True), "slow": _stub_stop_at(150, False)}
    res = campaigns.run_sequential_cell(cell, p0, p0, method_impls=impls)

    assert res.seed_master == runner.seed_master(cell)
    fast, slow = res.methods["fast"], res.methods["slow"]
    assert (fast.TP, fast.FN, fast.FP, fast.TN) == (25, 0, 0, 0)
    assert (slow.TP, slow.FN, slow.FP, slow.TN) == (0, 25, 0, 0)
    assert fast.n_used_mean == 40 and slow.n_used_median == 150
    assert fast.n_evaluated == slow.n_evaluated == 25

    baseline = Cell(campaign="I7seq", circuit="QFT", d=3, m=200, paradigm="marginal",
                    case="baseline", methods=("fast",), n_reps=25)
    flipped = campaigns.run_sequential_cell(baseline, p0, None, method_impls=impls)
    assert (flipped.methods["fast"].FP, flipped.methods["fast"].TN) == (25, 0)


def test_sequential_path_is_deterministic_under_the_seed_policy():
    cell = Cell(campaign="I7seq", circuit="QFT", d=3, m=200, paradigm="marginal",
                case="baseline", methods=("probe",), n_reps=10)
    p0 = _uniform_p0(3)
    seen = []

    def recorder(source, *, p0_marg, alpha, horizon, delta, rng):
        rows = source.prefix(20)
        seen.append(rows.sum())
        return {"p_value": 1.0, "reject": False, "n_used": 20}

    a = campaigns.run_sequential_cell(cell, p0, None, method_impls={"probe": recorder})
    first = list(seen)
    seen.clear()
    b = campaigns.run_sequential_cell(cell, p0, None, method_impls={"probe": recorder})
    assert first == seen
    assert a.methods["probe"].TN == b.methods["probe"].TN == 10


def test_sequential_path_runs_the_real_designs():
    cell = Cell(campaign="I7seq", circuit="QFT", d=2, m=300, paradigm="marginal",
                case="baseline", methods=campaigns.SEQ_METHODS, n_reps=8)
    p0 = _uniform_p0(2)
    res = campaigns.run_sequential_cell(cell, p0, None, alpha=0.05, delta=0.2)

    seq, sprt = res.methods["sequential"], res.methods["sprt"]
    assert seq.n_evaluated == sprt.n_evaluated == 8
    assert 0 < seq.n_used_mean <= 300
    assert sprt.n_used_mean > 0
    assert seq.FP + seq.TN == 8 and sprt.FP + sprt.TN == 8
    assert np.isnan(sprt.mean_p)


def test_run_any_cell_dispatches_on_the_cell_kind():
    p0 = _uniform_p0(2)
    fixed = Cell(campaign="I4E1B", circuit="QFT", d=2, m=20, paradigm="marginal",
                 case="baseline", methods=("stub",), n_reps=5)

    def stub(draw, *, alpha, rng):
        assert draw.X.shape == (20, 2)
        return {"p_value": 0.5, "reject": False}

    assert not campaigns.is_sequential_cell(fixed)
    out = campaigns.run_any_cell(fixed, p0, None, method_impls={"stub": stub})
    assert out.methods["stub"].TN == 5

    seq = Cell(campaign="I7seq", circuit="QFT", d=2, m=50, paradigm="marginal",
               case="baseline", methods=("fast",), n_reps=5)
    assert campaigns.is_sequential_cell(seq)
    out = campaigns.run_any_cell(seq, p0, None, method_impls={"fast": _stub_stop_at(20, False)})
    assert out.methods["fast"].n_used_mean == 20


def test_run_cells_uses_the_campaign_distribution_provider():
    cells = [c for c in campaigns.cells("I8", app=APP) if c.d == 3 and c.m == 50][:1]
    if not cells:
        cells = [Cell(campaign="I8", circuit="shapeC_ghz", d=3, m=50, paradigm="joint",
                      case="baseline", methods=(campaigns.mmdagg_method_name(500),),
                      n_reps=2)]
    cell = Cell(campaign="I8", circuit=cells[0].circuit, d=3, m=40, paradigm="joint",
                case="baseline", methods=("stub",), n_reps=3)

    def stub(draw, *, alpha, rng):
        assert draw.counts.sum() == 40
        return {"p_value": 0.9, "reject": False}

    out = campaigns.run_cells([cell], method_impls={"stub": stub})
    assert out[0].methods["stub"].TN == 3




def test_register_with_runner_overrides_and_is_reversible():
    campaigns.register_with_runner()
    try:
        assert "I7seq" in runner.CAMPAIGNS and campaigns.I9_CAMPAIGN in runner.CAMPAIGNS
        realized = runner.enumerate_cells("I4E1", n_reps=1000)
        assert len(realized) == len(campaigns.cells("I4E1"))
        assert len(realized) <= 8_400
        assert callable(runner._default_method(campaigns.mmdagg_method_name(500)))
    finally:
        campaigns.unregister_from_runner()
    assert len(runner.enumerate_cells("I4E1", n_reps=1000)) == 8_400
    assert "I7seq" not in runner.CAMPAIGNS
    assert callable(runner._default_method(campaigns.mmdagg_method_name(500)))
    with pytest.raises(KeyError):
        runner._default_method("definitely_not_a_method")


def test_runner_cli_executes_a_sequential_job_end_to_end(tmp_path):
    pytest.importorskip("qiskit")
    from harness import io_store

    rc = runner.main(["--campaign", "I7seq", "--circuit", "QFT", "--d", "2",
                      "--operator", campaigns.DELTA_OPERATOR, "--n-reps", "3",
                      "--n-jobs", "1", "--out", str(tmp_path)])
    assert rc == 0
    df = io_store.read_grid(tmp_path)
    assert set(df["method"]) == set(campaigns.SEQ_METHODS)
    assert (df["TP"] + df["FN"] == 3).all()
    assert (df["n_used_mean"] > 0).all()
    assert (df["m"] == campaigns.SEQ_HORIZON).all()
    seq = df[df["method"] == "sequential"].iloc[0]
    assert seq["n_used_mean"] <= campaigns.SEQ_HORIZON


def test_runner_cli_executes_an_i8_shape_arm_job(tmp_path):
    from harness import io_store

    rc = runner.main(["--campaign", "I8", "--circuit", "shapeC_ghz", "--d", "3",
                      "--m", "200", "--n-reps", "2", "--n-jobs", "1",
                      "--out", str(tmp_path)])
    assert rc == 0
    df = io_store.read_grid(tmp_path)
    assert set(df["method"]) == {campaigns.mmdagg_method_name(B) for B in campaigns.I8_B}
    assert (df["FP"] + df["TN"] == 2).all()
    assert df["seed_master"].nunique() == 1


def test_applicability_cache_round_trip_and_staleness(tmp_path):
    path = campaigns.write_applicability(APP, tmp_path / "app.json")
    back = campaigns.load_applicability(path)
    assert back.width_ok("Adder", 5) is False and back.width_ok("Adder", 7) is True
    assert back.operator_ok("BV", 6, "O1") is False
    assert back.null("QFT", 4).n_interior == 4
    assert not back.stale

    blob = json.loads(path.read_text())
    blob["meta"]["module_digest"] = "0" * 64
    path.write_text(json.dumps(blob))
    assert campaigns.load_applicability(path).stale is True

    missing = campaigns.load_applicability(tmp_path / "nope.json")
    assert missing.source == "structural"
    with pytest.raises(FileNotFoundError):
        campaigns.load_applicability(tmp_path / "nope.json", required=True)




def test_probe_applicability_smoke():
    pytest.importorskip("qiskit")
    app = campaigns.probe_applicability(("BV", "Adder"), (2, 3, 4), ("O1", "O2"))
    assert app.source == "probe"
    assert app.widths[("Adder", 2)] is False and app.widths[("Adder", 4)] is True
    assert app.widths[("BV", 3)] is True
    assert app.operators[("BV", 3, "O1")] is False
    assert app.operators[("BV", 3, "O2")] is True
    assert "OperatorNotApplicable" in app.reason("BV", "3", "O1")
    assert app.null("BV", 3).support == 1
    assert app.null("BV", 3).admits_c3 is False


def test_run_screen_smoke():
    pytest.importorskip("qiskit")
    rows = campaigns.run_screen(
        [{"circuit": "QFT", "d": 3, "operator": "O5", "strength": 0.4},
         {"circuit": "QFT", "d": 3, "operator": "O1", "strength": 0.0}],
        app=APP,
    )
    assert len(rows) == 2
    assert all(0.0 <= r["fidelity"] <= 1.0 + 1e-9 for r in rows)
    assert rows[1]["fidelity"] == pytest.approx(1.0)
    assert rows[1]["undetectable"] is True
