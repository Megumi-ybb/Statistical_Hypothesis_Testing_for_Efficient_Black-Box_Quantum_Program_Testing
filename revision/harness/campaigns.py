
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from harness import runner
from harness.runner import (
    ANGLE_OPERATORS,
    BASELINE_OPERATOR,
    CIRCUITS,
    DEFAULT_ALPHA,
    DEFAULT_N_REPS,
    E3_M,
    FRONTIER_D,
    FRONTIER_M,
    FRONTIER_STRENGTHS,
    JOINT_D,
    JOINT_METHODS,
    M_POINTS,
    MARGINAL_D,
    MARGINAL_METHODS,
    OPERATORS,
    Cell,
    CellResult,
    MethodResult,
)


I7SIZE_M: tuple[int, ...] = (2500, 3000, 4000, 5000)

MATCHED_DELTA: float = 0.02

SEQ_HORIZON: int = 6500

SPRT_BATCH: int = 50
SEQ_MAX_N: int = 50_000

SEQ_METHODS: tuple[str, ...] = ("sequential", "sequential_exact", "sprt")

I7SIZE_METHODS: tuple[str, ...] = ("exact_binomial", "g_test_marginal", "adaptive")

SHAPE_ARMS: tuple[str, ...] = ("shapeA_dense", "shapeB_sparse", "shapeC_ghz")
I8_D: tuple[int, ...] = (3, 6, 9)
I8_M: tuple[int, ...] = (200, 1000)
I8_B: tuple[int, ...] = (500, 2000)
I8_SPARSE_SUPPORT: int = 16
I8_SPARSE_TAIL: float = 0.05

I9_CAMPAIGN = "I9C3"
I9_OPERATOR = "c3dep"
I9_METHODS: tuple[str, ...] = tuple(JOINT_METHODS) + tuple(MARGINAL_METHODS)
I9_EPSILON: float = 0.10

DELTA_OPERATOR = "deltashift"

SEQUENTIAL_CAMPAIGNS: frozenset[str] = frozenset({"I7seq"})

INTERIOR_METHODS: frozenset[str] = frozenset({"bootstrap", "sequential", "sequential_exact"})

PROBE_SEED_CAMPAIGN = "I4E1"

DEFAULT_CACHE = Path(__file__).with_name("applicability.json")

PLAN_BOUNDS: dict[str, tuple[int | None, int | None]] = {
    "I4E1": (8_400, 8_400_000),
    "I4E1B": (None, None),
    "I4E2": (1_350, 1_350_000),
    "I4E2b": (1_350, 1_350_000),
    "I4E3": (378, 378_000),
    "I7size": (528, 528_000),
    "I7seq": (132, 132_000),
    "I8": (36, 36_000),
    I9_CAMPAIGN: (540, 540_000),
}



STRUCTURAL_WIDTHS: dict[str, Callable[[int], bool]] = {
    "BV": lambda d: d >= 2,
    "Grover": lambda d: d >= 2,
    "QFT": lambda d: d >= 1,
    "QPE": lambda d: d >= 1,
    "QWALK": lambda d: d >= 2,
    "Adder": lambda d: d >= 4 and d % 3 == 1,
}

NEWLY_CONSTRUCTED: frozenset[str] = frozenset({"QWALK"})

_DEGENERATE_TOL = 1e-12


@dataclass(frozen=True)
class NullInfo:

    support: int
    max_prob: float
    n_interior: int
    d: int
    c3_epsilon_max: float = float("nan")

    @property
    def marginal_degenerate(self) -> bool:
        return self.n_interior < self.d

    @property
    def admits_c3(self) -> bool:
        return self.n_interior >= 2

    def to_json(self) -> dict:
        return {"support": self.support, "max_prob": self.max_prob,
                "n_interior": self.n_interior, "d": self.d,
                "c3_epsilon_max": self.c3_epsilon_max}


@dataclass
class Applicability:

    widths: dict[tuple[str, int], bool] = field(default_factory=dict)
    operators: dict[tuple[str, int, str], bool] = field(default_factory=dict)
    nulls: dict[tuple[str, int], NullInfo] = field(default_factory=dict)
    reasons: dict[tuple, str] = field(default_factory=dict)
    source: str = "structural"
    stale: bool = False
    meta: dict = field(default_factory=dict)

    def width_ok(self, circuit: str, d: int) -> bool:
        if circuit in SHAPE_ARMS:
            return True
        key = (circuit, int(d))
        if key in self.widths:
            return self.widths[key]
        rule = STRUCTURAL_WIDTHS.get(circuit)
        return bool(rule(int(d))) if rule else True

    def operator_ok(self, circuit: str, d: int, operator: str) -> bool:
        if operator in (BASELINE_OPERATOR, DELTA_OPERATOR, I9_OPERATOR):
            return True
        key = (circuit, int(d), operator)
        if key in self.operators:
            return self.operators[key]
        return True

    def null(self, circuit: str, d: int) -> NullInfo | None:
        return self.nulls.get((circuit, int(d)))

    def marginal_degenerate(self, circuit: str, d: int) -> bool:
        info = self.null(circuit, d)
        return bool(info.marginal_degenerate) if info else False

    def admits_c3(self, circuit: str, d: int) -> bool | None:
        info = self.null(circuit, d)
        return None if info is None else info.admits_c3

    def reason(self, *key) -> str:
        return self.reasons.get(tuple(key), "")

    @property
    def probed(self) -> bool:
        return self.source != "structural"

    def to_json(self) -> dict:
        return {
            "meta": self.meta,
            "widths": {f"{c}|{d}": ok for (c, d), ok in sorted(self.widths.items())},
            "operators": {
                f"{c}|{d}|{o}": ok for (c, d, o), ok in sorted(self.operators.items())
            },
            "nulls": {f"{c}|{d}": n.to_json() for (c, d), n in sorted(self.nulls.items())},
            "reasons": {"|".join(str(p) for p in k): v for k, v in sorted(self.reasons.items())},
        }

    @classmethod
    def from_json(cls, blob: Mapping, *, source: str) -> "Applicability":
        widths = {}
        for key, ok in blob.get("widths", {}).items():
            c, d = key.split("|")
            widths[(c, int(d))] = bool(ok)
        operators = {}
        for key, ok in blob.get("operators", {}).items():
            c, d, o = key.split("|")
            operators[(c, int(d), o)] = bool(ok)
        nulls = {}
        for key, info in blob.get("nulls", {}).items():
            c, d = key.split("|")
            nulls[(c, int(d))] = NullInfo(
                support=int(info["support"]),
                max_prob=float(info["max_prob"]),
                n_interior=int(info["n_interior"]),
                d=int(info["d"]),
                c3_epsilon_max=float(info.get("c3_epsilon_max", float("nan"))),
            )
        reasons = {tuple(k.split("|")): v for k, v in blob.get("reasons", {}).items()}
        return cls(widths=widths, operators=operators, nulls=nulls, reasons=reasons,
                   source=source, meta=dict(blob.get("meta", {})))


def _module_digest() -> str:
    h = hashlib.sha256()
    for name in ("base_circuits.py", "mutants.py"):
        path = Path(__file__).with_name(name)
        h.update(path.read_bytes() if path.exists() else b"")
    return h.hexdigest()


def probe_applicability(
    circuits: Sequence[str] = CIRCUITS,
    widths: Sequence[int] = MARGINAL_D,
    operators: Sequence[str] = OPERATORS,
    *,
    probe_nulls: bool = True,
    seed_campaign: str = PROBE_SEED_CAMPAIGN,
) -> Applicability:
    from harness import base_circuits, mutants

    app = Applicability(source="probe")
    app.meta = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seed_campaign": seed_campaign,
        "circuits": list(circuits),
        "widths": [int(d) for d in widths],
        "operators": list(operators),
        "probe_nulls": bool(probe_nulls),
        "module_digest": _module_digest(),
    }
    for circuit in circuits:
        for d in widths:
            d = int(d)
            if not STRUCTURAL_WIDTHS.get(circuit, lambda _d: True)(d):
                app.widths[(circuit, d)] = False
                app.reasons[(circuit, str(d))] = "family width rule (G3): no benchmark instance"
                continue
            seed = runner.circuit_seed(
                Cell(campaign=seed_campaign, circuit=circuit, d=d, m=1,
                     operator=BASELINE_OPERATOR, case="baseline")
            )
            try:
                spec = base_circuits.build(circuit, d, seed=seed)
            except base_circuits.UnsupportedWidth as exc:
                app.widths[(circuit, d)] = False
                app.reasons[(circuit, str(d))] = f"UnsupportedWidth: {exc}"
                continue
            app.widths[(circuit, d)] = True

            for operator in operators:
                try:
                    mutants.mutate(spec, operator,
                                   rng=np.random.default_rng(abs(seed) % (2**32)))
                except mutants.OperatorNotApplicable as exc:
                    app.operators[(circuit, d, operator)] = False
                    app.reasons[(circuit, str(d), operator)] = f"OperatorNotApplicable: {exc}"
                else:
                    app.operators[(circuit, d, operator)] = True

            if probe_nulls:
                from harness import oracles

                p0 = oracles.p0_from_spec(spec)
                probs = np.fromiter(p0.values(), dtype=float, count=len(p0))
                marg = runner.marginal_null(p0, d)
                n_interior = int(
                    ((marg > _DEGENERATE_TOL) & (marg < 1 - _DEGENERATE_TOL)).sum()
                )
                app.nulls[(circuit, d)] = NullInfo(
                    support=int((probs > _DEGENERATE_TOL).sum()),
                    max_prob=float(probs.max()),
                    n_interior=n_interior,
                    d=d,
                    c3_epsilon_max=_c3_epsilon_max(p0, seed),
                )
    return app


def _c3_epsilon_max(p0: Mapping[str, float], seed: int) -> float:
    try:
        from harness import i9_c3
    except ImportError:
        return float("nan")
    verdict = i9_c3.admissibility(p0)
    marg = np.asarray(runner.marginal_null(p0, len(next(iter(p0)))), dtype=float)
    mine = int(((marg > _DEGENERATE_TOL) & (marg < 1 - _DEGENERATE_TOL)).sum()) >= 2
    if bool(verdict.admissible) != mine:
        raise RuntimeError(
            f"C3 admissibility disagreement: i9_c3 says {verdict.admissible} "
            f"({verdict.reason}), campaigns' interior-coordinate rule says {mine}"
        )
    if not verdict.admissible:
        return 0.0
    return float(i9_c3.constructible_epsilon(p0, np.random.default_rng(int(seed))))


def write_applicability(app: Applicability, path: str | Path = DEFAULT_CACHE) -> Path:
    path = Path(path)
    path.write_text(json.dumps(app.to_json(), indent=1, sort_keys=True) + "\n")
    return path


def load_applicability(path: str | Path | None = None, *, required: bool = False
                       ) -> Applicability:
    path = Path(DEFAULT_CACHE if path is None else path)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"no applicability cache at {path}; run "
                f"`python -m harness.campaigns --probe` first"
            )
        return Applicability(source="structural")
    app = Applicability.from_json(json.loads(path.read_text()), source=f"cache:{path.name}")
    app.stale = app.meta.get("module_digest") not in (None, _module_digest())
    return app


_APP_CACHE: dict[str, Applicability] = {}


def default_applicability() -> Applicability:
    if "app" not in _APP_CACHE:
        _APP_CACHE["app"] = load_applicability()
    return _APP_CACHE["app"]




@dataclass
class Enumeration:

    campaign: str
    cells: list[Cell] = field(default_factory=list)
    exclusions: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    plan_cells: int | None = None
    plan_runs: int | None = None
    candidates: int = 0

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_runs(self) -> int:
        return sum(c.n_runs for c in self.cells)

    def by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for exc in self.exclusions:
            if exc["level"] == "cell":
                out[exc["reason"]] = out.get(exc["reason"], 0) + 1
        return out

    def method_drops(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for exc in self.exclusions:
            if exc["level"] == "method":
                key = f"{exc['method']}:{exc['reason']}"
                out[key] = out.get(key, 0) + 1
        return out


def cell_provenance(cell: Cell, app: Applicability | None = None) -> dict:
    app = app or default_applicability()
    return {
        "newly_constructed": cell.circuit in NEWLY_CONSTRUCTED,
        "synthetic_null": cell.circuit in SHAPE_ARMS,
        "marginal_degenerate": app.marginal_degenerate(cell.circuit, cell.d),
        "admits_c3": app.admits_c3(cell.circuit, cell.d),
        "applicability_source": app.source,
    }


def _filter_methods(methods: Sequence[str], circuit: str, d: int, app: Applicability
                    ) -> tuple[tuple[str, ...], list[str]]:
    if not app.marginal_degenerate(circuit, d):
        return tuple(methods), []
    kept = tuple(m for m in methods if m not in INTERIOR_METHODS)
    dropped = [m for m in methods if m in INTERIOR_METHODS]
    return kept, dropped


def _finalize(campaign: str, candidates: Iterable[Mapping[str, Any]], *,
              n_reps: int, app: Applicability, enum: Enumeration,
              cell_level_veto: Callable[[Mapping, Applicability], str | None] | None = None,
              ) -> None:
    for kw in candidates:
        enum.candidates += 1
        kw = dict(kw)
        circuit, d = kw["circuit"], int(kw["d"])
        operator = kw.get("operator", BASELINE_OPERATOR)

        if not app.width_ok(circuit, d):
            enum.exclusions.append({
                "level": "cell", "circuit": circuit, "d": d, "operator": operator,
                "reason": "unsupported_width",
                "detail": app.reason(circuit, str(d)) or "family width rule",
            })
            continue
        if not app.operator_ok(circuit, d, operator):
            enum.exclusions.append({
                "level": "cell", "circuit": circuit, "d": d, "operator": operator,
                "reason": "operator_not_applicable",
                "detail": app.reason(circuit, str(d), operator),
            })
            continue
        if cell_level_veto is not None:
            veto = cell_level_veto(kw, app)
            if veto:
                enum.exclusions.append({
                    "level": "cell", "circuit": circuit, "d": d, "operator": operator,
                    "reason": veto, "detail": "",
                })
                continue

        methods = kw.pop("methods", None)
        if methods is None:
            methods = (JOINT_METHODS if kw.get("paradigm", "joint") == "joint"
                       else MARGINAL_METHODS)
        methods, dropped = _filter_methods(methods, circuit, d, app)
        for method in dropped:
            enum.exclusions.append({
                "level": "method", "circuit": circuit, "d": d, "operator": operator,
                "method": method, "reason": "degenerate_marginal_null",
                "detail": "p0 has a deterministic qubit; test requires p0 in (0,1)",
            })
        if not methods:
            enum.exclusions.append({
                "level": "cell", "circuit": circuit, "d": d, "operator": operator,
                "reason": "no_runnable_method", "detail": "",
            })
            continue
        enum.cells.append(Cell(campaign=campaign, n_reps=n_reps, methods=methods, **kw))




def _cand_e1() -> Iterator[dict]:
    for paradigm, ds in (("joint", JOINT_D), ("marginal", MARGINAL_D)):
        for circuit in CIRCUITS:
            for d in ds:
                for m in M_POINTS:
                    for operator in OPERATORS:
                        yield {"circuit": circuit, "d": d, "m": m, "paradigm": paradigm,
                               "operator": operator, "case": "falsecase"}


def _cand_e1b() -> Iterator[dict]:
    for paradigm, ds in (("joint", JOINT_D), ("marginal", MARGINAL_D)):
        for circuit in CIRCUITS:
            for d in ds:
                for m in M_POINTS:
                    yield {"circuit": circuit, "d": d, "m": m, "paradigm": paradigm,
                           "operator": BASELINE_OPERATOR, "case": "baseline"}


def _cand_frontier(paradigm: str) -> Iterator[dict]:
    for circuit in CIRCUITS:
        for d in FRONTIER_D:
            for m in FRONTIER_M:
                for operator in ANGLE_OPERATORS:
                    for strength in FRONTIER_STRENGTHS:
                        yield {"circuit": circuit, "d": d, "m": m, "paradigm": paradigm,
                               "operator": operator, "strength": strength,
                               "case": "falsecase"}


def _cand_e3() -> Iterator[dict]:
    for circuit in CIRCUITS:
        for d in JOINT_D:
            for operator in OPERATORS:
                yield {"circuit": circuit, "d": d, "m": E3_M, "paradigm": "joint",
                       "operator": operator, "case": "falsecase",
                       "methods": ("chi_square",)}


def _cand_i7size() -> Iterator[dict]:
    for circuit in CIRCUITS:
        for d in MARGINAL_D:
            for m in I7SIZE_M:
                for case in ("baseline", "falsecase"):
                    kw = {"circuit": circuit, "d": d, "m": m, "paradigm": "marginal",
                          "case": case, "methods": I7SIZE_METHODS}
                    if case == "falsecase":
                        kw["operator"] = DELTA_OPERATOR
                        kw["strength"] = MATCHED_DELTA
                    yield kw


def _cand_i7seq() -> Iterator[dict]:
    for circuit in CIRCUITS:
        for d in MARGINAL_D:
            for case in ("baseline", "falsecase"):
                kw = {"circuit": circuit, "d": d, "m": SEQ_HORIZON,
                      "paradigm": "marginal", "case": case, "methods": SEQ_METHODS}
                if case == "falsecase":
                    kw["operator"] = DELTA_OPERATOR
                    kw["strength"] = MATCHED_DELTA
                yield kw


def _cand_i8() -> Iterator[dict]:
    for arm in SHAPE_ARMS:
        for d in I8_D:
            for m in I8_M:
                for B in I8_B:
                    yield {"circuit": arm, "d": d, "m": m, "paradigm": "joint",
                           "operator": BASELINE_OPERATOR, "case": "baseline",
                           "methods": (mmdagg_method_name(B),)}


def _cand_i9(epsilons: Sequence[float] = (I9_EPSILON,)) -> Iterator[dict]:
    for epsilon in epsilons:
        for circuit in CIRCUITS:
            for d in JOINT_D:
                for m in M_POINTS:
                    yield {"circuit": circuit, "d": d, "m": m, "paradigm": "joint",
                           "operator": I9_OPERATOR, "strength": float(epsilon),
                           "case": "falsecase", "methods": I9_METHODS}


def mmdagg_method_name(n_permute: int) -> str:
    return f"mmdagg_B{int(n_permute)}"




def _veto_i7seq(kw: Mapping, app: Applicability) -> str | None:
    return "degenerate_marginal_null" if app.marginal_degenerate(kw["circuit"], kw["d"]) else None


def _veto_i9(kw: Mapping, app: Applicability) -> str | None:
    admits = app.admits_c3(kw["circuit"], kw["d"])
    if admits is None:
        return None
    if not admits:
        return "no_c3_alternative"
    info = app.null(kw["circuit"], kw["d"])
    eps_max = info.c3_epsilon_max if info else float("nan")
    epsilon = float(kw.get("strength") or I9_EPSILON)
    if np.isfinite(eps_max) and eps_max < epsilon:
        return "epsilon_above_constructible_max"
    return None


ENUMERATORS: dict[str, tuple[Callable[[], Iterator[dict]], Callable | None]] = {
    "I4E1": (_cand_e1, None),
    "I4E1B": (_cand_e1b, None),
    "I4E2": (lambda: _cand_frontier("joint"), None),
    "I4E2b": (lambda: _cand_frontier("marginal"), None),
    "I4E3": (_cand_e3, None),
    "I7size": (_cand_i7size, None),
    "I7seq": (_cand_i7seq, _veto_i7seq),
    "I8": (_cand_i8, None),
    I9_CAMPAIGN: (_cand_i9, _veto_i9),
}

CAMPAIGN_ORDER: tuple[str, ...] = tuple(ENUMERATORS)


def enumerate_campaign(campaign: str, *, n_reps: int = DEFAULT_N_REPS,
                       app: Applicability | None = None, **filters) -> Enumeration:
    if campaign not in ENUMERATORS:
        raise KeyError(f"unknown campaign {campaign!r}; known: {sorted(ENUMERATORS)}")
    app = app or default_applicability()
    candidates, veto = ENUMERATORS[campaign]
    plan_cells, plan_runs = PLAN_BOUNDS.get(campaign, (None, None))
    enum = Enumeration(campaign=campaign, plan_cells=plan_cells, plan_runs=plan_runs)
    _finalize(campaign, candidates(), n_reps=n_reps, app=app, enum=enum,
              cell_level_veto=veto)

    if not app.probed:
        enum.notes.append(
            "applicability source = structural rules only; counts are UPPER BOUNDS "
            "(run `python -m harness.campaigns --probe`)"
        )
    elif app.stale:
        enum.notes.append(
            "applicability cache predates the current base_circuits.py/mutants.py "
            "(module digest mismatch) -- re-probe before the campaign run"
        )
    if campaign == I9_CAMPAIGN and not any(app.admits_c3(c, d) is not None
                                           for c, d in app.nulls):
        enum.notes.append("C3 admissibility unprobed: cells are the plan upper bound")
    n_new = sum(1 for c in enum.cells if c.circuit in NEWLY_CONSTRUCTED)
    if n_new:
        enum.notes.append(f"{n_new:,d} cells use the newly constructed QWALK (disclose)")

    if filters:
        enum.cells = [c for c in enum.cells
                      if all(getattr(c, k) == v for k, v in filters.items() if v is not None)]
    return enum


def cells(campaign: str, *, n_reps: int = DEFAULT_N_REPS,
          app: Applicability | None = None, **filters) -> list[Cell]:
    return enumerate_campaign(campaign, n_reps=n_reps, app=app, **filters).cells


def screen_pairs(*, app: Applicability | None = None) -> list[dict]:
    app = app or default_applicability()
    enum = enumerate_campaign("I4E3", n_reps=1, app=app)
    return [{"circuit": c.circuit, "d": c.d, "operator": c.operator,
             "strength": c.strength} for c in enum.cells]


def run_screen(pairs: Sequence[Mapping] | None = None, *, limit: int | None = None,
               app: Applicability | None = None) -> list[dict]:
    from harness import base_circuits, mutants, oracles

    app = app or default_applicability()
    pairs = list(screen_pairs(app=app) if pairs is None else pairs)
    if limit is not None:
        pairs = pairs[:limit]
    rows = []
    for pair in pairs:
        cell = Cell(campaign="I4E3", circuit=pair["circuit"], d=int(pair["d"]), m=E3_M,
                    paradigm="joint", operator=pair["operator"],
                    strength=pair.get("strength"), case="falsecase",
                    methods=("chi_square",))
        spec = base_circuits.build(cell.circuit, cell.d, seed=runner.circuit_seed(cell))
        mutant = mutants.mutate(spec, cell.operator, strength=cell.strength,
                                rng=np.random.default_rng(runner.mutant_seed(cell)))
        rows.append({
            **pair,
            "fidelity": oracles.fidelity(spec, mutant),
            "tvd": oracles.tvd(spec, mutant),
            "equivalent": oracles.is_equivalent(spec, mutant),
            "undetectable": oracles.is_undetectable(spec, mutant),
            "newly_constructed": cell.circuit in NEWLY_CONSTRUCTED,
        })
    return rows




def manifest(*, n_reps: int = DEFAULT_N_REPS, app: Applicability | None = None,
             campaigns: Sequence[str] = CAMPAIGN_ORDER) -> dict:
    app = app or default_applicability()
    entries = []
    for name in campaigns:
        enum = enumerate_campaign(name, n_reps=n_reps, app=app)
        entries.append({
            "campaign": name,
            "cells": enum.n_cells,
            "runs": enum.n_runs,
            "plan_cells": enum.plan_cells,
            "plan_runs": enum.plan_runs,
            "candidates": enum.candidates,
            "excluded_cells": enum.candidates - enum.n_cells,
            "by_reason": enum.by_reason(),
            "method_drops": enum.method_drops(),
            "newly_constructed_cells": sum(1 for c in enum.cells
                                           if c.circuit in NEWLY_CONSTRUCTED),
            "sequential": name in SEQUENTIAL_CAMPAIGNS,
            "notes": enum.notes,
        })
    return {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "n_reps": int(n_reps),
        "applicability_source": app.source,
        "applicability_stale": bool(app.stale),
        "campaigns": entries,
        "totals": {
            "cells": sum(e["cells"] for e in entries),
            "runs": sum(e["runs"] for e in entries),
            "plan_cells": sum(e["plan_cells"] or 0 for e in entries),
            "plan_runs": sum(e["plan_runs"] or 0 for e in entries),
        },
        "shot_free_screen_pairs": len(screen_pairs(app=app)),
    }


def manifest_rows(**kwargs) -> list[dict]:
    return list(manifest(**kwargs)["campaigns"])


def format_manifest(man: Mapping) -> str:
    head = (f"{'campaign':<8} {'cells':>7} {'plan':>7} {'runs':>12} {'plan runs':>12} "
            f"{'excl':>6}  exclusions / notes")
    lines = [
        f"applicability: {man['applicability_source']}"
        + ("  [STALE]" if man.get("applicability_stale") else ""),
        f"n_reps: {man['n_reps']:,d}",
        "",
        head,
        "-" * len(head),
    ]
    for e in man["campaigns"]:
        detail = ", ".join(f"{k}={v:,d}" for k, v in sorted(e["by_reason"].items()))
        drops = ", ".join(f"-{k}({v:,d})" for k, v in sorted(e["method_drops"].items()))
        bits = " ".join(x for x in (detail, drops) if x)
        plan_c = f"{e['plan_cells']:,d}" if e["plan_cells"] is not None else "-"
        plan_r = f"{e['plan_runs']:,d}" if e["plan_runs"] is not None else "-"
        lines.append(
            f"{e['campaign']:<8} {e['cells']:>7,d} {plan_c:>7} "
            f"{e['runs']:>12,d} {plan_r:>12} "
            f"{e['excluded_cells']:>6,d}  {bits}"
        )
        for note in e["notes"]:
            lines.append(f"{'':<8} {'':>7} {'':>7} {'':>12} {'':>12} {'':>6}  ! {note}")
    t = man["totals"]
    unpriced = [e["campaign"] for e in man["campaigns"] if e["plan_cells"] is None]
    lines += [
        "-" * len(head),
        f"{'TOTAL':<8} {t['cells']:>7,d} {t['plan_cells']:>7,d} "
        f"{t['runs']:>12,d} {t['plan_runs']:>12,d}",
        "",
        f"shot-free E3 screen pairs: {man['shot_free_screen_pairs']:,d}",
    ]
    if unpriced:
        lines.append(
            "plan columns exclude " + ", ".join(unpriced)
            + " (never priced separately in impl_data.json)"
        )
    return "\n".join(lines)




def _keys(d: int) -> list[str]:
    return [format(i, f"0{d}b") for i in range(1 << d)]


def shape_distribution(arm: str, d: int, *, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    n = 1 << int(d)
    keys = _keys(int(d))
    if arm == "shapeA_dense":
        amps = rng.normal(size=n) + 1j * rng.normal(size=n)
        probs = np.abs(amps) ** 2
        probs /= probs.sum()
        return {k: float(p) for k, p in zip(keys, probs)}
    if arm == "shapeB_sparse":
        support = min(I8_SPARSE_SUPPORT, n)
        idx = rng.choice(n, size=support, replace=False)
        probs = np.zeros(n)
        probs[idx[0]] = 1.0 - I8_SPARSE_TAIL
        if support > 1:
            tail = rng.dirichlet(np.ones(support - 1)) * I8_SPARSE_TAIL
            probs[idx[1:]] = tail
        else:
            probs[idx[0]] = 1.0
        return {keys[i]: float(probs[i]) for i in np.flatnonzero(probs)}
    if arm == "shapeC_ghz":
        bits = rng.integers(0, 2, size=int(d))
        a = "".join(str(int(b)) for b in bits)
        b = "".join(str(1 - int(x)) for x in bits)
        return {a: 0.5, b: 0.5}
    raise ValueError(f"unknown shape arm {arm!r}; expected one of {SHAPE_ARMS}")


def bitflip_shift_alternative(p0: Mapping[str, float], d: int, delta: float
                              ) -> dict[str, float]:
    d = int(d)
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    marg = runner.marginal_null(p0, d)
    vec = np.zeros(1 << d, dtype=float)
    for key, prob in p0.items():
        vec[int(key, 2)] = prob
    total = vec.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("p0 does not sum to a positive finite value")
    vec /= total

    tensor = vec.reshape((2,) * d)
    for j in range(d):
        p_j = float(marg[j])
        if p_j <= 0.5:
            a = delta / (1.0 - p_j)
            T = np.array([[1.0 - a, a], [0.0, 1.0]])
        else:
            b = delta / p_j
            T = np.array([[1.0, 0.0], [b, 1.0 - b]])
        tensor = np.moveaxis(np.moveaxis(tensor, j, -1) @ T, -1, j)
    out = np.asarray(tensor).reshape(-1)
    return {k: float(v) for k, v in zip(_keys(d), out)}



_DIST_CACHE: dict[tuple, tuple[dict, dict | None]] = {}


def distributions(cell: Cell) -> tuple[Mapping[str, float], Mapping[str, float] | None]:
    key = (cell.campaign, cell.circuit, cell.d, cell.operator, cell.strength, cell.case)
    if key in _DIST_CACHE:
        return _DIST_CACHE[key]

    if cell.circuit in SHAPE_ARMS:
        p0 = shape_distribution(cell.circuit, cell.d, seed=runner.circuit_seed(cell))
        if cell.case != "baseline":
            raise ValueError(f"{cell.circuit} arms are Type I only, got case={cell.case!r}")
        out = (p0, None)
    elif cell.operator == I9_OPERATOR:
        out = _i9_distributions(cell)
    elif cell.operator == DELTA_OPERATOR:
        from harness import base_circuits, oracles

        spec = base_circuits.build(cell.circuit, cell.d, seed=runner.circuit_seed(cell))
        p0 = oracles.p0_from_spec(spec)
        delta = MATCHED_DELTA if cell.strength is None else float(cell.strength)
        out = (p0, bitflip_shift_alternative(p0, cell.d, delta))
    else:
        out = runner.cell_distributions(cell)

    _DIST_CACHE[key] = out
    return out


def _i9_distributions(cell: Cell) -> tuple[Mapping[str, float], Mapping[str, float]]:
    try:
        from harness import i9_c3
    except ImportError as exc:
        raise RuntimeError(
            "I9C3 cells need harness/i9_c3.py (dependence-only alternative "
            f"construction); enumeration works without it, execution does not: {exc}"
        ) from exc

    p0 = i9_c3.benchmark_null(cell.circuit, cell.d, seed=runner.circuit_seed(cell))
    epsilon = I9_EPSILON if cell.strength is None else float(cell.strength)
    built = i9_c3.make_dependence_only(
        p0, epsilon, np.random.default_rng(runner.mutant_seed(cell))
    )
    if built.clipped:
        raise RuntimeError(
            f"{cell.circuit} d={cell.d}: requested epsilon {epsilon} clipped to "
            f"{built.epsilon:.4f}; re-probe (--probe records the constructible max) "
            f"so the enumeration drops the cell instead of shrinking the fault"
        )
    return p0, built.q




def _mmdagg_impl(n_permute: int) -> Callable[..., dict]:
    def call(draw, *, alpha: float, rng: np.random.Generator) -> dict:
        from harness import stats_joint

        return stats_joint.mmdagg_counts(draw.states, draw.counts, draw.p0,
                                         alpha=alpha, n_permute=int(n_permute), rng=rng)

    return call


EXTRA_METHOD_IMPLS: dict[str, Callable[..., dict]] = {
    mmdagg_method_name(B): _mmdagg_impl(B) for B in I8_B
}


_RUNNER_ORIGINALS: dict[str, Callable | None] = {}


def register_with_runner(*, override: bool = True) -> None:
    runner.EXTRA_METHODS.update(EXTRA_METHOD_IMPLS)
    for name in ENUMERATORS:
        if override or name not in runner.CAMPAIGNS:
            _RUNNER_ORIGINALS.setdefault(name, runner.CAMPAIGNS.get(name))
            runner.CAMPAIGNS[name] = _runner_adapter(name)


def unregister_from_runner() -> None:
    for name, original in _RUNNER_ORIGINALS.items():
        if original is None:
            runner.CAMPAIGNS.pop(name, None)
        else:
            runner.CAMPAIGNS[name] = original
    _RUNNER_ORIGINALS.clear()
    for name in EXTRA_METHOD_IMPLS:
        runner.EXTRA_METHODS.pop(name, None)


def _runner_adapter(name: str) -> Callable[[str, int], Iterator[Cell]]:
    def gen(campaign: str, n_reps: int) -> Iterator[Cell]:
        return iter(cells(campaign, n_reps=n_reps))

    gen.__name__ = f"campaigns_{name}"
    return gen




def is_sequential_cell(cell: Cell) -> bool:
    return cell.campaign in SEQUENTIAL_CAMPAIGNS or "sprt" in cell.methods


class ShotSource:

    def __init__(self, p_draw: Mapping[str, float], d: int,
                 rng: np.random.Generator, *, block: int = 512):
        keys = [k for k, v in p_draw.items() if v > 0]
        probs = np.fromiter((p_draw[k] for k in keys), dtype=float, count=len(keys))
        probs = probs / probs.sum()
        self._bits = np.array([[int(c) for c in k] for k in keys], dtype=np.int8)
        self._probs = probs
        self._rng = rng
        self._d = int(d)
        self._block = int(block)
        self._buf = np.empty((0, self._d), dtype=np.int8)

    @property
    def n_drawn(self) -> int:
        return int(self._buf.shape[0])

    def _extend_to(self, n: int) -> None:
        if n <= self._buf.shape[0]:
            return
        need = n - self._buf.shape[0]
        take = max(need, self._block)
        idx = self._rng.choice(len(self._probs), size=take, p=self._probs)
        self._buf = np.vstack([self._buf, self._bits[idx]])

    def prefix(self, n: int) -> np.ndarray:
        self._extend_to(int(n))
        return self._buf[: int(n)]

    def cursor(self, *, max_n: int = SEQ_MAX_N) -> Callable[[int], np.ndarray]:
        state = {"i": 0}

        def stream(batch: int) -> np.ndarray:
            i = state["i"]
            j = min(i + int(batch), int(max_n))
            if j <= i:
                return np.empty((0, self._d), dtype=np.int8)
            self._extend_to(j)
            state["i"] = j
            return self._buf[i:j]

        return stream


def _sequential_impl(name: str) -> Callable[..., dict]:

    def paper_sequential(source: ShotSource, *, p0_marg, alpha, horizon, delta, rng):
        from harness import stats_marginal

        return stats_marginal.sequential_test(source.prefix(horizon), p0_marg, alpha=alpha)

    def paper_sequential_exact(source: ShotSource, *, p0_marg, alpha, horizon, delta, rng):
        from harness import stats_marginal

        return stats_marginal.sequential_test(source.prefix(horizon), p0_marg, alpha=alpha,
                                              branch="exact")

    def sprt(source: ShotSource, *, p0_marg, alpha, horizon, delta, rng):
        from harness import stats_marginal

        return stats_marginal.sprt_test(source.cursor(max_n=SEQ_MAX_N), p0_marg,
                                        delta=delta, alpha=alpha, batch=SPRT_BATCH,
                                        max_n=SEQ_MAX_N)

    registry = {"sequential": paper_sequential, "sequential_exact": paper_sequential_exact, "sprt": sprt}
    if name not in registry:
        raise KeyError(
            f"unknown sequential method {name!r}; known: {sorted(registry)} "
            f"(inject others via run_sequential_cell(method_impls=...))"
        )
    return registry[name]


def run_sequential_cell(
    cell: Cell,
    p0: Mapping[str, float],
    p0_alt: Mapping[str, float] | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    delta: float = MATCHED_DELTA,
    horizon: int | None = None,
    method_impls: Mapping[str, Callable[..., dict]] | None = None,
) -> CellResult:
    if cell.case == "falsecase" and p0_alt is None:
        raise ValueError("falsecase cells require p0_alt")
    p_draw = p0 if cell.case == "baseline" else p0_alt
    horizon = int(cell.m if horizon is None else horizon)

    master = runner.seed_master(cell)
    children = runner.rep_seed_sequences(cell)
    impls = {
        name: (method_impls or {}).get(name) or _sequential_impl(name)
        for name in cell.methods
    }
    p0_marg = runner.marginal_null(p0, cell.d)
    results = {name: MethodResult(method=name) for name in cell.methods}
    pvals = {name: np.full(cell.n_reps, np.nan) for name in cell.methods}
    n_used = {name: [] for name in cell.methods}

    t_cell = time.perf_counter()
    for rep, child in enumerate(children):
        source = ShotSource(p_draw, cell.d, np.random.default_rng(child))
        for name in cell.methods:
            t0 = time.perf_counter()
            out = impls[name](source, p0_marg=p0_marg, alpha=alpha, horizon=horizon,
                              delta=delta, rng=runner._method_rng(master, rep, name))
            res = results[name]
            res.wall_s += time.perf_counter() - t0
            res.n_evaluated += 1
            reject = bool(out["reject"])
            if cell.case == "baseline":
                res.FP += reject
                res.TN += not reject
            else:
                res.TP += reject
                res.FN += not reject
            pvals[name][rep] = float(out.get("p_value", np.nan))
            n_used[name].append(float(out.get("n_used", np.nan)))

    for name, res in results.items():
        col = pvals[name]
        res.mean_p = float(np.nanmean(col)) if np.any(np.isfinite(col)) else float("nan")
        arr = np.asarray(n_used[name], dtype=float)
        if np.any(np.isfinite(arr)):
            res.n_used_mean = float(np.nanmean(arr))
            res.n_used_median = float(np.nanmedian(arr))

    return CellResult(cell=cell, seed_master=master, methods=results,
                      wall_s=time.perf_counter() - t_cell)


def run_any_cell(cell: Cell, p0: Mapping[str, float],
                 p0_alt: Mapping[str, float] | None = None, **kwargs) -> CellResult:
    if is_sequential_cell(cell):
        draw_fn = kwargs.pop("draw_fn", None)
        if draw_fn is not None:
            raise ValueError(
                f"{cell.campaign} is sequential: it consumes a ShotSource, not a "
                f"fixed-m draw_fn (inject via source_factory instead)"
            )
        return run_sequential_cell(cell, p0, p0_alt, **kwargs)
    kwargs.pop("delta", None)
    kwargs.pop("horizon", None)
    return runner.run_cell(cell, p0, p0_alt, **kwargs)


def run_cells(cells_: Sequence[Cell], *, alpha: float = DEFAULT_ALPHA,
              dist_provider: Callable[[Cell], tuple] | None = None,
              **kwargs) -> list[CellResult]:
    provider = dist_provider or distributions
    out = []
    for cell in cells_:
        p0, p0_alt = provider(cell)
        out.append(run_any_cell(cell, p0, p0_alt, alpha=alpha, **kwargs))
    return out




def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness.campaigns",
        description="Enumerate revision campaigns, probe applicability, emit the manifest.",
    )
    parser.add_argument("--manifest", action="store_true",
                        help="realized vs plan cell/run counts per campaign")
    parser.add_argument("--probe", action="store_true",
                        help="probe base_circuits/mutants and write the applicability cache")
    parser.add_argument("--screen", action="store_true",
                        help="I4 E3 shot-free fidelity/TVD screen (use --limit locally)")
    parser.add_argument("--campaign", choices=sorted(ENUMERATORS),
                        help="print one campaign's cells")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    parser.add_argument("--applicability", type=Path, default=None,
                        help=f"applicability cache (default {DEFAULT_CACHE.name})")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the probe/screen/manifest output here")
    parser.add_argument("--limit", type=int, default=None, help="cap printed/screened items")
    parser.add_argument("--widths", type=str, default=None,
                        help="probe only these widths, e.g. 2,3 (smoke)")
    parser.add_argument("--circuits", type=str, default=None,
                        help="probe only these circuits (smoke)")
    args = parser.parse_args(argv)

    if args.probe:
        widths = ([int(x) for x in args.widths.split(",")] if args.widths else MARGINAL_D)
        circuits = (tuple(args.circuits.split(",")) if args.circuits else CIRCUITS)
        t0 = time.perf_counter()
        app = probe_applicability(circuits, widths)
        path = write_applicability(app, args.out or (args.applicability or DEFAULT_CACHE))
        n_ok = sum(1 for v in app.widths.values() if v)
        n_op = sum(1 for v in app.operators.values() if v)
        print(f"probed {len(app.widths)} (circuit, d) pairs [{n_ok} supported], "
              f"{len(app.operators)} operator sites [{n_op} applicable] "
              f"in {time.perf_counter() - t0:.1f}s -> {path}")
        _APP_CACHE.pop("app", None)
        return 0

    app = (load_applicability(args.applicability) if args.applicability
           else default_applicability())

    if args.screen:
        rows = run_screen(limit=args.limit, app=app)
        if args.json or args.out:
            text = json.dumps(rows, indent=1)
            if args.out:
                Path(args.out).write_text(text + "\n")
                print(f"{len(rows)} screen rows -> {args.out}")
            else:
                print(text)
        else:
            for row in rows:
                print(f"{row['circuit']:<7} d={row['d']:<2} {row['operator']:<3} "
                      f"F={row['fidelity']:.6f} TVD={row['tvd']:.3e} "
                      f"{'EQUIV' if row['equivalent'] else ''}"
                      f"{' UNDETECTABLE' if row['undetectable'] else ''}")
        return 0

    if args.campaign:
        enum = enumerate_campaign(args.campaign, n_reps=args.n_reps, app=app)
        if args.json:
            print(json.dumps({
                "campaign": enum.campaign, "cells": enum.n_cells, "runs": enum.n_runs,
                "plan_cells": enum.plan_cells, "candidates": enum.candidates,
                "by_reason": enum.by_reason(), "method_drops": enum.method_drops(),
                "notes": enum.notes,
            }, indent=1))
            return 0
        limit = enum.n_cells if args.limit in (None, 0) else min(args.limit, enum.n_cells)
        for cell in enum.cells[:limit]:
            methods = (",".join(cell.methods) if len(cell.methods) <= 3
                       else f"{len(cell.methods)} methods")
            print(f"{cell.label()}  master={runner.seed_master(cell)}  {methods}")
        if limit < enum.n_cells:
            print(f"... {enum.n_cells - limit:,d} more cells (raise --limit)")
        print(f"{enum.campaign}: {enum.n_cells:,d} cells / {enum.n_runs:,d} runs "
              f"(plan {enum.plan_cells or 0:,d} / {enum.plan_runs or 0:,d}), "
              f"{enum.candidates - enum.n_cells:,d} excluded")
        for reason, n in sorted(enum.by_reason().items()):
            print(f"  excluded {reason}: {n:,d}")
        for note in enum.notes:
            print(f"  ! {note}")
        return 0

    man = manifest(n_reps=args.n_reps, app=app)
    text = json.dumps(man, indent=1) if args.json else format_manifest(man)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"manifest -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
