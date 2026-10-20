
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


CIRCUITS: tuple[str, ...] = ("BV", "Grover", "QFT", "QPE", "QWALK", "Adder")
M_POINTS: tuple[int, ...] = (50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 2000)
JOINT_D: tuple[int, ...] = tuple(range(2, 11))
MARGINAL_D: tuple[int, ...] = tuple(range(2, 13))
OPERATORS: tuple[str, ...] = ("O1", "O2", "O3", "O4", "O5", "O6", "O7")

ANGLE_OPERATORS: tuple[str, ...] = ("O1", "O3", "O5")
FRONTIER_D: tuple[int, ...] = (3, 6, 9)
FRONTIER_M: tuple[int, ...] = (50, 200, 500, 1000, 2000)
FRONTIER_STRENGTHS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.4)
E3_M: int = 2000

PARADIGMS: tuple[str, ...] = ("joint", "marginal")
CASES: tuple[str, ...] = ("baseline", "falsecase")
BASELINE_OPERATOR = "baseline"

JOINT_METHODS: tuple[str, ...] = (
    "mmdagg",
    "chi_square",
    "g_test",
    "multinomial_mc",
    "energy_distance",
    "tv_bound",
    "tolerance",
)
MARGINAL_METHODS: tuple[str, ...] = (
    "exact_binomial",
    "adaptive",
    "bootstrap",
    "sequential",
    "g_test_marginal",
)

DEFAULT_ALPHA = 0.05
DEFAULT_N_REPS = 1000




@dataclass(frozen=True)
class Cell:

    campaign: str
    circuit: str
    d: int
    m: int
    paradigm: str = "joint"
    operator: str = BASELINE_OPERATOR
    strength: float | None = None
    case: str = "falsecase"
    methods: tuple[str, ...] = ()
    n_reps: int = DEFAULT_N_REPS

    def __post_init__(self) -> None:
        if self.paradigm not in PARADIGMS:
            raise ValueError(f"paradigm must be one of {PARADIGMS}, got {self.paradigm!r}")
        if self.case not in CASES:
            raise ValueError(f"case must be one of {CASES}, got {self.case!r}")
        if self.case == "baseline" and self.operator != BASELINE_OPERATOR:
            raise ValueError("baseline cells must carry operator='baseline'")
        if self.case == "falsecase" and self.operator == BASELINE_OPERATOR:
            raise ValueError("falsecase cells must carry a mutation operator")
        if not self.methods:
            default = JOINT_METHODS if self.paradigm == "joint" else MARGINAL_METHODS
            object.__setattr__(self, "methods", default)
        else:
            object.__setattr__(self, "methods", tuple(self.methods))
        if self.strength is not None:
            object.__setattr__(self, "strength", float(self.strength))

    @property
    def n_runs(self) -> int:
        return self.n_reps

    @property
    def job_key(self) -> "JobKey":
        return JobKey(self.campaign, self.circuit, self.d, self.operator)

    def label(self) -> str:
        strength = "-" if self.strength is None else f"{self.strength:g}"
        return (
            f"{self.campaign} {self.paradigm:<8} {self.circuit:<7} d={self.d:<2} "
            f"m={self.m:<5} op={self.operator:<8} strength={strength:<5} case={self.case}"
        )


@dataclass(frozen=True)
class JobKey:

    campaign: str
    circuit: str
    d: int
    operator: str

    def filename(self, suffix: str = "") -> str:
        return f"{self.campaign}__{self.circuit}__d{self.d:02d}__{self.operator}{suffix}.parquet"




def seed_key(cell: Cell) -> str:
    return (
        f"{cell.campaign}|{cell.circuit}|{cell.d}|{cell.m}|"
        f"{cell.operator}|{cell.strength}|{cell.case}"
    )


def seed_master(cell: Cell) -> int:
    return int.from_bytes(hashlib.sha256(seed_key(cell).encode("utf-8")).digest()[:8], "big")


def rep_seed_sequences(cell: Cell) -> list[np.random.SeedSequence]:
    return np.random.SeedSequence(seed_master(cell)).spawn(cell.n_reps)


def _method_key(method: str) -> int:
    return int.from_bytes(hashlib.sha256(method.encode("utf-8")).digest()[:4], "big")


def _method_rng(master: int, rep: int, method: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([master, rep, _method_key(method)]))


def mutant_seed(cell: Cell) -> int:
    key = (
        f"{cell.campaign}|{cell.circuit}|{cell.d}|{cell.operator}|{cell.strength}|mutant"
    )
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def circuit_seed(cell: Cell) -> int:
    key = f"{cell.campaign}|{cell.circuit}|{cell.d}|base"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")




@dataclass
class Draw:

    states: np.ndarray
    counts: np.ndarray
    X: np.ndarray
    p0: Mapping[str, float]
    p0_marg: np.ndarray
    d: int
    m: int


def marginal_null(p0: Mapping[str, float], d: int) -> np.ndarray:
    keys = list(p0)
    if not keys:
        raise ValueError("empty p0")
    probs = np.fromiter((p0[k] for k in keys), dtype=float, count=len(keys))
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("p0 does not sum to a positive finite value")
    bits = np.array([[int(c) for c in k] for k in keys], dtype=float)
    if bits.shape[1] != d:
        raise ValueError(f"p0 keys have width {bits.shape[1]}, expected d={d}")
    marg = (probs @ bits) / total
    if marg.min() < -1e-9 or marg.max() > 1.0 + 1e-9:
        raise ValueError(f"marginal null outside [0,1] beyond fp tolerance: {marg}")
    return np.clip(marg, 0.0, 1.0)


def _default_draw(p_draw: Mapping[str, float], m: int, rng: np.random.Generator):
    from harness import sampling

    states, counts = sampling.draw_counts(p_draw, m, rng)
    X = sampling.X_from_counts(states, counts, rng)
    return states, counts, X


EXTRA_METHODS: dict[str, Callable[..., dict]] = {}


def _default_method(name: str) -> Callable[..., dict]:
    if name in EXTRA_METHODS:
        return EXTRA_METHODS[name]

    b_match = re.fullmatch(r"mmdagg_B(\d+)", name)
    if b_match:
        n_permute = int(b_match.group(1))

        def mmdagg_budget(draw: Draw, *, alpha: float, rng: np.random.Generator) -> dict:
            from harness import stats_joint

            return stats_joint.mmdagg_counts(
                draw.states, draw.counts, draw.p0,
                alpha=alpha, n_permute=n_permute, rng=rng)

        return mmdagg_budget

    def joint(fn_name: str, needs_rng: bool, **extra):
        def call(draw: Draw, *, alpha: float, rng: np.random.Generator) -> dict:
            from harness import stats_joint

            fn = getattr(stats_joint, fn_name)
            kwargs = dict(alpha=alpha, **extra)
            if needs_rng:
                kwargs["rng"] = rng
            return fn(draw.states, draw.counts, draw.p0, **kwargs)

        return call

    def marginal(fn_name: str, needs_rng: bool, **extra):
        def call(draw: Draw, *, alpha: float, rng: np.random.Generator) -> dict:
            from harness import stats_marginal

            fn = getattr(stats_marginal, fn_name)
            kwargs = dict(alpha=alpha, **extra)
            if needs_rng:
                kwargs["rng"] = rng
            return fn(draw.X, draw.p0_marg, **kwargs)

        return call

    registry: dict[str, Callable[..., dict]] = {
        "mmdagg": joint("mmdagg_counts", True),
        "chi_square": joint("chi_square_joint_counts", False),
        "g_test": joint("g_test_counts", False),
        "multinomial_mc": joint("multinomial_mc_counts", True),
        "energy_distance": joint("energy_distance_counts", True),
        "tv_bound": joint("tv_bound_counts", False),
        "tolerance": joint("tolerance_counts", False),
        "exact_binomial": marginal("exact_binomial_bonferroni", False),
        "adaptive": marginal("adaptive_test", False),
        "bootstrap": marginal("bootstrap_test", True),
        "sequential": marginal("sequential_test", False),
        "sequential_exact": marginal("sequential_test", False, branch="exact"),
        "g_test_marginal": marginal("g_test_marginal", False),
    }
    if name not in registry:
        raise KeyError(
            f"unknown method {name!r}; register it in runner._default_method or inject "
            f"it via run_cell(method_impls=...)"
        )
    return registry[name]




@dataclass
class MethodResult:

    method: str
    TP: int = 0
    FP: int = 0
    TN: int = 0
    FN: int = 0
    n_evaluated: int = 0
    mean_p: float = float("nan")
    error: str | None = None
    n_used_mean: float | None = None
    n_used_median: float | None = None
    wall_s: float = 0.0

    @property
    def n_reject(self) -> int:
        return self.TP + self.FP


@dataclass
class CellResult:
    cell: Cell
    seed_master: int
    methods: dict[str, MethodResult] = field(default_factory=dict)
    wall_s: float = 0.0


def run_cell(
    cell: Cell,
    p0: Mapping[str, float],
    p0_alt: Mapping[str, float] | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    method_impls: Mapping[str, Callable[..., dict]] | None = None,
    draw_fn: Callable[..., tuple] | None = None,
) -> CellResult:
    if cell.case == "falsecase" and p0_alt is None:
        raise ValueError("falsecase cells require p0_alt")
    p_draw = p0 if cell.case == "baseline" else p0_alt

    master = seed_master(cell)
    children = rep_seed_sequences(cell)
    draw_fn = draw_fn or _default_draw
    impls = {
        name: (method_impls or {}).get(name) or _default_method(name)
        for name in cell.methods
    }

    p0_marg = marginal_null(p0, cell.d)
    results = {name: MethodResult(method=name) for name in cell.methods}
    pvals = {name: np.full(cell.n_reps, np.nan) for name in cell.methods}
    n_used = {name: [] for name in cell.methods}

    t_cell = time.perf_counter()
    for rep, child in enumerate(children):
        rng = np.random.default_rng(child)
        states, counts, X = draw_fn(p_draw, cell.m, rng)
        draw = Draw(
            states=states,
            counts=counts,
            X=X,
            p0=p0,
            p0_marg=p0_marg,
            d=cell.d,
            m=cell.m,
        )
        for name in cell.methods:
            t0 = time.perf_counter()
            out = impls[name](draw, alpha=alpha, rng=_method_rng(master, rep, name))
            results[name].wall_s += time.perf_counter() - t0
            reject = bool(out["reject"])
            res = results[name]
            res.n_evaluated += 1
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
            if out.get("n_used") is not None:
                n_used[name].append(float(out["n_used"]))

    for name, res in results.items():
        col = pvals[name]
        res.mean_p = float(np.nanmean(col)) if np.any(np.isfinite(col)) else float("nan")
        if n_used[name]:
            arr = np.asarray(n_used[name], dtype=float)
            res.n_used_mean = float(arr.mean())
            res.n_used_median = float(np.median(arr))

    return CellResult(
        cell=cell,
        seed_master=master,
        methods=results,
        wall_s=time.perf_counter() - t_cell,
    )



_DIST_CACHE: dict[tuple, tuple[dict, dict | None]] = {}


def cell_distributions(cell: Cell) -> tuple[Mapping[str, float], Mapping[str, float] | None]:
    key = (cell.campaign, cell.circuit, cell.d, cell.operator, cell.strength, cell.case)
    if key in _DIST_CACHE:
        return _DIST_CACHE[key]

    from harness import base_circuits, mutants, oracles

    spec = base_circuits.build(cell.circuit, cell.d, seed=circuit_seed(cell))
    p0 = oracles.p0_from_spec(spec)
    p0_alt = None
    if cell.case == "falsecase":
        mutant = mutants.mutate(
            spec,
            cell.operator,
            strength=cell.strength,
            rng=np.random.default_rng(mutant_seed(cell)),
        )
        p0_alt = oracles.p0_from_spec(mutant)
    _DIST_CACHE[key] = (p0, p0_alt)
    return p0, p0_alt




def _e1_cells(campaign: str, n_reps: int) -> Iterator[Cell]:
    for paradigm, ds in (("joint", JOINT_D), ("marginal", MARGINAL_D)):
        for circuit in CIRCUITS:
            for d in ds:
                for m in M_POINTS:
                    for operator in OPERATORS:
                        yield Cell(
                            campaign=campaign,
                            circuit=circuit,
                            d=d,
                            m=m,
                            paradigm=paradigm,
                            operator=operator,
                            case="falsecase",
                            n_reps=n_reps,
                        )


def _e1b_cells(campaign: str, n_reps: int) -> Iterator[Cell]:
    for paradigm, ds in (("joint", JOINT_D), ("marginal", MARGINAL_D)):
        for circuit in CIRCUITS:
            for d in ds:
                for m in M_POINTS:
                    yield Cell(
                        campaign=campaign,
                        circuit=circuit,
                        d=d,
                        m=m,
                        paradigm=paradigm,
                        operator=BASELINE_OPERATOR,
                        case="baseline",
                        n_reps=n_reps,
                    )


def _frontier_cells(campaign: str, paradigm: str, n_reps: int) -> Iterator[Cell]:
    for circuit in CIRCUITS:
        for d in FRONTIER_D:
            for m in FRONTIER_M:
                for operator in ANGLE_OPERATORS:
                    for strength in FRONTIER_STRENGTHS:
                        yield Cell(
                            campaign=campaign,
                            circuit=circuit,
                            d=d,
                            m=m,
                            paradigm=paradigm,
                            operator=operator,
                            strength=strength,
                            case="falsecase",
                            n_reps=n_reps,
                        )


def _e3_cells(campaign: str, n_reps: int) -> Iterator[Cell]:
    for circuit in CIRCUITS:
        for d in JOINT_D:
            for operator in OPERATORS:
                yield Cell(
                    campaign=campaign,
                    circuit=circuit,
                    d=d,
                    m=E3_M,
                    paradigm="joint",
                    operator=operator,
                    case="falsecase",
                    methods=("chi_square",),
                    n_reps=n_reps,
                )


CAMPAIGNS: dict[str, Callable[[str, int], Iterator[Cell]]] = {
    "I4E1": _e1_cells,
    "I4E1B": _e1b_cells,
    "I4E2": lambda c, n: _frontier_cells(c, "joint", n),
    "I4E2b": lambda c, n: _frontier_cells(c, "marginal", n),
    "I4E3": _e3_cells,
}


def enumerate_cells(
    campaign: str,
    *,
    circuit: str | None = None,
    d: int | None = None,
    m: int | None = None,
    operator: str | None = None,
    paradigm: str | None = None,
    case: str | None = None,
    n_reps: int = DEFAULT_N_REPS,
) -> list[Cell]:
    if campaign not in CAMPAIGNS:
        raise KeyError(f"unknown campaign {campaign!r}; known: {sorted(CAMPAIGNS)}")
    cells = CAMPAIGNS[campaign](campaign, n_reps)
    out = []
    for cell in cells:
        if circuit is not None and cell.circuit != circuit:
            continue
        if d is not None and cell.d != d:
            continue
        if m is not None and cell.m != m:
            continue
        if operator is not None and cell.operator != operator:
            continue
        if paradigm is not None and cell.paradigm != paradigm:
            continue
        if case is not None and cell.case != case:
            continue
        out.append(cell)
    return out


def total_runs(cells: Iterable[Cell]) -> int:
    return sum(c.n_runs for c in cells)


def iter_jobs(cells: Iterable[Cell]) -> Iterator[tuple[JobKey, list[Cell]]]:
    grouped: dict[JobKey, list[Cell]] = {}
    for cell in cells:
        grouped.setdefault(cell.job_key, []).append(cell)
    for key, group in grouped.items():
        yield key, group




def _run_one(cell: Cell, alpha, dist_provider, method_impls, draw_fn,
             cell_runner=None) -> CellResult:
    try:
        p0, p0_alt = dist_provider(cell)
        return (cell_runner or run_cell)(
            cell,
            p0,
            p0_alt,
            alpha=alpha,
            method_impls=method_impls,
            draw_fn=draw_fn,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[cell-error] {cell.campaign}/{cell.circuit} d={cell.d} m={cell.m} "
              f"op={cell.operator} case={cell.case}: {msg}", file=sys.stderr)
        return CellResult(
            cell=cell,
            seed_master=seed_master(cell),
            methods={"_error": MethodResult(method="_error", error=msg)},
        )


def run_job(
    cells: Sequence[Cell],
    out_dir: str | Path | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    n_jobs: int = -1,
    dist_provider: Callable[[Cell], tuple] | None = None,
    method_impls: Mapping[str, Callable[..., dict]] | None = None,
    draw_fn: Callable[..., tuple] | None = None,
    cell_runner: Callable[..., CellResult] | None = None,
    filename_suffix: str = "",
) -> tuple[list[CellResult], Path | None]:
    from joblib import Parallel, delayed

    cells = list(cells)
    if not cells:
        return [], None
    keys = {c.job_key for c in cells}
    if len(keys) != 1:
        raise ValueError(f"run_job expects one (campaign, circuit, d, operator) slice, got {keys}")
    dist_provider = dist_provider or cell_distributions

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(cell, alpha, dist_provider, method_impls, draw_fn, cell_runner)
        for cell in cells
    )
    results = list(results)

    path = None
    if out_dir is not None:
        from harness import io_store

        path = Path(out_dir) / next(iter(keys)).filename(filename_suffix)
        io_store.write_job(path, results)
    return results, path




def _summarise(cells: Sequence[Cell]) -> list[str]:
    lines = []
    by_paradigm: dict[str, list[Cell]] = {}
    for cell in cells:
        by_paradigm.setdefault(cell.paradigm, []).append(cell)
    for paradigm in PARADIGMS:
        group = by_paradigm.get(paradigm)
        if group:
            lines.append(
                f"  {paradigm:<8} {len(group):>7,d} cells {total_runs(group):>12,d} runs"
            )
    lines.append(f"  {'jobs':<8} {len(list(iter_jobs(cells))):>7,d}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    import importlib.util

    dist_provider = None
    cell_runner = None

    if importlib.util.find_spec("harness.campaigns") is not None:
        from harness import campaigns

        campaigns.register_with_runner()
        dist_provider = campaigns.distributions
        cell_runner = campaigns.run_any_cell

        canonical = sys.modules.get("harness.runner")
        if canonical is not None and canonical is not sys.modules[__name__]:
            raise RuntimeError(
                "harness.runner is loaded twice (as "
                f"{__name__!r} and 'harness.runner'), so campaigns.register_with_runner() "
                "did not reach this module's CAMPAIGNS. Invoke the CLI as "
                "`python -m harness.runner` or `runner.main()` on the imported module."
            )

    parser = argparse.ArgumentParser(
        prog="python -m harness.runner",
        description="Run (or enumerate) revision-campaign cells.",
    )
    parser.add_argument("--campaign", required=True, choices=sorted(CAMPAIGNS))
    parser.add_argument("--circuit", help="circuit / arm name (campaign-dependent)")
    parser.add_argument("--d", type=int)
    parser.add_argument("--m", type=int, help="optional finer shard than the SPEC job key")
    parser.add_argument("--operator")
    parser.add_argument("--paradigm", choices=PARADIGMS)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--out", type=Path, help="output directory for job parquet files")
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS)
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip fine chunks whose output parquet already exists "
                             "(makes coarse-job resubmission an idempotent retry)")
    parser.add_argument("--methods", type=lambda s: tuple(s.split(",")),
                        help="comma-separated method names: run ONLY these on the "
                             "enumerated cells (draws are seed-identical to a full run; "
                             "the names are appended to the output filename)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true", help="print cells and run counts")
    parser.add_argument(
        "--list-limit",
        type=int,
        default=200,
        help="max cells printed by --dry-run (0 = print all)",
    )
    args = parser.parse_args(argv)

    if args.circuit is not None or args.operator is not None:
        universe = enumerate_cells(args.campaign, n_reps=args.n_reps)
    for flag, value, field in (
        ("--circuit", args.circuit, "circuit"),
        ("--operator", args.operator, "operator"),
    ):
        if value is not None:
            valid = sorted({getattr(c, field) for c in universe})
            if value not in valid:
                parser.error(
                    f"argument {flag}: {value!r} does not occur in campaign "
                    f"{args.campaign} (valid: {', '.join(valid)})"
                )

    cells = enumerate_cells(
        args.campaign,
        circuit=args.circuit,
        d=args.d,
        m=args.m,
        operator=args.operator,
        paradigm=args.paradigm,
        case=args.case,
        n_reps=args.n_reps,
    )

    if args.methods:
        cells = [dataclasses.replace(c, methods=args.methods) for c in cells]

    if args.dry_run:
        limit = len(cells) if args.list_limit == 0 else min(args.list_limit, len(cells))
        for cell in cells[:limit]:
            print(f"{cell.label()}  master={seed_master(cell)}  methods={len(cell.methods)}")
        if limit < len(cells):
            print(f"... {len(cells) - limit:,d} more cells (raise --list-limit to print)")
        print(f"campaign {args.campaign}: {len(cells):,d} cells, "
              f"{total_runs(cells):,d} runs (n_reps={args.n_reps})")
        for line in _summarise(cells):
            print(line)
        return 0

    if not cells:
        print("no cells match the filters", file=sys.stderr)
        return 1
    if args.out is None:
        parser.error("--out is required unless --dry-run")

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "".join(
        f"__{part}"
        for part in (
            f"m{args.m}" if args.m is not None else "",
            args.paradigm or "",
            args.case or "",
            "-".join(args.methods) if args.methods else "",
        )
        if part
    )
    n_written = 0
    n_skipped = 0
    for key, group in iter_jobs(cells):
        if args.skip_existing and (Path(args.out) / key.filename(suffix)).exists():
            n_skipped += 1
            continue
        _, path = run_job(
            group,
            args.out,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            dist_provider=dist_provider,
            cell_runner=cell_runner,
            filename_suffix=suffix,
        )
        n_written += 1
    if n_skipped:
        print(f"skipped {n_skipped} chunk(s) with existing parquet (--skip-existing)")
        print(f"wrote {path} ({len(group)} cells, {total_runs(group):,d} runs)")
    print(f"{n_written} job file(s) in {args.out}")
    return 0


if __name__ == "__main__":
    try:
        from harness.runner import main as _canonical_main
    except ImportError:
        sys.exit(main())
    sys.exit(_canonical_main())
