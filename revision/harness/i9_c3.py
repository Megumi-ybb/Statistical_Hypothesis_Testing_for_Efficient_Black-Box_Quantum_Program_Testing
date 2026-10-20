
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


MARGINAL_TOL: float = 1e-12
DEGENERATE_TOL: float = 1e-12
SUPPORT_TOL: float = 1e-12
EXACT_FREE_COORDS: int = 4

STYLES: tuple[str, ...] = ("comonotone", "independent", "pairwise")
AUTO_ORDER: tuple[str, ...] = ("comonotone", "independent", "pairwise")

BENCHMARKS: tuple[str, ...] = ("BV", "Grover", "QFT", "QPE", "QWALK", "Adder")
DEFAULT_CLASSIFY_DIMS: tuple[int, ...] = (2, 3, 4)

I9_CAMPAIGN: str = "I9"
C3_OPERATOR: str = "C3"
EPSILON_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.20)
MARGINAL_METHODS_INTERIOR_SAFE: tuple[str, ...] = (
    "exact_binomial", "adaptive", "g_test_marginal",
)


class NotAdmissible(ValueError):
    pass


class MarginalDrift(RuntimeError):
    pass




def _keys_probs(p0: Mapping[str, float]) -> tuple[list[str], np.ndarray, int]:
    keys = list(p0)
    if not keys:
        raise ValueError("empty p0")
    d = len(keys[0])
    if any(len(k) != d for k in keys):
        raise ValueError("p0 keys have inconsistent width")
    probs = np.fromiter((float(p0[k]) for k in keys), dtype=float, count=len(keys))
    if np.any(probs < -1e-12) or not np.all(np.isfinite(probs)):
        raise ValueError("p0 has negative or non-finite entries")
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("p0 does not normalise")
    return keys, np.maximum(probs, 0.0) / total, d


def _bit_matrix(keys: Sequence[str]) -> np.ndarray:
    return np.array([[float(c) for c in k] for k in keys], dtype=float)


def marginals(p0: Mapping[str, float]) -> np.ndarray:
    keys, probs, _ = _keys_probs(p0)
    return probs @ _bit_matrix(keys)


def tvd(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * float(sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys))


def free_coordinates(p0: Mapping[str, float], *,
                     tol: float = DEGENERATE_TOL) -> tuple[int, ...]:
    mu = marginals(p0)
    return tuple(int(j) for j in range(mu.size) if tol < mu[j] < 1.0 - tol)


def support(p0: Mapping[str, float], *, tol: float = SUPPORT_TOL) -> list[str]:
    keys, probs, _ = _keys_probs(p0)
    return [k for k, p in zip(keys, probs) if p > tol]


@dataclass(frozen=True)
class Admissibility:

    admissible: bool
    d: int
    n_free: int
    free_coords: tuple[int, ...]
    n_support: int
    reason: str
    witness: tuple[str, str] | None = None


def admissibility(p0: Mapping[str, float], *, tol: float = DEGENERATE_TOL) -> Admissibility:
    keys, probs, d = _keys_probs(p0)
    supp = [k for k, p in zip(keys, probs) if p > SUPPORT_TOL]
    free = free_coordinates(p0, tol=tol)
    witness = _hamming_witness(supp)

    if (len(free) >= 2) != (witness is not None):
        raise RuntimeError(
            "applicability theorem violated numerically: "
            f"n_free={len(free)} but witness={witness}; tighten the tolerances"
        )

    if len(supp) == 1:
        reason = ("concentrated null (|supp| = 1): the joint IS the product of its "
                  "degenerate marginals, so no dependence-only fault exists")
    elif len(free) == 0:
        reason = "every coordinate degenerate: M(p0) = {p0}"
    elif len(free) == 1:
        reason = ("one non-degenerate coordinate: its marginal pins the joint "
                  "(support 2 at Hamming distance 1), so M(p0) = {p0}")
    else:
        reason = (f"{len(free)} non-degenerate coordinates; dependence directions "
                  f"available: 2^{len(free)} - {len(free)} - 1 = "
                  f"{2 ** len(free) - len(free) - 1}")
    return Admissibility(
        admissible=len(free) >= 2,
        d=d,
        n_free=len(free),
        free_coords=free,
        n_support=len(supp),
        reason=reason,
        witness=witness,
    )


def _hamming_witness(supp: Sequence[str]) -> tuple[str, str] | None:
    for a, b in itertools.combinations(supp[:3], 2):
        if sum(1 for u, v in zip(a, b) if u != v) >= 2:
            return (a, b)
    return None




def comonotone_target(p0: Mapping[str, float]) -> dict[str, float]:
    keys, _, d = _keys_probs(p0)
    mu = marginals(p0)
    q = {k: 0.0 for k in keys}
    cuts = sorted(set(mu.tolist()) | {0.0, 1.0})
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        w = hi - lo
        if w <= 0:
            continue
        u = 0.5 * (lo + hi)
        key = "".join("1" if u <= mu[j] else "0" for j in range(d))
        q[key] = q.get(key, 0.0) + w
    return q


def independent_target(p0: Mapping[str, float]) -> dict[str, float]:
    keys, _, _ = _keys_probs(p0)
    mu = marginals(p0)
    B = _bit_matrix(keys)
    probs = np.prod(np.where(B > 0.5, mu, 1.0 - mu), axis=1)
    return {k: float(v) for k, v in zip(keys, probs)}




@dataclass(frozen=True)
class Move:

    lose: tuple[str, str]
    gain: tuple[str, str]
    capacity: float
    kind: str


def _meet(a: str, b: str) -> str:
    return "".join("1" if u == "1" and v == "1" else "0" for u, v in zip(a, b))


def _apply_subset(base: str, diff: Sequence[int], subset: frozenset[int]) -> str:
    out = list(base)
    for c in diff:
        out[c] = "1" if c in subset else "0"
    return "".join(out)


def _pair_move(y: str, z: str, capacity: float, rng: np.random.Generator,
               *, prefer_align: bool, allowed: set[str] | None = None) -> Move | None:
    diff = [c for c in range(len(y)) if y[c] != z[c]]
    if len(diff) < 2:
        return None
    base = _meet(y, z)
    forbidden = {frozenset(c for c in diff if y[c] == "1"),
                 frozenset(c for c in diff if z[c] == "1")}
    singles = [frozenset({int(c)}) for c in rng.permutation(np.asarray(diff))]
    options = [frozenset()] + singles if prefer_align else singles + [frozenset()]
    tried = [a for a in options if a not in forbidden]
    for subset in tried:
        w = _apply_subset(base, diff, subset)
        x = _apply_subset(base, diff, frozenset(diff) - subset)
        if {w, x} == {y, z}:
            continue
        if allowed is not None and (w not in allowed or x not in allowed):
            continue
        kind = "meet-join" if not subset else "split"
        return Move(lose=(y, z), gain=(w, x), capacity=float(capacity), kind=kind)
    return None


def candidate_moves(p0: Mapping[str, float], rng: np.random.Generator, *,
                    n_moves: int = 1, prefer_align: bool = True,
                    support_preserving: bool = False,
                    pool: int = 256) -> list[Move]:
    keys, probs, _ = _keys_probs(p0)
    supp = sorted(((p, k) for k, p in zip(keys, probs) if p > SUPPORT_TOL), reverse=True)
    allowed = {k for _, k in supp} if support_preserving else None
    heads = supp[:pool]
    moves: list[Move] = []
    for candidates in (heads, supp if len(supp) > len(heads) else []):
        used: set[str] = {s for mv in moves for s in mv.lose}
        pairs = sorted(
            ((min(pa, pb), ka, kb)
             for (pa, ka), (pb, kb) in itertools.combinations(candidates, 2)),
            key=lambda t: -t[0],
        )
        for capacity, ka, kb in pairs:
            if len(moves) >= n_moves:
                break
            if ka in used or kb in used:
                continue
            mv = _pair_move(ka, kb, capacity, rng, prefer_align=prefer_align,
                            allowed=allowed)
            if mv is None:
                continue
            moves.append(mv)
            used.update(mv.lose)
        if len(moves) >= n_moves:
            break
    return moves


def _pairwise_direction(p0: Mapping[str, float], rng: np.random.Generator, *,
                        n_moves: int, prefer_align: bool, support_preserving: bool
                        ) -> tuple[dict[str, float], list[Move]]:
    moves = candidate_moves(p0, rng, n_moves=n_moves, prefer_align=prefer_align,
                            support_preserving=support_preserving)
    if not moves:
        raise NotAdmissible("no marginal-preserving block move exists on supp(p0)")
    v: dict[str, float] = {}
    for mv in moves:
        for s in mv.lose:
            v[s] = v.get(s, 0.0) - mv.capacity
        for s in mv.gain:
            v[s] = v.get(s, 0.0) + mv.capacity
    return v, moves




@dataclass
class DependenceOnly:

    q: dict[str, float]
    style: str
    epsilon: float
    epsilon_requested: float
    epsilon_max_style: float
    clipped: bool
    marginal_error: float
    t: float
    escaped_mass: float = 0.0
    moves: tuple[Move, ...] = ()
    d: int = 0

    @property
    def escapes_support(self) -> bool:
        return self.escaped_mass > SUPPORT_TOL

    def as_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "epsilon": self.epsilon,
            "epsilon_requested": self.epsilon_requested,
            "epsilon_max_style": self.epsilon_max_style,
            "clipped": self.clipped,
            "marginal_error": self.marginal_error,
            "t": self.t,
            "escaped_mass": self.escaped_mass,
            "escapes_support": self.escapes_support,
            "n_moves": len(self.moves),
            "d": self.d,
        }


def _check_marginals(q: Mapping[str, float], mu0: np.ndarray, *,
                     tol: float = MARGINAL_TOL) -> float:
    err = float(np.max(np.abs(marginals(q) - mu0))) if mu0.size else 0.0
    if not (err < tol):
        raise MarginalDrift(
            f"constructed q drifts from the null marginals by {err:.3e} (tol {tol:.0e})"
        )
    return err


def make_dependence_only(p0: Mapping[str, float], epsilon: float,
                         rng: np.random.Generator, *, style: str = "auto",
                         n_moves: int = 1, prefer_align: bool = True,
                         support_preserving: bool = False) -> DependenceOnly:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    verdict = admissibility(p0)
    if not verdict.admissible:
        raise NotAdmissible(f"{verdict.reason} (n_free={verdict.n_free})")

    keys, probs, d = _keys_probs(p0)
    p0n = {k: float(p) for k, p in zip(keys, probs)}
    mu0 = marginals(p0)

    styles = AUTO_ORDER if style == "auto" else (style,)
    if any(s not in STYLES for s in styles):
        raise ValueError(f"style must be 'auto' or one of {STYLES}, got {style!r}")

    last: DependenceOnly | None = None
    for candidate in styles:
        try:
            built = _build_style(p0n, mu0, d, epsilon, rng, candidate, n_moves,
                                 prefer_align, support_preserving)
        except NotAdmissible:
            continue
        last = built
        if built.epsilon_max_style <= SUPPORT_TOL:
            continue
        if support_preserving and built.escapes_support:
            continue
        return built
    if last is None or support_preserving:
        raise NotAdmissible(
            f"no {'support-preserving ' if support_preserving else ''}dependence-only "
            f"alternative available for this null under style={style!r}"
        )
    return last


def _build_style(p0n: dict[str, float], mu0: np.ndarray, d: int, epsilon: float,
                 rng: np.random.Generator, style: str, n_moves: int,
                 prefer_align: bool, support_preserving: bool = False) -> DependenceOnly:
    moves: tuple[Move, ...] = ()
    if style in ("comonotone", "independent"):
        target = (comonotone_target(p0n) if style == "comonotone"
                  else independent_target(p0n))
        reach = tvd(p0n, target)
        t = 0.0 if reach <= 0 else min(epsilon, reach) / reach
        q = {k: (1.0 - t) * p0n.get(k, 0.0) + t * float(target.get(k, 0.0))
             for k in set(p0n) | set(target)}
    else:
        v, mv = _pairwise_direction(p0n, rng, n_moves=n_moves, prefer_align=prefer_align,
                                    support_preserving=support_preserving)
        moves = tuple(mv)
        l1_half = 0.5 * sum(abs(x) for x in v.values())
        losses = [(-x, p0n.get(k, 0.0)) for k, x in v.items() if x < 0]
        t_max = min((mass / loss for loss, mass in losses), default=0.0)
        reach = t_max * l1_half
        t = 0.0 if l1_half <= 0 else min(epsilon, reach) / l1_half
        q = {k: p0n.get(k, 0.0) + t * v.get(k, 0.0) for k in set(p0n) | set(v)}
        worst = min(q.values(), default=0.0)
        if worst < -1e-14:
            raise RuntimeError(f"pairwise step left q({worst:.3e}) < 0")
        q = {k: max(x, 0.0) for k, x in q.items()}

    realised = tvd(p0n, q)
    err = _check_marginals(q, mu0)
    escaped = float(sum(x for k, x in q.items() if p0n.get(k, 0.0) <= SUPPORT_TOL))
    return DependenceOnly(
        q=q,
        style=style,
        epsilon=realised,
        epsilon_requested=float(epsilon),
        epsilon_max_style=float(reach),
        clipped=bool(epsilon > reach + 1e-12),
        marginal_error=err,
        t=float(t),
        escaped_mass=escaped,
        moves=moves,
        d=d,
    )


def dependence_only_q(p0: Mapping[str, float], epsilon: float,
                      rng: np.random.Generator, **kwargs) -> dict[str, float]:
    return make_dependence_only(p0, epsilon, rng, **kwargs).q


def constructible_epsilon(p0: Mapping[str, float], rng: np.random.Generator, *,
                          style: str = "auto", **kwargs) -> float:
    try:
        built = make_dependence_only(p0, 1.0, rng, style=style, **kwargs)
    except NotAdmissible:
        return 0.0
    return built.epsilon_max_style




def _free_cube(p0: Mapping[str, float]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    keys, probs, d = _keys_probs(p0)
    mu = marginals(p0)
    free = free_coordinates(p0)
    fixed = {j: ("1" if mu[j] >= 0.5 else "0") for j in range(d) if j not in free}
    cube: list[str] = []
    for bits in itertools.product("01", repeat=len(free)):
        chars = ["0"] * d
        for j, b in zip(free, bits):
            chars[j] = b
        for j, b in fixed.items():
            chars[j] = b
        cube.append("".join(chars))
    lookup = {k: p for k, p in zip(keys, probs)}
    p_cube = np.array([lookup.get(k, 0.0) for k in cube], dtype=float)
    return cube, p_cube, mu[list(free)], np.array(free, dtype=int)


def max_epsilon_info(p0: Mapping[str, float], *,
                     exact_free_coords: int = EXACT_FREE_COORDS,
                     rng: np.random.Generator | None = None) -> dict[str, Any]:
    verdict = admissibility(p0)
    if not verdict.admissible:
        return {"value": 0.0, "exact": True, "method": "theorem",
                "n_free": verdict.n_free, "n_states": 0, "reason": verdict.reason}

    k = verdict.n_free
    if k <= exact_free_coords:
        cube, p_cube, mu_free, free_idx = _free_cube(p0)
        outside = max(0.0, 1.0 - float(p_cube.sum()))
        n = len(cube)
        C = np.vstack([np.ones(n)] +
                      [np.array([float(s[j]) for s in cube]) for j in free_idx])
        b = np.concatenate([[1.0], np.asarray(mu_free, dtype=float)])
        best = 0.0
        for size in range(1, k + 2):
            for cols in itertools.combinations(range(n), size):
                A = C[:, cols]
                if np.linalg.matrix_rank(A, tol=1e-10) < size:
                    continue
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                if sol.min() < -1e-10 or np.linalg.norm(A @ sol - b) > 1e-8:
                    continue
                q = np.zeros(n)
                q[list(cols)] = np.maximum(sol, 0.0)
                best = max(best, 0.5 * float(np.abs(q - p_cube).sum()) + 0.5 * outside)
        return {"value": float(best), "exact": True, "method": "vertex-enumeration",
                "n_free": k, "n_states": n, "reason": verdict.reason}

    rng = rng or np.random.default_rng(0)
    reach = {s: constructible_epsilon(p0, rng, style=s) for s in STYLES}
    best_style = max(reach, key=lambda s: reach[s])
    return {"value": float(reach[best_style]), "exact": False,
            "method": f"achievable-lower-bound({best_style})", "n_free": k,
            "n_states": 2 ** k, "reason": verdict.reason, "by_style": reach}


def max_epsilon(p0: Mapping[str, float], **kwargs) -> float:
    return float(max_epsilon_info(p0, **kwargs)["value"])


def frechet_max_epsilon_d2(p0: Mapping[str, float]) -> float:
    mu = marginals(p0)
    if mu.size != 2:
        raise ValueError("frechet_max_epsilon_d2 is for d = 2")
    a0 = float(p0.get("11", 0.0))
    lo, hi = max(0.0, mu[0] + mu[1] - 1.0), min(mu[0], mu[1])
    return float(2.0 * max(a0 - lo, hi - a0))




@dataclass(frozen=True)
class C3Row:

    circuit: str
    d: int
    supported: bool
    width: int
    n_support: int
    n_free: int
    admissible: bool
    max_epsilon: float
    max_epsilon_exact: bool
    eps_comonotone: float
    eps_independent: float
    eps_pairwise: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def benchmark_null(circuit: str, d: int, *, seed: int = 0,
                   variant: str = "benchmark") -> dict[str, float]:
    return _benchmark_null_cached(circuit, d, seed, variant)


_NULL_CACHE: dict[tuple[str, int, int, str], dict[str, float]] = {}


def _benchmark_null_cached(circuit: str, d: int, seed: int, variant: str) -> dict[str, float]:
    key = (circuit, d, seed, variant)
    if key not in _NULL_CACHE:
        from harness import base_circuits, oracles

        spec = base_circuits.build(circuit, d, seed=seed, variant=variant)
        _NULL_CACHE[key] = oracles.p0_from_spec(spec)
    return _NULL_CACHE[key]


def classify_benchmarks(dims: Sequence[int] = DEFAULT_CLASSIFY_DIMS,
                        circuits: Sequence[str] = BENCHMARKS, *, seed: int = 0,
                        variant: str = "benchmark",
                        rng: np.random.Generator | None = None) -> list[C3Row]:
    from harness.base_circuits import UnsupportedWidth

    rng = rng or np.random.default_rng(20260726)
    rows: list[C3Row] = []
    for circuit in circuits:
        for d in dims:
            try:
                p0 = benchmark_null(circuit, d, seed=seed, variant=variant)
            except UnsupportedWidth as exc:
                rows.append(C3Row(circuit, int(d), False, 0, 0, 0, False, 0.0, True,
                                  0.0, 0.0, 0.0, f"width unsupported: {exc}"))
                continue
            rows.append(_classify_one(circuit, d, p0, rng))
    return rows


def _classify_one(circuit: str, d: int, p0: Mapping[str, float],
                  rng: np.random.Generator) -> C3Row:
    verdict = admissibility(p0)
    info = max_epsilon_info(p0, rng=rng)
    reach = {s: constructible_epsilon(p0, rng, style=s) for s in STYLES} if \
        verdict.admissible else {s: 0.0 for s in STYLES}
    return C3Row(
        circuit=circuit,
        d=int(d),
        supported=True,
        width=verdict.d,
        n_support=verdict.n_support,
        n_free=verdict.n_free,
        admissible=verdict.admissible,
        max_epsilon=float(info["value"]),
        max_epsilon_exact=bool(info["exact"]),
        eps_comonotone=float(reach["comonotone"]),
        eps_independent=float(reach["independent"]),
        eps_pairwise=float(reach["pairwise"]),
        reason=verdict.reason,
    )


def classification_table(rows: Sequence[C3Row] | None = None, **kwargs) -> str:
    rows = list(rows) if rows is not None else classify_benchmarks(**kwargs)
    head = (f"{'circuit':<8}{'d':>3}{'|supp|':>8}{'k_free':>8}{'admits C3':>11}"
            f"{'max_eps':>10}{'exact':>7}  reason")
    lines = [head, "-" * (len(head) + 30)]
    for r in rows:
        if not r.supported:
            lines.append(f"{r.circuit:<8}{r.d:>3}{'-':>8}{'-':>8}{'n/a':>11}"
                         f"{'-':>10}{'-':>7}  {r.reason}")
            continue
        lines.append(
            f"{r.circuit:<8}{r.d:>3}{r.n_support:>8}{r.n_free:>8}"
            f"{('YES' if r.admissible else 'NO'):>11}{r.max_epsilon:>10.4f}"
            f"{('yes' if r.max_epsilon_exact else 'LB'):>7}  {r.reason}"
        )
    return "\n".join(lines)


def fallback_report(dims: Sequence[int] = DEFAULT_CLASSIFY_DIMS, *, seed: int = 0
                    ) -> list[dict[str, Any]]:
    from harness.base_circuits import UnsupportedWidth

    rng = np.random.default_rng(20260726)
    out: list[dict[str, Any]] = []
    for circuit in BENCHMARKS:
        for d in dims:
            row: dict[str, Any] = {"circuit": circuit, "d": int(d)}
            for variant in ("benchmark", "textbook"):
                try:
                    p0 = benchmark_null(circuit, d, seed=seed, variant=variant)
                except UnsupportedWidth:
                    row[f"{variant}_admissible"] = None
                    row[f"{variant}_max_epsilon"] = None
                    continue
                verdict = admissibility(p0)
                row[f"{variant}_admissible"] = verdict.admissible
                row[f"{variant}_max_epsilon"] = max_epsilon(p0, rng=rng)
            out.append(row)
    return out




def c3_seed(circuit: str, d: int, style: str) -> int:
    key = f"{I9_CAMPAIGN}|{circuit}|{d}|{style}|c3"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def distributions_for(circuit: str, d: int, epsilon: float, *, seed: int = 0,
                      variant: str = "benchmark", style: str = "auto",
                      n_moves: int = 1, support_preserving: bool = False
                      ) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    p0 = benchmark_null(circuit, d, seed=seed, variant=variant)
    rng = np.random.default_rng(c3_seed(circuit, d, style))
    built = make_dependence_only(p0, epsilon, rng, style=style, n_moves=n_moves,
                                 support_preserving=support_preserving)
    info = built.as_dict()
    info.update({"circuit": circuit, "variant": variant, "seed": seed,
                 "construction_seed": c3_seed(circuit, d, style)})
    return p0, built.q, info


def distributions_for_cell(cell, *, seed: int = 0, variant: str = "benchmark",
                           style: str = "auto", support_preserving: bool = False):
    if cell.case == "baseline":
        return benchmark_null(cell.circuit, cell.d, seed=seed, variant=variant), None
    p0, q, _ = distributions_for(cell.circuit, cell.d, float(cell.strength),
                                 seed=seed, variant=variant, style=style,
                                 support_preserving=support_preserving)
    return p0, q


def admissible_grid(dims: Sequence[int] | None = None,
                    circuits: Sequence[str] = BENCHMARKS, *,
                    epsilon_levels: Sequence[float] = EPSILON_LEVELS,
                    seed: int = 0, variant: str = "benchmark",
                    style: str = "auto",
                    support_preserving: bool = False) -> list[dict[str, Any]]:
    from harness.base_circuits import UnsupportedWidth
    from harness import runner

    dims = tuple(runner.JOINT_D) if dims is None else tuple(int(x) for x in dims)
    out: list[dict[str, Any]] = []
    for circuit in circuits:
        for d in dims:
            entry: dict[str, Any] = {"circuit": circuit, "d": int(d), "supported": True,
                                     "admissible": False, "reach": 0.0,
                                     "epsilons": (), "reason": ""}
            try:
                p0 = benchmark_null(circuit, d, seed=seed, variant=variant)
            except UnsupportedWidth as exc:
                entry.update(supported=False, reason=f"width unsupported: {exc}")
                out.append(entry)
                continue
            verdict = admissibility(p0)
            entry["reason"] = verdict.reason
            entry["n_free"] = verdict.n_free
            entry["n_support"] = verdict.n_support
            entry["interior_marginals"] = bool(verdict.n_free == verdict.d)
            if not verdict.admissible:
                out.append(entry)
                continue
            rng = np.random.default_rng(c3_seed(circuit, d, style))
            reach = constructible_epsilon(p0, rng, style=style,
                                          support_preserving=support_preserving)
            entry.update(
                admissible=reach > SUPPORT_TOL,
                reach=float(reach),
                epsilons=tuple(float(e) for e in epsilon_levels if e <= reach + 1e-12),
            )
            if not entry["admissible"]:
                entry["reason"] = ("no constructible alternative under "
                                   f"support_preserving={support_preserving}")
            out.append(entry)
    return out


def cells(campaign: str = I9_CAMPAIGN, *, dims: Sequence[int] | None = None,
          circuits: Sequence[str] = BENCHMARKS,
          m_points: Sequence[int] | None = None,
          epsilon_levels: Sequence[float] = EPSILON_LEVELS,
          n_reps: int = 1000, include_baseline: bool = True,
          paradigms: Sequence[str] = ("joint", "marginal"),
          seed: int = 0, variant: str = "benchmark", style: str = "auto",
          support_preserving: bool = False,
          grid: Sequence[Mapping[str, Any]] | None = None) -> list:
    from harness import runner

    m_points = tuple(runner.M_POINTS) if m_points is None else tuple(int(m) for m in m_points)
    grid = grid if grid is not None else admissible_grid(
        dims, circuits, epsilon_levels=epsilon_levels, seed=seed, variant=variant,
        style=style, support_preserving=support_preserving)

    out = []
    for entry in grid:
        if not entry.get("admissible"):
            continue
        for m in m_points:
            for paradigm in paradigms:
                methods: tuple[str, ...] = ()
                if paradigm == "marginal" and not entry.get("interior_marginals", True):
                    methods = MARGINAL_METHODS_INTERIOR_SAFE
                if include_baseline:
                    out.append(runner.Cell(campaign=campaign, circuit=entry["circuit"],
                                           d=int(entry["d"]), m=int(m),
                                           paradigm=paradigm, case="baseline",
                                           methods=methods, n_reps=n_reps))
                for eps in entry["epsilons"]:
                    out.append(runner.Cell(campaign=campaign, circuit=entry["circuit"],
                                           d=int(entry["d"]), m=int(m),
                                           paradigm=paradigm, operator=C3_OPERATOR,
                                           strength=float(eps), case="falsecase",
                                           methods=methods, n_reps=n_reps))
    return out


def manifest(**kwargs) -> dict[str, Any]:
    grid_kw = {k: kwargs[k] for k in ("dims", "circuits", "epsilon_levels", "seed",
                                      "variant", "style", "support_preserving")
               if k in kwargs}
    grid = admissible_grid(**grid_kw)
    cell_list = cells(grid=grid, **kwargs)
    admitted = [e for e in grid if e.get("admissible")]
    excluded = [{"circuit": e["circuit"], "d": e["d"], "reason": e["reason"]}
                for e in grid if not e.get("admissible")]
    n_reps = kwargs.get("n_reps", 1000)
    shared = {(c.campaign, c.circuit, c.d, c.m, c.operator, c.strength, c.case)
              for c in cell_list}
    return {
        "campaign": I9_CAMPAIGN,
        "n_cells": len(cell_list),
        "n_runs": sum(c.n_runs for c in cell_list),
        "n_cells_falsecase": sum(1 for c in cell_list if c.case == "falsecase"),
        "n_cells_plan_equivalent": len(shared),
        "n_runs_plan_equivalent": len(shared) * n_reps,
        "n_admissible_circuit_widths": len(admitted),
        "n_excluded_circuit_widths": len(excluded),
        "excluded": excluded,
        "n_cells_marginal_restricted": sum(
            1 for c in cell_list if c.paradigm == "marginal"
            and tuple(c.methods) == MARGINAL_METHODS_INTERIOR_SAFE),
        "marginal_restriction_reason": (
            "nulls with a degenerate qubit marginal drop bootstrap/sequential, which "
            "validate strict_interior; see MARGINAL_METHODS_INTERIOR_SAFE"),
        "circuits_with_interior_marginals": sorted(
            {e["circuit"] for e in admitted if e.get("interior_marginals")}),
        "epsilon_levels": tuple(kwargs.get("epsilon_levels", EPSILON_LEVELS)),
        "plan_upper_bound_cells": 540,
        "plan_upper_bound_runs": 540 * n_reps,
        "note": ("marginal-paradigm falsecase rows measure the rejection rate under a "
                 "MARGINAL-EQUIVALENT alternative (a true null for that paradigm), not "
                 "power; runner books them as TP by its case convention"),
    }


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="C3 dependence-only alternatives (I9)")
    ap.add_argument("--classify", action="store_true", help="print the theorem table")
    ap.add_argument("--dims", type=int, nargs="*", default=None)
    ap.add_argument("--manifest", action="store_true", help="print realised I9 counts")
    ap.add_argument("--fallback", action="store_true",
                    help="diagnostic: what superposed-input variants would buy")
    args = ap.parse_args(argv)

    if args.classify or not (args.manifest or args.fallback):
        dims = tuple(args.dims) if args.dims else DEFAULT_CLASSIFY_DIMS
        print(classification_table(dims=dims))
    if args.fallback:
        for row in fallback_report(tuple(args.dims) if args.dims else DEFAULT_CLASSIFY_DIMS):
            print(row)
    if args.manifest:
        info = manifest(**({"dims": tuple(args.dims)} if args.dims else {}))
        for key, value in info.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
