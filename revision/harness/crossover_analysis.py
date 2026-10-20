
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from harness.ci_reanalysis import (
    CI_ALPHA,
    LEGACY_N_REPS,
    REPO_ROOT,
    load_harness_cells,
    load_legacy_cells,
    wilson_ci,
)

POWER_TARGET = 0.80

DESIGN_ALPHA = 0.05
DESIGN_BETA = 0.20

DESIGN_P0 = 0.5
DESIGN_DELTA = 0.02

DESIGN_W = 0.40

EXPECTED_BINOMIAL_WORST = 4904
EXPECTED_BINOMIAL_BEST = 484
EXPECTED_BONFERRONI_RANGE = (5900, 8600)
EXPECTED_JOINT_RANGE = (65, 900)

CANONICAL_FALSECASE_ROWS = {"marginal": 2640, "joint": 1620}

_GROUP_KEYS = ("source", "paradigm", "campaign", "circuit", "method", "fault",
               "operator", "strength", "d")




def _crossover_one(m, k, n, *, target: float = POWER_TARGET, ci_alpha: float = CI_ALPHA):
    power = k / n
    lo, hi = wilson_ci(k, n, alpha=ci_alpha)
    out = {
        "n_m_points": int(len(m)),
        "m_min": int(m[0]),
        "m_max": int(m[-1]),
        "power_at_m_min": float(power[0]),
        "power_at_m_max": float(power[-1]),
        "monotone_in_m": bool(np.all(np.diff(power) >= 0.0)),
        "m_cross_first": np.nan,
        "power_at_cross": np.nan,
        "wilson_lo_at_cross": np.nan,
        "wilson_hi_at_cross": np.nan,
        "m_cross_interp": np.nan,
        "m_cross_wilson_lo": np.nan,
        "censoring": "interior",
    }
    reached = np.flatnonzero(power >= target)
    if reached.size == 0:
        out["censoring"] = "right"
        return out
    i = int(reached[0])
    out["m_cross_first"] = float(m[i])
    out["power_at_cross"] = float(power[i])
    out["wilson_lo_at_cross"] = float(lo[i])
    out["wilson_hi_at_cross"] = float(hi[i])
    if i == 0:
        out["censoring"] = "left"
        out["m_cross_interp"] = float(m[0])
    else:
        p_lo, p_hi = power[i - 1], power[i]
        span = p_hi - p_lo
        frac = 0.0 if span <= 0 else (target - p_lo) / span
        out["m_cross_interp"] = float(m[i - 1] + frac * (m[i] - m[i - 1]))
    solid = np.flatnonzero(lo >= target)
    if solid.size:
        out["m_cross_wilson_lo"] = float(m[int(solid[0])])
    return out


def empirical_crossovers(cells, *, target: float = POWER_TARGET,
                         ci_alpha: float = CI_ALPHA):
    import pandas as pd

    false = cells[cells["case"] == "falsecase"]
    if false.empty:
        raise ValueError("no falsecase rows: nothing to cross over")

    keys = [k for k in _GROUP_KEYS if k in false.columns]
    rows = []
    for key, group in false.groupby(keys, dropna=False, sort=True):
        group = group.sort_values("m")
        if group["m"].duplicated().any():
            raise ValueError(f"duplicate m values in curve {key}")
        summary = _crossover_one(
            group["m"].to_numpy(dtype=float),
            group["k"].to_numpy(dtype=float),
            group["n"].to_numpy(dtype=float),
            target=target,
            ci_alpha=ci_alpha,
        )
        rows.append(dict(zip(keys, key)) | summary)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)




@lru_cache(maxsize=None)
def apriori_marginal(d: int, *, p0: float = DESIGN_P0, delta: float = DESIGN_DELTA,
                     alpha: float = DESIGN_ALPHA, beta: float = DESIGN_BETA,
                     exact: bool = False) -> float:
    from harness import sample_size

    if exact:
        return float(sample_size.bonferroni_marginal_n(p0, delta, alpha, beta, d=d))
    return float(np.ceil(sample_size.bonferroni_marginal_n_normal(p0, delta, alpha, beta, d=d)))


@lru_cache(maxsize=None)
def apriori_joint(d: int, *, w: float = DESIGN_W, alpha: float = DESIGN_ALPHA,
                  beta: float = DESIGN_BETA) -> float:
    from harness import sample_size

    return float(sample_size.chisq_joint_n(w, 2 ** int(d) - 1, alpha, beta))


@lru_cache(maxsize=None)
def implied_delta(m: int, d: int, *, p0: float = DESIGN_P0, alpha: float = DESIGN_ALPHA,
                  beta: float = DESIGN_BETA, tol: float = 1e-5) -> float:
    from harness import sample_size

    if not np.isfinite(m) or m < 2:
        return float("nan")
    level = alpha / max(int(d), 1)
    target = 1.0 - beta

    def power(delta: float) -> float:
        return sample_size.exact_binomial_power(int(m), p0, delta, level)

    hi = min(0.5 - 1e-6, 1.0 - p0 - 1e-6)
    if power(hi) < target:
        return float("nan")
    lo = 1e-6
    if power(lo) >= target:
        return float(lo)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if power(mid) >= target:
            hi = mid
        else:
            lo = mid
    return float(hi)


@lru_cache(maxsize=None)
def implied_w(m: int, d: int, *, alpha: float = DESIGN_ALPHA, beta: float = DESIGN_BETA,
              tol: float = 1e-7) -> float:
    from harness import sample_size

    if not np.isfinite(m) or m < 2:
        return float("nan")
    df = 2 ** int(d) - 1
    target = 1.0 - beta

    def power(w: float) -> float:
        return sample_size.chisq_joint_power(int(m), w, df, alpha)

    lo, hi = 1e-6, 1.0
    while power(hi) < target:
        hi *= 2.0
        if hi > 1e3:
            return float("nan")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if power(mid) >= target:
            hi = mid
        else:
            lo = mid
    return float(hi)


def attach_apriori(crossovers, *, exact_bonferroni: bool = False):
    out = crossovers.copy()
    a_m, a_conv, a_eff, i_eff, i_kind = [], [], [], [], []
    for row in out.itertuples(index=False):
        d = int(row.d)
        m_cross = getattr(row, "m_cross_first", np.nan)
        usable = np.isfinite(m_cross) and row.censoring == "interior"
        if row.paradigm == "marginal":
            a_m.append(apriori_marginal(d, exact=exact_bonferroni))
            a_conv.append(
                f"exact-binomial @ alpha/d, p0={DESIGN_P0}, |dp|={DESIGN_DELTA}"
                if exact_bonferroni else
                f"normal-approx @ alpha/d, p0={DESIGN_P0}, |dp|={DESIGN_DELTA}"
            )
            a_eff.append(DESIGN_DELTA)
            i_eff.append(implied_delta(int(m_cross), d) if usable else np.nan)
            i_kind.append("delta_p")
        else:
            a_m.append(apriori_joint(d))
            a_conv.append(f"noncentral-chi2, w={DESIGN_W}, df=2^d-1")
            a_eff.append(DESIGN_W)
            i_eff.append(implied_w(int(m_cross), d) if usable else np.nan)
            i_kind.append("cohen_w")
    out["a_priori_m"] = a_m
    out["a_priori_convention"] = a_conv
    out["a_priori_effect"] = a_eff
    out["implied_effect"] = i_eff
    out["implied_effect_kind"] = i_kind
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ratio_empirical_over_apriori"] = out["m_cross_first"] / out["a_priori_m"]
    return out




def apriori_reference_table(*, exact_bonferroni: bool = False):
    import pandas as pd

    from harness import sample_size

    rows = [
        {"design": "single-qubit exact binomial", "cell": "p0=0.5, |dp|=0.02",
         "normal": sample_size.normal_approx_n(0.5, 0.02, DESIGN_ALPHA, DESIGN_BETA),
         "exact": float(sample_size.exact_binomial_n(0.5, 0.02, DESIGN_ALPHA, DESIGN_BETA))},
        {"design": "single-qubit exact binomial", "cell": "p0=0.1, |dp|=0.04",
         "normal": sample_size.normal_approx_n(0.1, 0.04, DESIGN_ALPHA, DESIGN_BETA),
         "exact": float(sample_size.exact_binomial_n(0.1, 0.04, DESIGN_ALPHA, DESIGN_BETA))},
    ]
    for d in range(2, 13):
        rows.append({
            "design": "Bonferroni marginal", "cell": f"d={d}, p0=0.5, |dp|=0.02",
            "normal": apriori_marginal(d, exact=False),
            "exact": apriori_marginal(d, exact=True) if exact_bonferroni else np.nan,
        })
    for d in range(2, 11):
        rows.append({"design": "joint chi-square", "cell": f"d={d}, w={DESIGN_W}",
                     "normal": np.nan, "exact": apriori_joint(d)})
    return pd.DataFrame(rows)


def check_expected(reference, crossovers=None) -> list[tuple[str, bool, str]]:
    checks = []
    binom = reference[reference["design"] == "single-qubit exact binomial"]
    worst = float(binom.iloc[0]["normal"])
    best = float(binom.iloc[1]["normal"])
    checks.append((f"binomial m ~ {EXPECTED_BINOMIAL_WORST} at (0.5, 0.02)",
                   abs(worst - EXPECTED_BINOMIAL_WORST) <= 5, f"{worst:.1f}"))
    checks.append((f"binomial m ~ {EXPECTED_BINOMIAL_BEST} at (0.1, 0.04)",
                   abs(best - EXPECTED_BINOMIAL_BEST) <= 5, f"{best:.1f}"))
    bonf = reference[reference["design"] == "Bonferroni marginal"]["normal"]
    lo, hi = EXPECTED_BONFERRONI_RANGE
    checks.append((f"Bonferroni m in {lo}-{hi} for d=2..12",
                   bool(bonf.min() >= lo and bonf.max() <= hi),
                   f"{bonf.min():.0f}-{bonf.max():.0f}"))
    joint = reference[reference["design"] == "joint chi-square"]["exact"]
    lo, hi = EXPECTED_JOINT_RANGE
    checks.append((f"joint chi-square m in {lo}-{hi} for d=2..10 (w={DESIGN_W})",
                   bool(joint.min() >= lo and joint.max() <= hi),
                   f"{joint.min():.0f}-{joint.max():.0f}"))
    checks.append(("p0=0.5 sizing exceeds the manuscript's 4,165 cap",
                   worst > 4165, f"{worst:.1f} > 4165"))
    if crossovers is not None:
        legacy = crossovers[crossovers["source"] == "legacy"]
        canonical = legacy[legacy["fault"] == "bitflip"]
        for paradigm, expected in CANONICAL_FALSECASE_ROWS.items():
            got = int((canonical["paradigm"] == paradigm).sum()) * _M_POINTS_LEGACY
            checks.append((f"canonical {paradigm} falsecase rows = {expected}",
                           got == expected, f"{got}"))
    return checks


_M_POINTS_LEGACY = 10




def build(root=REPO_ROOT, campaign=None, *, target: float = POWER_TARGET,
          exact_bonferroni: bool = False):
    import pandas as pd

    frames = []
    if root is not None:
        frames.append(load_legacy_cells(root))
    if campaign is not None:
        frames.append(load_harness_cells(campaign))
    if not frames:
        raise ValueError("nothing to load: pass a legacy root or a campaign glob")
    cells = pd.concat(frames, ignore_index=True)
    crossovers = empirical_crossovers(cells, target=target)
    return attach_apriori(crossovers, exact_bonferroni=exact_bonferroni)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='A priori sample sizes against the empirical power-0.80 crossover.')
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--campaign", default=None,
                        help="parquet glob / directory of new-harness results (optional)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "revision" / "analysis_out")
    parser.add_argument("--target", type=float, default=POWER_TARGET)
    parser.add_argument("--no-legacy", action="store_true")
    parser.add_argument("--exact-bonferroni", action="store_true",
                        help="discrete search for the Bonferroni sizes (~1 s per width)")
    args = parser.parse_args(argv)

    table = build(None if args.no_legacy else args.root, args.campaign,
                  target=args.target, exact_bonferroni=args.exact_bonferroni)
    reference = apriori_reference_table(exact_bonferroni=args.exact_bonferroni)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "a_priori_vs_empirical_crossover.csv"
    ref_path = args.out_dir / "a_priori_sizes.csv"
    table.to_csv(out_path, index=False)
    reference.to_csv(ref_path, index=False)

    print(f"crossover curves : {len(table):,d} rows -> {out_path}")
    print(f"a-priori sizes   : {len(reference):,d} rows -> {ref_path}")
    counts = table["censoring"].value_counts().to_dict()
    print(f"censoring        : {counts}")
    interior = table[table["censoring"] == "interior"]
    if len(interior):
        print(f"interior crossovers: median m = {interior['m_cross_first'].median():.0f}, "
              f"range {interior['m_cross_first'].min():.0f}-"
              f"{interior['m_cross_first'].max():.0f}")
    print(f"N per cell       : {LEGACY_N_REPS} (legacy)")
    failed = 0
    for claim, ok, detail in check_expected(reference, table):
        print(f"  [{'PASS' if ok else 'FAIL'}] {claim}  ({detail})")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
