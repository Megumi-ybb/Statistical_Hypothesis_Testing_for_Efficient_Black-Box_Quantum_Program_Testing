
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

LEGACY_N_REPS = 1000

NOMINAL_ALPHA = 0.05

CI_ALPHA = 0.05

CIRCUIT_ALIASES = {
    "BV": "BV",
    "GROVER": "Grover",
    "grovertest": "Grover",
    "grover": "Grover",
    "QFT": "QFT",
    "QPE": "QPE",
    "QWALK": "QWALK",
    "add": "Adder",
    "Adder": "Adder",
}

FAULT_COLUMNS = ("power_bitflip", "power_asymmetric", "power_depolarize")

_TIDY_COLUMNS = (
    "source",
    "paradigm",
    "circuit",
    "case",
    "d",
    "m",
    "method",
    "metric",
    "fault",
    "campaign",
    "operator",
    "strength",
    "k",
    "n",
    "rate",
    "reconstructed",
    "source_dir",
    "source_file",
)



def wilson_ci(k, n, alpha: float = CI_ALPHA):
    from scipy.stats import norm

    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = float(norm.isf(alpha / 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / denom
        half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
        lo = np.clip(centre - half, 0.0, 1.0)
        hi = np.clip(centre + half, 0.0, 1.0)
    bad = ~(n > 0)
    lo = np.where(bad, np.nan, lo)
    hi = np.where(bad, np.nan, hi)
    return lo, hi


def clopper_pearson_ci(k, n, alpha: float = CI_ALPHA):
    from scipy.stats import beta

    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = beta.ppf(alpha / 2.0, k, n - k + 1.0)
        hi = beta.isf(alpha / 2.0, k + 1.0, n - k)
    lo = np.where(k <= 0, 0.0, lo)
    hi = np.where(k >= n, 1.0, hi)
    bad = ~(n > 0)
    lo = np.where(bad, np.nan, np.nan_to_num(lo, nan=0.0))
    hi = np.where(bad, np.nan, np.nan_to_num(hi, nan=1.0))
    return lo, hi


def mc_se(k, n):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        se = np.sqrt(p * (1.0 - p) / n)
    return np.where(n > 0, se, np.nan)


def wilson_half_width(p: float, n: int, alpha: float = CI_ALPHA) -> float:
    lo, hi = wilson_ci(round(p * n), n, alpha=alpha)
    return float((hi - lo) / 2.0)


def counts_from_rate(rate, n: int = LEGACY_N_REPS, *, atol: float = 1e-6,
                     what: str = "rate"):
    rate = np.asarray(rate, dtype=float)
    scaled = rate * n
    k = np.round(scaled)
    finite = np.isfinite(scaled)
    if not np.allclose(scaled[finite], k[finite], atol=atol, rtol=0.0):
        worst = np.nanmax(np.abs(scaled[finite] - k[finite])) if finite.any() else np.nan
        raise ValueError(
            f"{what}: cannot reconstruct counts at N={n}; worst |rate*N - round| = "
            f"{worst:.3g} > {atol:g}.  The on-disk rates are not multiples of 1/{n}."
        )
    return np.where(finite, k, np.nan)




def _canonical_circuit(raw: str) -> str:
    try:
        return CIRCUIT_ALIASES[raw]
    except KeyError:
        raise ValueError(f"unknown circuit label {raw!r}") from None


def legacy_directories(root: Path | str = REPO_ROOT) -> list[tuple[Path, str, str, str]]:
    root = Path(root)
    out: list[tuple[Path, str, str, str]] = []

    joint_dir = root / "Joint" / "Experiment"
    for path in sorted(joint_dir.glob("joint_experiment_*_results")):
        if not path.is_dir():
            continue
        stem = path.name[len("joint_experiment_"):-len("_results")]
        case = "baseline"
        if stem.endswith("_falsecase"):
            case = "falsecase"
            stem = stem[: -len("_falsecase")]
        out.append((path, "joint", _canonical_circuit(stem), case))

    marg_dir = root / "Marginal" / "Experiment"
    for path in sorted(marg_dir.glob("marginal_experiment_*")):
        if not path.is_dir():
            continue
        name = path.name
        if "flase_" in name or "false_" in name:
            key = "flase_" if "flase_" in name else "false_"
            stem = name[name.find(key) + len(key): name.find("_nested_parallel_results")]
            out.append((path, "marginal", _canonical_circuit(stem), "falsecase"))
            continue
        prefix = "marginal_experiment_nested_parallel_results"
        if not name.startswith(prefix):
            continue
        stem = name[len(prefix):].lstrip("_") or "grovertest"
        out.append((path, "marginal", _canonical_circuit(stem), "baseline"))

    return out


def _tidy_baseline(df, meta: dict, *, strict: bool):
    import pandas as pd

    need = {"TP", "TN", "FP", "FN"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{meta['source_file']}: baseline file missing {sorted(missing)}")
    n = df["TP"] + df["TN"] + df["FP"] + df["FN"]
    if strict and not (n == LEGACY_N_REPS).all():
        raise ValueError(
            f"{meta['source_file']}: confusion counts do not sum to {LEGACY_N_REPS} "
            f"(min {int(n.min())}, max {int(n.max())})"
        )
    if "type_i_error" in df.columns:
        recorded = df["type_i_error"].to_numpy(dtype=float)
        implied = (df["FP"] / n).to_numpy(dtype=float)
        ok = np.isclose(recorded, implied, atol=1e-9, rtol=0.0) | ~np.isfinite(recorded)
        if strict and not ok.all():
            raise ValueError(
                f"{meta['source_file']}: recorded type_i_error disagrees with FP/N in "
                f"{int((~ok).sum())} rows"
            )
    return pd.DataFrame(
        {
            "source": "legacy",
            "paradigm": meta["paradigm"],
            "circuit": meta["circuit"],
            "case": "baseline",
            "d": df["n_qubits"].astype(int),
            "m": df["m_samples"].astype(int),
            "method": df["test_method"].astype(str),
            "metric": "type_i_error",
            "fault": "",
            "campaign": "legacy",
            "operator": "baseline",
            "strength": np.nan,
            "k": df["FP"].astype(int),
            "n": n.astype(int),
            "reconstructed": False,
            "source_dir": meta["source_dir"],
            "source_file": meta["source_file"],
        }
    )


def _tidy_falsecase(df, meta: dict, *, strict: bool):
    import pandas as pd

    frames = []
    for column in FAULT_COLUMNS:
        if column not in df.columns:
            continue
        rate = df[column].to_numpy(dtype=float)
        try:
            k = counts_from_rate(rate, LEGACY_N_REPS, what=f"{meta['source_file']}:{column}")
        except ValueError:
            if strict:
                raise
            k = np.round(rate * LEGACY_N_REPS)
        frames.append(
            pd.DataFrame(
                {
                    "source": "legacy",
                    "paradigm": meta["paradigm"],
                    "circuit": meta["circuit"],
                    "case": "falsecase",
                    "d": df["n_qubits"].astype(int),
                    "m": df["m_samples"].astype(int),
                    "method": df["test_method"].astype(str),
                    "metric": column,
                    "fault": column[len("power_"):],
                    "campaign": "legacy",
                    "operator": "bit_corruption",
                    "strength": np.nan,
                    "k": k.astype(int),
                    "n": LEGACY_N_REPS,
                    "reconstructed": True,
                    "source_dir": meta["source_dir"],
                    "source_file": meta["source_file"],
                }
            )
        )
    if not frames:
        raise ValueError(f"{meta['source_file']}: no power_* columns")
    return pd.concat(frames, ignore_index=True)


def load_legacy_cells(root: Path | str = REPO_ROOT, *, strict: bool = True):
    import pandas as pd

    frames = []
    for path, paradigm, circuit, case in legacy_directories(root):
        for csv in sorted(path.glob("*.csv")):
            raw = pd.read_csv(csv)
            meta = {
                "paradigm": paradigm,
                "circuit": circuit,
                "source_dir": path.name,
                "source_file": csv.name,
            }
            tidy = (_tidy_baseline if case == "baseline" else _tidy_falsecase)(
                raw, meta, strict=strict
            )
            frames.append(tidy)
    if not frames:
        raise FileNotFoundError(f"no legacy result CSVs under {root}")
    df = pd.concat(frames, ignore_index=True)
    df["rate"] = df["k"] / df["n"]
    return df[list(_TIDY_COLUMNS)]




def load_harness_cells(pattern, *, strict: bool = True):
    import pandas as pd

    from harness import io_store

    grid = io_store.read_grid(pattern)
    n = grid[["TP", "FP", "TN", "FN"]].sum(axis=1)
    if strict and "n_evaluated" in grid.columns:
        recorded = grid["n_evaluated"].to_numpy(dtype=float)
        ok = np.isclose(recorded, n.to_numpy(dtype=float)) | ~np.isfinite(recorded)
        if not ok.all():
            raise ValueError(
                f"parquet grid: n_evaluated disagrees with TP+FP+TN+FN in "
                f"{int((~ok).sum())} rows"
            )
    out = pd.DataFrame(
        {
            "source": "harness",
            "paradigm": grid.get("paradigm", pd.Series("", index=grid.index)),
            "circuit": grid["circuit"].astype(str),
            "case": grid["case"].astype(str),
            "d": grid["d"].astype(int),
            "m": grid["m"].astype(int),
            "method": grid["method"].astype(str),
            "metric": np.where(grid["case"].to_numpy() == "baseline", "type_i_error", "power"),
            "fault": grid.get("operator", pd.Series("", index=grid.index)).astype(str),
            "campaign": grid.get("campaign", pd.Series("", index=grid.index)).astype(str),
            "operator": grid.get("operator", pd.Series("", index=grid.index)).astype(str),
            "strength": grid.get("strength", pd.Series(np.nan, index=grid.index)),
            "k": (grid["TP"] + grid["FP"]).astype(int),
            "n": n.astype(int),
            "reconstructed": False,
            "source_dir": "",
            "source_file": "",
        }
    )
    out.loc[out["case"] == "baseline", "fault"] = ""
    out["rate"] = out["k"] / out["n"]
    return out[list(_TIDY_COLUMNS)]




def add_intervals(cells, *, alpha: float = CI_ALPHA):
    out = cells.copy()
    k = out["k"].to_numpy(dtype=float)
    n = out["n"].to_numpy(dtype=float)
    w_lo, w_hi = wilson_ci(k, n, alpha=alpha)
    c_lo, c_hi = clopper_pearson_ci(k, n, alpha=alpha)
    out["wilson_lo"] = w_lo
    out["wilson_hi"] = w_hi
    out["wilson_half"] = (w_hi - w_lo) / 2.0
    out["cp_lo"] = c_lo
    out["cp_hi"] = c_hi
    out["cp_half"] = (c_hi - c_lo) / 2.0
    out["mc_se"] = mc_se(k, n)
    return out


REPRESENTATIVE_RULES: tuple[tuple[str, str], ...] = (
    ("max_type_i", "largest Type I cell -- backs the reported worst case"),
    ("median_type_i", "cell realising the grid median Type I (ties -> smallest d, m)"),
    ("widest_wilson", "largest Wilson half-width -- the global precision claim"),
    ("max_power_small_m", "highest power at the smallest m -- the 'detected at m=100' claim"),
    ("conservative_corner", "m=50, d in {9,10}: the lumped-chi-square corner (I9)"),
)

CORNER_M = 50
CORNER_D = (9, 10)


def representative_cells(ci):
    import pandas as pd

    picks = []
    group_keys = ["source", "paradigm", "method"]
    sort_keys = ["circuit", "d", "m", "fault"]

    def _take(frame, rule, by, ascending):
        if frame.empty:
            return
        ordered = frame.sort_values(by + sort_keys, ascending=[ascending] * len(by) + [True] * 4)
        first = ordered.groupby(group_keys, dropna=False, sort=True).head(1).copy()
        first["rule"] = rule
        picks.append(first)

    baseline = ci[ci["metric"] == "type_i_error"]
    power = ci[ci["metric"] != "type_i_error"]

    _take(baseline, "max_type_i", ["rate"], False)

    if not baseline.empty:
        med = baseline.groupby(group_keys, dropna=False)["rate"].transform("median")
        near = baseline.assign(_gap=(baseline["rate"] - med).abs())
        ordered = near.sort_values(["_gap"] + sort_keys)
        first = ordered.groupby(group_keys, dropna=False, sort=True).head(1).drop(columns="_gap")
        first = first.copy()
        first["rule"] = "median_type_i"
        picks.append(first)

    _take(ci, "widest_wilson", ["wilson_half"], False)

    if not power.empty:
        small_m = power[power["m"] == power["m"].min()]
        _take(small_m, "max_power_small_m", ["rate"], False)

    corner = ci[(ci["m"] == CORNER_M) & (ci["d"].isin(CORNER_D))]
    _take(corner, "conservative_corner", ["rate"], False)

    if not picks:
        return ci.head(0).assign(rule=pd.Series(dtype=str))
    out = pd.concat(picks, ignore_index=True)
    return out.sort_values(["rule"] + group_keys + sort_keys).reset_index(drop=True)




def precision_summary(ci) -> dict:
    n_cell = ci["n"].to_numpy(dtype=float)
    lo, hi = wilson_ci(n_cell / 2.0, n_cell)
    worst_at_own_n = (hi - lo) / 2.0
    return {
        "wilson_half_at_p50": wilson_half_width(0.5, LEGACY_N_REPS),
        "wilson_half_at_p05": wilson_half_width(0.05, LEGACY_N_REPS),
        "max_wilson_half_observed": float(np.nanmax(ci["wilson_half"].to_numpy())),
        "max_cp_half_observed": float(np.nanmax(ci["cp_half"].to_numpy())),
        "max_half_excess": float(np.nanmax(ci["wilson_half"].to_numpy() - worst_at_own_n)),
        "n_min": int(np.nanmin(n_cell)),
        "n_max": int(np.nanmax(n_cell)),
        "n_cells": int(len(ci)),
        "n_reconstructed": int(ci["reconstructed"].sum()),
    }


def check_expected(summary: dict) -> list[tuple[str, bool, str]]:
    checks = []
    half50 = summary["wilson_half_at_p50"]
    checks.append(
        ("Wilson half-width <= 0.031 at p=0.5, N=1000", half50 <= 0.031,
         f"{half50:.4f}")
    )
    half05 = summary["wilson_half_at_p05"]
    checks.append(
        ("Wilson half-width ~ 0.014 near p=0.05, N=1000", abs(half05 - 0.014) <= 0.002,
         f"{half05:.4f}")
    )
    excess = summary["max_half_excess"]
    checks.append(
        ("no cell exceeds the p=0.5 worst case at its own N", excess <= 1e-12,
         f"max excess {excess:.2e}, worst observed "
         f"{summary['max_wilson_half_observed']:.4f} at N in "
         f"[{summary['n_min']}, {summary['n_max']}]")
    )
    return checks




def build(root: Path | str = REPO_ROOT, campaign=None, *, strict: bool = True,
          alpha: float = CI_ALPHA):
    import pandas as pd

    frames = []
    if root is not None:
        frames.append(load_legacy_cells(root, strict=strict))
    if campaign is not None:
        frames.append(load_harness_cells(campaign, strict=strict))
    if not frames:
        raise ValueError("nothing to load: pass a legacy root or a campaign glob")
    cells = pd.concat(frames, ignore_index=True)
    return add_intervals(cells, alpha=alpha)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Per-cell binomial confidence intervals for every stored rejection rate.')
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="repository root holding Joint/ and Marginal/")
    parser.add_argument("--campaign", default=None,
                        help="parquet glob / directory of new-harness results (optional)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "revision" / "analysis_out")
    parser.add_argument("--alpha", type=float, default=CI_ALPHA)
    parser.add_argument("--no-legacy", action="store_true", help="skip the legacy CSVs")
    parser.add_argument("--lax", action="store_true",
                        help="warn instead of raising on count/rate inconsistencies")
    args = parser.parse_args(argv)

    ci = build(None if args.no_legacy else args.root, args.campaign,
               strict=not args.lax, alpha=args.alpha)
    reps = representative_cells(ci)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_cell = args.out_dir / "per_cell_ci.csv"
    rep_path = args.out_dir / "representative_cells_ci.csv"
    ci.to_csv(per_cell, index=False)
    reps.to_csv(rep_path, index=False)

    summary = precision_summary(ci)
    print(f"per-cell CIs      : {len(ci):,d} rows -> {per_cell}")
    print(f"representative    : {len(reps):,d} rows -> {rep_path}")
    print(f"reconstructed k   : {summary['n_reconstructed']:,d} rows "
          f"(falsecase power x {LEGACY_N_REPS}, allclose-guarded)")
    print(f"Wilson half-width : {summary['wilson_half_at_p50']:.4f} at p=0.5, "
          f"{summary['wilson_half_at_p05']:.4f} at p=0.05 (N={LEGACY_N_REPS})")
    print(f"worst observed    : Wilson {summary['max_wilson_half_observed']:.4f}, "
          f"Clopper-Pearson {summary['max_cp_half_observed']:.4f} "
          f"(N in [{summary['n_min']}, {summary['n_max']}])")
    failed = 0
    for claim, ok, detail in check_expected(summary):
        print(f"  [{'PASS' if ok else 'FAIL'}] {claim}  ({detail})")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
