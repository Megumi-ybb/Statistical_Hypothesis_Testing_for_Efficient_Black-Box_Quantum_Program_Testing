
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

N_REPS: int = 1000
ALPHA: float = 0.05
FAULT_COLUMNS: dict[str, str] = {
    "power_bitflip": "bitflip",
    "power_asymmetric": "asymmetric",
    "power_depolarize": "depolarize",
}
CIRCUIT_NAMES: dict[str, str] = {
    "BV": "BV", "GROVER": "Grover", "QFT": "QFT",
    "QPE": "QPE", "QWALK": "QWALK", "add": "Adder",
}
CORNER_POWER_FLOOR: float = 0.68

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = _REPO_ROOT / "Joint" / "combined_results.csv"
DEFAULT_OUT = Path(__file__).resolve().parent


def wilson_interval(successes, n, alpha: float = ALPHA):
    from scipy.stats import norm

    k = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    denom = n + z ** 2
    centre = (k + z ** 2 / 2.0) / denom
    half = (z / denom) * np.sqrt(k * (n - k) / n + z ** 2 / 4.0)
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def load(csv_path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"n_qubits", "m_samples", "test_method", "experiment", "case"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
    df = df.copy()
    df["circuit"] = df["experiment"].map(CIRCUIT_NAMES).fillna(df["experiment"])
    return df


def reconstruct_counts(power: pd.Series, n_reps: int = N_REPS,
                       *, atol: float = 1e-9, strict: bool = True) -> pd.Series:
    successes = np.round(power.to_numpy(dtype=float) * n_reps)
    back = successes / n_reps
    bad = ~np.isclose(back, power.to_numpy(dtype=float), atol=atol, rtol=0.0)
    if bad.any():
        worst = float(np.max(np.abs(back - power.to_numpy(dtype=float))[bad]))
        msg = (f"{int(bad.sum())} stored power values are not multiples of 1/{n_reps} "
               f"(worst deviation {worst:.3e}); the count reconstruction is unsafe")
        if strict:
            raise ValueError(msg)
        raise UserWarning(msg)
    return pd.Series(successes.astype(np.int64), index=power.index)


def power_by_faultmodel(df: pd.DataFrame, *, n_reps: int = N_REPS,
                        alpha: float = ALPHA) -> pd.DataFrame:
    fc = df[df["case"] == "falsecase"]
    if fc.empty:
        raise ValueError("no falsecase rows in the input grid")
    keep = ["circuit", "experiment", "n_qubits", "m_samples", "test_method",
            "measurement_reduction_ratio", *FAULT_COLUMNS]
    long = fc[keep].melt(
        id_vars=[c for c in keep if c not in FAULT_COLUMNS],
        value_vars=list(FAULT_COLUMNS),
        var_name="fault_column", value_name="power",
    )
    long = long.dropna(subset=["power"]).reset_index(drop=True)
    long["fault_model"] = long["fault_column"].map(FAULT_COLUMNS)
    long["n"] = int(n_reps)
    long["successes"] = reconstruct_counts(long["power"], n_reps)
    lo, hi = wilson_interval(long["successes"], long["n"], alpha)
    long["wilson_lo"], long["wilson_hi"] = lo, hi
    long["mc_se"] = np.sqrt(long["power"] * (1.0 - long["power"]) / long["n"])

    key = ["circuit", "n_qubits", "m_samples", "fault_model"]
    wide = long.pivot_table(index=key, columns="test_method", values="power")
    for method in ("chi_square", "mmdagg"):
        if method in wide.columns:
            long = long.merge(wide[[method]].rename(columns={method: f"{method}_power"}),
                              on=key, how="left")
    if {"chi_square_power", "mmdagg_power"} <= set(long.columns):
        long["mmdagg_minus_chisq"] = long["mmdagg_power"] - long["chi_square_power"]
    long = long.drop(columns=["fault_column"]).rename(
        columns={"n_qubits": "d", "m_samples": "m"})
    return long.sort_values(["circuit", "d", "m", "fault_model", "test_method"]
                            ).reset_index(drop=True)


def typei_conservative_corner(df: pd.DataFrame, *, n_reps: int = N_REPS,
                              alpha: float = ALPHA,
                              power_floor: float = CORNER_POWER_FLOOR) -> pd.DataFrame:
    bl = df[df["case"] == "baseline"].copy()
    if bl.empty:
        raise ValueError("no baseline rows in the input grid")
    counts = bl[["TP", "TN", "FP", "FN"]].sum(axis=1)
    if not np.allclose(counts.dropna(), n_reps):
        raise ValueError(f"baseline confusion counts do not sum to {n_reps}: "
                         f"{sorted(set(counts.dropna().tolist()))[:5]}")
    bl["n"] = counts.astype(np.int64)
    bl["false_positives"] = bl["FP"].astype(np.int64)
    bl["type_i"] = bl["false_positives"] / bl["n"]
    stored = bl["type_i_error"].to_numpy(dtype=float)
    if not np.allclose(stored, bl["type_i"].to_numpy(dtype=float), atol=1e-9,
                       equal_nan=True):
        raise ValueError("stored type_i_error disagrees with FP/(FP+TN)")
    lo, hi = wilson_interval(bl["false_positives"], bl["n"], alpha)
    bl["wilson_lo"], bl["wilson_hi"] = lo, hi
    bl["conservative"] = bl["wilson_hi"] < alpha
    bl["liberal"] = bl["wilson_lo"] > alpha

    fc = df[df["case"] == "falsecase"]
    key = ["circuit", "n_qubits", "m_samples"]
    powers = (fc[fc["test_method"] == "mmdagg"][key + list(FAULT_COLUMNS)]
              .rename(columns={c: f"mmdagg_{v}" for c, v in FAULT_COLUMNS.items()}))
    chi = (fc[fc["test_method"] == "chi_square"][key + ["power_depolarize"]]
           .rename(columns={"power_depolarize": "chi_square_depolarize"}))
    out = (bl[key + ["experiment", "test_method", "n", "false_positives", "type_i",
                     "wilson_lo", "wilson_hi", "conservative", "liberal",
                     "measurement_reduction_ratio"]]
           .merge(powers, on=key, how="left").merge(chi, on=key, how="left"))
    out["corner"] = (out["test_method"].eq("chi_square") & out["conservative"]
                     & out["mmdagg_depolarize"].ge(power_floor))
    return out.rename(columns={"n_qubits": "d", "m_samples": "m"}).sort_values(
        ["test_method", "circuit", "d", "m"]).reset_index(drop=True)


def summarise(power: pd.DataFrame, typei: pd.DataFrame, *,
              alpha: float = ALPHA) -> list[str]:
    lines = [f"rows: power_by_faultmodel={len(power)}  typei_corner={len(typei)}"]
    chi = typei[typei["test_method"] == "chi_square"]
    cons = chi[chi["conservative"]]
    lines.append(
        f"chi-square baseline: {len(cons)}/{len(chi)} cells conservative at alpha={alpha} "
        f"(Wilson upper < alpha); {int(chi['liberal'].sum())} liberal"
    )
    if not cons.empty:
        lines.append(
            "  conservative corner: d in "
            f"{sorted(cons['d'].unique().tolist())}, m in "
            f"{sorted(cons['m'].unique().tolist())}, circuits "
            f"{sorted(cons['circuit'].unique().tolist())}"
        )
        corner = typei[typei["corner"]]
        if not corner.empty:
            lines.append(
                f"  of those, {len(corner)} keep MMDAgg depolarize power >= "
                f"{CORNER_POWER_FLOOR} (min {corner['mmdagg_depolarize'].min():.3f})"
            )
        gap = cons["mmdagg_depolarize"].dropna()
        if not gap.empty:
            lines.append(f"  MMDAgg depolarize power over conservative cells: "
                         f"min {gap.min():.3f}, median {gap.median():.3f}")
    for fault in sorted(power["fault_model"].unique()):
        sub = power[power["fault_model"] == fault]
        med = sub.pivot_table(index="test_method", values="power", aggfunc="median")
        digest = ", ".join(f"{m}={v:.3f}" for m, v in med["power"].items())
        lines.append(f"median power [{fault}]: {digest}")
    if "mmdagg_minus_chisq" in power.columns:
        diff = power.drop_duplicates(["circuit", "d", "m", "fault_model"])[
            "mmdagg_minus_chisq"].dropna()
        lines.append(f"mmdagg - chi_square power: median {diff.median():+.3f}, "
                     f"mmdagg ahead in {int((diff > 0).sum())}/{len(diff)} cells")
    return lines


def run(csv_path: Path | str = DEFAULT_CSV, out_dir: Path | str = DEFAULT_OUT, *,
        n_reps: int = N_REPS, alpha: float = ALPHA) -> dict[str, Path]:
    df = load(csv_path)
    power = power_by_faultmodel(df, n_reps=n_reps, alpha=alpha)
    typei = typei_conservative_corner(df, n_reps=n_reps, alpha=alpha)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "power": out_dir / "r19_power_by_faultmodel.csv",
        "typei": out_dir / "r19_typei_conservative_corner.csv",
    }
    power.to_csv(paths["power"], index=False)
    typei.to_csv(paths["typei"], index=False)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Lumped chi-square against MMDAgg on the original joint grid.')
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-reps", type=int, default=N_REPS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    df = load(args.csv)
    power = power_by_faultmodel(df, n_reps=args.n_reps, alpha=args.alpha)
    typei = typei_conservative_corner(df, n_reps=args.n_reps, alpha=args.alpha)
    args.out.mkdir(parents=True, exist_ok=True)
    power.to_csv(args.out / "r19_power_by_faultmodel.csv", index=False)
    typei.to_csv(args.out / "r19_typei_conservative_corner.csv", index=False)
    if not args.quiet:
        print(f"read {args.csv} ({len(df)} rows)")
        for line in summarise(power, typei, alpha=args.alpha):
            print(line)
        print(f"wrote {args.out / 'r19_power_by_faultmodel.csv'}")
        print(f"wrote {args.out / 'r19_typei_conservative_corner.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
