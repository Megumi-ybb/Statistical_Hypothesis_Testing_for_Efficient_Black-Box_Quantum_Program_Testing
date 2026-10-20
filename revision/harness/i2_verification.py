
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

from harness.ci_reanalysis import (
    CIRCUIT_ALIASES,
    LEGACY_N_REPS,
    REPO_ROOT,
    load_legacy_cells,
    wilson_ci,
)

DEFAULT_CSV = REPO_ROOT / "Marginal" / "combined_results.csv"

METHOD_LABELS = {
    "auto": "adaptive",
    "bootstrap": "bootstrap",
    "sequential": "sequential",
    "tolerance": "tolerance",
}

INFERENTIAL = ("auto", "bootstrap", "sequential")

ENTANGLING = ("Grover", "QWALK", "QPE", "QFT")

CIRCUITS = ("BV", "Grover", "QFT", "QPE", "QWALK", "Adder")

MANUSCRIPT = {
    "adaptive_median": (0.060, 0.0005),
    "adaptive_max": (0.087, 0.0005),
    "bootstrap_median": (0.055, 0.0005),
    "sequential_median": (0.048, 0.0005),
    "partition_adaptive_entangling": (0.0599, 0.0005),
    "partition_adaptive_non_entangling": (0.0607, 0.0005),
    "partition_difference": (0.002, 0.0020),
}




def load_baseline(csv_path=DEFAULT_CSV):
    import pandas as pd

    df = pd.read_csv(csv_path)
    baseline = df[df["case"] == "baseline"].copy()
    if baseline.empty:
        raise ValueError(f"{csv_path}: no rows with case='baseline'")
    baseline["circuit"] = baseline["experiment"].map(CIRCUIT_ALIASES)
    if baseline["circuit"].isna().any():
        unknown = sorted(baseline.loc[baseline["circuit"].isna(), "experiment"].unique())
        raise ValueError(f"unmapped experiment labels: {unknown}")
    baseline["method_label"] = baseline["test_method"].map(METHOD_LABELS)
    baseline = baseline.rename(columns={"n_qubits": "d", "m_samples": "m",
                                        "type_i_error": "type_i"})
    return baseline[["circuit", "d", "m", "test_method", "method_label", "type_i"]]


def cross_check_against_raw(baseline, root=REPO_ROOT) -> tuple[bool, str]:
    raw = load_legacy_cells(root)
    raw = raw[(raw["paradigm"] == "marginal") & (raw["metric"] == "type_i_error")]
    merged = baseline.merge(
        raw[["circuit", "d", "m", "method", "k", "n", "rate"]],
        left_on=["circuit", "d", "m", "test_method"],
        right_on=["circuit", "d", "m", "method"],
        how="outer",
        indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        return False, f"{unmatched} rows do not join 1:1 with the raw CSVs"
    same = np.allclose(merged["type_i"].to_numpy(dtype=float),
                       merged["rate"].to_numpy(dtype=float), atol=1e-12)
    return same, ("combined CSV matches raw FP/N in all "
                  f"{len(merged):,d} cells" if same else "rate mismatch vs raw counts")




def per_method_summary(baseline):
    import pandas as pd

    rows = []
    for method, group in baseline.groupby("test_method", sort=True):
        rates = group["type_i"].to_numpy(dtype=float)
        imax = int(np.argmax(rates))
        peak = group.iloc[imax]
        k = round(float(peak["type_i"]) * LEGACY_N_REPS)
        lo, hi = wilson_ci(k, LEGACY_N_REPS)
        rows.append({
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "n_cells": int(len(rates)),
            "median": float(np.median(rates)),
            "iqr_lo": float(np.percentile(rates, 25)),
            "iqr_hi": float(np.percentile(rates, 75)),
            "mean": float(rates.mean()),
            "max": float(rates.max()),
            "max_cell": f"{peak['circuit']} d={int(peak['d'])} m={int(peak['m'])}",
            "max_wilson_lo": float(lo),
            "max_wilson_hi": float(hi),
            "min": float(rates.min()),
        })
    return pd.DataFrame(rows)


def d_trend(baseline, method: str = "auto"):
    import pandas as pd
    from scipy.stats import spearmanr

    sub = baseline[baseline["test_method"] == method]
    per_d = sub.groupby("d", sort=True)["type_i"].median().reset_index(name="median_type_i")
    rho, pval = spearmanr(sub["d"].to_numpy(dtype=float), sub["type_i"].to_numpy(dtype=float))
    fit = np.polyfit(sub["d"].to_numpy(dtype=float), sub["type_i"].to_numpy(dtype=float), 1)
    return per_d, {"spearman_rho": float(rho), "spearman_p": float(pval),
                   "ols_slope_per_qubit": float(fit[0])}


def partition_means(baseline, entangling=ENTANGLING, methods=INFERENTIAL):
    import pandas as pd

    rows = []
    ent = set(entangling)
    for method in methods:
        sub = baseline[baseline["test_method"] == method]
        is_ent = sub["circuit"].isin(ent)
        a = float(sub.loc[is_ent, "type_i"].mean())
        b = float(sub.loc[~is_ent, "type_i"].mean())
        rows.append({"method": method, "label": METHOD_LABELS.get(method, method),
                     "entangling": a, "non_entangling": b, "difference": a - b,
                     "abs_difference": abs(a - b)})
    return pd.DataFrame(rows)


def partition_brittleness(baseline, method: str = "auto"):
    sub = baseline[baseline["test_method"] == method]
    means = sub.groupby("circuit", sort=True)["type_i"].mean()
    counts = sub.groupby("circuit", sort=True)["type_i"].size()
    total, holds, diffs = 0, 0, []
    for size in range(1, len(CIRCUITS)):
        for combo in itertools.combinations(sorted(CIRCUITS), size):
            other = tuple(c for c in sorted(CIRCUITS) if c not in combo)
            if combo > other:
                continue
            a = float((means[list(combo)] * counts[list(combo)]).sum()
                      / counts[list(combo)].sum())
            b = float((means[list(other)] * counts[list(other)]).sum()
                      / counts[list(other)].sum())
            total += 1
            holds += a >= b
            diffs.append(abs(a - b))
    return {"n_partitions": total, "frac_entangling_larger": holds / total,
            "max_abs_difference": float(max(diffs)),
            "median_abs_difference": float(np.median(diffs))}




def build_claims(baseline):
    import pandas as pd

    summary = per_method_summary(baseline).set_index("method")
    partition = partition_means(baseline).set_index("method")

    recomputed = {
        "adaptive_median": float(summary.loc["auto", "median"]),
        "adaptive_max": float(summary.loc["auto", "max"]),
        "bootstrap_median": float(summary.loc["bootstrap", "median"]),
        "sequential_median": float(summary.loc["sequential", "median"]),
        "partition_adaptive_entangling": float(partition.loc["auto", "entangling"]),
        "partition_adaptive_non_entangling": float(partition.loc["auto", "non_entangling"]),
        "partition_difference": float(partition["abs_difference"].mean()),
    }
    descriptions = {
        "adaptive_median": "adaptive (auto) Type I, median over the grid",
        "adaptive_max": "adaptive (auto) Type I, max over the grid",
        "bootstrap_median": "bootstrap Type I, median over the grid",
        "sequential_median": "sequential Type I, median over the grid",
        "partition_adaptive_entangling": "adaptive Type I, entangling circuits (mean)",
        "partition_adaptive_non_entangling": "adaptive Type I, non-entangling (mean)",
        "partition_difference": "mean |entangling - non-entangling| over 3 methods",
    }
    rows = []
    for key, (quoted, tol) in MANUSCRIPT.items():
        got = recomputed[key]
        rows.append({"claim": key, "description": descriptions[key],
                     "manuscript": quoted, "recomputed": got,
                     "abs_error": abs(got - quoted), "tolerance": tol,
                     "verdict": "PASS" if abs(got - quoted) <= tol else "FAIL"})
    return pd.DataFrame(rows)


def report(baseline, *, root=REPO_ROOT, check_raw: bool = True) -> tuple[list[str], int]:
    lines: list[str] = []
    claims = build_claims(baseline)
    summary = per_method_summary(baseline)
    partition = partition_means(baseline)
    per_d, trend = d_trend(baseline)
    brittle = partition_brittleness(baseline)

    lines.append("I2 verification -- marginal Type I error, recomputed from on-disk cells")
    lines.append(f"source cells: {len(baseline):,d} baseline rows, N={LEGACY_N_REPS} each")
    if check_raw:
        ok, detail = cross_check_against_raw(baseline, root)
        lines.append(f"raw-count cross-check: [{'PASS' if ok else 'FAIL'}] {detail}")
        if not ok:
            claims = claims.copy()
    lines.append("")
    lines.append("manuscript vs recomputed")
    lines.append(f"  {'claim':<34}{'paper':>8}{'recomputed':>12}{'|err|':>9}  verdict")
    for row in claims.itertuples(index=False):
        lines.append(f"  {row.claim:<34}{row.manuscript:>8.4f}{row.recomputed:>12.4f}"
                     f"{row.abs_error:>9.4f}  {row.verdict}")
    lines.append("")
    lines.append("per-method summary (IQR is ACROSS GRID CELLS, not a CI -- I11)")
    lines.append(f"  {'method':<12}{'median':>8}{'IQR':>18}{'max':>8}"
                 f"{'  max-cell Wilson 95%':>24}  max cell")
    for row in summary.itertuples(index=False):
        lines.append(
            f"  {row.label:<12}{row.median:>8.4f}"
            f"{f'[{row.iqr_lo:.3f}, {row.iqr_hi:.3f}]':>18}{row.max:>8.4f}"
            f"{f'[{row.max_wilson_lo:.3f}, {row.max_wilson_hi:.3f}]':>24}  {row.max_cell}"
        )
    lines.append("")
    lines.append(f"adaptive d-trend: Spearman rho = {trend['spearman_rho']:+.3f} "
                 f"(p = {trend['spearman_p']:.2g}), OLS slope = "
                 f"{trend['ols_slope_per_qubit']:+.5f} per qubit")
    lines.append("  per-d median: " + ", ".join(
        f"d={int(r.d)}:{r.median_type_i:.3f}" for r in per_d.itertuples(index=False)))
    lines.append("")
    lines.append(f"entangling partition {ENTANGLING} vs the rest")
    for row in partition.itertuples(index=False):
        lines.append(f"  {row.label:<12}entangling {row.entangling:.4f}  "
                     f"non-entangling {row.non_entangling:.4f}  "
                     f"difference {row.difference:+.4f}")
    signs = {np.sign(v) for v in partition["difference"] if v != 0.0}
    lines.append(f"  directional? {'NO -- signs disagree across methods' if len(signs) > 1 else 'yes -- consistent sign'}")
    lines.append(f"  over all {brittle['n_partitions']} two-way circuit partitions "
                 f"(adaptive): entangling side larger in "
                 f"{brittle['frac_entangling_larger']:.0%}, "
                 f"max |difference| {brittle['max_abs_difference']:.4f}")
    failed = int((claims["verdict"] == "FAIL").sum())
    lines.append("")
    lines.append(f"{len(claims) - failed}/{len(claims)} claims reproduce")
    return lines, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Recompute the marginal Type I error rates quoted in the article.')
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="also write i2_claims.csv / i2_per_method.csv here")
    parser.add_argument("--no-raw-check", action="store_true",
                        help="skip the join back onto the raw per-experiment CSVs")
    args = parser.parse_args(argv)

    baseline = load_baseline(args.csv)
    lines, failed = report(baseline, root=args.root, check_raw=not args.no_raw_check)
    print("\n".join(lines))

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        build_claims(baseline).to_csv(args.out_dir / "i2_claims.csv", index=False)
        per_method_summary(baseline).to_csv(args.out_dir / "i2_per_method.csv", index=False)
        partition_means(baseline).to_csv(args.out_dir / "i2_partition.csv", index=False)
        print(f"\nwrote i2_claims.csv, i2_per_method.csv, i2_partition.csv -> {args.out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
