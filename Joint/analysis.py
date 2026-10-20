from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "combined_results_long.csv"
OUT_DIR = SCRIPT_DIR / "analysis_outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

BASELINE_METRICS = ["type_i_error", "type_ii_error", "power", "accuracy", "precision", "f1_score"]
FALSECASE_METRICS = ["power_bitflip", "power_asymmetric", "power_depolarize"]
CONFUSION_COLS = ["TP", "TN", "FP", "FN"]
ID_COLS = ["experiment", "case", "n_qubits", "m_samples", "test_method", "measurement_reduction_ratio", "alpha"]

THRESHOLDS = {
    "power_min": 0.95,
    "type_i_max": 0.05,
    "accuracy_min": 0.95,
}


def load_long_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols = ["n_qubits", "m_samples", "measurement_reduction_ratio", "alpha", "value"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pivot_metrics(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    wanted = set(metrics) | set(CONFUSION_COLS)
    filtered = df[df["metric"].isin(wanted)]
    pivot = (
        filtered.pivot_table(index=ID_COLS, columns="metric", values="value", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return pivot


def summarise(df: pd.DataFrame, group_cols: List[str], metrics: Iterable[str]) -> pd.DataFrame:
    agg_map = {m: "mean" for m in metrics}
    return df.groupby(group_cols, as_index=False).agg(agg_map)


def compute_pass_rates(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    pass_mask = (
        (df["power"] >= thresholds["power_min"])
        & (df["type_i_error"] <= thresholds["type_i_max"])
        & (df["accuracy"] >= thresholds["accuracy_min"])
    )
    out = (
        pd.DataFrame({"test_method": df["test_method"], "pass_flag": pass_mask})
        .groupby("test_method", as_index=False)["pass_flag"]
        .mean()
        .rename(columns={"pass_flag": "pass_rate"})
        .sort_values("pass_rate", ascending=False)
    )
    return out


def compute_falsecase_pass_rates(df: pd.DataFrame, power_threshold: float = 0.95) -> pd.DataFrame:
    metrics = [m for m in FALSECASE_METRICS if m in df.columns]
    if not metrics:
        return pd.DataFrame(columns=["test_method", "pass_rate"])
    pass_mask = (df[metrics] >= power_threshold).all(axis=1)
    return (
        pd.DataFrame({"test_method": df["test_method"], "pass_flag": pass_mask})
        .groupby("test_method", as_index=False)["pass_flag"]
        .mean()
        .rename(columns={"pass_flag": "pass_rate"})
        .sort_values("pass_rate", ascending=False)
    )


def min_m_samples_table(df: pd.DataFrame, thresholds: Dict[str, float], by_qubits: bool = True) -> pd.DataFrame:
    grouping = ["test_method", "n_qubits"] if by_qubits else ["test_method"]
    rows = []
    for key, subset in df.groupby(grouping):
        subset = subset.sort_values("m_samples")
        min_m = None
        for m in sorted(subset["m_samples"].unique()):
            tail = subset[subset["m_samples"] >= m]
            if (
                (tail["power"] >= thresholds["power_min"]).all()
                and (tail["type_i_error"] <= thresholds["type_i_max"]).all()
                and (tail["accuracy"] >= thresholds["accuracy_min"]).all()
            ):
                min_m = int(m)
                break
        row = {"test_method": key[0] if by_qubits else key}
        if by_qubits:
            row["n_qubits"] = int(key[1])
        row["min_m_samples"] = min_m
        rows.append(row)
    return pd.DataFrame(rows).sort_values(grouping)


def min_m_falsecase(df: pd.DataFrame, power_threshold: float = 0.95) -> pd.DataFrame:
    metrics = [m for m in FALSECASE_METRICS if m in df.columns]
    rows = []
    for method, subset in df.groupby("test_method"):
        subset = subset.sort_values("m_samples")
        min_m = None
        for m in sorted(subset["m_samples"].unique()):
            tail = subset[subset["m_samples"] >= m]
            if (tail[metrics] >= power_threshold).all(axis=None):
                min_m = int(m)
                break
        rows.append({"test_method": method, "min_m_samples": min_m})
    return pd.DataFrame(rows).sort_values("test_method")


def tolerance_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("test_method")[["power", "type_i_error", "accuracy", "precision", "f1_score"]]
        .mean()
        .reset_index()
    )
    best_power = summary["power"].max()
    summary["power_gap_to_best"] = best_power - summary["power"]
    return summary.sort_values("power", ascending=False)


def plot_metric_vs_samples(df: pd.DataFrame, out_dir: Path, metrics: Iterable[str], title_suffix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = df.groupby(["m_samples", "test_method"])[metric].mean().reset_index()
        for method in grouped["test_method"].unique():
            g = grouped[grouped["test_method"] == method]
            ax.plot(g["m_samples"], g[metric], marker="o", label=method)
        ax.set_xscale("log")
        ax.set_xlabel("m_samples (log scale)")
        ax.set_ylabel(metric)
        title = f"{metric} vs m_samples"
        if title_suffix:
            title = f"{title} {title_suffix}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_vs_m_samples{title_suffix.replace(' ', '_')}.png", dpi=200)
        plt.close(fig)


def plot_falsecase_power(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in FALSECASE_METRICS if m in df.columns]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = df.groupby(["m_samples", "test_method"])[metric].mean().reset_index()
        for method in grouped["test_method"].unique():
            g = grouped[grouped["test_method"] == method]
            ax.plot(g["m_samples"], g[metric], marker="o", label=method)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("m_samples (log scale)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs m_samples (falsecase)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_vs_m_samples_falsecase.png", dpi=200)
        plt.close(fig)

    overall = df.groupby("test_method")[metrics].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(overall["test_method"]))
    width = 0.25
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, overall[metric], width=width, label=metric)
    ax.set_xticks(x + width)
    ax.set_xticklabels(overall["test_method"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean detection power")
    ax.set_title("Falsecase detection power by method and error type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "falsecase_power_bar.png", dpi=200)
    plt.close(fig)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    df_long = load_long_dataframe(DATA_PATH)
    baseline_long = df_long[df_long["case"] == "baseline"]
    falsecase_long = df_long[df_long["case"] == "falsecase"]

    baseline_wide = pivot_metrics(baseline_long, BASELINE_METRICS)
    falsecase_wide = pivot_metrics(falsecase_long, FALSECASE_METRICS)

    overall_summary = summarise(baseline_wide, ["test_method"], BASELINE_METRICS)
    qubit_summary = summarise(baseline_wide, ["n_qubits", "test_method"], BASELINE_METRICS)
    sample_summary = summarise(baseline_wide, ["m_samples", "test_method"], BASELINE_METRICS)
    pass_rates = compute_pass_rates(baseline_wide, THRESHOLDS)
    tol_diag = tolerance_diagnostic(baseline_wide)
    min_m_by_qubit = min_m_samples_table(baseline_wide, THRESHOLDS, by_qubits=True)
    min_m_overall = min_m_samples_table(baseline_wide, THRESHOLDS, by_qubits=False)
    falsecase_summary = summarise(falsecase_wide, ["test_method"], FALSECASE_METRICS)
    falsecase_pass = compute_falsecase_pass_rates(falsecase_wide)
    falsecase_min_m = min_m_falsecase(falsecase_wide)

    save_table(overall_summary, TABLE_DIR / "baseline_summary_by_method.csv")
    save_table(qubit_summary, TABLE_DIR / "baseline_summary_by_qubits.csv")
    save_table(sample_summary, TABLE_DIR / "baseline_summary_by_m_samples.csv")
    save_table(pass_rates, TABLE_DIR / "baseline_pass_rates.csv")
    save_table(tol_diag, TABLE_DIR / "tolerance_diagnostic.csv")
    save_table(min_m_by_qubit, TABLE_DIR / "min_m_samples_by_qubit.csv")
    save_table(min_m_overall, TABLE_DIR / "min_m_samples_overall.csv")
    save_table(falsecase_summary, TABLE_DIR / "falsecase_power_by_method.csv")
    save_table(falsecase_pass, TABLE_DIR / "falsecase_pass_rates.csv")
    save_table(falsecase_min_m, TABLE_DIR / "falsecase_min_m_samples.csv")

    plot_metric_vs_samples(baseline_wide, FIG_DIR / "baseline", ["power", "type_i_error", "accuracy"])
    plot_metric_vs_samples(
        baseline_wide, FIG_DIR / "baseline_by_qubits", ["power", "type_i_error", "accuracy"], title_suffix=" (by qubits)"
    )
    plot_falsecase_power(falsecase_wide, FIG_DIR / "falsecase")

    print(f"Loaded {len(df_long):,} rows from {DATA_PATH.name}")
    print("\nOverall baseline metrics (means):")
    print(overall_summary.to_string(index=False))

    print("\nPass rates against thresholds (power>={THRESHOLDS['power_min']}, type_i<={THRESHOLDS['type_i_max']}, accuracy>={THRESHOLDS['accuracy_min']}):")
    print(pass_rates.to_string(index=False))

    print("\nTolerance vs statistical tests (power gap to best):")
    print(tol_diag.to_string(index=False))

    print("\nMin m_samples required (by qubits):")
    print(min_m_by_qubit.to_string(index=False))

    print("\nMin m_samples required (overall per method):")
    print(min_m_overall.to_string(index=False))

    print(f"\nFalsecase detection power (mean):")
    print(falsecase_summary.to_string(index=False))
    print("\nFalsecase pass rates (power >= 0.95 on all noise types):")
    print(falsecase_pass.to_string(index=False))
    print("\nFalsecase min m_samples to sustain high power across noise types:")
    print(falsecase_min_m.to_string(index=False))

    print(f"\nTables written to {TABLE_DIR}")
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
