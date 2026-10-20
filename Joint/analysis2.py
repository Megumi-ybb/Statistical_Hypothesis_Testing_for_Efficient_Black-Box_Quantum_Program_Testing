from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import (
    BASELINE_METRICS,
    FALSECASE_METRICS,
    THRESHOLDS,
    DATA_PATH,
    FIG_DIR,
    TABLE_DIR,
    load_long_dataframe,
    pivot_metrics,
    summarise,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "analysis_outputs"


def min_m_by_experiment(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    grouping = ["test_method", "experiment"]
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
        rows.append({"test_method": key[0], "experiment": key[1], "min_m_samples": min_m})
    return pd.DataFrame(rows).sort_values(["test_method", "experiment"])


def pass_rate_by_experiment(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    pass_mask = (
        (df["power"] >= thresholds["power_min"])
        & (df["type_i_error"] <= thresholds["type_i_max"])
        & (df["accuracy"] >= thresholds["accuracy_min"])
    )
    return (
        pd.DataFrame({"test_method": df["test_method"], "experiment": df["experiment"], "pass_flag": pass_mask})
        .groupby(["experiment", "test_method"], as_index=False)["pass_flag"]
        .mean()
        .rename(columns={"pass_flag": "pass_rate"})
        .sort_values(["experiment", "test_method"])
    )


def min_m_falsecase_by_experiment(df: pd.DataFrame, power_threshold: float = 0.95) -> pd.DataFrame:
    metrics = [m for m in FALSECASE_METRICS if m in df.columns]
    rows = []
    for key, subset in df.groupby(["test_method", "experiment"]):
        subset = subset.sort_values("m_samples")
        min_m = None
        for m in sorted(subset["m_samples"].unique()):
            tail = subset[subset["m_samples"] >= m]
            if (tail[metrics] >= power_threshold).all(axis=None):
                min_m = int(m)
                break
        rows.append({"test_method": key[0], "experiment": key[1], "min_m_samples": min_m})
    return pd.DataFrame(rows).sort_values(["test_method", "experiment"])


def plot_min_m_heatmap(min_m_df: pd.DataFrame, out_path: Path) -> None:
    methods = sorted(min_m_df["test_method"].unique())
    experiments = sorted(min_m_df["experiment"].unique())
    data = np.full((len(methods), len(experiments)), np.nan)
    for i, method in enumerate(methods):
        for j, exp in enumerate(experiments):
            match = min_m_df[(min_m_df["test_method"] == method) & (min_m_df["experiment"] == exp)]
            if not match.empty and pd.notna(match["min_m_samples"].iloc[0]):
                data[i, j] = match["min_m_samples"].iloc[0]

    fig, ax = plt.subplots(figsize=(1.2 * len(experiments), 4.5))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(experiments)))
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Min m_samples meeting thresholds (per experiment)")
    for i in range(len(methods)):
        for j in range(len(experiments)):
            text = "" if np.isnan(data[i, j]) else f"{int(data[i,j])}"
            ax.text(j, i, text, ha="center", va="center", color="white" if not np.isnan(data[i, j]) else "black")
    fig.colorbar(im, ax=ax, label="m_samples")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_power_curves_by_experiment(df: pd.DataFrame, out_dir: Path, metrics: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for exp, subset in df.groupby("experiment"):
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            grouped = subset.groupby(["m_samples", "test_method"])[metric].mean().reset_index()
            for method in grouped["test_method"].unique():
                g = grouped[grouped["test_method"] == method]
                ax.plot(g["m_samples"], g[metric], marker="o", label=method)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("m_samples (log scale)")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs m_samples ({exp})")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"{exp}_{metric}_vs_m_samples.png", dpi=200)
            plt.close(fig)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    df_long = load_long_dataframe(DATA_PATH)
    baseline_long = df_long[df_long["case"] == "baseline"]
    baseline_wide = pivot_metrics(baseline_long, BASELINE_METRICS)
    falsecase_long = df_long[df_long["case"] == "falsecase"]
    falsecase_wide = pivot_metrics(falsecase_long, FALSECASE_METRICS)

    experiment_summary = summarise(baseline_wide, ["experiment", "test_method"], BASELINE_METRICS)
    experiment_pass = pass_rate_by_experiment(baseline_wide, THRESHOLDS)
    min_m_exp = min_m_by_experiment(baseline_wide, THRESHOLDS)
    falsecase_min_m = min_m_falsecase_by_experiment(falsecase_wide)
    falsecase_summary = summarise(falsecase_wide, ["experiment", "test_method"], FALSECASE_METRICS)

    save_table(experiment_summary, TABLE_DIR / "analysis2_experiment_summary.csv")
    save_table(experiment_pass, TABLE_DIR / "analysis2_pass_rate_by_experiment.csv")
    save_table(min_m_exp, TABLE_DIR / "analysis2_min_m_by_experiment.csv")
    save_table(falsecase_min_m, TABLE_DIR / "analysis2_falsecase_min_m_by_experiment.csv")
    save_table(falsecase_summary, TABLE_DIR / "analysis2_falsecase_power_by_experiment.csv")

    plot_min_m_heatmap(min_m_exp, FIG_DIR / "analysis2" / "min_m_heatmap.png")
    plot_power_curves_by_experiment(baseline_wide, FIG_DIR / "analysis2" / "power_curves", ["power", "type_i_error"])
    plot_power_curves_by_experiment(falsecase_wide, FIG_DIR / "analysis2" / "falsecase_power_curves", FALSECASE_METRICS)

    tolerance_rows = experiment_pass[experiment_pass["test_method"] == "tolerance"]
    worst_tolerance = tolerance_rows["pass_rate"].min()
    best_stats = experiment_pass[experiment_pass["test_method"].isin(["chi_square", "mmdagg"])]["pass_rate"].max()

    print(f"Loaded {len(df_long):,} rows from {DATA_PATH.name}")
    print("\nExperiment-level mean metrics:")
    print(experiment_summary.to_string(index=False))
    print("\nPass rates by experiment (power>={THRESHOLDS['power_min']}, type_i<={THRESHOLDS['type_i_max']}, accuracy>={THRESHOLDS['accuracy_min']}):")
    print(experiment_pass.to_string(index=False))
    print("\nMinimum m_samples per experiment/method:")
    print(min_m_exp.to_string(index=False))
    print("\nFalsecase: min m_samples per experiment/method to keep power>=0.95 across noise types:")
    print(falsecase_min_m.to_string(index=False))
    print("\nFalsecase power (mean) per experiment/method:")
    print(falsecase_summary.to_string(index=False))
    print(f"\nTolerance worst pass_rate: {worst_tolerance:.3f} | Best statistical test pass_rate: {best_stats:.3f}")
    print(f"Tables saved to {TABLE_DIR}, figures to {FIG_DIR/'analysis2'}")


if __name__ == "__main__":
    main()
